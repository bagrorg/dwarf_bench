#pragma once

#include "dpcpp_common.hpp"
#include <CL/sycl.hpp>
#include <algorithm>
#include <math.h>
#include <optional>

namespace SlabHash {
constexpr size_t SUBGROUP_SIZE = 32;
constexpr size_t SLAB_SIZE_MULTIPLIER = 8;
constexpr size_t SLAB_SIZE = SLAB_SIZE_MULTIPLIER * SUBGROUP_SIZE;

constexpr size_t CLUSTER_SIZE = 4192000;

constexpr size_t BUCKETS_COUNT = 1024;

constexpr size_t EMPTY_UINT32_T = std::numeric_limits<uint32_t>::max();

template <size_t A, size_t B, size_t P> struct DefaultHasher {
  size_t operator()(const uint32_t &k) {
    return ((A * k + B) % P) % BUCKETS_COUNT;
  };
};

template <typename T> struct SlabNode {
  SlabNode(T el) {
    for (int i = 0; i < SLAB_SIZE; i++) {
      data[i] = el;
    }
  }

  T data[SLAB_SIZE];
  sycl::device_ptr<SlabNode<T>> next = nullptr;
};

template <typename T> struct SlabList {
  SlabList() = default;

  sycl::device_ptr<SlabNode<T>> root;
};

template <typename T> struct HeapMaster {
  HeapMaster(size_t buckets_count, sycl::queue &q) : _q(q) {
    list_group_size = buckets_count / num_of_list_groups; //mb problems
    memory_for_group = CLUSTER_SIZE / num_of_list_groups;
    _heap = sycl::malloc_device<SlabNode<T>>(CLUSTER_SIZE, q);
    for (int i = 0; i < num_of_list_groups; i++) {
        _head[i] = (_heap + memory_for_group * i);
    }
    sycl::device_ptr<uint32_t> tmp_lock = sycl::malloc_device<uint32_t>(num_of_list_groups, q);

    q.parallel_for(sycl::range<1> {num_of_list_groups}, [=](auto &i) { *(tmp_lock + i) = 0; });

    _lock = tmp_lock;
  }

  ~HeapMaster() { sycl::free(_heap, _q); }

  sycl::device_ptr<SlabNode<T>> malloc_node(size_t i) {
    int ind = i / list_group_size;
    sycl::device_ptr<SlabNode<T>> ret;
    while (sycl::atomic<uint32_t,
                        sycl::access::address_space::global_device_space>(_lock + ind)
               .fetch_or(1)) {
    }
    ret = _head[ind];
    _head[ind]++;
    sycl::atomic<uint32_t, sycl::access::address_space::global_device_space>(
        _lock + ind)
        .fetch_and(0);
    return ret;
  }

  sycl::device_ptr<uint32_t> _lock;
  static constexpr size_t num_of_list_groups = 1024;
  size_t list_group_size;
  size_t memory_for_group;
  sycl::device_ptr<SlabNode<T>> _heap;
  sycl::device_ptr<SlabNode<T>> _head[num_of_list_groups];
  sycl::queue &_q;
};
template <typename T> struct AllocAdapter {
  AllocAdapter(size_t bucket_size, T empty, sycl::queue &q, int work_size) : _q(q), _heap(bucket_size, q) {
    sycl::device_ptr<SlabList<T>> _data_tmp = sycl::malloc_device<SlabList<T>>(bucket_size, q);
    sycl::device_ptr<uint32_t> _lock_tmp = sycl::malloc_device<uint32_t>(ceil((float)bucket_size / sizeof(uint32_t)), q);
    _its = sycl::malloc_device<sycl::device_ptr<SlabHash::SlabNode<std::pair<uint32_t, uint32_t>>>>(work_size, q);

    q.parallel_for(bucket_size, [=](auto &i) {
      *(_data_tmp + i) = SlabList<T>();
      *(_lock_tmp + i) = 0;
    });

    _data = _data_tmp;
    _lock = _lock_tmp;
  }

  ~AllocAdapter() {
    sycl::free(_data, _q);
    sycl::free(_lock, _q);
  }

  sycl::device_ptr<SlabList<T>> _data;
  sycl::device_ptr<uint32_t> _lock;
  HeapMaster<T> _heap;
  sycl::device_ptr<sycl::device_ptr<SlabHash::SlabNode<std::pair<uint32_t, uint32_t>>>>
          _its;
  sycl::queue &_q;
};

template <typename K, typename T, typename Hash> class SlabHashTable {
public:
  SlabHashTable() = default;
  SlabHashTable(K empty,
                sycl::nd_item<1> &it,
                SlabHash::AllocAdapter<std::pair<K, T>> &adap)
      : _lists(adap._data), _gr(it.get_sub_group()), _it(it), _empty(empty),
         _iter(adap._its[it.get_group().get_id()]), _ind(_it.get_local_id()), _lock(adap._lock),
        _heap(adap._heap){};

  void insert(K key, T val) {
    _key = key;
    _val = val;

    if (_ind == 0) {
      if ((_lists + _hasher(key))->root == nullptr) {
        alloc_node((_lists + _hasher(key))->root);
      } else {
        _iter = (_lists + _hasher(key))->root;
      }
    }
    sycl::group_barrier(_gr);

    while (1) {
      while (_iter != nullptr) {
        if (insert_in_node()) {
          return;
        } else if (_ind == 0) {
          _prev = _iter;
          _iter = _iter->next;
        }

        sycl::group_barrier(_gr);
      }
      if (_ind == 0)
        alloc_node(_prev->next);
      sycl::group_barrier(_gr);
    }
  }

  std::optional<T> find(K key) {
    _key = key;
    _ans = std::nullopt;

    if (_ind == 0) {
      _iter = (_lists + _hasher(key))->root;
    }
    sycl::group_barrier(_gr);

    while (_iter != nullptr) {
      if (find_in_node()) {
        break;
      } else if (_ind == 0) {
        _iter = _iter->next;
      }

      sycl::group_barrier(_gr);
    }
    return _ans;
  }

private:
  void alloc_node(sycl::device_ptr<SlabNode<std::pair<K, T>>> &src) {
    lock();
    if (src == nullptr) {
      auto tmp = _heap.malloc_node(_hasher(_key));
      *tmp = SlabNode<std::pair<K, T>>({_empty, T()}); //!!!!!!
      src = tmp;
    }
    unlock();
    _iter = src;
  }

  void lock() {
    auto tmp = _hasher(_key);
    int i = 0;
    while (
        sycl::atomic<uint32_t, sycl::access::address_space::global_device_space>((_lock + (tmp / (CHAR_BIT * sizeof(uint32_t)))))
            .fetch_or(1 << (tmp % (CHAR_BIT * sizeof(uint32_t)))) &
        (1 << (tmp % (CHAR_BIT * sizeof(uint32_t))))) {
    }
  }

  void unlock() {
    auto tmp = _hasher(_key);
    sycl::atomic<uint32_t, sycl::access::address_space::global_device_space>((_lock + (tmp / (CHAR_BIT * sizeof(uint32_t)))))
        .fetch_and(~(1 << (tmp % (CHAR_BIT * sizeof(uint32_t)))));
  }

  bool insert_in_node() {
    bool total_found = false;
    bool find = false;

    for (int i = _ind; i <= _ind + SUBGROUP_SIZE * (SLAB_SIZE_MULTIPLIER - 1);
         i += SUBGROUP_SIZE) {
      find = ((_iter->data[i].first) == _empty);
      sycl::group_barrier(_gr);
      total_found = sycl::any_of_group(_gr, find);

      if (total_found) {
        if (insert_in_subgroup(find, i)) {
          return true;
        }
      }
    }

    return false;
  }

  bool insert_in_subgroup(bool find, int i) {
    for (int j = 0; j < SUBGROUP_SIZE; j++) {
      if (cl::sycl::group_broadcast(_gr, find, j)) {
        K tmp_empty = _empty;
        bool done = _ind == j
                        ? sycl::ext::oneapi::atomic_ref<
                              K, sycl::ext::oneapi::memory_order::acq_rel,
                              sycl::ext::oneapi::memory_scope::device,
                              sycl::access::address_space::global_device_space>(
                              _iter->data[i].first)
                              .compare_exchange_strong(tmp_empty, _key)
                        : false;
        sycl::group_barrier(_gr);
        if (done) {
          _iter->data[i].second = _val;
        }
        if (cl::sycl::group_broadcast(_gr, done, j)) {
          return true;
        }
      }
    }

    return false;
  }

  bool find_in_node() {
    bool find = false;
    bool total_found = false;

    for (int i = _ind; i <= _ind + SUBGROUP_SIZE * (SLAB_SIZE_MULTIPLIER - 1);
         i += SUBGROUP_SIZE) {
      find = ((_iter->data[i].first) == _key);
      sycl::group_barrier(_gr);
      total_found = sycl::any_of_group(_gr, find);

      if (total_found) {
        find_in_subgroup(find, i);
        return true;
      }
    }

    return false;
  }

  void find_in_subgroup(bool find, int i) {
    for (int j = 0; j < SUBGROUP_SIZE; j++) {
      if (cl::sycl::group_broadcast(_gr, find, j)) {
        T tmp;
        if (_ind == j)
          tmp = _iter->data[i].second; // todo index shuffle

        _ans = std::optional<T>{cl::sycl::group_broadcast(_gr, tmp, j)};
        break;
      }
    }
  }

  sycl::device_ptr<SlabList<std::pair<K, T>>> _lists;
  sycl::device_ptr<uint32_t> _lock;
  sycl::device_ptr<SlabNode<std::pair<K, T>>> &_iter;
  sycl::device_ptr<SlabNode<std::pair<K, T>>> _prev;
  HeapMaster<std::pair<K, T>> &_heap;
  sycl::sub_group _gr;
  sycl::nd_item<1> &_it;
  size_t _ind;

  K _empty;
  Hash _hasher;

  K _key;
  T _val;

  std::optional<T> _ans;
};

} // namespace SlabHash