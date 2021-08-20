#pragma once

#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/iterator>

#include "dpcpp_common.hpp"
#include <CL/sycl.hpp>
#include <algorithm>
#include <cmath>
#include <optional>

namespace SlabHash {
using sycl::access::address_space::global_device_space;
using sycl::ext::oneapi::memory_order::acq_rel;
using sycl::ext::oneapi::memory_scope::device;

template <typename K>
using atomic_ref_device =
    sycl::ext::oneapi::atomic_ref<K, acq_rel, device, global_device_space>;

constexpr size_t UINT32_T_BIT = CHAR_BIT * sizeof(uint32_t);

constexpr size_t SUBGROUP_SIZE = 16;
constexpr size_t SLAB_SIZE_MULTIPLIER = 16;
constexpr size_t SLAB_SIZE = SLAB_SIZE_MULTIPLIER * SUBGROUP_SIZE;

constexpr size_t CLUSTER_SIZE = 20480;

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

namespace detail {
template <typename T> struct HeapMaster {
  HeapMaster(size_t cluster_size, sycl::queue &q) : _q(q) {
    _heap = sycl::malloc_device<SlabNode<T>>(cluster_size, q);
    _head = _heap;
    sycl::device_ptr<uint32_t> tmp_lock = sycl::malloc_device<uint32_t>(1, q);

    q.single_task([=]() { *tmp_lock = 0; });

    _lock = tmp_lock;
  }

  ~HeapMaster() { sycl::free(_heap, _q); }

  sycl::device_ptr<SlabNode<T>> malloc_node() {
    sycl::device_ptr<SlabNode<T>> ret;
    while (sycl::atomic<uint32_t,
                        sycl::access::address_space::global_device_space>(_lock)
               .fetch_or(1)) {
    }
    ret = _head;
    _head++;
    sycl::atomic<uint32_t, sycl::access::address_space::global_device_space>(
        _lock)
        .fetch_and(0);
    return ret;
  }

  sycl::device_ptr<uint32_t> _lock = nullptr;
  sycl::device_ptr<SlabNode<T>> _heap;
  sycl::device_ptr<SlabNode<T>> _head;
  sycl::queue &_q;
};
} // namespace detail

template <typename T> struct AllocAdapter {
  AllocAdapter(size_t cluster_size, size_t work_size, size_t bucket_size,
               T empty, sycl::queue &q)
      : _q(q), _heap(cluster_size, q) {
    sycl::device_ptr<SlabList<T>> _data_tmp =
        sycl::malloc_device<SlabList<T>>(bucket_size, q);
    sycl::device_ptr<uint32_t> _lock_tmp = sycl::malloc_device<uint32_t>(
        ceil((float)bucket_size / sizeof(uint32_t)), q);
    _its = sycl::malloc_device<
        sycl::device_ptr<SlabHash::SlabNode<std::pair<uint32_t, uint32_t>>>>(
        work_size, q);
    v = sycl::malloc_device<sycl::vec<uint8_t, 16>>(work_size, q);

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
    sycl::free(v, _q);
  }

  sycl::device_ptr<SlabList<T>> _data;
  sycl::device_ptr<uint32_t> _lock;
  detail::HeapMaster<T> _heap;
  sycl::device_ptr<
      sycl::device_ptr<SlabHash::SlabNode<std::pair<uint32_t, uint32_t>>>>
      _its;
  sycl::device_ptr<sycl::vec<uint8_t, 16>> v;
  sycl::queue &_q;
};

template <typename K, typename T, typename Hash> class SlabHashTable {
public:
  SlabHashTable() = default;
  SlabHashTable(K empty, sycl::nd_item<1> &it,
                SlabHash::AllocAdapter<std::pair<K, T>> &adap)
      : _lists(adap._data), _gr(it.get_sub_group()), _it(it), _empty(empty),
        _iter(adap._its[it.get_group().get_id()]), _ind(_it.get_local_id()),
        _lock(adap._lock), _heap(adap._heap), _v(adap.v + it.get_group().get_id()) {};

  void insert(K key, T val) {
    _key = key;
    _val = val;

    if (_ind == 0) {
      if ((_lists + _hasher(key))->root == nullptr) {
        alloc_node((_lists + _hasher(key))->root);
      }
      _iter = (_lists + _hasher(key))->root;
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
      if (_ind == 0) {
        alloc_node(_prev->next);
        _iter = _prev->next;
      }

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
      auto allocated_pointer = _heap.malloc_node();
      *allocated_pointer = SlabNode<std::pair<K, T>>({_empty, T()}); //!!!!!!

      src = allocated_pointer;
    }
    unlock();
  }

  void lock() {
    auto list_index = _hasher(_key);
    while (sycl::atomic<uint32_t,
                        sycl::access::address_space::global_device_space>(
               (_lock + (list_index / (UINT32_T_BIT))))
               .fetch_or(1 << (list_index % (UINT32_T_BIT))) &
           (1 << (list_index % (UINT32_T_BIT)))) {
    }
  }

  void unlock() {
    auto list_index = _hasher(_key);
    sycl::atomic<uint32_t, sycl::access::address_space::global_device_space>(
        (_lock + (list_index / (UINT32_T_BIT))))
        .fetch_and(~(1 << (list_index % (UINT32_T_BIT))));
  }

  bool insert_in_node() {
    bool total_found = true;

    for (int i = _ind; i < SUBGROUP_SIZE * SLAB_SIZE_MULTIPLIER;
         i += SUBGROUP_SIZE) {
      (*_v)[_ind] = ((_iter->data[i].first) == _empty);
      sycl::group_barrier(_gr);
      uint8_t find = sycl::inclusive_scan_over_group(_gr, (*_v)[_ind],
                                                       sycl::plus<uint8_t>());
      int idx = 1;
      while (sycl::any_of_group(_gr, find == idx)) {
        bool done = false;
        if (find == idx)
          done = insert_in_subgroup(i);
        if (cl::sycl::any_of_group(_gr, done)) {
          return true;
        }
        idx++;
      }
    }

    return false;
  }

  bool insert_in_subgroup(int i) {

    K tmp_empty = _empty;
    bool done = atomic_ref_device<K>(_iter->data[i].first)
                    .compare_exchange_strong(tmp_empty, _key);
    if (done) {
      _iter->data[i].second = _val;
      return true;
    }

    return false;
  }

  bool find_in_node() {
    sycl::multi_ptr<T, sycl::access::address_space::local_space> _data_ans = sycl::group_local_memory<T>(_it.get_group());
    bool total_found = true;

    for (int i = _ind; i < SUBGROUP_SIZE * SLAB_SIZE_MULTIPLIER;
         i += SUBGROUP_SIZE) {

      
        (*_v)[_ind] = ((_iter->data[i].first) == _key);
        sycl::group_barrier(_gr);
        uint8_t find = sycl::inclusive_scan_over_group(_gr, (*_v)[_ind],
                                                       sycl::plus<uint8_t>());
        sycl::group_barrier(_gr);
        if (sycl::any_of_group(_gr, find == 1)) {
          if (find == 1 && (*_v)[_ind] == 1) {
            *_data_ans = _iter->data[i].second;
    
          }
          _ans = std::optional<T>{*_data_ans};
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
  detail::HeapMaster<std::pair<K, T>> &_heap;
  sycl::sub_group _gr;
  sycl::nd_item<1> &_it;
  size_t _ind;

  K _empty;
  Hash _hasher;

  K _key;
  T _val;

  std::optional<T> _ans;

  sycl::device_ptr<sycl::vec<uint8_t, 16>> _v;
};

} // namespace SlabHash