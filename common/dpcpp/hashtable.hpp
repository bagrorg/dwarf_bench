#pragma once
#include "dpcpp_common.hpp"
#include "hashfunctions.hpp"

template <class Key, class T, class Hash> class SimpleNonOwningHashTable {
public:
  explicit SimpleNonOwningHashTable(size_t size, sycl::global_ptr<Key> keys,
                                    sycl::global_ptr<T> vals,
                                    sycl::global_ptr<uint32_t> bitmask,
                                    Hash hash)
      : _keys(keys), _vals(vals), _bitmask(bitmask), _size(size),
        _hasher(hash) {}

  std::pair<uint32_t, bool> insert(Key key, T val) {
    uint32_t pos = update_bitmask(_hasher(key));
    _keys[pos] = key;
    _vals[pos] = val;
    // todo
    return {pos, true};
  }

  const std::pair<T, bool> at(const Key &key) const {
    uint32_t pos = _hasher(key);
    const auto start = pos;
    bool present = (_bitmask[pos / elem_sz] & (uint32_t(1) << pos % elem_sz));
    while (present) {
      if (_keys[pos] == key) {
        return {_vals[pos], true};
      }

      pos = (++pos) % _size;
      if (pos == start)
        break;

      present = (_bitmask[pos / elem_sz] & (uint32_t(1) << pos % elem_sz));
    }

    return {{}, false};
  }

  bool has(const Key &key) const {
    uint32_t pos = _hasher(key);
    const auto start = pos;
    bool present = (_bitmask[pos / elem_sz] & (uint32_t(1) << pos % elem_sz));
    while (present) {
      if (_keys[pos] == key)
        return true;

      pos = (++pos) % _size;
      if (pos == start)
        break;

      present = (_bitmask[pos / elem_sz] & (uint32_t(1) << pos % elem_sz));
    }

    return false;
  }

private:
  sycl::global_ptr<Key> _keys;
  sycl::global_ptr<T> _vals;
  sycl::global_ptr<uint32_t> _bitmask;
  size_t _size;
  Hash _hasher;

  static constexpr uint32_t elem_sz = CHAR_BIT * sizeof(uint32_t);

  uint32_t update_bitmask(uint32_t at) {
    uint32_t major_idx = at / elem_sz;
    uint8_t minor_idx = at % elem_sz;

    while (true) {
      uint32_t mask = uint32_t(1) << minor_idx;
      uint32_t present =
          sycl::atomic<uint32_t>(_bitmask + major_idx).fetch_or(mask);
      if (!(present & mask)) {
        return major_idx * elem_sz + minor_idx;
      }

      minor_idx++;
      uint32_t occupied = sycl::intel::ctz<uint32_t>(~(present >> minor_idx));
      if (occupied + minor_idx == elem_sz) {
        major_idx = (++major_idx) % _size;
        minor_idx = 0;
      } else {
        minor_idx += occupied;
      }
    }
  }
};

template <class Key, class T, class Hash> class NonOwningHashTableNonBitmask {
public:
  explicit NonOwningHashTableNonBitmask(size_t size, sycl::global_ptr<Key> keys,
                                        sycl::global_ptr<T> vals, Hash hash,
                                        Key empty_key)
      : _keys(keys), _vals(vals), _size(size), _hasher(hash),
        _empty_key(empty_key) {}

  bool add(Key key, T val) {
    bool success = add_update(key, val);
    return success;
  }

  bool insert(Key key, T val) {
    bool success = insert_update(key, val);
    return success;
  }

  const std::pair<T, bool> at(const Key &key) const {
    uint32_t pos = _hasher(key);
    bool present = !(_keys[pos] == _empty_key);
    while (present) {
      if (_keys[pos] == key) {
        return {_vals[pos], true};
      }

      pos = (++pos) % _size;
      if (pos == _hasher(key))
        break;

      present = !(_keys[pos] == _empty_key);
    }

    return {{}, false};
  }

  bool has(const Key &key) const {
    uint32_t pos = _hasher(key);
    const auto start = pos;
    bool present = !(_keys[pos] == _empty_key);
    while (present) {
      if (_keys[pos] == key)
        return true;

      pos = (++pos) % _size;
      if (pos == start)
        break;

      present = !(_keys[pos] == _empty_key);
    }

    return false;
  }

private:
  sycl::global_ptr<Key> _keys;
  sycl::global_ptr<T> _vals;
  size_t _size;
  Hash _hasher;
  Key _empty_key;

  static constexpr uint32_t elem_sz = CHAR_BIT * sizeof(uint32_t);

  bool add_update(Key key, T val) {
    uint32_t at = _hasher(key);

    while (true) {
      Key expected_key = _empty_key;
      bool success = sycl::atomic<uint32_t>(_keys + at)
                         .compare_exchange_strong(expected_key, key);
      if (success || expected_key == key) {
        sycl::atomic<uint32_t>(_vals + at).fetch_add(val);
        return true;
      }

      at = (at + 1) % _size;
      if (at == _hasher(key)) {
        return false;
      }
    }
  }

  bool insert_update(Key key, T val) {
    uint32_t at = _hasher(key);

    while (true) {
      Key expected_key = _empty_key;
      bool success = sycl::atomic<uint32_t>(_keys + at)
                         .compare_exchange_strong(expected_key, key);
      if (success || expected_key == key) {
        sycl::atomic<uint32_t>(_vals + at).store(val);
        return true;
      }

      at = (at + 1) % _size;
      if (at == _hasher(key)) {
        return false;
      }
    }
  }
};