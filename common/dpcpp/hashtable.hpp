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

  std::pair<uint32_t, bool> insert_group_by(Key key, T val, const sycl::stream & s) {
    uint32_t pos = update_bitmask_group_by(key, s);
    sycl::atomic<uint32_t>(_keys + pos).store(key);
    sycl::atomic<uint32_t>(_vals + pos).fetch_add(val);
    s << key << ' ' << val << ' ' << pos << ' ' << _vals[pos] << ' ' << _hasher(key) << sycl::endl;
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

  uint32_t update_bitmask_group_by(Key key, const sycl::stream & s) {
    uint32_t at = _hasher(key);
    uint32_t major_idx = at / elem_sz;
    uint8_t minor_idx = at % elem_sz;

    while (true) {
      uint32_t mask = uint32_t(1) << minor_idx;
      uint32_t present =
          sycl::atomic<uint32_t>(_bitmask + major_idx).fetch_or(mask);
      s << !(present & mask) << ' ' << key << ' ' << _keys[major_idx * elem_sz + minor_idx] << sycl::endl;
      if (!(present & mask)) {
        return major_idx * elem_sz + minor_idx;
      }
      auto cur_k = sycl::atomic<uint32_t>(_keys + (major_idx * elem_sz + minor_idx)).load(); 
      if (cur_k == key) {
        return major_idx * elem_sz + minor_idx;
      } else {

      }
      
      if (1 + minor_idx == elem_sz) {
        major_idx = (++major_idx) % _size;
        minor_idx = 0;
      } else {
        minor_idx++;
      }
    }
  }
};

template <class Key, class T, class Hash> class SimpleNonOwningHashTableForGroupBy {
public:
  explicit SimpleNonOwningHashTableForGroupBy(size_t size, sycl::global_ptr<Key> keys,
                                    sycl::global_ptr<T> vals,
                                    Hash hash,
                                    Key empty_key)
      : _keys(keys), _vals(vals), _size(size),
        _hasher(hash), _empty_key(empty_key) {}

  bool insert_group_by(Key key, T val) {
    bool pos = update_bitmask_group_by(key, val);
    return pos;
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

private:
  sycl::global_ptr<Key> _keys;
  sycl::global_ptr<T> _vals;
  size_t _size;
  Hash _hasher;
  Key _empty_key;

  static constexpr uint32_t elem_sz = CHAR_BIT * sizeof(uint32_t);

  bool update_bitmask_group_by(Key key, T val) {
    uint32_t at = _hasher(key);

    while (true) {
      Key expected_key = _empty_key;
      bool success =
          sycl::atomic<uint32_t>(_keys + at).compare_exchange_strong(expected_key, key);
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
};