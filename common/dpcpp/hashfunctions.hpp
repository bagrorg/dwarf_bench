#include <random>

struct PolynomialHasher {
  PolynomialHasher(size_t sz) {
    _sz = sz;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(0, 13);
    p = possible_p[dist(gen)];
  }

  size_t operator()(const uint32_t &v) const {
    uint32_t v_copy = v;
    int res = 0;
    int pow_p = p;
    while (v_copy > 0) {
      res += ((v_copy % 10) * pow_p) % _sz;
      res = res % _sz;
      pow_p *= p;
      v_copy /= 10;
    }
    return res;
  }

private:
  size_t _sz;
  int p;
  const int possible_p[14] = {2,  3,  5,  7,  11, 13, 17,
                              19, 23, 29, 31, 37, 41, 43};
};

template <size_t Size> struct StaticSimpleHasher {
  size_t operator()(const uint32_t &v) const { return v % Size; }
};

template <size_t Size, size_t Offset> struct StaticSimpleHasherWithOffset {
  size_t operator()(const uint32_t &v) const {
    return (v % Size + Offset) % Size;
  }
};

template <class Integral> struct SimpleHasher {
  SimpleHasher(size_t sz) : _sz(sz) {}
  size_t operator()(const Integral &v) const { return v % _sz; }

private:
  const size_t _sz;
};

struct SimpleHasherWithOffset {
  SimpleHasherWithOffset(size_t sz, size_t offset)
      : _sz(sz), _offset(offset % sz) {}
  size_t operator()(const uint32_t &v) const {
    return (v % _sz + _offset) % _sz;
  }
  size_t get_offset() { return _offset; }

private:
  size_t _sz;
  size_t _offset;
};

struct MurmurHash3_x86_32 {
  MurmurHash3_x86_32(size_t sz, int len, uint32_t seed)
      : _sz(sz), _len(len), _seed(seed) {}

  inline uint32_t rotl32(uint32_t x, int8_t r) const {
    return (x << r) | (x >> (32 - r));
  }

  inline __attribute__((always_inline)) uint32_t getblock32(const uint32_t *p,
                                                            int i) const {
    return p[i];
  }

  inline __attribute__((always_inline)) uint32_t fmix32(uint32_t h) const {
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;

    return h;
  }

  size_t operator()(const uint32_t &v) const {
    const void *key = &v;
    const uint8_t *data = (const uint8_t *)key;
    const int nblocks = _len / 4;

    uint32_t h1 = _seed;

    const uint32_t c1 = 0xcc9e2d51;
    const uint32_t c2 = 0x1b873593;

    const uint32_t *blocks = (const uint32_t *)(data + nblocks * 4);

    for (int i = -nblocks; i; i++) {
      uint32_t k1 = getblock32(blocks, i);

      k1 *= c1;
      k1 = rotl32(k1, 15);
      k1 *= c2;

      h1 ^= k1;
      h1 = rotl32(h1, 13);
      h1 = h1 * 5 + 0xe6546b64;
    }

    const uint8_t *tail = (const uint8_t *)(data + nblocks * 4);

    uint32_t k1 = 0;

    switch (_len & 3) {
    case 3:
      k1 ^= tail[2] << 16;
    case 2:
      k1 ^= tail[1] << 8;
    case 1:
      k1 ^= tail[0];
      k1 *= c1;
      k1 = rotl32(k1, 15);
      k1 *= c2;
      h1 ^= k1;
    };

    h1 ^= _len;
    h1 = fmix32(h1);
    return h1 % _sz;
  }

private:
  size_t _sz;
  int _len;
  uint32_t _seed;
};
