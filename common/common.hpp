#pragma once 

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "dwarf.hpp"
#include "meter.hpp"
#include "options.hpp"

template <class Collection>
void dump_collection(const Collection &c, std::ostream &os = std::cout) {
  bool first = true;
  for (const auto &e : c) {
    if (!first) {
      os << " ";
    }
    os << e;
    first = false;
  }
  os << "\n";
}

namespace helpers {
std::vector<uint32_t> make_unique_random(size_t size);

template <class T> std::vector<T> make_random(size_t size, size_t l = 1, size_t r = 10000) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::vector<T> out(size);
  std::uniform_int_distribution<T> dist(l, r);
  std::generate(out.begin(), out.end(), [&]() { return dist(gen); });
  return out;
}

std::vector<int> make_random_uniform_binary(size_t size);
std::string get_kernels_root_env(const char *argv0);
void set_dpcpp_filter_env_no_overwrite(const char *filter);
void set_dpcpp_filter_env(const RunOptions &opts);

template <typename T, typename U>
bool check_first(const T &v1, const U &v2, size_t sz) {
  for (size_t i = 0; i < sz; ++i) {
    if (v1[i] != v2[i])
      return false;
  }
  return true;
}

} // namespace helpers