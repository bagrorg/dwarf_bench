#include "common.hpp"
#include "boost/dll.hpp"
#include <cstdlib>

namespace helpers {
std::vector<uint32_t> make_unique_random(size_t size) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dist(1, std::min((long) size * 10, (long) ((uint32_t) -1)));

  std::set<uint32_t> s;
  while (s.size() < size) {
    s.insert(dist(gen) % (size * 10));
  }
  return std::vector<uint32_t>(s.begin(), s.end());
}

std::vector<int> make_random_uniform_binary(size_t size) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::vector<int> out(size);
  std::uniform_int_distribution<int> dist(0, 1);
  std::generate(out.begin(), out.end(), [&]() { return dist(gen); });
  return out;
}

std::string get_kernels_root_env(const char *argv0) {
  auto *val = std::getenv("DWARF_BENCH_ROOT");
  return val ? val : boost::dll::program_location().parent_path().c_str();
}

void set_dpcpp_filter_env(const RunOptions &opts) {
  switch (opts.device_ty) {
  case RunOptions::DeviceType::GPU:
    set_dpcpp_filter_env_no_overwrite("cuda");
    break;

  default:
    break;
  }
}

void set_dpcpp_filter_env_no_overwrite(const char *filter) {
  setenv("SYCL_DEVICE_FILTER", filter, 0);
}
} // namespace helpers