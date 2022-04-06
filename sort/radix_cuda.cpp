#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/iterator>

#include "sort/radix.hpp"

#include "common/dpcpp/dpcpp_common.hpp"

namespace {
template <typename T> std::vector<T> expected_out(const std::vector<T> &v) {
  std::vector<int> out = v;
  std::sort(out.begin(), out.end());
  return out;
}
} // namespace

RadixCuda::RadixCuda() : Dwarf("RadixCuda") {}

void RadixCuda::_run(const size_t buf_size, Meter &meter) {
  auto opts = meter.opts();
  const std::vector<int> host_src = helpers::make_random<int>(buf_size);
  const std::vector<int> expected = expected_out(host_src);

  auto sel = get_device_selector(opts);
  sycl::queue q{*sel};
  std::cout << "Selected device: "
            << q.get_device().get_info<sycl::info::device::name>() << "\n";

  auto dev_policy = oneapi::dpl::execution::device_policy{*sel};

  for (auto it = 0; it < opts.iterations; ++it) {
    sycl::buffer<int> src(host_src.data(), sycl::range<1>{buf_size});

    auto host_start = std::chrono::steady_clock::now();
    std::sort(dev_policy, oneapi::dpl::begin(src), oneapi::dpl::end(src));
    auto host_end = std::chrono::steady_clock::now();
    auto host_exe_time = std::chrono::duration_cast<std::chrono::microseconds>(
                             host_end - host_start)
                             .count();
#ifndef NDEBUG
    {
      sycl::host_accessor res(src, sycl::read_only);
      std::cout << "Input:    ";
      dump_collection(host_src);
      std::cout << "Output:    ";
      for (int i = 0; i < expected.size(); ++i) {
        std::cout << res[i] << " ";
      }
      std::cout << std::endl;
      std::cout << "Expected:  ";
      dump_collection(expected);
    }
#endif
    std::unique_ptr<Result> result = std::make_unique<Result>();
    result->host_time = host_end - host_start;

    {
      sycl::host_accessor res(src, sycl::read_only);
      if (!helpers::check_first(res, expected, expected.size())) {
        std::cerr << "incorrect results" << std::endl;
        result->valid = false;
      }
    }

    DwarfParams params{{"buf_size", std::to_string(buf_size)}};
    meter.add_result(std::move(params), std::move(result));
  }
}

void RadixCuda::run(const RunOptions &opts) {
  for (auto size : opts.input_size) {
    _run(size, meter());
  }
}

void RadixCuda::init(const RunOptions &opts) {
  meter().set_opts(opts);
  DwarfParams params = {{"device_type", to_string(opts.device_ty)}};
  meter().set_params(params);
}