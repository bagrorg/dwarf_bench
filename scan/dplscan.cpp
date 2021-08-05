#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/iterator>

#include "scan/scan.hpp"
#include <functional>
#include <iostream>

#include "common/dpcpp/dpcpp_common.hpp"

namespace {
template <typename T> using Func = std::function<bool(T)>;

template <typename T>
std::vector<T> expected_out(const std::vector<T> &v, Func<T> f) {
  std::vector<int> out;
  std::copy_if(v.begin(), v.end(), std::back_inserter(out), f);
  return out;
}
} // namespace

DPLScan::DPLScan() : Dwarf("DPLScan") {}

void DPLScan::run_scan(const size_t buf_size, Meter &meter) {
  auto opts = meter.opts();
  const int buffer_size = buf_size;
  const std::vector<int> host_src = helpers::make_random<int>(buffer_size);

  std::vector<int> expected =
      expected_out<int>(host_src, [](int x) { return x < 5; });

  auto sel = get_device_selector(opts);
  sycl::queue q{*sel};
  std::cout << "Selected device: "
            << q.get_device().get_info<sycl::info::device::name>() << "\n";
  auto dev_policy =
      oneapi::dpl::execution::device_policy<class Dev_Policy_Kernel>{*sel};

  for (auto it = 0; it < opts.iterations; ++it) {
    sycl::buffer<int> src_buf(host_src.data(), sycl::range<1>{buf_size});
    sycl::buffer<int> out_buf{sycl::range<1>{buf_size}};

    auto host_start = std::chrono::steady_clock::now();

    auto end_it = std::copy_if(
        dev_policy, oneapi::dpl::begin(src_buf), oneapi::dpl::end(src_buf),
        oneapi::dpl::begin(out_buf), [](auto &x) { return x < 5; });

    auto host_end = std::chrono::steady_clock::now();
    auto host_exe_time = std::chrono::duration_cast<std::chrono::microseconds>(
                             host_end - host_start)
                             .count();
#ifndef NDEBUG
    {
      sycl::host_accessor res(out_buf, sycl::read_only);
      std::cout << "Input:    ";
      dump_collection(host_src);
      std::cout << "Output:    ";
      for (int i = 0; i < expected.size(); ++i) {
        std::cout << res[i] << " ";
      }
      std::cout << std::endl;
      std::cout << "Expected: ";
      dump_collection(expected);
    }
#endif
    Result result;
    result.host_time = host_end - host_start;
    DwarfParams params{{"buf_size", std::to_string(buffer_size)}};
    meter.add_result(std::move(params), std::move(result));

    {
      sycl::host_accessor res(out_buf, sycl::read_only);
      if (!helpers::check_first(res, expected, expected.size())) {
        std::cerr << "incorrect results" << std::endl;
        result.valid = false;
      }
    }
  }
}

void DPLScan::run(const RunOptions &opts) {
  for (auto size : opts.input_size) {
    run_scan(size, meter());
  }
}

void DPLScan::init(const RunOptions &opts) {
  meter().set_opts(opts);
  DwarfParams params = {{"device_type", to_string(opts.device_ty)}};
  meter().set_params(params);
}