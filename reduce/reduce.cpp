#include "reduce.hpp"
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <numeric>

#include "common/dpcpp/dpcpp_common.hpp"

namespace {
template <typename T> T expected_out(const std::vector<T> &v) {
  /* addition is not cumulative */
  auto max_T = std::numeric_limits<T>::max();
  auto min_T = std::numeric_limits<T>::min();
  assert(std::all_of(v.begin(), v.end(), [&](T e) {
    return e < max_T / static_cast<int>(v.size());
  }));
  assert(std::all_of(v.begin(), v.end(), [&](T e) {
    return e > min_T / static_cast<int>(v.size());
  }));

  return std::accumulate(v.begin(), v.end(), 0);
}
} // namespace

ReduceDPCPP::ReduceDPCPP() : Dwarf("ReduceDPCPP") {}

void ReduceDPCPP::_run(const size_t buf_size, Meter &meter) {
  auto opts = meter.opts();
  const std::vector<int> host_src = helpers::make_random<int>(buf_size);
  int host_out = 0;
  const int expected = expected_out(host_src);

  auto sel = get_device_selector(opts);
  sycl::queue q{*sel};
  std::cout << "Selected device: "
            << q.get_device().get_info<sycl::info::device::name>() << "\n";

  auto wg_size =
      q.get_device().get_info<sycl::info::device::max_work_group_size>();

  sycl::buffer<int> src(host_src.data(), sycl::range<1>{host_src.size()});
  sycl::buffer<int> out(&host_out, 1);

  auto rng = (buf_size < wg_size) ? sycl::nd_range<1>{buf_size, buf_size}
                                  : sycl::nd_range<1>{buf_size, wg_size};

  for (auto it = 0; it < opts.iterations; ++it) {
    auto host_start = std::chrono::steady_clock::now();

    q.submit([&](sycl::handler &cgh) {
      auto s = src.get_access<sycl::access::mode::read>(cgh);
      auto o = out.get_access<sycl::access::mode::discard_write>(cgh);
      auto reducer =
          sycl::ext::oneapi::reduction(o, sycl::ext::oneapi::plus<>());

      cgh.parallel_for<class dpcreduction>(
          rng, reducer, [=](sycl::nd_item<1> it, auto &reducer_arg) {
            auto gid = it.get_global_id(0);
            reducer_arg += s[gid];
          });
    });

    auto host_end = std::chrono::steady_clock::now();
    auto host_exe_time = std::chrono::duration_cast<std::chrono::microseconds>(
                             host_end - host_start)
                             .count();

    Result result;
    result.host_time = host_end - host_start;
    out.get_access<sycl::access::mode::read>();
    if (expected != host_out) {
      std::cerr << "Incorrect results" << std::endl;
      result.valid = false;
    }
#ifndef NDEBUG
    {
      std::cout << "Input:    ";
      dump_collection(host_src);
      std::cout << "Output:    " << host_out;
      std::cout << std::endl;
      std::cout << "Expected:  " << expected;
      std::cout << std::endl;
    }
#endif
    DwarfParams params{{"buf_size", std::to_string(buf_size)}};
    meter.add_result(std::move(params), std::move(result));
  }
}

void ReduceDPCPP::run(const RunOptions &opts) {
  for (auto size : opts.input_size) {
    _run(size, meter());
  }
}

void ReduceDPCPP::init(const RunOptions &opts) {
  meter().set_opts(opts);
  DwarfParams params = {{"device_type", to_string(opts.device_ty)}};
  meter().set_params(params);

  // workaround for tbb cpu backend
  if (opts.device_ty == RunOptions::CPU) {
    setenv("CL_CONFIG_CPU_FORCE_PRIVATE_MEM_SIZE", "1MB", 0);
  }
}