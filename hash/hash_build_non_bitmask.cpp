#include "hash_build_non_bitmask.hpp"

#include "common/dpcpp/hashtable.hpp"
#include <limits>

HashBuildNonBitmask::HashBuildNonBitmask() : Dwarf("HashBuildNonBitmask") {}
void HashBuildNonBitmask::_run(const size_t buf_size, Meter &meter) {
  auto opts = meter.opts();
  const std::vector<uint32_t> host_src =
      helpers::make_random<uint32_t>(buf_size);
  const uint32_t empty_element = std::numeric_limits<uint32_t>::max();

  auto sel = get_device_selector(opts);
  sycl::queue q{*sel};
  std::cout << "Selected device: "
            << q.get_device().get_info<sycl::info::device::name>() << "\n";

  SimpleHasher<uint32_t> hasher(buf_size);

  for (auto it = 0; it < opts.iterations; ++it) {
    std::vector<uint32_t> data(buf_size, 0);
    std::vector<uint32_t> keys(buf_size, empty_element);
    std::vector<uint32_t> output(buf_size, 0);
    std::vector<uint32_t> expected(buf_size, 1);

    sycl::buffer<uint32_t> data_buf(data);
    sycl::buffer<uint32_t> keys_buf(keys);
    sycl::buffer<uint32_t> src(host_src);

    auto host_start = std::chrono::steady_clock::now();
    q.submit([&](sycl::handler &h) {
       auto s = src.get_access(h);
       auto data_acc = data_buf.get_access(h);
       auto keys_acc = keys_buf.get_access(h);

       h.parallel_for<class hash_build>(buf_size, [=](auto &idx) {
         NonOwningHashTableNonBitmask<uint32_t, uint32_t, SimpleHasher<uint32_t>>
             ht(buf_size, keys_acc.get_pointer(), data_acc.get_pointer(), hasher,
             empty_element);

         ht.insert(s[idx], s[idx]);
       });
     }).wait();

    auto host_end = std::chrono::steady_clock::now();
    auto host_exe_time = std::chrono::duration_cast<std::chrono::microseconds>(
                             host_end - host_start)
                             .count();
    std::unique_ptr<Result> result = std::make_unique<Result>();
    result->host_time = host_end - host_start;

    sycl::buffer<uint32_t> out_buf(output);

    q.submit([&](sycl::handler &h) {
       auto s = src.get_access(h);
       auto o = out_buf.get_access(h);
       auto data_acc = data_buf.get_access(h);
       auto keys_acc = keys_buf.get_access(h);

       h.parallel_for<class hash_build_check>(buf_size, [=](auto &idx) {
         NonOwningHashTableNonBitmask<uint32_t, uint32_t, SimpleHasher<uint32_t>> ht(
             buf_size, keys_acc.get_pointer(), data_acc.get_pointer(), hasher,
             empty_element);

         o[idx] = ht.has(s[idx]);
       });
     }).wait();

    out_buf.get_access<sycl::access::mode::read>();
    if (output != expected) {
      std::cerr << "Incorrect results" << std::endl;
      result->valid = false;
    }

    DwarfParams params{{"buf_size", std::to_string(buf_size)}};
    meter.add_result(std::move(params), std::move(result));
  }
}

void HashBuildNonBitmask::run(const RunOptions &opts) {
  for (auto size : opts.input_size) {
    _run(size, meter());
  }
}
void HashBuildNonBitmask::init(const RunOptions &opts) {
  meter().set_opts(opts);
  DwarfParams params = {{"device_type", to_string(opts.device_ty)}};
  meter().set_params(params);
}
