#include "slab_hash_build.hpp"
#include "common/dpcpp/slab_hash.hpp"
#include <math.h>

using std::pair;

SlabHashBuild::SlabHashBuild() : Dwarf("SlabHashBuild") {}

void SlabHashBuild::_run(const size_t buf_size, Meter &meter) {
  const int scale = 16; // todo how to get through options

  auto opts = meter.opts();
  const std::vector<uint32_t> host_src =
      helpers::make_unique_random(buf_size);

  auto sel = get_device_selector(opts);
  sycl::queue q{*sel};
  std::cout << "Selected device: "
            << q.get_device().get_info<sycl::info::device::name>() << "\n";

  for (auto it = 0; it < opts.iterations; ++it) {
    int work_size = ceil((float)buf_size / scale);

    sycl::nd_range<1> r{SlabHash::SUBGROUP_SIZE * work_size,
                        SlabHash::SUBGROUP_SIZE};
    SlabHash::AllocAdapter<std::pair<uint32_t, uint32_t>> adap(
        SlabHash::BUCKETS_COUNT, {SlabHash::EMPTY_UINT32_T, 0}, q, work_size);

    std::vector<uint32_t> output(buf_size, 0);
    std::vector<uint32_t> expected(buf_size, 1);

    {
      sycl::buffer<SlabHash::AllocAdapter<std::pair<uint32_t, uint32_t>>>
          adap_buf(&adap, sycl::range<1>{1});
      sycl::buffer<uint32_t> src(host_src);
      std::cout << "STARTED" << std::endl;
      auto host_start = std::chrono::steady_clock::now();

      q.submit([&](sycl::handler &h) {
         auto adap_acc = sycl::accessor(adap_buf, h, sycl::read_write);
         auto s = sycl::accessor(src, h, sycl::read_only);

         h.parallel_for<class slab_hash_build>(r, [=](sycl::nd_item<1> it) [[intel::reqd_sub_group_size(32)]] {
           size_t ind = it.get_group().get_id();

           SlabHash::SlabHashTable<uint32_t, uint32_t,
                                   SlabHash::DefaultHasher<5, 11, 1031>>
               ht(SlabHash::EMPTY_UINT32_T, it, *(adap_acc.get_pointer()));

           for (int i = ind * scale; i < ind * scale + scale && i < buf_size;
                i++) {
             ht.insert(s[i], s[i]);
           }
         });
       })
          .wait();

      auto host_end = std::chrono::steady_clock::now();
      std::cout << "END" << std::endl;
      auto host_exe_time =
          std::chrono::duration_cast<std::chrono::microseconds>(host_end -
                                                                host_start)
              .count();
      Result result;
      result.host_time = host_end - host_start;

      sycl::buffer<uint32_t> out_buf(output);

      q.submit([&](sycl::handler &h) {
         auto adap_acc = sycl::accessor(adap_buf, h, sycl::read_write);
         auto s = sycl::accessor(src, h, sycl::read_only);
         auto o = sycl::accessor(out_buf, h, sycl::read_write);

         h.parallel_for<class slab_hash_build_check>(
             r, [=](sycl::nd_item<1> it) [[intel::reqd_sub_group_size(32)]] {
               size_t ind = it.get_group().get_id();
               SlabHash::DefaultHasher<5, 11, 1031> h;
               SlabHash::SlabHashTable<uint32_t, uint32_t,
                                       SlabHash::DefaultHasher<5, 11, 1031>>
                   ht(SlabHash::EMPTY_UINT32_T, it, *(adap_acc.get_pointer()));

               for (int i = ind * scale;
                    i < ind * scale + scale && i < buf_size; i++) {
                 auto ans = ht.find(s[i]);
                 if (it.get_local_id() == 0) {
                   o[i] = static_cast<bool>(ans);
                 }
               }
             });
       })
          .wait();

      out_buf.get_access<sycl::access::mode::read>();
      if (output != expected) {
        std::cerr << "Incorrect results" << std::endl;
        result.valid = false;
      }

      DwarfParams params{{"buf_size", std::to_string(buf_size)}};
      meter.add_result(std::move(params), std::move(result));
    }
  }
}

void SlabHashBuild::run(const RunOptions &opts) {
  for (auto size : opts.input_size) {
    _run(size, meter());
  }
}
void SlabHashBuild::init(const RunOptions &opts) {
  meter().set_opts(opts);
  DwarfParams params = {{"device_type", to_string(opts.device_ty)}};
  meter().set_params(params);
}
