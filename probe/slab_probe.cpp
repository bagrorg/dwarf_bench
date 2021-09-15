#include "slab_probe.hpp"
#include "common/dpcpp/slab_hash.hpp"
#include <math.h>

using std::pair;

SlabProbe::SlabProbe() : Dwarf("SlabProbe") {}

void SlabProbe::_run(const size_t buf_size, Meter &meter) {

  auto opts = meter.opts();
  const int scale = opts.scale;
  const std::vector<uint32_t> host_src = helpers::make_unique_random(buf_size);

  auto sel = get_device_selector(opts);
  sycl::queue q{*sel};
  std::cout << "Selected device: "
            << q.get_device().get_info<sycl::info::device::name>() << "\n";

  for (auto it = 0; it < opts.iterations; ++it) {
    int num_of_groups = ceil((float)buf_size / scale);

    sycl::nd_range<1> r{SlabHash::SUBGROUP_SIZE * num_of_groups,
                        SlabHash::SUBGROUP_SIZE};
    SlabHash::AllocAdapter<std::pair<uint32_t, uint32_t>> adap(
        SlabHash::CLUSTER_SIZE, num_of_groups, opts.buckets_count,
        {SlabHash::EMPTY_UINT32_T, 0}, q);

    std::vector<uint32_t> output(buf_size, 0);
    std::vector<uint32_t> expected(buf_size, 1);

    {
      sycl::buffer<SlabHash::AllocAdapter<std::pair<uint32_t, uint32_t>>>
          adap_buf(&adap, sycl::range<1>{1});
      sycl::buffer<uint32_t> src(host_src);

      q.submit([&](sycl::handler &h) {
         auto s = sycl::accessor(src, h, sycl::read_only);

         auto adap_acc = sycl::accessor(adap_buf, h, sycl::read_write);

         h.parallel_for<class slab_hash_build>(
             r, [=](sycl::nd_item<1> it)[
                    [intel::reqd_sub_group_size(SlabHash::SUBGROUP_SIZE)]] {
               size_t ind = it.get_group().get_id();

               SlabHash::SlabHashTable<uint32_t, uint32_t,
                                       SlabHash::DefaultHasher<242792922, 653019598, 2147483647>>
                   ht(SlabHash::EMPTY_UINT32_T, it, *adap_acc.get_pointer());

               for (int i = ind * scale; i < (ind + 1) * scale && i < buf_size;
                    i++) {
                 ht.insert(s[i], s[i]);
               }
             });
       }).wait();

      sycl::buffer<uint32_t> out_buf(output);
      auto host_start = std::chrono::steady_clock::now();
      q.submit([&](sycl::handler &h) {
         auto s = sycl::accessor(src, h, sycl::read_only);
         auto o = sycl::accessor(out_buf, h, sycl::read_write);
         auto adap_acc = sycl::accessor(adap_buf, h, sycl::read_write);

         h.parallel_for<class slab_hash_build_check>(
             r, [=](sycl::nd_item<1> it)[
                    [intel::reqd_sub_group_size(SlabHash::SUBGROUP_SIZE)]] {
               size_t ind = it.get_group().get_id();

               SlabHash::SlabHashTable<uint32_t, uint32_t,
                                       SlabHash::DefaultHasher<242792922, 653019598, 2147483647>>
                   ht(SlabHash::EMPTY_UINT32_T, it, *adap_acc.get_pointer());

               for (int i = ind * scale; i < (ind + 1) * scale && i < buf_size;
                    i++) {
                 auto ans = ht.find(s[i]);
                 if (it.get_local_id() == 0) {
                   o[i] = static_cast<bool>(ans);
                 }
               }
             });
       }).wait();

      auto host_end = std::chrono::steady_clock::now();
      auto host_exe_time =
          std::chrono::duration_cast<std::chrono::microseconds>(host_end -
                                                                host_start)
              .count();
      std::unique_ptr<Result> result = std::make_unique<Result>();
      result->host_time = host_end - host_start;
      out_buf.get_access<sycl::access::mode::read>();
      double memory_util = (double) sizeof(uint32_t) * 2 * buf_size / (adap._heap._count * (sizeof(SlabHash::SlabNode<std::pair<uint32_t, uint32_t>>)));
      double average_slab = (double) buf_size / (SlabHash::SLAB_SIZE * opts.buckets_count);
      if (output != expected) {
        std::cerr << "Incorrect results" << std::endl;
        result->valid = false;
      }

      DwarfParams params{{"buf_size", std::to_string(buf_size)}, {"memory_utilization", std::to_string(memory_util)}, {"buckets_count", std::to_string(opts.buckets_count)},
        {"scale", std::to_string(opts.scale)}, {"avg_slab", std::to_string(average_slab)}, {"subgroup_size", std::to_string(SlabHash::SLAB_SIZE)}};
      meter.add_result(std::move(params), std::move(result));
      std::cout << "AVG_SLAB - " << average_slab << ' ' << opts.buckets_count << std::endl;
    }
  }
}

void SlabProbe::run(const RunOptions &opts) {
  for (auto size : opts.input_size) {
    _run(size, meter());
  }
}
void SlabProbe::init(const RunOptions &opts) {
  meter().set_opts(opts);
  DwarfParams params = {{"device_type", to_string(opts.device_ty)}};
  meter().set_params(params);
}
