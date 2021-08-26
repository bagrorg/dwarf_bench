#include "slab_join.hpp"
#include "common/dpcpp/slab_hash.hpp"
#include "join_helpers/join_helpers.hpp"
#include <math.h>

using std::pair;
using namespace join_helpers;
SlabJoin::SlabJoin() : Dwarf("SlabJoin") {}

void SlabJoin::_run(const size_t buf_size, Meter &meter) {
  const int scale = 16;
  auto opts = meter.opts();

  const std::vector<uint32_t> table_a_keys =
      helpers::make_unique_random(buf_size);
  const std::vector<uint32_t> table_a_values =
      helpers::make_unique_random(table_a_keys.size());

  const std::vector<uint32_t> table_b_keys =
      helpers::make_unique_random(buf_size);
  const std::vector<uint32_t> table_b_values =
      helpers::make_unique_random(table_b_keys.size());

  auto sel = get_device_selector(opts);
  sycl::queue q{*sel};
  std::cout << "Selected device: "
            << q.get_device().get_info<sycl::info::device::name>() << "\n";

  //auto expected = join_helpers::seq_join(table_a_keys, table_a_values,
   //                                      table_b_keys, table_b_values);

  for (auto it = 0; it < opts.iterations; ++it) {
    int num_of_groups = ceil((float)buf_size / scale);
    sycl::nd_range<1> r{(SlabHash::SUBGROUP_SIZE + 1) * num_of_groups,
                        SlabHash::SUBGROUP_SIZE + 1};

    SlabHash::AllocAdapter<std::pair<uint32_t, uint32_t>> adap(
        SlabHash::CLUSTER_SIZE, num_of_groups, SlabHash::BUCKETS_COUNT,
        {SlabHash::EMPTY_UINT32_T, 0}, q);
    std::vector<uint32_t> key_out(buf_size, 0);
    std::vector<uint32_t> val1_out(buf_size, -1);
    std::vector<uint32_t> val2_out(buf_size, -1);

    std::unique_ptr<HashJoinResult> result = std::make_unique<HashJoinResult>();

    {
      sycl::buffer<SlabHash::AllocAdapter<std::pair<uint32_t, uint32_t>>>
          adap_buf(&adap, sycl::range<1>{1});

      sycl::buffer<uint32_t> key_a(table_a_keys);
      sycl::buffer<uint32_t> val_a(table_a_values);
      sycl::buffer<uint32_t> key_b(table_b_keys);
      sycl::buffer<uint32_t> val_b(table_b_values);

      sycl::buffer<uint32_t> out_key_b(key_out);
      sycl::buffer<uint32_t> out_val1_b(val1_out);
      sycl::buffer<uint32_t> out_val2_b(val2_out);

      auto host_start = std::chrono::steady_clock::now();
      q.submit([&](sycl::handler &h) {
         auto adap_acc = sycl::accessor(adap_buf, h, sycl::read_write);
         auto key_a_acc = sycl::accessor(key_a, h, sycl::read_only);
         auto val_a_acc = sycl::accessor(val_a, h, sycl::read_only);

         h.parallel_for<class join_build>(
             r, [=](sycl::nd_item<1> it)[
                    [intel::reqd_sub_group_size(SlabHash::SUBGROUP_SIZE)]] {
               int idx = it.get_local_id();
               size_t ind = it.get_group().get_id();

               SlabHash::SlabHashTable<uint32_t, uint32_t,
                                       SlabHash::DefaultHasher<32, 48, 1031>>
                   ht(SlabHash::EMPTY_UINT32_T, it, *adap_acc.get_pointer());

               // todo: pick smaller one
               for (int i = ind * scale; i < (ind + 1) * scale && i < buf_size;
                    i++) {
                 if (it.get_local_id() != SlabHash::SUBGROUP_SIZE) ht.insert(key_a_acc[i], val_a_acc[i]);
               }
             });
       }).wait();
      auto build_end = std::chrono::steady_clock::now();
      auto probe_start = std::chrono::steady_clock::now();
      q.submit([&](sycl::handler &h) {
         auto key_b_acc = key_b.get_access(h);
         auto val_b_acc = val_b.get_access(h);

         auto out_key_a = out_key_b.get_access(h);
         auto out_val1_a = out_val1_b.get_access(h);
         auto out_val2_a = out_val2_b.get_access(h);

         auto adap_acc = sycl::accessor(adap_buf, h, sycl::read_write);

         h.parallel_for<class join_probe>(
             r, [=](sycl::nd_item<1> it)[
                    [intel::reqd_sub_group_size(SlabHash::SUBGROUP_SIZE)]] {
               size_t ind = it.get_group().get_id();

               SlabHash::SlabHashTable<uint32_t, uint32_t,
                                       SlabHash::DefaultHasher<32, 48, 1031>>
                   ht(SlabHash::EMPTY_UINT32_T, it, *adap_acc.get_pointer());
              if (it.get_local_id() != SlabHash::SUBGROUP_SIZE) {
               for (int i = ind * scale; i < (ind + 1) * scale && i < buf_size;
                    i++) {
                 auto ans = ht.find(key_b_acc[i]);

                 if (static_cast<bool>(ans)) {
                   out_key_a[i] = key_b_acc[i];
                   out_val1_a[i] = ans.value_or(-1);
                   out_val2_a[i] = val_b_acc[i];
                 }
               }
              }
             });
       }).wait();
      auto host_end = std::chrono::steady_clock::now();

      result->host_time = host_end - host_start;
      result->build_time = build_end - host_start;
      result->probe_time = host_end - probe_start;
    }

    std::vector<uint32_t> res_k;
    std::vector<uint32_t> res1;
    std::vector<uint32_t> res2;

    for (int i = 0; i < buf_size; i++) {
      if (key_out[i] != ((uint32_t)0)) {
        res_k.push_back(key_out[i]);
        res1.push_back(val1_out[i]);
        res2.push_back(val2_out[i]);
      }
    }

    join_helpers::ColJoinedTableTy<uint32_t, uint32_t, uint32_t> output = {
        res_k, {res1, res2}};

    //if (output != expected) {
    //  std::cerr << "Incorrect results" << std::endl;
    //  result->valid = false;
    //}

    DwarfParams params{{"buf_size", std::to_string(buf_size)}, {"type", std::to_string(opts.type)}, {"threads_count", std::to_string(opts.threads_count)}, {"algo", "slab_join"}};
    meter.add_result(std::move(params), std::move(result));
    // todo: scale factor?
  }
}

void SlabJoin::run(const RunOptions &opts) {
  for (auto size : opts.input_size) {
    _run(size, meter());
  }
}
void SlabJoin::init(const RunOptions &opts) {
  meter().set_opts(opts);
  DwarfParams params = {{"device_type", to_string(opts.device_ty)}};
  meter().set_params(params);
}
