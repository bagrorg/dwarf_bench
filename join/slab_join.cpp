#include "join.hpp"

#include "common/dpcpp/slab_hash.hpp"
#include <math.h>

using std::pair;
using namespace join_helpers;
Join::Join() : Dwarf("Join") {}

void Join::_run(const size_t buf_size, Meter &meter) {
  const int scale = 16;
  auto opts = meter.opts();

  // todo: sizes
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

  // hash table
  int work_size = ceil((float) buf_size / scale);
  sycl::nd_range<1> r{SlabHash::SUBGROUP_SIZE * work_size,
                      SlabHash::SUBGROUP_SIZE};

  SlabHash::AllocAdapter<pair<uint32_t, uint32_t>> data(
      SlabHash::BUCKETS_COUNT, {SlabHash::EMPTY_UINT32_T, 0}, q);

  // testing
  std::vector<uint32_t> key_out(buf_size, 0);
  std::vector<uint32_t> val1_out(buf_size, -1);
  std::vector<uint32_t> val2_out(buf_size, -1);

  auto expected = join_helpers::seq_join(table_a_keys, table_a_values,
                                         table_b_keys, table_b_values);

    std::cout << "expected done\n";
  Result result;

  {

    sycl::buffer<SlabHash::SlabList<pair<uint32_t, uint32_t>>> data_buf(
        data._data);
    sycl::buffer<sycl::device_ptr<SlabHash::SlabNode<pair<uint32_t, uint32_t>>>>
        its(work_size);

    sycl::buffer<uint32_t> key_a(table_a_keys);
    sycl::buffer<uint32_t> val_a(table_a_values);
    sycl::buffer<uint32_t> key_b(table_b_keys);
    sycl::buffer<uint32_t> val_b(table_b_values);

    sycl::buffer<uint32_t> out_key_b(key_out);
    sycl::buffer<uint32_t> out_val1_b(val1_out);
    sycl::buffer<uint32_t> out_val2_b(val2_out);

    //const size_t ht_size = buf_size;
    //SimpleHasher<uint32_t> hasher(ht_size);

    auto host_start = std::chrono::steady_clock::now();
    q.submit([&](sycl::handler &h) {
       auto key_a_acc = sycl::accessor(key_a, h, sycl::read_only);
       auto val_a_acc = sycl::accessor(val_a, h, sycl::read_only);


       // ht data accessors
       auto data_acc = sycl::accessor(data_buf, h, sycl::read_write);
       auto itrs = sycl::accessor(its, h, sycl::read_write);

       h.parallel_for<class join_build>(r, [=](sycl::nd_item<1> it) {
         int idx = it.get_local_id();
         size_t ind = it.get_group().get_id();
         SlabHash::DefaultHasher<32, 48, 1031> h;
         SlabHash::SlabHashTable<uint32_t, uint32_t,
                                 SlabHash::DefaultHasher<32, 48, 1031>>
             ht(SlabHash::EMPTY_UINT32_T, h, data_acc.get_pointer(), it,
                itrs[it.get_group().get_id()]);

         // todo: pick smaller one

         for (int i = ind * scale; i < ind * scale + scale && i < buf_size;
              i++) {
           ht.insert(key_a_acc[i], val_a_acc[i]);
         }
       });
     })
        .wait();
    auto build_end = std::chrono::steady_clock::now();
    std::cout << "Builded\n";
    auto probe_start = std::chrono::steady_clock::now();
    q.submit([&](sycl::handler &h) {
       auto key_b_acc = key_b.get_access(h);
       auto val_b_acc = val_b.get_access(h);

       auto out_key_a = out_key_b.get_access(h);
       auto out_val1_a = out_val1_b.get_access(h);
       auto out_val2_a = out_val2_b.get_access(h);

       // ht data accessors
       auto data_acc = sycl::accessor(data_buf, h, sycl::read_write);
       auto itrs = sycl::accessor(its, h, sycl::read_write);

       h.parallel_for<class join_probe>(r, [=](sycl::nd_item<1> it) {
         //int idx = it.get_local_id();
         size_t ind = it.get_group().get_id();
         SlabHash::DefaultHasher<32, 48, 1031> h;
         SlabHash::SlabHashTable<uint32_t, uint32_t,
                                 SlabHash::DefaultHasher<32, 48, 1031>>
             ht(SlabHash::EMPTY_UINT32_T, h, data_acc.get_pointer(), it,
                itrs[it.get_group().get_id()]);

        for (int idx = ind * scale; idx < ind * scale + scale && idx < buf_size;
              idx++) {
         auto ans = ht.find(key_b_acc[idx]);

         if (static_cast<bool>(ans)) {
           out_key_a[idx] = key_b_acc[idx];
           out_val1_a[idx] = ans.value_or(-1);
           out_val2_a[idx] = val_b_acc[idx];
         }
              }
       });
     })
        .wait();
    auto host_end = std::chrono::steady_clock::now();

    result.isJoin = true;
    result.host_time = host_end - host_start;
    result.build_time = build_end - host_start;
    result.probe_time = host_end - probe_start;
    std::cout << "end\n";
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

  if (output != expected) {
    std::cerr << "Incorrect results" << std::endl;
    result.valid = false;
  }

  DwarfParams params{{"buf_size", std::to_string(buf_size)}};
  meter.add_result(std::move(params), std::move(result));
  // todo: scale factor?
}

void Join::run(const RunOptions &opts) {
  for (auto size : opts.input_size) {
    _run(size, meter());
  }
}
void Join::init(const RunOptions &opts) {
  meter().set_opts(opts);
  DwarfParams params = {{"device_type", to_string(opts.device_ty)}};
  meter().set_params(params);
}
