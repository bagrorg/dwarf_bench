
#include "join.hpp"
#include "common/dpcpp/hashtable.hpp"
#include "join_helpers/join_helpers.hpp"

Join::Join() : Dwarf("Join") {}
using namespace join_helpers;
void Join::_run(const size_t buf_size, Meter &meter) {
  auto opts = meter.opts();

  constexpr uint32_t empty_element = std::numeric_limits<uint32_t>::max();
  // todo: sizes
  const std::vector<uint32_t> table_a_keys =
      helpers::make_unique_random(buf_size);
  const std::vector<uint32_t> table_a_values =
      helpers::make_random<uint32_t>(table_a_keys.size());

  const std::vector<uint32_t> table_b_keys =
      helpers::make_unique_random(buf_size);
  const std::vector<uint32_t> table_b_values =
      helpers::make_random<uint32_t>(table_b_keys.size());

  auto sel = get_device_selector(opts);
  sycl::queue q{*sel};
  std::cout << "Selected device: "
            << q.get_device().get_info<sycl::info::device::name>() << "\n";
  auto expected =
      seq_join(table_a_keys, table_a_values, table_b_keys, table_b_values);

  const size_t ht_size = buf_size * 2;
  const size_t bitmask_sz = ht_size / 32 + 1;
  SimpleHasher<uint32_t> hasher(ht_size);

  for (unsigned it = 0; it < opts.iterations; ++it) {
    // hash table
    std::vector<uint32_t> bitmask(bitmask_sz, 0);
    std::vector<uint32_t> data(ht_size, 0);
    std::vector<uint32_t> keys(ht_size, empty_element);

    // testing
    std::vector<uint32_t> key_out(buf_size, -1);
    std::vector<uint32_t> key_present_out(buf_size, -1);
    std::vector<uint32_t> val_out(buf_size, -1);
    std::unique_ptr<HashJoinResult> result = std::make_unique<HashJoinResult>();
    {
      sycl::buffer<uint32_t> bitmask_buf(bitmask);
      sycl::buffer<uint32_t> data_buf(data);
      sycl::buffer<uint32_t> keys_buf(keys);

      sycl::buffer<uint32_t> key_a(table_a_keys);
      sycl::buffer<uint32_t> val_a(table_a_values);
      sycl::buffer<uint32_t> key_b(table_b_keys);
      sycl::buffer<uint32_t> val_b(table_b_values);

      sycl::buffer<uint32_t> out_key_buf(key_out);
      sycl::buffer<uint32_t> out_key_present_buf(key_present_out);
      sycl::buffer<uint32_t> out_val_buf(val_out);

      auto host_start = std::chrono::steady_clock::now();
      q.submit([&](sycl::handler &h) {
         auto key_a_acc = key_a.get_access(h);
         auto val_a_acc = val_a.get_access(h);

         // ht data accessors
         auto bitmask_acc = bitmask_buf.get_access(h);
         auto data_acc = data_buf.get_access(h);
         auto keys_acc = keys_buf.get_access(h);

         h.parallel_for<class join_build>(buf_size, [=](auto &idx) {
           NonOwningHashTableNonBitmask<uint32_t, uint32_t, SimpleHasher<uint32_t>> ht(
             buf_size, keys_acc.get_pointer(), data_acc.get_pointer(), hasher,
             empty_element);

           ht.insert(key_a_acc[idx], val_a_acc[idx]);
         });
         q.wait();
       }).wait();
      auto build_end = std::chrono::steady_clock::now();

      q.submit([&](sycl::handler &h) {
         auto key_b_acc = key_b.get_access(h);
         auto val_b_acc = val_b.get_access(h);

         auto out_key_acc = out_key_buf.get_access(h);
         auto out_key_present_acc = out_key_present_buf.get_access(h);
         auto out_val_acc = out_val_buf.get_access(h);

         // ht data accessors
         auto bitmask_acc = bitmask_buf.get_access(h);
         auto data_acc = data_buf.get_access(h);
         auto keys_acc = keys_buf.get_access(h);

         h.parallel_for<class join_probe>(buf_size, [=](auto &idx) {
           NonOwningHashTableNonBitmask<uint32_t, uint32_t, SimpleHasher<uint32_t>> ht(
             buf_size, keys_acc.get_pointer(), data_acc.get_pointer(), hasher,
             empty_element);
           auto ans = ht.at(key_b_acc[idx]);
           if (ans.second) {
             out_key_acc[idx] = key_b_acc[idx];
             out_key_present_acc[idx] = ans.first;
             out_val_acc[idx] = val_b_acc[idx];
           }
         });
       }).wait();
      auto host_end = std::chrono::steady_clock::now();
      auto host_exe_time =
          std::chrono::duration_cast<std::chrono::microseconds>(host_end -
                                                                host_start)
              .count();

      result->host_time = host_end - host_start;
      result->build_time = build_end - host_start;
      result->probe_time = host_end - build_end;

      std::vector<uint32_t> res_k;
      std::vector<uint32_t> res_present;
      std::vector<uint32_t> res_val;

      for (int i = 0; i < buf_size; i++) {
        if (key_out[i] != empty_element) {
          res_k.push_back(key_out[i]);
          res_present.push_back(key_present_out[i]);
          res_val.push_back(val_out[i]);
        }
      }
      ColJoinedTableTy<uint32_t, uint32_t, uint32_t> output = {
          res_k, {res_present, res_val}};

      if (output != expected) {
       std::cerr << "Incorrect results" << std::endl;
       result->valid = false;
      }
    }

    DwarfParams params{{"buf_size", std::to_string(buf_size)}};
    meter.add_result(std::move(params), std::move(result));
    // todo: scale factor?
  }
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