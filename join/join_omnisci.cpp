
#include "join_omnisci.hpp"
#include "common/dpcpp/omnisci_hashtable.hpp"
#include "join_helpers/join_helpers.hpp"
#include <unordered_set>

using JoinOneToManySet = JoinOneToMany<std::unordered_set<size_t>>;

size_t count_distinct(const std::vector<uint32_t> &v) {
  std::unordered_set<uint32_t> s{v.begin(), v.end()};
  return s.size();
}

std::vector<JoinOneToManySet>
build_join_id_buffer(const std::vector<uint32_t> &a,
                     const std::vector<uint32_t> &b) {
  std::vector<JoinOneToManySet> ans(b.size());
  for (int i = 0; i < b.size(); i++) {
    std::unordered_set<size_t> ids;
    for (int j = 0; j < a.size(); j++) {
      if (a[j] == b[i])
        ids.insert(j);
    }

    ans[i] = {ids, ids.size()};
  }
  return ans;
}

bool are_equal(const std::vector<JoinOneToManySet> &expected,
               const std::vector<JoinOneToManyPtrs> &result, sycl::queue &q) {
  for (int i = 0; i < expected.size(); i++) {
    if (expected[i].size != result[i].size) {
      return false;
    }
    for (int j = 0; j < result[i].size; j++) {
      auto vals = get_from_device(result[i].vals, result[i].size, q);
      if (expected[i].vals.find(vals[j]) == expected[i].vals.end()) {
        return false;
      }
    }
  }
  return true;
}

JoinOmnisci::JoinOmnisci() : Dwarf("JoinOmnisci") {}
using namespace join_helpers;
void JoinOmnisci::_run(const size_t buf_size, Meter &meter) {
  auto opts = meter.opts();

  constexpr uint32_t empty_element = std::numeric_limits<uint32_t>::max();
  const std::vector<uint32_t> table_a_keys =
      helpers::make_random<uint32_t>(buf_size);
  size_t unique_keys = count_distinct(table_a_keys);

  const std::vector<uint32_t> table_b_keys =
      helpers::make_random<uint32_t>(buf_size);

  auto sel = get_device_selector(opts);
  sycl::queue q{*sel};
  std::cout << "Selected device: "
            << q.get_device().get_info<sycl::info::device::name>() << "\n";
  auto expected = build_join_id_buffer(table_a_keys, table_b_keys);

  const size_t ht_size = unique_keys * 2;
  SimpleHasher<uint32_t> hasher(ht_size);

  for (unsigned it = 0; it < opts.iterations; ++it) {
    std::unique_ptr<HashJoinResult> result = std::make_unique<HashJoinResult>();
    OmniSci::HashTable<uint32_t, uint32_t, SimpleHasher<uint32_t>> ht(
        table_a_keys, std::numeric_limits<uint32_t>::max(), hasher, ht_size,
        unique_keys, q);
    auto host_start = std::chrono::steady_clock::now();

    ht.build_table();
    ht.build_id_buffer();

    auto build_end = std::chrono::steady_clock::now();
    auto res = ht.lookup(table_b_keys);

    auto host_end = std::chrono::steady_clock::now();
    auto host_exe_time = std::chrono::duration_cast<std::chrono::microseconds>(
                             host_end - host_start)
                             .count();

    result->host_time = host_end - host_start;
    result->build_time = build_end - host_start;
    result->probe_time = host_end - build_end;

    if (!(are_equal(expected, res, q))) {
      std::cerr << "Incorrect results" << std::endl;
      result->valid = false;
    }

    DwarfParams params{{"buf_size", std::to_string(buf_size)}};
    meter.add_result(std::move(params), std::move(result));
    // todo: scale factor?
  }
}

void JoinOmnisci::run(const RunOptions &opts) {
  for (auto size : opts.input_size) {
    _run(size, meter());
  }
}
void JoinOmnisci::init(const RunOptions &opts) {
  meter().set_opts(opts);
  DwarfParams params = {{"device_type", to_string(opts.device_ty)}};
  meter().set_params(params);
}