#include "common/dpcpp/hashtable.hpp"
#include <gtest/gtest.h>
#include <vector>

TEST(HashTable, Build) {
  using namespace sycl;
  cpu_selector sel;
  queue q{sel};

  constexpr int input_size = 64;
  std::vector<uint32_t> bitmask(input_size / 32, 0);
  std::vector<uint32_t> data(input_size, 0);
  std::vector<uint32_t> keys(input_size, 0);

  buffer<uint32_t> bitmask_buf(bitmask.data(), bitmask.size());
  buffer<uint32_t> data_buf(data.data(), data.size());
  buffer<uint32_t> keys_buf(keys.data(), keys.size());

  StaticSimpleHasher<input_size> hasher;

  q.submit([&](handler &h) {
    auto bitmask_acc = bitmask_buf.get_access<access::mode::read_write>(h);
    auto data_acc = data_buf.get_access<access::mode::read_write>(h);
    auto keys_acc = keys_buf.get_access<access::mode::read_write>(h);

    h.parallel_for<class test_hash>(range<1>{2}, [=](auto &idx) {
      SimpleNonOwningHashTable<uint32_t, uint32_t,
                               StaticSimpleHasher<input_size>>
          ht(input_size, keys_acc.get_pointer(), data_acc.get_pointer(),
             bitmask_acc.get_pointer(), hasher);
      int id = idx.get_id(0);
      if (id == 0) {
        ht.insert(2, 2);
        ht.insert(65, 3);
        ht.insert(66, 8);
        ht.insert(1, 9);
      }

      ht.insert(10, id + 1);
    });
  });

  auto result = data_buf.get_access<access::mode::read>();
  for (int i = 0; i < input_size; ++i) {
    std::cout << result[i] << " ";
  }
  std::cout << std::endl;

  ASSERT_EQ(result[1], 3);
  ASSERT_EQ(result[2], 2);
  ASSERT_EQ(result[3], 8);
  ASSERT_EQ(result[4], 9);
  ASSERT_EQ(result[10] + result[11], 3);
}

TEST(HashTable, Probe) {
  using namespace sycl;
  cpu_selector sel;
  queue q{sel};

  constexpr int input_size = 64;
  std::vector<uint32_t> bitmask(input_size / 32, 0);
  std::vector<uint32_t> data(input_size, 0);
  std::vector<uint32_t> keys(input_size, 0);
  std::vector<uint32_t> output(input_size, 0);

  buffer<uint32_t> bitmask_buf(bitmask);
  buffer<uint32_t> data_buf(data);
  buffer<uint32_t> keys_buf(keys);

  buffer<uint32_t> out_buf(output);

  StaticSimpleHasher<input_size> hasher;

  q.submit([&](handler &h) {
    auto bitmask_acc = bitmask_buf.get_access(h);
    auto data_acc = data_buf.get_access(h);
    auto keys_acc = keys_buf.get_access(h);
    auto out_acc = out_buf.get_access(h);

    h.parallel_for<class test_hash_probe>(range{1}, [=](auto &idx) {
      SimpleNonOwningHashTable<uint32_t, uint32_t,
                               StaticSimpleHasher<input_size>>
          ht(input_size, keys_acc.get_pointer(), data_acc.get_pointer(),
             bitmask_acc.get_pointer(), hasher);
      int id = idx.get_id(0);

      ht.insert(1, 1);
      ht.insert(1, 5);
      ht.insert(4, 55);

      auto [v1, b1] = ht.at(1);
      auto [v2, b2] = ht.at(4);
      out_acc[idx] = b1 ? v1 : 0;
      out_acc[idx + 1] = b2 ? v2 : 0;
    });
  });

  auto result = data_buf.get_access<access::mode::read>();
  auto outer = out_buf.get_access<access::mode::read>();
  for (int i = 0; i < input_size; ++i) {
    std::cout << result[i] << " ";
  }
  std::cout << std::endl;
  for (int i = 0; i < input_size; ++i) {
    std::cout << outer[i] << " ";
  }
  std::cout << std::endl;

  ASSERT_EQ(outer[0], 1);
  ASSERT_EQ(outer[1], 55);
}

TEST(HashTable, Has) {
  using namespace sycl;
  cpu_selector sel;
  queue q{sel};

  constexpr int input_size = 64;
  std::vector<uint32_t> bitmask(input_size / 32, 0);
  std::vector<uint32_t> data(input_size, 0);
  std::vector<uint32_t> keys(input_size, 0);
  std::vector<uint32_t> output(input_size, 0);

  buffer<uint32_t> bitmask_buf(bitmask);
  buffer<uint32_t> data_buf(data);
  buffer<uint32_t> keys_buf(keys);

  buffer<uint32_t> out_buf(output);

  StaticSimpleHasher<input_size> hasher;

  q.submit([&](handler &h) {
    auto bitmask_acc = bitmask_buf.get_access(h);
    auto data_acc = data_buf.get_access(h);
    auto keys_acc = keys_buf.get_access(h);
    auto out_acc = out_buf.get_access(h);

    h.parallel_for<class test_hash_has>(range{1}, [=](auto &idx) {
      SimpleNonOwningHashTable<uint32_t, uint32_t,
                               StaticSimpleHasher<input_size>>
          ht(input_size, keys_acc.get_pointer(), data_acc.get_pointer(),
             bitmask_acc.get_pointer(), hasher);
      int id = idx.get_id(0);

      ht.insert(1, 1);
      ht.insert(65, 5);
      ht.insert(1 + 2 * input_size, 6);
      ht.insert(1 + 3 * input_size, 7);
      ht.insert(4, 55);

      out_acc[0] = ht.has(1);
      out_acc[1] = ht.has(65);
      out_acc[2] = ht.has(64);
      out_acc[3] = ht.has(4);
      out_acc[4] = ht.has(129);
      out_acc[5] = ht.has(193);
    });
  });

  auto result = data_buf.get_access<access::mode::read>();
  auto outer = out_buf.get_access<access::mode::read>();
  for (int i = 0; i < input_size; ++i) {
    std::cout << result[i] << " ";
  }
  std::cout << std::endl;
  for (int i = 0; i < input_size; ++i) {
    std::cout << outer[i] << " ";
  }
  std::cout << std::endl;

  ASSERT_EQ(outer[0], 1);
  ASSERT_EQ(outer[1], 1);
  ASSERT_EQ(outer[2], 0);
  ASSERT_EQ(outer[3], 1);
  ASSERT_EQ(outer[4], 1);
  ASSERT_EQ(outer[5], 1);
}

TEST(HashTable, BigBuild) {
  using namespace sycl;
  gpu_selector sel;
  queue q{sel};

  constexpr int buf_size = 500;

  std::vector<uint32_t> host_src_tmp;
  for (int i = 0; i < buf_size; i++) {
    host_src_tmp.push_back(i);
  }
  std::vector<uint32_t> host_src(host_src_tmp.begin(), host_src_tmp.end());

  StaticSimpleHasher<buf_size> hasher;
  size_t bitmask_sz = buf_size / 32 + 1;
  std::vector<uint32_t> bitmask(bitmask_sz, 0);
  std::vector<uint32_t> data(buf_size, 0);
  std::vector<uint32_t> keys(buf_size, 0);

  {
    sycl::buffer<uint32_t> bitmask_buf(bitmask);
    sycl::buffer<uint32_t> data_buf(data);
    sycl::buffer<uint32_t> keys_buf(keys);
    sycl::buffer<uint32_t> src(host_src);

    q.submit([&](sycl::handler &h) {
       auto s = sycl::accessor(src, h, sycl::read_write);

       auto bitmask_acc = sycl::accessor(bitmask_buf, h, sycl::read_write);
       auto data_acc = sycl::accessor(data_buf, h, sycl::read_write);
       auto keys_acc = sycl::accessor(keys_buf, h, sycl::read_write);

       h.parallel_for<class hash_big_build_test>(buf_size, [=](auto &idx) {
         SimpleNonOwningHashTable<uint32_t, uint32_t,
                                  StaticSimpleHasher<buf_size>>
             ht(buf_size, keys_acc.get_pointer(), data_acc.get_pointer(),
                bitmask_acc.get_pointer(), hasher);
         ht.insert(s[idx], s[idx]);
       });
     }).wait();
  }

  std::set<uint32_t> s(data.begin(), data.end());
  std::cout << s.size() << std::endl;
  ASSERT_EQ(s.size(), host_src.size());
}

TEST(GroupByHashTable, GroupByFunctions) {
  using namespace sycl;
  gpu_selector sel;
  queue q{sel};

  constexpr int buf_size = 50;
  constexpr int groups = 9;

  std::vector<uint32_t> host_src_keys = {0, 1, 2, 0, 3, 4, 0, 5, 0, 1, 8, 7, 2,
                                         4, 5, 7, 1, 2, 4, 6, 2, 4, 1, 4, 6, 2,
                                         4, 6, 8, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8,
                                         1, 1, 2, 3, 4, 5, 6, 7, 8, 0, 0};
  std::vector<uint32_t> host_src_vals = {
      12, 19, 1,  4,  30, 21, 3,  8,  6,  19,  1,  1, 2,  0,  4, 4, 0,
      5,  0,  1,  0,  1,  2,  0,  3,  4,  0,   5,  0, 1,  1,  3, 1, 3,
      1,  0,  23, 11, 0,  1,  33, 91, 12, 321, 12, 9, 99, 65, 7, 4};
  std::vector<uint32_t> answers(groups, 0);
  for (int i = 0; i < buf_size; i++) {
    answers[host_src_keys[i]] += host_src_vals[i];
  }

  PolynomialHasher hasher(buf_size);
  size_t bitmask_sz = buf_size / 32 + 1;
  std::vector<uint32_t> bitmask(bitmask_sz, 0);
  std::vector<uint32_t> data(buf_size, 0);
  std::vector<uint32_t> data_answers(groups, 0);
  std::vector<uint32_t> keys(buf_size, -1);

  {
    sycl::buffer<uint32_t> bitmask_buf(bitmask);
    sycl::buffer<uint32_t> data_buf(data);
    sycl::buffer<uint32_t> keys_buf(keys);
    sycl::buffer<uint32_t> src_keys(host_src_keys);
    sycl::buffer<uint32_t> src_vals(host_src_vals);
    sycl::buffer<uint32_t> data_ans(data_answers);

    q.submit([&](sycl::handler &h) {
       auto sk = sycl::accessor(src_keys, h, sycl::read_write);
       auto sv = sycl::accessor(src_vals, h, sycl::read_write);

       auto bitmask_acc = sycl::accessor(bitmask_buf, h, sycl::read_write);
       auto data_acc = sycl::accessor(data_buf, h, sycl::read_write);
       auto keys_acc = sycl::accessor(keys_buf, h, sycl::read_write);

       h.parallel_for<class hash_group_by_build_test>(buf_size, [=](auto &idx) {
         SimpleNonOwningHashTableForGroupBy<uint32_t, uint32_t,
                                            PolynomialHasher>
             ht(buf_size, keys_acc.get_pointer(), data_acc.get_pointer(),
                hasher, -1);
         ht.insert_group_by(sk[idx], sv[idx]);
       });
     }).wait();
  }
  {
    sycl::buffer<uint32_t> bitmask_buf(bitmask);
    sycl::buffer<uint32_t> data_buf(data);
    sycl::buffer<uint32_t> keys_buf(keys);
    sycl::buffer<uint32_t> src_keys(host_src_keys);
    sycl::buffer<uint32_t> src_vals(host_src_vals);
    sycl::buffer<uint32_t> data_ans(data_answers);

    q.submit([&](sycl::handler &h) {
       auto sk = sycl::accessor(src_keys, h, sycl::read_write);
       auto sv = sycl::accessor(src_vals, h, sycl::read_write);

       auto bitmask_acc = sycl::accessor(bitmask_buf, h, sycl::read_write);
       auto data_acc = sycl::accessor(data_buf, h, sycl::read_write);
       auto keys_acc = sycl::accessor(keys_buf, h, sycl::read_write);
       auto data_ans_acc = sycl::accessor(data_ans, h, sycl::read_write);

       h.parallel_for<class hash_group_by_lookup_test>(
           buf_size, [=](auto &idx) {
             SimpleNonOwningHashTableForGroupBy<uint32_t, uint32_t,
                                                PolynomialHasher>
                 ht(buf_size, keys_acc.get_pointer(), data_acc.get_pointer(),
                    hasher, -1);

             std::pair<uint32_t, bool> ans = ht.at(sk[idx]);
             sycl::atomic<uint32_t>(data_ans_acc.get_pointer() + sk[idx])
                 .store(ans.first);
           });
     }).wait();
  }

  for (int i = 0; i < groups; i++) {
    ASSERT_EQ(data_answers[i], answers[i]);
  }
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}