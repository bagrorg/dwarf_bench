#include "common/dpcpp/slab_hash.hpp"
#include <gtest/gtest.h>
#include <math.h>
#include <vector>

using std::pair;
using namespace SlabHash;

TEST(SlabHash, insert) {
  std::vector<pair<uint32_t, uint32_t>> testUniv = {{1, 2}, {5, 2}, {101, 3},
                                                    {5, 5}, {3, 0}, {10, 10}};

  sycl::queue q{sycl::gpu_selector()};
  sycl::nd_range<1> r{SUBGROUP_SIZE * 3, SUBGROUP_SIZE};

  SlabHash::AllocAdapter<std::pair<uint32_t, uint32_t>> adap(SlabHash::CLUSTER_SIZE, 3,
        SlabHash::BUCKETS_COUNT, {SlabHash::EMPTY_UINT32_T, 0}, q);

  std::vector<uint8_t> checks(6, 0);

  {
    sycl::buffer<SlabHash::AllocAdapter<std::pair<uint32_t, uint32_t>>>
        adap_buf(&adap, sycl::range<1>{1});
    sycl::buffer<uint8_t> checks_buf(checks);
    sycl::buffer<pair<uint32_t, uint32_t>> buffTestUniv(testUniv);

    q.submit([&](sycl::handler &cgh) {
       auto tests = sycl::accessor(buffTestUniv, cgh, sycl::read_only);
       auto adap_acc = sycl::accessor(adap_buf, cgh, sycl::read_write);

       cgh.parallel_for<class insert_test_slab>(r, [=](sycl::nd_item<1> it) [[intel::reqd_sub_group_size(SlabHash::SUBGROUP_SIZE)]] {
         size_t ind = it.get_group().get_id();

         SlabHashTable<uint32_t, uint32_t, DefaultHasher<13, 24, 343>> ht(
             SlabHash::EMPTY_UINT32_T, it, *(adap_acc.get_pointer()));

         for (int i = ind * 2; i < ind * 2 + 2; i++) {
           ht.insert(tests[i].first, tests[i].second);
         }
       });
     })
        .wait();

    q.submit([&](sycl::handler &cgh) {
       auto tests = sycl::accessor(buffTestUniv, cgh, sycl::read_only);
       auto checks = sycl::accessor(checks_buf, cgh, sycl::read_write);
       auto adap_acc = sycl::accessor(adap_buf, cgh, sycl::read_write);

       cgh.parallel_for<class insert_test_slab_check>(6, [=](auto &it) {
         size_t ind = it;

         DefaultHasher<13, 24, 343> h;
         auto list = (*(adap_acc.get_pointer()))._data[h(tests[ind].first)];
         for (int i = 0; i < SLAB_SIZE; i++) {
           if (list.root->data[i] == tests[ind]) {
             checks[ind] = 1;
             break;
           }
         }
       });
     })
        .wait();
  }

  for (auto e : checks) {
    EXPECT_EQ(e, 1);
  }
}

TEST(SlabHash, find) {
  std::vector<pair<uint32_t, uint32_t>> testUniv = {
      {1, 2}, {5, 2}, {101, 3}, {21312, 5}, {3, 0}, {10, 10}};

  sycl::queue q{sycl::gpu_selector()};
  sycl::nd_range<1> r{SUBGROUP_SIZE * 3, SUBGROUP_SIZE};

  SlabHash::AllocAdapter<std::pair<uint32_t, uint32_t>> adap(SlabHash::CLUSTER_SIZE, 3,
        SlabHash::BUCKETS_COUNT, {SlabHash::EMPTY_UINT32_T, 0}, q);

  std::vector<pair<bool, bool>> checks(6, {false, false});

  {
    sycl::buffer<SlabHash::AllocAdapter<std::pair<uint32_t, uint32_t>>>
        adap_buf(&adap, sycl::range<1>{1});
    sycl::buffer<pair<bool, bool>> checks_buf(checks);
    sycl::buffer<pair<uint32_t, uint32_t>> buffTestUniv(testUniv);

    q.submit([&](sycl::handler &cgh) {
       auto tests = sycl::accessor(buffTestUniv, cgh, sycl::read_only);
       auto checks = sycl::accessor(checks_buf, cgh, sycl::read_write);
       auto adap_acc = sycl::accessor(adap_buf, cgh, sycl::read_write);
       cgh.single_task<class find_test_slab_check>([=]() {
         DefaultHasher<13, 24, 343> h;

         for (int j = 0; j < 6; j++) {
           auto e = tests[j];
           auto list = &((*(adap_acc.get_pointer()))._data[h(e.first)]);
           if (list->root == nullptr) {
             list->root = (*(adap_acc.get_pointer()))._heap.malloc_node();
             *list->root = SlabNode<pair<uint32_t, uint32_t>>(
                 {SlabHash::EMPTY_UINT32_T, 0});
           }
           for (int i = 0; i < SLAB_SIZE; i++) {
             if (list->root->data[i].first == EMPTY_UINT32_T) {
               list->root->data[i] = e;
               break;
             }
           }
         }
       });
     })
        .wait();

    q.submit([&](sycl::handler &cgh) {
       auto tests = sycl::accessor(buffTestUniv, cgh, sycl::read_only);
       auto accChecks = sycl::accessor(checks_buf, cgh, sycl::write_only);
       auto adap_acc = sycl::accessor(adap_buf, cgh, sycl::read_write);

       cgh.parallel_for<class find_test_slab>(r, [=](sycl::nd_item<1> it) [[intel::reqd_sub_group_size(SlabHash::SUBGROUP_SIZE)]] {
         size_t ind = it.get_group().get_id();

         SlabHashTable<uint32_t, uint32_t, DefaultHasher<13, 24, 343>> ht(
             SlabHash::EMPTY_UINT32_T, it, *(adap_acc.get_pointer()));

         for (int i = ind * 2; i < ind * 2 + 2; i++) {
           auto ans = ht.find(tests[i].first);

           if (it.get_local_id() == 0)
             accChecks[i] = {static_cast<bool>(ans),
                             ans.value_or(-1) == tests[i].second};
         }
       });
     })
        .wait();
  }

  for (int i = 0; i < 6; i++) {
    EXPECT_TRUE(checks[i].first && checks[i].second);
  }
}

TEST(SlabHash, find_and_insert_together) {
  std::vector<pair<uint32_t, uint32_t>> testUniv = {
      {1, 2}, {5, 2}, {101, 3}, {10932, 5}, {3, 0}, {10, 10}};

  sycl::queue q{sycl::gpu_selector()};
  sycl::nd_range<1> r{SUBGROUP_SIZE * 3, SUBGROUP_SIZE};

  SlabHash::AllocAdapter<std::pair<uint32_t, uint32_t>> adap(SlabHash::CLUSTER_SIZE, 3,
        SlabHash::BUCKETS_COUNT, {SlabHash::EMPTY_UINT32_T, 0}, q);
  std::vector<pair<bool, bool>> checks(6);

  {

    sycl::buffer<SlabHash::AllocAdapter<std::pair<uint32_t, uint32_t>>>
        adap_buf(&adap, sycl::range<1>{1});
    sycl::buffer<pair<bool, bool>> checks_buf(checks);
    sycl::buffer<pair<uint32_t, uint32_t>> buffTestUniv(testUniv);

    q.submit([&](sycl::handler &cgh) {
       auto tests = sycl::accessor(buffTestUniv, cgh, sycl::read_only);
       auto accChecks = sycl::accessor(checks_buf, cgh, sycl::write_only);
       auto adap_acc = sycl::accessor(adap_buf, cgh, sycl::read_write);

       cgh.parallel_for<class insert_test_slab_both>(
           r, [=](sycl::nd_item<1> it) [[intel::reqd_sub_group_size(SlabHash::SUBGROUP_SIZE)]] {
             size_t ind = it.get_group().get_id();

             SlabHashTable<uint32_t, uint32_t, DefaultHasher<13, 24, 343>> ht(
                 SlabHash::EMPTY_UINT32_T, it, *(adap_acc.get_pointer()));

             for (int i = ind * 2; i < ind * 2 + 2; i++) {
               ht.insert(tests[i].first, tests[i].second);
             }
           });
     })
        .wait();

    q.submit([&](sycl::handler &cgh) {
       auto tests = sycl::accessor(buffTestUniv, cgh, sycl::read_only);
       auto accChecks = sycl::accessor(checks_buf, cgh, sycl::write_only);
       auto adap_acc = sycl::accessor(adap_buf, cgh, sycl::read_write);

       cgh.parallel_for<class find_test_slab_both>(r, [=](sycl::nd_item<1> it) [[intel::reqd_sub_group_size(SlabHash::SUBGROUP_SIZE)]] {
         size_t ind = it.get_group().get_id();

         SlabHashTable<uint32_t, uint32_t, DefaultHasher<13, 24, 343>> ht(
             SlabHash::EMPTY_UINT32_T, it, *(adap_acc.get_pointer()));

         for (int i = ind * 2; i < ind * 2 + 2; i++) {
           auto ans = ht.find(tests[i].first);

           if (it.get_local_id() == 0)
             accChecks[i] = {static_cast<bool>(ans),
                             ans.value_or(-1) == tests[i].second};
         }
       });
     })
        .wait();
  }

  for (auto &e : checks) {
    EXPECT_TRUE(e.first && e.second);
  }
}

TEST(SlabHash, find_and_insert_together_big) {
  std::vector<pair<uint32_t, uint32_t>> testUniv;
  auto f = [&](int i) { return i * i; };

  for (int i = 0; i < 1000; i++) {
    testUniv.push_back({f(i), f(i)});
  }

  sycl::queue q{sycl::gpu_selector()};
  sycl::nd_range<1> r{SUBGROUP_SIZE * 25, SUBGROUP_SIZE};

  SlabHash::AllocAdapter<std::pair<uint32_t, uint32_t>> adap(SlabHash::CLUSTER_SIZE, 25,
        SlabHash::BUCKETS_COUNT, {SlabHash::EMPTY_UINT32_T, 0}, q);

  std::vector<pair<bool, bool>> checks(1000);

  {

    sycl::buffer<SlabHash::AllocAdapter<std::pair<uint32_t, uint32_t>>>
        adap_buf(&adap, sycl::range<1>{1});
    sycl::buffer<pair<bool, bool>> checks_buf(checks);
    sycl::buffer<pair<uint32_t, uint32_t>> buffTestUniv(testUniv);

    q.submit([&](sycl::handler &cgh) {
       auto tests = sycl::accessor(buffTestUniv, cgh, sycl::read_only);
       auto accChecks = sycl::accessor(checks_buf, cgh, sycl::write_only);
       auto adap_acc = sycl::accessor(adap_buf, cgh, sycl::read_write);

       cgh.parallel_for<class insert_test_slab_both_big>(
           r, [=](sycl::nd_item<1> it) [[intel::reqd_sub_group_size(SlabHash::SUBGROUP_SIZE)]] {
             size_t ind = it.get_group().get_id();

             SlabHashTable<uint32_t, uint32_t, DefaultHasher<13, 24, 343>> ht(
                 SlabHash::EMPTY_UINT32_T, it, *(adap_acc.get_pointer()));

             for (int i = ind * 40; i < ind * 40 + 40; i++) {
               ht.insert(tests[i].first, tests[i].second);
             }
           });
     })
        .wait();

    q.submit([&](sycl::handler &cgh) {
       auto tests = sycl::accessor(buffTestUniv, cgh, sycl::read_only);
       auto accChecks = sycl::accessor(checks_buf, cgh, sycl::write_only);
       auto adap_acc = sycl::accessor(adap_buf, cgh, sycl::read_write);

       cgh.parallel_for<class find_test_slab_both_big>(
           r, [=](sycl::nd_item<1> it) [[intel::reqd_sub_group_size(SlabHash::SUBGROUP_SIZE)]] {
             size_t ind = it.get_group().get_id();

             SlabHashTable<uint32_t, uint32_t, DefaultHasher<13, 24, 343>> ht(
                 SlabHash::EMPTY_UINT32_T, it, *(adap_acc.get_pointer()));

             for (int i = ind * 40; i < ind * 40 + 40; i++) {
               auto ans = ht.find(tests[i].first);

               if (it.get_local_id() == 0)
                 accChecks[i] = {static_cast<bool>(ans),
                                 ans.value_or(-1) == tests[i].second};
             }
           });
     })
        .wait();
  }

  for (auto &e : checks) {
    EXPECT_TRUE(e.first && e.second);
  }
}

TEST(SlabHash, find_all_and_sum_with_insert) {
  const size_t test_size = 9;
  std::vector<pair<uint32_t, uint32_t>> testUniv = {
      {0, 2}, {0, 2}, {1, 3}, {2, 5}, {1, 0}, {3, 10}, {0, 22}, {0, 1}, {3, 5}};
  std::vector<uint32_t> keysUniv = {0, 1, 2, 3};

  sycl::queue q{sycl::gpu_selector()};
  sycl::nd_range<1> r_ins{SUBGROUP_SIZE, SUBGROUP_SIZE};
  sycl::nd_range<1> r_sum{SUBGROUP_SIZE * 4, SUBGROUP_SIZE};

  SlabHash::AllocAdapter<std::pair<uint32_t, uint32_t>> adap(SlabHash::CLUSTER_SIZE, 3,
        SlabHash::BUCKETS_COUNT, {SlabHash::EMPTY_UINT32_T, 0}, q);
  std::vector<uint32_t> checks(4);

  {

    sycl::buffer<SlabHash::AllocAdapter<std::pair<uint32_t, uint32_t>>>
        adap_buf(&adap, sycl::range<1>{1});
    sycl::buffer<uint32_t> checks_buf(checks);
    sycl::buffer<pair<uint32_t, uint32_t>> buffTestUniv(testUniv);
    sycl::buffer<uint32_t> buffKeysUniv(keysUniv);

    q.submit([&](sycl::handler &cgh) {
       auto tests = sycl::accessor(buffTestUniv, cgh, sycl::read_only);
       auto accChecks = sycl::accessor(checks_buf, cgh, sycl::write_only);
       auto adap_acc = sycl::accessor(adap_buf, cgh, sycl::read_write);

       cgh.parallel_for<class insert_test_slab_sum>(
           r_ins, [=](sycl::nd_item<1> it) [[intel::reqd_sub_group_size(SlabHash::SUBGROUP_SIZE)]] {
             size_t ind = it.get_group().get_id();

             SlabHashTable<uint32_t, uint32_t, DefaultHasher<13, 24, 343>> ht(
                 SlabHash::EMPTY_UINT32_T, it, *(adap_acc.get_pointer()));

             for (int i = 0; i < test_size; i++) {
               ht.insert(tests[i].first, tests[i].second);
             }
           });
     })
        .wait();

    q.submit([&](sycl::handler &cgh) {
       auto tests = sycl::accessor(buffKeysUniv, cgh, sycl::read_only);
       auto accChecks = sycl::accessor(checks_buf, cgh, sycl::write_only);
       auto adap_acc = sycl::accessor(adap_buf, cgh, sycl::read_write);

       cgh.parallel_for<class find_test_slab_sum>(r_sum, [=](sycl::nd_item<1> it) [[intel::reqd_sub_group_size(SlabHash::SUBGROUP_SIZE)]] {
         size_t ind = it.get_group().get_id();

         SlabHashTable<uint32_t, uint32_t, DefaultHasher<13, 24, 343>> ht(
             SlabHash::EMPTY_UINT32_T, it, *(adap_acc.get_pointer()));

         uint32_t gr_ind = tests[ind];
         auto ans = ht.find_all_and_sum(gr_ind);
         SlabHash::atomic_ref_device<uint32_t> total_ans_ref(accChecks[gr_ind]);
            total_ans_ref.fetch_add(ans);
       });
     })
        .wait();
  }

  EXPECT_EQ(checks[0], 27);
  EXPECT_EQ(checks[1], 3);
  EXPECT_EQ(checks[2], 5);
  EXPECT_EQ(checks[3], 15);
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
