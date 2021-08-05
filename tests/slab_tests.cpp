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

  std::vector<SlabList<pair<uint32_t, uint32_t>>> lists(BUCKETS_COUNT);
  for (auto &e : lists) {
    e.root = sycl::global_ptr<SlabNode<pair<uint32_t, uint32_t>>>(
        sycl::malloc_shared<SlabNode<pair<uint32_t, uint32_t>>>(CLUSTER_SIZE,
                                                                q));

    for (int i = 0; i < CLUSTER_SIZE - 1; i++) {
      *(e.root + i) = SlabNode<pair<uint32_t, uint32_t>>({EMPTY_UINT32_T, 0});
      (e.root + i)->next = (e.root + i + 1);
    }
  }

  {
    sycl::buffer<SlabList<pair<uint32_t, uint32_t>>> ls(lists);
    sycl::buffer<sycl::device_ptr<SlabNode<pair<uint32_t, uint32_t>>>> its(3);
    sycl::buffer<pair<uint32_t, uint32_t>> buffTestUniv(testUniv);

    q.submit([&](sycl::handler &cgh) {
       auto l = sycl::accessor(ls, cgh, sycl::read_write);
       auto tests = sycl::accessor(buffTestUniv, cgh, sycl::read_only);
       auto itrs = sycl::accessor(its, cgh, sycl::read_write);

       cgh.parallel_for<class insert_test_slab>(r, [=](sycl::nd_item<1> it) {
         size_t ind = it.get_group().get_id();
         DefaultHasher<13, 24, 343> h;
         SlabHashTable<uint32_t, uint32_t, DefaultHasher<13, 24, 343>> ht(
             EMPTY_UINT32_T, h, l.get_pointer(), it,
             itrs[it.get_group().get_id()]);

         for (int i = ind * 2; i < ind * 2 + 2; i++) {
           ht.insert(tests[i].first, tests[i].second);
         }
       });
     }).wait();
  }

  bool allOk = true;
  DefaultHasher<13, 24, 343> h;
  for (auto &e : testUniv) {
    auto r = lists[h(e.first)].root;

    for (int i = 0; i < SLAB_SIZE; i++) {
      if (r->data[i] == e)
        break;
      if (i == SLAB_SIZE - 1)
        allOk = false;
    }
  }

  EXPECT_TRUE(allOk);

  for (auto &e : lists) {
    sycl::free(e.root, q);
  }
}

TEST(SlabHash, find) {
  sycl::queue q{sycl::gpu_selector()};
  sycl::nd_range<1> r{SUBGROUP_SIZE * 3, SUBGROUP_SIZE};

  std::vector<SlabList<pair<uint32_t, uint32_t>>> lists(BUCKETS_COUNT);
  for (auto &e : lists) {
    e.root = sycl::global_ptr<SlabNode<pair<uint32_t, uint32_t>>>(
        sycl::malloc_shared<SlabNode<pair<uint32_t, uint32_t>>>(CLUSTER_SIZE,
                                                                q));

    for (int i = 0; i < CLUSTER_SIZE - 1; i++) {
      *(e.root + i) = SlabNode<pair<uint32_t, uint32_t>>({EMPTY_UINT32_T, 0});
      (e.root + i)->next = (e.root + i + 1);
    }
  }

  std::vector<pair<uint32_t, uint32_t>> testUniv = {
      {1, 2}, {5, 2}, {101, 3}, {21312, 5}, {3, 0}, {10, 10}};

  DefaultHasher<13, 24, 343> h;

  for (auto &e : testUniv) {
    auto r = lists[h(e.first)].root;

    for (int i = 0; i < SLAB_SIZE; i++) {
      if (r->data[i].first == EMPTY_UINT32_T) {
        r->data[i] = e;
        break;
      }
    }
  }
  std::vector<pair<bool, bool>> checks(6);
  {
    sycl::buffer<SlabList<pair<uint32_t, uint32_t>>> ls(lists);
    sycl::buffer<sycl::device_ptr<SlabNode<pair<uint32_t, uint32_t>>>> its(3);
    sycl::buffer<pair<uint32_t, uint32_t>> buffTestUniv(testUniv);
    sycl::buffer<pair<bool, bool>> buffChecks(checks);

    q.submit([&](sycl::handler &cgh) {
       auto l = sycl::accessor(ls, cgh, sycl::read_write);
       auto itrs = sycl::accessor(its, cgh, sycl::read_write);
       auto tests = sycl::accessor(buffTestUniv, cgh, sycl::read_only);
       auto accChecks = sycl::accessor(buffChecks, cgh, sycl::write_only);

       cgh.parallel_for<class find_test_slab>(r, [=](sycl::nd_item<1> it) {
         size_t ind = it.get_group().get_id();
         DefaultHasher<13, 24, 343> h;
         SlabHashTable<uint32_t, uint32_t, DefaultHasher<13, 24, 343>> ht(
             EMPTY_UINT32_T, h, l.get_pointer(), it,
             itrs[it.get_group().get_id()]);

         for (int i = ind * 2; i < ind * 2 + 2; i++) {
           auto ans = ht.find(tests[i].first);

           if (it.get_local_id() == 0)
             accChecks[i] = {static_cast<bool>(ans),
                             ans.value_or(-1) == tests[i].second};
         }
       });
     }).wait();
  }

  for (int i = 0; i < 6; i++) {
    EXPECT_TRUE(checks[i].first && checks[i].second);
  }

  for (auto &e : lists) {
    sycl::free(e.root, q);
  }
}

TEST(SlabHash, find_and_insert_together) {
  std::vector<pair<uint32_t, uint32_t>> testUniv = {
      {1, 2}, {5, 2}, {101, 3}, {10932, 5}, {3, 0}, {10, 10}};

  sycl::queue q{sycl::gpu_selector()};
  sycl::nd_range<1> r{SUBGROUP_SIZE * 3, SUBGROUP_SIZE};

  AllocAdapter<std::pair<uint32_t, uint32_t>> adap(BUCKETS_COUNT, {EMPTY_UINT32_T, 0}, q);

  {
    sycl::buffer<SlabList<pair<uint32_t, uint32_t>>> ls(adap._data);
    sycl::buffer<sycl::device_ptr<SlabNode<pair<uint32_t, uint32_t>>>> its(3);
    sycl::buffer<pair<uint32_t, uint32_t>> buffTestUniv(testUniv);

    q.submit([&](sycl::handler &cgh) {
       auto l = sycl::accessor(ls, cgh, sycl::read_write);
       auto tests = sycl::accessor(buffTestUniv, cgh, sycl::read_only);
       auto itrs = sycl::accessor(its, cgh, sycl::read_write);

       cgh.parallel_for<class insert_test_slab_both>(
           r, [=](sycl::nd_item<1> it) {
             size_t ind = it.get_group().get_id();
             DefaultHasher<13, 24, 343> h;
             SlabHashTable<uint32_t, uint32_t, DefaultHasher<13, 24, 343>> ht(
                 EMPTY_UINT32_T, h, l.get_pointer(), it,
                 itrs[it.get_group().get_id()]);

             for (int i = ind * 2; i < ind * 2 + 2; i++) {
               ht.insert(tests[i].first, tests[i].second);
             }
           });
     }).wait();
  }

  std::vector<pair<bool, bool>> checks(6);
  {
    sycl::buffer<SlabList<pair<uint32_t, uint32_t>>> ls(adap._data);
    sycl::buffer<sycl::device_ptr<SlabNode<pair<uint32_t, uint32_t>>>> its(3);
    sycl::buffer<pair<uint32_t, uint32_t>> buffTestUniv(testUniv);
    sycl::buffer<pair<bool, bool>> buffChecks(checks);

    q.submit([&](sycl::handler &cgh) {
       auto l = sycl::accessor(ls, cgh, sycl::read_write);
       auto itrs = sycl::accessor(its, cgh, sycl::read_write);
       auto tests = sycl::accessor(buffTestUniv, cgh, sycl::read_only);
       auto accChecks = sycl::accessor(buffChecks, cgh, sycl::write_only);

       cgh.parallel_for<class find_test_slab_both>(r, [=](sycl::nd_item<1> it) {
         size_t ind = it.get_group().get_id();
         DefaultHasher<13, 24, 343> h;
         SlabHashTable<uint32_t, uint32_t, DefaultHasher<13, 24, 343>> ht(
             EMPTY_UINT32_T, h, l.get_pointer(), it,
             itrs[it.get_group().get_id()]);

         for (int i = ind * 2; i < ind * 2 + 2; i++) {
           auto ans = ht.find(tests[i].first);

           if (it.get_local_id() == 0)
             accChecks[i] = {static_cast<bool>(ans),
                             ans.value_or(-1) == tests[i].second};
         }
       });
     }).wait();
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

  AllocAdapter<std::pair<uint32_t, uint32_t>> adap(BUCKETS_COUNT, {EMPTY_UINT32_T, 0}, q);

  {
    sycl::buffer<SlabList<pair<uint32_t, uint32_t>>> ls(adap._data);
    sycl::buffer<sycl::device_ptr<SlabNode<pair<uint32_t, uint32_t>>>> its(25);
    sycl::buffer<pair<uint32_t, uint32_t>> buffTestUniv(testUniv);

    q.submit([&](sycl::handler &cgh) {
       auto l = sycl::accessor(ls, cgh, sycl::read_write);
       auto tests = sycl::accessor(buffTestUniv, cgh, sycl::read_only);
       auto itrs = sycl::accessor(its, cgh, sycl::read_write);

       cgh.parallel_for<class insert_test_slab_both_big>(
           r, [=](sycl::nd_item<1> it) {
             size_t ind = it.get_group().get_id();
             DefaultHasher<13, 24, 343> h;
             SlabHashTable<uint32_t, uint32_t, DefaultHasher<13, 24, 343>> ht(
                 EMPTY_UINT32_T, h, l.get_pointer(), it,
                 itrs[it.get_group().get_id()]);

             for (int i = ind * 40; i < ind * 40 + 40; i++) {
               ht.insert(tests[i].first, tests[i].second);
             }
           });
     }).wait();
  }

  std::vector<pair<bool, bool>> checks(1000);
  {
    sycl::buffer<SlabList<pair<uint32_t, uint32_t>>> ls(adap._data);
    sycl::buffer<sycl::device_ptr<SlabNode<pair<uint32_t, uint32_t>>>> its(25);
    sycl::buffer<pair<uint32_t, uint32_t>> buffTestUniv(testUniv);
    sycl::buffer<pair<bool, bool>> buffChecks(checks);

    q.submit([&](sycl::handler &cgh) {
       auto l = sycl::accessor(ls, cgh, sycl::read_write);
       auto itrs = sycl::accessor(its, cgh, sycl::read_write);
       auto tests = sycl::accessor(buffTestUniv, cgh, sycl::read_only);
       auto accChecks = sycl::accessor(buffChecks, cgh, sycl::write_only);

       cgh.parallel_for<class find_test_slab_both_big>(
           r, [=](sycl::nd_item<1> it) {
             size_t ind = it.get_group().get_id();
             DefaultHasher<13, 24, 343> h;
             SlabHashTable<uint32_t, uint32_t, DefaultHasher<13, 24, 343>> ht(
                 EMPTY_UINT32_T, h, l.get_pointer(), it,
                 itrs[it.get_group().get_id()]);

             for (int i = ind * 40; i < ind * 40 + 40; i++) {
               auto ans = ht.find(tests[i].first);

               if (it.get_local_id() == 0)
                 accChecks[i] = {static_cast<bool>(ans),
                                 ans.value_or(-1) == tests[i].second};
             }
           });
     }).wait();
  }

  for (auto &e : checks) {
    EXPECT_TRUE(e.first && e.second);
  }
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
