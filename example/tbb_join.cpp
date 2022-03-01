#include <oneapi/tbb/concurrent_unordered_map.h>
#include "oneapi/tbb/blocked_range.h"
#include "oneapi/tbb/parallel_for.h"
#include "oneapi/tbb/concurrent_vector.h"
#include "oneapi/tbb/parallel_for.h"

#include <iostream>

struct triple {
    int x, y, z;
};


int main() {
    tbb::concurrent_unordered_multimap<int, int> ht;

    std::vector<int> keys_a = {1, 2, 3};
    std::vector<int> vals_a = {10, 20, 30};

    std::vector<int> keys_b = {3, 2, 2, 1, 1, 3, 2, 5};
    std::vector<int> vals_b = {1, 1, 2, 1, 2, 2, 3, 1};

    tbb::concurrent_vector<triple> ans;


    tbb::parallel_for( tbb::blocked_range<int>(0, keys_a.size()),
                       [&](tbb::blocked_range<int> r)
    {
        for (int i=r.begin(); i<r.end(); ++i)
        {
            std::pair<int, int> pp {keys_a[i], vals_a[i]};
            ht.insert(std::move(pp));
        }
    });


    tbb::parallel_for( tbb::blocked_range<int>(0, keys_b.size()),
                       [&](tbb::blocked_range<int> r)
    {
        for (int i=r.begin(); i<r.end(); ++i)
        {
            auto it = ht.find(keys_b[i]);

            while (it != ht.end() && it->first == keys_b[i]) {
                triple row = {keys_b[i], vals_b[i], it->second};
                ans.push_back(row);
                it++;
            }
        }
    });

    for (auto e: ans) {
        std::cout << e.x << ' ' << e.y << ' ' << e.z << std::endl;
    }
}