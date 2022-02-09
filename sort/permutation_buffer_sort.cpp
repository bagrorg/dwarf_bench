#include <oneapi/tbb/parallel_sort.h>

#include "sort/permutation_buffer_sort.hpp"

namespace {
template <typename T> std::vector<T> expected_out(const std::vector<T> &v) {
  std::vector<int> out = v;
  std::sort(out.begin(), out.end());
  return out;
}

void in_place_permutation(std::vector<int> &v, std::vector<size_t> &permutation) {
    for (size_t i = 0; i < v.size(); i++) {
        int current = i;
        int next = permutation[i];

        while (next != i) {
            std::swap(v[current], v[next]);
            permutation[current] = current;
            current = next;
            next = permutation[current];
        }
        permutation[current] = current;
    }
}
} // namespace

PermutationBufferSort::PermutationBufferSort() : Dwarf("PermutationBufferSort") {}

void PermutationBufferSort::_run(const size_t buf_size, Meter &meter) {
  auto opts = meter.opts();
  std::vector<int> host_src = helpers::make_random<int>(buf_size);
  const std::vector<int> expected = expected_out(host_src);

  for (auto it = 0; it < opts.iterations; ++it) {
    auto host_start = std::chrono::steady_clock::now();
    std::vector<size_t> permutation_buffer(buf_size);
    std::iota(permutation_buffer.begin(), permutation_buffer.end(), 0);
    oneapi::tbb::parallel_sort(permutation_buffer.begin(), permutation_buffer.end(),
        [&host_src](size_t left, size_t right) {
            return host_src[left] < host_src[right];
    });
    in_place_permutation(host_src, permutation_buffer);
    auto host_end = std::chrono::steady_clock::now();
    auto host_exe_time = std::chrono::duration_cast<std::chrono::microseconds>(
                             host_end - host_start)
                             .count();

    std::unique_ptr<Result> result = std::make_unique<Result>();
    result->host_time = host_end - host_start;
    DwarfParams params{{"buf_size", std::to_string(buf_size)}};

    {
      if (!helpers::check_first(host_src, expected, expected.size())) {
        std::cerr << "incorrect results" << std::endl;
        result->valid = false;
      }
    }
    meter.add_result(std::move(params), std::move(result));
  }
}

void PermutationBufferSort::run(const RunOptions &opts) {
  for (auto size : opts.input_size) {
    _run(size, meter());
  }
}

void PermutationBufferSort::init(const RunOptions &opts) {
  meter().set_opts(opts);
  DwarfParams params = {{"device_type", to_string(opts.device_ty)}};
  meter().set_params(params);
}