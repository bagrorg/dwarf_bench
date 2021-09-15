#include "result.hpp"
#include <fstream>

std::ostream &operator<<(std::ostream &os, const Result &res) {
  return res.print_to_stream(os);
}

std::ostream &Result::print_to_stream(std::ostream &os) const {
  os << "Kernel duration: " << ((double)kernel_time) / 1000.0 << " us\n"
     << "Host duration:   " << host_time.count() << " us\n";

  return os;
}

std::ostream &HashJoinResult::print_to_stream(std::ostream &os) const {
  Result::print_to_stream(os);

  os << "Build time: " << build_time.count() << " us\n"
       << "Probe time: " << probe_time.count() << " us\n";

  return os;
}

MeasureResults::const_iterator MeasureResults::begin() const {
  return results_.begin();
}

MeasureResults::const_iterator MeasureResults::end() const {
  return results_.end();
}

void MeasureResults::add_result(DwarfParams params, std::unique_ptr<Result> result) {
  results_.push_back({params, std::move(result)});
}

void MeasureResults::write_csv(const std::string &filename) const {
  bool exists = std::ifstream(filename).good();
  std::ofstream of(filename, std::ios::app);
  if (of.is_open()) {
    if (!exists)
      of << "device_type,buf_size_bytes,host_time_ms,kernel_time_ms,scale,buckets_count,memory_utilization,avg_slab,subgroup_size\n";
    for (const auto &res : results_) {
      of << res.params.at("device_type") << ","
         << std::stoi(res.params.at("buf_size")) * 8 << ",";
      of << res.result->host_time.count() / 1000.0 << ","
         << ((double)res.result->kernel_time) / (1000.0 * 1000.0) << ",";
      of << res.params.at("scale") << ",";
      of << res.params.at("buckets_count") << ",";
      of << res.params.at("memory_utilization") << ",";
      of << res.params.at("avg_slab") << ",";
      of << res.params.at("subgroup_size") << "\n";
    }
  } else {
    throw std::runtime_error("Could not open the file at " + filename);
  }
}