#include "result.hpp"
#include <fstream>

std::ostream &operator<<(std::ostream &os, const Result &res) {
  os << "Kernel duration: " << ((double)res.kernel_time) / 1000.0 << " us\n"
     << "Host duration:   " << res.host_time.count() << " us\n";
  if (res.isJoin) {
    os << "Build time: " << res.build_time.count() << " us\n"
       << "Probe time: " << res.probe_time.count() << " us\n";
  }
  return os;
}

MeasureResults::const_iterator MeasureResults::begin() const {
  return results_.begin();
}

MeasureResults::const_iterator MeasureResults::end() const {
  return results_.end();
}

void MeasureResults::add_result(DwarfParams params, Result &&result) {
  results_.push_back({params, std::move(result)});
}

void MeasureResults::write_csv(const std::string &filename) const {
  bool exists = std::ifstream(filename).good();
  std::ofstream of(filename, std::ios::app);
  if (of.is_open()) {
    if (!exists)
      of << "device_type,buf_size_bytes,host_time_ms,kernel_time_ms\n";
    for (const auto &res : results_) {
      of << res.params.at("device_type") << ","
         << std::stoi(res.params.at("buf_size")) * 4 << ",";
      of << res.result.host_time.count() / 1000.0 << ","
         << ((double)res.result.kernel_time) / (1000.0 * 1000.0) << "\n";
    }
  } else {
    throw std::runtime_error("Could not open the file at " + filename);
  }
}