#pragma once
#include "meter.hpp"
#include "options.hpp"
#include "result.hpp"

class Dwarf {
public:
  Dwarf(const std::string &name)
      : name_(name), results_(name), meter_(name, results_) {}
  virtual ~Dwarf() = default;

  virtual void init(std::unique_ptr<RunOptions> optsPtr) = 0;

  const std::string &name() const { return name_; }

  void run() {
    generateData_();
    measurePerformance_();
    fillResults_();
  }

  void report() {
    std::shared_ptr<RunOptions> opts = meter_.opts();
    if (opts->report_path.empty()) {
      for (const auto &res : results_) {
        std::cout << *res.result;
      }
    } else {
      results_.write_csv(opts->report_path);
    }
  }

  Meter &meter() { return meter_; }
protected:
  virtual void run_() = 0;
  virtual void generateData_() = 0;
  virtual void fillResults_() = 0;
  virtual void measurePerformance_() = 0;

private:
  std::string name_;
  MeasureResults results_;
  Meter meter_;
};