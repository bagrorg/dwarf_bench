#pragma once
#include "options.hpp"
#include "result.hpp"

class Meter {
public:
  Meter(const std::string &dwarf_name, MeasureResults &result)
      : dwarf_name_(dwarf_name), result_(result) {}
  void add_result(DwarfParams &&params, std::unique_ptr<Result> result);
  void set_params(DwarfParams params);
  void set_opts(std::unique_ptr<RunOptions> ptr);
  std::shared_ptr<RunOptions> opts() const;

private:
  const std::string dwarf_name_;
  MeasureResults &result_;
  DwarfParams params_;
  std::shared_ptr<RunOptions> opts_;
};