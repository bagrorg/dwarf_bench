#include "register_dwarfs.hpp"
#include "common/registry.hpp"
#include "constant/constant.hpp"
#include "groupby/groupby.hpp"
#include "hash/cuckoo_hash_build.hpp"
#include "hash/hash_build.hpp"
#include "hash/hash_build_non_bitmask.hpp"
#include "hash/slab_hash_build.hpp"
#include "join/nested_join.hpp"
#include "join/slab_join.hpp"
#include "probe/slab_probe.hpp"
#include "radix/radix.hpp"
#include "reduce/reduce.hpp"
#include "scan/scan.hpp"

void populate_registry() {
  auto registry = Registry::instance();
  registry->registerd(new TwoPassScan());
  registry->registerd(new ConstantExample());
  registry->registerd(new ConstantExampleCAPI());

#ifdef DPCPP_ENABLED
  registry->registerd(new ConstantExampleDPCPP());
  registry->registerd(new DPLScan());
  registry->registerd(new Radix());
  registry->registerd(new ReduceDPCPP());
  registry->registerd(new HashBuild());
  registry->registerd(new SlabHashBuild());
  registry->registerd(new NestedLoopJoin());
  registry->registerd(new SlabJoin());
  registry->registerd(new SlabProbe());
  registry->registerd(new CuckooHashBuild());
  registry->registerd(new GroupBy());
  registry->registerd(new HashBuildNonBitmask());
#ifdef CUDA_ENABLED
  registry->registerd(new ConstantExampleDPCPPCuda());
  registry->registerd(new DPLScanCuda());
  registry->registerd(new RadixCuda());
#endif
#endif
}