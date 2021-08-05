#include "register_dwarfs.hpp"
#include "common/registry.hpp"
#include "constant/constant.hpp"
#include "hash/hash_build.hpp"
#include "hash/slab_hash_build.hpp"
#include "radix/radix.hpp"
#include "reduce/reduce.hpp"
#include "scan/scan.hpp"
#include "hash/slab_hash_build.hpp"
#include "join/join.hpp"

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
  registry->registerd(new Join());
#ifdef CUDA_ENABLED
  registry->registerd(new ConstantExampleDPCPPCuda());
  registry->registerd(new DPLScanCuda());
  registry->registerd(new RadixCuda());
#endif
#endif
}