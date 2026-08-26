//===- NvFullInt8.cpp -----------------------------------------------------===//
//
//                             The ONNC Project
//
// See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// INT8 precision variant of the nv_full config. Same RTL geometry as
// Config/NvFull.cpp, but ELEMENT_SIZE = 1 and DATA_TYPE = DataType_INT8 so the
// loadable is laid out and tagged as int8. Selected via ConfigSet::nv_full_int8
// (see NvDlaBackend.cpp; ONNC_NVDLA_PRECISION=int8). The numeric quantization
// (weights/bias/converters) is added by later patches in patches/onnc/.
//
#include "NvDlaDefine.h"
#include "include/nvdla/IRuntime.h"

namespace onnc {

template <>
NvDlaConstants getConfig<nvdla::ConfigSet::nv_full_int8>(nvdla::ExecutionMode mode, bool enableLayerFusion)
{
  NvDlaConstants config;

  config.CONFIG_SET              = nvdla::ConfigSet::nv_full_int8;
  config.EXECUTION_MODE          = mode;
  config.IS_LAYER_FUSION_ENABLED = enableLayerFusion;
  config.DLA_PRECISION           = PRECISION_INT8;
  config.FEATURE_ATOM_CUBE_SIZE  = 32;
  config.WEIGHT_ATOM_CUBE_SIZE   = 128;
  config.ELEMENT_SIZE            = 1;
  config.MAC_ATOMIC_C            = 64;
  config.MAC_ATOMIC_K            = 32; // nv_full atomicKSize=32 (int8 does NOT halve like fp16; == FEATURE_ATOM_CUBE_SIZE/ELEMENT_SIZE)
  config.CBUF_BANK_NUM           = 16;
  config.CBUF_BANK_WIDTH         = 128;
  config.CBUF_BANK_DEPTH         = 256;
  config.MAX_MEM_TRANSACTION_NUM = 8;
  config.DATA_TYPE               = ::nvdla::loadable::DataType::DataType_INT8;
  config.INPUT_PIXEL_FORMAT  = TENSOR_PIXEL_FORMAT_FEATURE;
  config.OUTPUT_PIXEL_FORMAT = TENSOR_PIXEL_FORMAT_FEATURE;

  return config;
}

} // namespace onnc
