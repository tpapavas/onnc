//===- NvSmall.cpp ---------------------------------------------------------===//
//
//                             The ONNC Project
//
// See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include "NvDlaDefine.h"
#include "include/nvdla/IRuntime.h"

#include <onnc/Support/IOStream.h>

namespace onnc {

template <>
NvDlaConstants getConfig<nvdla::ConfigSet::nv_small>(nvdla::ExecutionMode mode, bool enableLayerFusion)
{
  NvDlaConstants config;

  outs() << "[NvDla][int8] nv_small config loaded.\n";

  config.CONFIG_SET              = nvdla::ConfigSet::nv_small;
  config.EXECUTION_MODE          = mode;
  config.IS_LAYER_FUSION_ENABLED = enableLayerFusion;
  config.DLA_PRECISION           = PRECISION_INT8; // PRECISION_FP16;
  config.FEATURE_ATOM_CUBE_SIZE  = 8; // 32;
  // Weight data sizes are rounded up to this many bytes (CodeEmitVisitor::packWeight), and that
  // size is what the driver programs into CDMA's and CSC's D_WEIGHT_BYTES. nv_small's CSC frees
  // weights per group of MAC_ATOMIC_K (8) kernels, rounded up to whole 8-byte CBUF entries; a full
  // group is always a multiple of 8 bytes, so rounding the total to 8 equals exactly what CSC
  // frees. nv_full's 128 made CDMA fetch entries CSC never frees, so CDMA waited forever before
  // the next layer's weights (LeNet conv1: 512 B = 64 entries fetched, 63 freed -> hang at conv2).
  // 8 gives the sizes nvdla_compiler emits for nv_small (LeNet: 504, 25000, 400000, 5000).
  config.WEIGHT_ATOM_CUBE_SIZE   = 8; // nv_full: 128
  config.ELEMENT_SIZE            = 1; // 2;
  config.MAC_ATOMIC_C            = 8; // 64;
  config.MAC_ATOMIC_K            = 8; // 16;
  config.CBUF_BANK_NUM           = 32; // 16;
  config.CBUF_BANK_WIDTH         = 8; // 128;
  config.CBUF_BANK_DEPTH         = 512; // 256;
  config.MAX_MEM_TRANSACTION_NUM = 8;
  config.DATA_TYPE               = ::nvdla::loadable::DataType::DataType_INT8;
  // Layout label of the network input/output tensors. The runtime places the input image (and
  // reads the output) by this label, not by the strides: FEATURE = 32-byte atoms (nv_full, pixel x
  // at byte x*32), FEATURE_X8 = 8-byte atoms (nv_small, x*8), which is what the strides describe
  // and what nvdla_compiler emits for nv_small. With FEATURE, conv1 saw only input columns 0-6.
  config.INPUT_PIXEL_FORMAT  = TENSOR_PIXEL_FORMAT_FEATURE_X8; // nv_full: FEATURE
  config.OUTPUT_PIXEL_FORMAT = TENSOR_PIXEL_FORMAT_FEATURE_X8; // nv_full: FEATURE

  return config;
}

} // namespace onnc
