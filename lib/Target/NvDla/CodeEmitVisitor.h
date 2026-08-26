//===- CodeEmitVisitor.h --------------------------------------------------===//
//
//                             The ONNC Project
//
// See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#ifndef TARGET_NVDLA_CODE_EMIT_VISITOR_H
#define TARGET_NVDLA_CODE_EMIT_VISITOR_H

#include "NvDlaDefine.h"
#include "NvDlaMeta.h"
#include "Compute/NvDlaShuffle.h"

#include <onnc/IR/Compute/Initializer.h>
#include <onnc/IR/Compute/InputOperator.h>
#include <onnc/IR/Compute/OutputOperator.h>
#include <onnc/IR/Compute/Tensor.h>
#include <onnc/IR/CustomVisitor.h>
#include <onnc/Support/Preprocessor.h>
#include <onnc/Support/Span.h>

#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#ifndef PP_NVDLA_OP_LIST
#  define PP_NVDLA_OP_LIST                                                                              \
    (Add, AveragePool, Concat, Conv, Initializer, InputOperator, LRN, MaxPool, Mul, \
     OutputOperator, Relu, Reshape, Softmax, Split, Sum, Transpose)
#else
#  error "already defined PP_NVDLA_OP_LIST macro before"
#endif

#define PP_DECL_VISIT(type)         \
  void visit(const type&) override; \
  void visitImpl(const type&);

namespace onnc {
namespace nvdla {

class CodeEmitVisitor : public CustomVisitor<CodeEmitVisitor>, private NvDlaConstants
{
public:
  using CbufAllocTypeGetter =
    std::function<CbufAllocType(const NvDlaCubeInfo&, const NvDlaCubeInfo&, Tensor::Dimension, unsigned&)>;

public:
  CodeEmitVisitor(const NvDlaConstants& constants, NvDlaBackendMeta& meta, CbufAllocTypeGetter cbufAllocTypeGetter,
                  unsigned verboseLevel) noexcept
    : NvDlaConstants{constants}
    , m_pMeta{meta}
    , m_CbufAllocTypeGetter{std::move(cbufAllocTypeGetter)}
    , m_VerboseLevel{verboseLevel}
  {
    loadCalibrationTable();
  }

  PP_LIST_FOR_EACH(PP_DECL_VISIT, PP_UNWRAP(PP_NVDLA_OP_LIST))

  void visit(NvDlaShuffle& op) { visit(const_cast<const NvDlaShuffle&>(op)); }
  void visit(const NvDlaShuffle&);
  void visitImpl(const NvDlaShuffle&);

  // Per-output-filter INT8 weight scales SF_wt[k] = max_abs(filter_k) / 127,
  // recorded when a weight tensor is packed as INT8 (see packWeight). The SDP
  // converters need these to realize NVDLA's requantization
  // out_rescale = SF_in * SF_wt / SF_out (programmed in a later patch).
  // Returns nullptr if the tensor was not packed as INT8 weights.
  const std::vector<float>* getWeightScales(const Tensor* weight) const;

  // Per-tensor INT8 activation scale SF for `tensorName`, from the calibration
  // table (calib.json). Returns 0.0f if the table has no entry for that name
  // (callers treat 0 as "no scale" -> fall back, see the converter patch).
  float getActivationScale(const std::string& tensorName) const;

  // Like getActivationScale, but if `tensor` has no scale under its own name,
  // follow a single shaping consumer (Reshape/Squeeze/Flatten/Unsqueeze/Transpose)
  // or a Relu (gemm->conv+relu -> relu-output scale) to inherit its scale. ReplaceGemmByConv names the conv
  // output `replace_gemm_by_conv_N` (absent from calib.json) and reshapes it to
  // the calibrated graph output; without this the final layer falls back to an
  // identity converter and its int32 accumulator saturates to +/-127. Returns 0
  // if no scale is reachable.
  float getActivationScaleResolved(const Tensor* tensor) const;

private:
  MemoryListEntryId packWeight(const Tensor& weight, NvDlaDims destDims, Tensor::Dimension numFrontPaddingChannels,
                               Tensor::Dimension outputChannelOffset);
  MemoryListEntryId packImageWeight(const Tensor& weight, NvDlaDims destDims, Tensor::Dimension outputChannelOffset);
  MemoryListEntryId packBias(const Tensor& bias, Tensor::Dimension numDestChannels,
                             Tensor::Dimension srcChannelOffset = 0);
  // Emit a per-kernel INT8 requantization operand cube: one int16 multiplier per
  // output channel (the SDP per-kernel MUL operand realizing SF_in*SF_wt[k]/SF_out).
  MemoryListEntryId packRequantMul(const std::vector<std::int16_t>& mult, Tensor::Dimension numDestChannels,
                                   Tensor::Dimension srcChannelOffset = 0);
  MemoryListEntryId packSDPOperand(const Tensor* aluTensor, const Tensor* mulTensor, const NvDlaCubeInfo& cubeInfo);

  MemoryListEntryId  packFeature(const Tensor& tensor, const NvDlaCubeInfo& cube);
  void               issueEmuOp(NvDlaEmuOperation* op);
  AddressListEntryId issueEmuAddr(MemoryListEntryId mid);
  void               issueDlaOp(NvDlaDlaOperation* op, NvDlaDlaOperation* op_fuse, NvDlaDlaOperation* op_prev);
  void               issueDlaOp(std::unique_ptr<NvDlaDlaOperation> op);
  AddressListEntryId issueDlaAddr(const Tensor& tensor, const NvDlaCubeInfo& cube, Tensor::Dimension channelOffset,
                                  NvDlaBackendMeta::Offset hOffset);
  AddressListEntryId issueDlaAddr(const Tensor& tensor, const NvDlaCubeInfo& cube);
  AddressListEntryId issueDlaAddr(MemoryListEntryId memoryId, const NvDlaCubeInfo& cube);
  AddressListEntryId issueSDPOperand(const Tensor& tensor, const NvDlaCubeInfo& cube, MemoryListEntryId& memoryId);

  void SetLUTParam(dla_lut_param* lut_param, float alpha, float beta, float bias, int size, float outdata_scale, float outdata_offset);

  // Perform SDP for 2 input tensors and an output tensor,
  // the possible value for parameter 'opType' is:
  //
  //   1. SDP_OP_ALU
  //   2. SDP_OP_MUL
  //
  void emitSdp(const ComputeOperator& op, const Tensor& first, const Tensor& second, const Tensor& output);
  std::pair<unsigned, bool> tryAllocateDataAndWeightsIntoCBuf(const NvDlaCubeInfo& data, NvDlaCubeInfo& weight,
                                                              Tensor::Dimension yDilation) const;

private:
  MemoryListEntryId packWeight(span<const float> weight, const Tensor* weightTensor, NvDlaDims srcDims,
                               NvDlaDims destDims, Tensor::Dimension numFrontPaddingChannels,
                               Tensor::Dimension outputChannelOffset);

  MemoryListEntryId packWeight(const Tensor& weight, NvDlaDims srcDims, NvDlaDims destDims,
                               Tensor::Dimension numFrontPaddingChannels, Tensor::Dimension outputChannelOffset);

  // Compute (once, memoized in m_WeightScales) the per-output-filter INT8 weight
  // scale for `weightTensor`. `weight`/`srcDims` describe the full fp32 weights.
  const std::vector<float>& computeWeightScales(const Tensor* weightTensor, span<const float> weight,
                                                NvDlaDims srcDims);

  // Load per-tensor activation scales from the calibration table named by the
  // ONNC_NVDLA_CALIB env var (the calib.json this repo's 05b produces) into
  // m_ActivationScales. INT8 only; no-op (and never throws) otherwise.
  void loadCalibrationTable();

  template <typename Type>
  void packWeightImpl(Type* destData, NvDlaDims destDimsWithFrontPadding, const Tensor* tensor, const float* srcData,
                      NvDlaDims srcDims, Tensor::Dimension numFrontPaddingChannels,
                      Tensor::Dimension outputChannelOffset);

  template <typename Type>
  void packImageWeightImpl(Type* blob, NvDlaDims blobDims, const Tensor* tensor, const float* srcData,
                           NvDlaDims srcDims, Tensor::Dimension outputChannelOffset);
  template <typename Type>
  void packBiasImpl(Type* destData, Tensor::Dimension numDestChannels, const Tensor* tensor, const float* srcData,
                    Tensor::Dimension srcChannelOffset);
  void packSDPOperandImpl(NvU8* blob, const Tensor* aluTensor, const float* aluData, const Tensor* mulTensor,
                          const float* mulData, const NvDlaCubeInfo& cubeInfo);

private:
  NvDlaBackendMeta&         m_pMeta;
  const CbufAllocTypeGetter m_CbufAllocTypeGetter;
  const unsigned            m_VerboseLevel;

  // weight tensor -> per-output-filter INT8 scale SF_wt[k]. Populated by
  // computeWeightScales during INT8 weight packing.
  std::unordered_map<const Tensor*, std::vector<float>> m_WeightScales;

  // ONNX tensor name -> per-tensor INT8 activation scale SF. Populated by
  // loadCalibrationTable from calib.json (empty when not INT8 / no table).
  std::unordered_map<std::string, float> m_ActivationScales;
};

} // namespace nvdla
} // namespace onnc

#undef PP_DECL_VISIT
#undef PP_NVDLA_OP_LIST

#endif
