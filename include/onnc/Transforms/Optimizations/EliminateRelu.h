//===- EliminateRelu.h ------------------------------------------------===//
//
//                             The ONNC Project
//
// See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#ifndef ONNC_ELIMINATE_RELU
#define ONNC_ELIMINATE_RELU
#include <onnc/Core/CustomPass.h>

namespace onnc {

/** \class EliminateRelu
 *  \brief Remove Relu nodes in the graph
 */
class EliminateRelu: public CustomPass<EliminateRelu>
{
public:
  EliminateRelu() = default;

  ReturnType runOnModule(Module& pModule) override;

private:
  ReturnType runOnComputeGraph(ComputeGraph& pCG);
};

} // namespace of onnc

#endif // ONNC_ELIMINATE_RELU
