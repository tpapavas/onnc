//===- EliminateSqueeze.h ------------------------------------------------===//
//
//                             The ONNC Project
//
// See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#ifndef ONNC_ELIMINATE_SQUEEZE
#define ONNC_ELIMINATE_SQUEEZE
#include <onnc/Core/CustomPass.h>

namespace onnc {

/** \class EliminateSqueeze
 *  \brief Remove identity nodes in the graph
 */
class EliminateSqueeze: public CustomPass<EliminateSqueeze>
{
public:
  EliminateSqueeze() = default;

  ReturnType runOnModule(Module& pModule) override;

private:
  ReturnType runOnComputeGraph(ComputeGraph& pCG);
};

} // namespace of onnc

#endif // ONNC_ELIMINATE_SQUEEZE
