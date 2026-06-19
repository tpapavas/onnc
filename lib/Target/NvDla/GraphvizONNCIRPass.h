//===- GraphvizONNCIRPass.h --------------------------------------------------===//
//
//                             The ONNC Project
//
// See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#ifndef ONNC_NVDLA_GRAPHVIZ_ONNC_IR_PASS_H
#define ONNC_NVDLA_GRAPHVIZ_ONNC_IR_PASS_H
#include <onnc/Core/CustomPass.h>

namespace onnc {

class GraphvizONNCIRPass : public CustomPass<GraphvizONNCIRPass>
{
public:
  GraphvizONNCIRPass() = default;

  ReturnType runOnModule(Module& pModule) override;
};

}

#endif