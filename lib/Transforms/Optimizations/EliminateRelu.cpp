//===- EliminateRelu.cpp ----------------------------------------------===//
//
//                             The ONNC Project
//
// See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include <onnc/Core/PassSupport.h>
#include <onnc/IR/Compute/Conv.h>
#include <onnc/IR/Compute/Relu.h>
#include <onnc/Transforms/Optimizations/EliminateRelu.h>

#include <vector>

using namespace onnc;

//===----------------------------------------------------------------------===//
// EliminateRelu
//===----------------------------------------------------------------------===//
Pass::ReturnType EliminateRelu::runOnModule(Module& pModule)
{
  Pass::ReturnType    ret = Pass::kModuleNoChanged;
  Module::cg_iterator cg, cgEnd = pModule.cgEnd();
  for (cg = pModule.cgBegin(); cg != cgEnd; ++cg) {
    ret |= runOnComputeGraph(*cg->value());
  }

  if (ret != Pass::kModuleNoChanged) {
    pModule.eraseUnusedValues();
  }

  return ret;
}

Pass::ReturnType EliminateRelu::runOnComputeGraph(ComputeGraph& pCG)
{
  std::vector<ComputeOperator*> listOfReluNodes;
  for (ComputeOperator& node : pCG) {
    if (isa<Relu>(&node)) {
      listOfReluNodes.emplace_back(&node);
    }
  }

  // Early return
  if (listOfReluNodes.empty()) {
    return Pass::kModuleNoChanged;
  }

  // Erase nodes here
  for (auto* pReluNode : listOfReluNodes) {
    assert(pReluNode->getNumOfInputs() == 1 && "Relu must have exactly one input");
    assert(pReluNode->getNumOfOutputs() == 1 && "Relu must have exactly one output");

    Value* inV  = pReluNode->getInput(0);
    Define* def = inV->getDefine();     // producer of this input value
    ComputeOperator* pred = dynamic_cast<ComputeOperator*>(def);
    if (!pred || !isa<Conv>(pred)) {
      continue;
    }
    Conv* conv_op = dynamic_cast<Conv*>(pred);
    conv_op->addRelu();

    Value* outV = pReluNode->getOutput(0);
    outV->replaceAllUsesWith(*inV);

    pReluNode->removeAllInputs();
    pReluNode->removeAllOutputs();
    pCG.erase(*pReluNode);
  }

  return Pass::kModuleChanged;
}
