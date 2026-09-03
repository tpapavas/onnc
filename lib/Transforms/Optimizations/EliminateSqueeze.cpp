//===- EliminateSqueeze.cpp ----------------------------------------------===//
//
//                             The ONNC Project
//
// See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include <onnc/Core/PassSupport.h>
#include <onnc/IR/Compute/Conv.h>
#include <onnc/IR/Compute/Squeeze.h>
#include <onnc/Transforms/Optimizations/EliminateSqueeze.h>

#include <vector>

using namespace onnc;

//===----------------------------------------------------------------------===//
// EliminateSqueeze
//===----------------------------------------------------------------------===//
Pass::ReturnType EliminateSqueeze::runOnModule(Module& pModule)
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

Pass::ReturnType EliminateSqueeze::runOnComputeGraph(ComputeGraph& pCG)
{
  std::vector<ComputeOperator*> listOfSqueezeNodes;
  for (ComputeOperator& node : pCG) {
    if (isa<Squeeze>(&node)) {
      listOfSqueezeNodes.emplace_back(&node);
    }
  }

  // Early return
  if (listOfSqueezeNodes.empty()) {
    return Pass::kModuleNoChanged;
  }

  // Erase nodes here
  for (auto* pSqueezeNode : listOfSqueezeNodes) {
    assert(pSqueezeNode->getNumOfInputs() == 1 && "Squeeze must have exactly one input");
    assert(pSqueezeNode->getNumOfOutputs() == 1 && "Squeeze must have exactly one output");

    Value* inV  = pSqueezeNode->getInput(0);
    Value* outV = pSqueezeNode->getOutput(0);
    outV->replaceAllUsesWith(*inV);

    pSqueezeNode->removeAllInputs();
    pSqueezeNode->removeAllOutputs();
    pCG.erase(*pSqueezeNode);
  }

  return Pass::kModuleChanged;
}
