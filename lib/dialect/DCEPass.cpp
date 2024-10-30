#include "dialect/Dialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"


#include "dialect/Passes.h"
#include "dialect/EffectInterface.h"
#include "dialect/LiveVarAnalysis.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include <stdexcept>

namespace mlir {
namespace sysy {

class DCEPass
    : public PassWrapper<DCEPass, OperationPass<mlir::func::FuncOp>> {
private:
  void runOnOperation() override;
};


void DCEPass::runOnOperation() {
  auto funcOp = getOperation();
  bool changed = true;

  while(changed){
    changed = false;
    DataFlowSolver solver;
    solver.load<LiveVarAnalysis>();
    if (failed(solver.initializeAndRun(funcOp)))
      return signalPassFailure();

    llvm::SmallVector<Operation*, 4> opToRemove;
    
    funcOp.walk([&](Operation* op) {
      if(!llvm::isa<sysy::SysyDialect>(op->getDialect()))
        return WalkResult::skip();

      const LiveVarState* state;
      if (Operation *next = op->getNextNode())
        state = solver.lookupState<LiveVarState>(next);
      else
        state = solver.lookupState<LiveVarState>(op->getBlock());

      // llvm::outs() << *op << "\n";
      // for(auto LiveVar : state->getValue()){
      //   llvm::outs() << LiveVar << "\n";
      // }
      // llvm::outs() << "\n";

      auto curOp = dyn_cast<sysy::EffectInterface>(op);
      auto writeValue = curOp.getWriteValue();
      auto liveVars = state->getValue();
      if(writeValue && liveVars.find(writeValue) == liveVars.end()){
        opToRemove.push_back(op);
      }
      return WalkResult::advance();
    });

    for(auto op : opToRemove){
      // llvm::outs() << op << ": " << *op << "\n";
      op->erase();
      changed = true;
    }
  }
}

std::unique_ptr<Pass> createDCEPass() {
  return std::make_unique<DCEPass>();
}

} // namespace sysy
} // namespace mlir