#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"


#include "dialect/Passes.h"
#include "dialect/EffectInterface.h"
#include "dialect/ReachingDefAnalysis.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include <stdexcept>


namespace mlir {
namespace sysy {

class SimplifyRedundantTransposePass
    : public PassWrapper<SimplifyRedundantTransposePass, OperationPass<mlir::func::FuncOp>> {
private:
  void runOnOperation() override;
};


void SimplifyRedundantTransposePass::runOnOperation() {
  auto funcOp = getOperation();
	DataFlowSolver solver;
	solver.load<ReachingDefAnalysis>();
	if (failed(solver.initializeAndRun(funcOp)))
    return signalPassFailure();

	llvm::SmallVector<sysy::TransposeOp, 4> transposesToRemove;

	// funcOp.walk([&](Operation* op) {
	// 	OpBuilder builder(op);
	// 	const ReachingDefState* state = nullptr;
	// 	if (Operation *prev = op->getPrevNode())
	// 		state = solver.lookupState<ReachingDefState>(prev);
	// 	else
	// 		state = solver.lookupState<ReachingDefState>(op->getBlock());
	// 	if(!state) 
	// 		return WalkResult::skip();
	// 	llvm::outs() << *op << "\n";
	// 	for(auto reachingDef : state->getValue()){
	// 		llvm::outs() << *reachingDef << "\n";
	// 	}
	// 	llvm::outs() << "\n";
	// 	return WalkResult::advance();
	// });

	funcOp.walk([&](sysy::TransposeOp transposeOp) {
    OpBuilder builder(transposeOp);
		Location loc = transposeOp.getLoc();
		const ReachingDefState* state;
		if (Operation *prev = transposeOp->getPrevNode())
			state = solver.lookupState<ReachingDefState>(prev);
		else
			state = solver.lookupState<ReachingDefState>(transposeOp->getBlock());

		Value input = transposeOp.getInput();
		Value output = transposeOp.getOutput();
		for(auto reachingDef : state->getValue()){
			if(auto reachingDefTransposeOp = llvm::dyn_cast<sysy::TransposeOp>(reachingDef)){
				if(reachingDefTransposeOp.getOutput() == input){
					builder.setInsertionPoint(transposeOp);
					builder.create<sysy::CopyOp>(loc, output, reachingDefTransposeOp.getInput());
					transposesToRemove.push_back(transposeOp);
				}
			}
		}
		return WalkResult::advance();
	});

	for(auto transposeOp : transposesToRemove)
		transposeOp->erase();
}

std::unique_ptr<Pass> createSimplifyRedundantTransposePass() {
  return std::make_unique<SimplifyRedundantTransposePass>();
}

} // namespace sysy
} // namespace mlir