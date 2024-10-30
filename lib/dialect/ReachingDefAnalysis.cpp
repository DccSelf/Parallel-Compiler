#include "dialect/ReachingDefAnalysis.h"
#include "dialect/EffectInterface.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {

LogicalResult ReachingDefAnalysis::initialize(Operation *top) {
  if (top->getNumRegions() != 1)
    return top->emitError("expected a single region top-level op");

  if (top->getRegion(0).getBlocks().empty())
    return success();

  // Initialize the top-level state.
  getOrCreate<ReachingDefState>(&top->getRegion(0).front());

  // Visit all nested blocks and operations.
  for (Block &block : top->getRegion(0)) {
    visitBlock(&block);
    for (Operation &op : block) {
      if (op.getNumRegions())
        return op.emitError("unexpected op with regions");
      visitOperation(&op);
    }
  }
  return success();
}

LogicalResult ReachingDefAnalysis::visit(ProgramPoint point) {
  if (auto *op = llvm::dyn_cast_if_present<Operation *>(point)) {
    visitOperation(op);
    return success();
  }
  if (auto *block = llvm::dyn_cast_if_present<Block *>(point)) {
    visitBlock(block);
    return success();
  }
  return emitError(point.getLoc(), "unknown point kind");
}

void ReachingDefAnalysis::visitBlock(Block *block) {
  if (block->isEntryBlock()) {
    // This is the initial state. Let the framework default-initialize it.
    return;
  }
  ReachingDefState *state = getOrCreate<ReachingDefState>(block);
  ChangeResult result = ChangeResult::NoChange;
  for (Block *pred : block->getPredecessors()) {
    // Join the state at the terminators of all predecessors.
    const ReachingDefState *predState =
        getOrCreateFor<ReachingDefState>(block, pred->getTerminator());
    result |= state->join(*predState);
  }
  propagateIfChanged(state, result);
}

void ReachingDefAnalysis::visitOperation(Operation *op) {
  ReachingDefState *state = getOrCreate<ReachingDefState>(op);
	ReachingDefState originalState = *state;
  ChangeResult result = ChangeResult::NoChange;

  // Copy the state across the operation.
  const ReachingDefState *prevState;
  if (Operation *prev = op->getPrevNode())
    prevState = getOrCreateFor<ReachingDefState>(op, prev);
  else
    prevState = getOrCreateFor<ReachingDefState>(op, op->getBlock());
  state->set(*prevState);

  if(isa<mlir::sysy::SysyDialect>(op->getDialect())){
    auto curOp = dyn_cast<sysy::EffectInterface>(op);
    auto tensor = curOp.getWriteValue();
    if(tensor){
      // Remove operations that write to the same tensor from the reaching definition set
      for(auto user : tensor.getUsers()){
        auto reachDefOp = dyn_cast<sysy::EffectInterface>(user);
        if(tensor == reachDefOp.getWriteValue() && curOp != reachDefOp){
          state->kill(reachDefOp);
          break;
        }
      }
      state->gen(op);
    }
  }
  
  // llvm::outs() << *op << "\n";
  // originalState->print(llvm::outs());
  // state->print(llvm::outs());
  // llvm::outs() << "\n";

	result |= state->equal(originalState);
  propagateIfChanged(state, result);
}
}