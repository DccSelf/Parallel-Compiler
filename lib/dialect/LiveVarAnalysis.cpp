
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "dialect/LiveVarAnalysis.h"
#include "dialect/EffectInterface.h"
#include "mlir/IR/Block.h"
#include "mlir/Support/LogicalResult.h"
#include <cassert>

namespace mlir {

LogicalResult LiveVarAnalysis::initialize(Operation *top) {
  if (top->getNumRegions() != 1)
    return top->emitError("expected a single region top-level op");

  if (top->getRegion(0).getBlocks().empty())
    return success();

  // Initialize the top-level state.
  Block* retBlock = nullptr;
  for (auto &block : top->getRegion(0)) {
    if (auto returnOp = dyn_cast<func::ReturnOp>(block.getTerminator())) {
      retBlock = &block;
      break;
    }
  }
  getOrCreate<LiveVarState>(retBlock);

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

LogicalResult LiveVarAnalysis::visit(ProgramPoint point) {
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

void LiveVarAnalysis::visitBlock(Block *block) {
  if (isa<func::ReturnOp>(block->back())) {
    // This is the initial state. Let the framework default-initialize it.
    return;
  }
  LiveVarState *state = getOrCreate<LiveVarState>(block);
  ChangeResult result = ChangeResult::NoChange;
  for (Block *succ : block->getSuccessors()) {
    // Join the state at the terminators of all successors.
    const LiveVarState *SuccState =
        getOrCreateFor<LiveVarState>(block, &succ->front());
    result |= state->join(*SuccState);
  }
  propagateIfChanged(state, result);
}

void LiveVarAnalysis::visitOperation(Operation *op) {
  LiveVarState *state = getOrCreate<LiveVarState>(op);
	LiveVarState originalState = *state;
  ChangeResult result = ChangeResult::NoChange;

  // Copy the state across the operation.
  const LiveVarState *nextState;
  if (Operation *next = op->getNextNode())
    nextState = getOrCreateFor<LiveVarState>(op, next);
  else
    nextState = getOrCreateFor<LiveVarState>(op, op->getBlock());
  state->set(*nextState);

  if(isa<mlir::sysy::SysyDialect>(op->getDialect())){
    auto curOp = dyn_cast<sysy::EffectInterface>(op);
    auto readValues = curOp.getReadValues();

    auto writeValue = curOp.getWriteValue();
    if(writeValue)
      state->kill(writeValue);

    for(auto readValue : readValues)
      state->gen(readValue);
  }

  // llvm::outs() << *op << "\n";
  // originalState.print(llvm::outs());
  // state->print(llvm::outs());
  // llvm::outs() << "\n";

	result |= state->equal(originalState);
  propagateIfChanged(state, result);
}

}
