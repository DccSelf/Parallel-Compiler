#include "mlir/Analysis/DataFlowFramework.h"

#include "dialect/Dialect.h"

#include <unordered_set>

namespace mlir {

class ReachingDefState : public AnalysisState {
public:
  using AnalysisState::AnalysisState;

	ReachingDefState(const ReachingDefState &other) : AnalysisState(other) {
			state = other.state;
	}

  ChangeResult join(const ReachingDefState &rhs) {
    unsigned size = state.size();
    std::unordered_set<Operation*> union_set;
    state.insert(rhs.state.begin(), rhs.state.end());
    return size == state.size() ? ChangeResult::NoChange : ChangeResult::Change;
  }

  void kill(Operation* op) {
    state.erase(op);
  }

	void gen(Operation* op) {
    state.insert(op);
  }

  ChangeResult set(const ReachingDefState &rhs) {
    if (state == rhs.state)
      return ChangeResult::NoChange;
    state = rhs.state;
    return ChangeResult::Change;
  }

	ChangeResult equal(const ReachingDefState &rhs) {
		return state == rhs.state ? ChangeResult::NoChange : ChangeResult::Change;
	}

	void print(raw_ostream &os) const override {
    os << "ReachingDefState: ";
    for (auto *op : state) {
      os << *op << " ";
    }
    os << "\n";
  }

  std::unordered_set<Operation*> getValue() const { return state; }

private:
  std::unordered_set<Operation*> state;
};


class ReachingDefAnalysis : public DataFlowAnalysis {
public:
  using DataFlowAnalysis::DataFlowAnalysis;

  LogicalResult initialize(Operation *top) override;
  LogicalResult visit(ProgramPoint point) override;

private:
  void visitBlock(Block *block);
  void visitOperation(Operation *op);
};

}