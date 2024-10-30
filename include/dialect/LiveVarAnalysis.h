#include "mlir/Analysis/DataFlowFramework.h"

#include "dialect/Dialect.h"

#include <unordered_set>

namespace std {
  template <>
  struct hash<mlir::Value> {
    std::size_t operator()(const mlir::Value& v) const {
      return hash<void*>()(v.getAsOpaquePointer());
    }
  };
}

namespace mlir {

class LiveVarState : public AnalysisState {
public:
  using AnalysisState::AnalysisState;

  LiveVarState(const LiveVarState &other) : 
    AnalysisState(other), state(other.state) {}

  ChangeResult join(const LiveVarState &rhs) {
    unsigned size = state.size();
    std::unordered_set<Operation*> union_set;
    state.insert(rhs.state.begin(), rhs.state.end());
    return size == state.size() ? ChangeResult::NoChange : ChangeResult::Change;
  }

  void kill(Value var) {
    state.erase(var);
  }

	void gen(Value var) {
    state.insert(var);
  }

  ChangeResult set(const LiveVarState &rhs) {
    if (state == rhs.state)
      return ChangeResult::NoChange;
    state = rhs.state;
    return ChangeResult::Change;
  }

	ChangeResult equal(const LiveVarState &rhs) {
		return state == rhs.state ? ChangeResult::NoChange : ChangeResult::Change;
	}

	void print(raw_ostream &os) const override {
    os << "LiveVarState: ";
    for (auto var : state) {
      os << var << " ";
    }
    os << "\n";
  }

  std::unordered_set<Value> getValue() const { return state; }

private:
  std::unordered_set<Value> state;
};

class LiveVarAnalysis : public DataFlowAnalysis {
public:
  using DataFlowAnalysis::DataFlowAnalysis;

  LogicalResult initialize(Operation *top) override;
  LogicalResult visit(ProgramPoint point) override;

private:
  void visitBlock(Block *block);
  void visitOperation(Operation *op);
};

}