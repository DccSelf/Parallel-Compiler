#include "frontend/SymbolTable.h"


using namespace std;
using namespace mlir;
using namespace func;

mlir::FunctionOpInterface FunctionTable::resolve(const string &name) {
    auto it = mapping.find(name);
    if (it != mapping.end())
        return it->second;
    else
        return mlir::FunctionOpInterface();
}

void FunctionTable::registerFunc(const string &name, mlir::FunctionOpInterface func) {
    mapping[name] = func;
}

LLVM::GlobalOp GlobalVarTable::resolve(const string &name) {
    auto it = mapping.find(name);
    if (it != mapping.end())
        return it->second;
    else
        return LLVM::GlobalOp();
}

llvm::Constant* GlobalVarTable::loadValue(mlir::LLVM::GlobalOp var){
    auto it = valueMap.find(var);
    if (it != valueMap.end()) {
        return it->second;
    } else {
        return nullptr;
    }
}

void GlobalVarTable::registerVar(const string &name, mlir::LLVM::GlobalOp var) {
    mapping[name] = var;
}

void GlobalVarTable::registerConst(const string &name, mlir::LLVM::GlobalOp var, llvm::Constant* value) {
    mapping[name] = var;
    valueMap[var] = value;
}

mlir::Value LocalVarTable::resolve(const string &name) {
    auto it = mapping.find(name);
    if (it != mapping.end()) return it->second;
    if (parent) return parent->resolve(name);
    return mlir::Value();
}

llvm::Constant* LocalVarTable::loadValue(mlir::Value var){
    auto it = valueMap.find(var);
    if (it != valueMap.end()) {
        return it->second;
    } else {
        return nullptr;
    }
}

void LocalVarTable::registerVar(const string &name, mlir::Value var) {
    mapping[name] = var;
}

void LocalVarTable::registerConst(const string &name, mlir::Value var, llvm::Constant* value) {
    mapping[name] = var;
    valueMap[var] = value;
}