#pragma once

#include <unordered_map>
#include <string>

#include "llvm/IR/Constant.h"

#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

namespace std {
  template <>
  struct hash<mlir::Value> {
    std::size_t operator()(const mlir::Value& v) const {
      return hash<void*>()(v.getAsOpaquePointer());
    }
  };
}

struct FunctionTable{
    std::unordered_map<std::string, mlir::FunctionOpInterface> mapping;

    mlir::FunctionOpInterface resolve(const std::string &name);
    void registerFunc(const std::string &name, mlir::FunctionOpInterface func);
};

struct GlobalVarTable{
    std::unordered_map<std::string, mlir::LLVM::GlobalOp> mapping;
    std::map<mlir::LLVM::GlobalOp, llvm::Constant*> valueMap;

    mlir::LLVM::GlobalOp resolve(const std::string &name);
    llvm::Constant* loadValue(mlir::LLVM::GlobalOp var);
    void registerVar(const std::string &name, mlir::LLVM::GlobalOp var);
    void registerConst(const std::string &name, mlir::LLVM::GlobalOp var, llvm::Constant* value);
};

struct LocalVarTable{
    LocalVarTable *parent;
    std::unordered_map<std::string, mlir::Value> mapping;
    std::unordered_map<mlir::Value, llvm::Constant*> valueMap;

    LocalVarTable(LocalVarTable *parentTable) : parent(parentTable){}
    mlir::Value resolve(const std::string &name);
    llvm::Constant* loadValue(mlir::Value var);
    void registerVar(const std::string &name, mlir::Value var);
    void registerConst(const std::string &name, mlir::Value var, llvm::Constant* value);
};
