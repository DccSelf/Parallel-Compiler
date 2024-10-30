#ifndef MLIR_TUTORIAL_SYSY_DIALECT_H_
#define MLIR_TUTORIAL_SYSY_DIALECT_H_

#include "mlir/IR/Dialect.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "dialect/Dialect.h.inc"
#include "dialect/ShapeInferenceInterface.h"
#include "dialect/EffectInterface.h"

#define GET_OP_CLASSES
#include "dialect/Ops.h.inc"

#endif // MLIR_TUTORIAL_SYSY_DIALECT_H_