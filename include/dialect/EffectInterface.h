#ifndef SYSY_EFFECT_INTERFACE_H_
#define SYSY_EFFECT_INTERFACE_H_

#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace sysy {

/// Include the auto-generated declarations.
#include "dialect/EffectInterface.h.inc"

} // namespace sysy
} // namespace mlir

#endif // SYSY_EFFECT_INTERFACE_H_