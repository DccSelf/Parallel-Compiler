#ifndef EFFECT_INTERFACE
#define EFFECT_INTERFACE

include "mlir/IR/OpBase.td"

def EffectOpInterface : OpInterface<"EffectInterface"> {
  let description = "Interface for operations with effects";

  let methods = [
    InterfaceMethod<"Get the value written by this operation.",
                    "mlir::Value", "getWriteValue">, 
    InterfaceMethod<"Get the values read by this operation.",
                    "llvm::SmallVector<mlir::Value, 3>", "getReadValues">
  ];
}

#endif // EFFECT_INTERFACE