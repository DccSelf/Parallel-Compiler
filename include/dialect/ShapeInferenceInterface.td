#ifndef SHAPE_INFERENCE_INTERFACE
#define SHAPE_INFERENCE_INTERFACE

include "mlir/IR/OpBase.td"

def ShapeInferenceOpInterface : OpInterface<"ShapeInference"> {
  let description = [{
    Interface to access a registered method to infer the return types for an
    operation that can be used during type inference.
  }];

  let methods = [
    InterfaceMethod<"Check if the operation requires shape inference.",
                    "bool", "needsShapeInference">, 
    InterfaceMethod<"Check if all inputs have been inferred.",
                    "bool", "allInputsInferred">, 
    InterfaceMethod<"Infer and set the result shape for the current operation.",
                    "void", "inferShapes">
  ];
}

#endif // SHAPE_INFERENCE_INTERFACE