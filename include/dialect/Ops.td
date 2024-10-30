#ifndef SYSY_OPS
#define SYSY_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"

include "dialect/ShapeInferenceInterface.td"
include "dialect/EffectInterface.td"

//===----------------------------------------------------------------------===//
// SysyDialect
//===----------------------------------------------------------------------===//

def Sysy_Dialect : Dialect {
  let name = "sysy";
  let cppNamespace = "::mlir::sysy";
}

//===----------------------------------------------------------------------===//
// Sysy Operations
//===----------------------------------------------------------------------===//

class Sysy_Op<string mnemonic, list<Trait> traits = []> :
    Op<Sysy_Dialect, mnemonic, traits>;

def ElementwiseBinaryOpTrait : NativeOpTrait<"ElementwiseBinaryOpTrait">;

//===----------------------------------------------------------------------===//
// InitTensorOp
//===----------------------------------------------------------------------===//

def InitTensorOp : Sysy_Op<"init_tensor", [
    // DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
    DeclareOpInterfaceMethods<EffectOpInterface>
  ]> {
  let summary = "Initialize a tensor with a constant value";
  let description = [{
    Example:
    ```mlir
    sysy.init_tensor %1, dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf32>
    ```
  }];

  let arguments = (ins F32Tensor:$dst, ElementsAttr:$value);
  let assemblyFormat = "$dst `,` $value attr-dict `:` type($dst)";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// DefTensorOp
//===----------------------------------------------------------------------===//

def DefTensorOp : Sysy_Op<"def_tensor", [
    Pure,
    DeclareOpInterfaceMethods<EffectOpInterface>
  ]> {
  let summary = "def_tensor";
  let description = [{
    ```mlir
    %0 = sysy.def_tensor [2, 3] : tensor<2x3xf32>
    ```
  }];
  let arguments = (ins I64ArrayAttr:$shape);
  let results = (outs F32Tensor:$result);
  let assemblyFormat = "$shape attr-dict `:` type($result)";
  let builders = [
    OpBuilder<(ins "ArrayRef<int64_t>":$shape), [{
      auto tensorType = RankedTensorType::get(shape, $_builder.getF32Type());
      build($_builder, $_state, tensorType, $_builder.getI64ArrayAttr(shape));
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// DeclTensorOp
//===----------------------------------------------------------------------===//

def DeclTensorOp : Sysy_Op<"decl_tensor", [
    Pure,
    DeclareOpInterfaceMethods<EffectOpInterface>
  ]> {
  let summary = "tensor with undefined shape and content";
  let results = (outs F32Tensor);
  let assemblyFormat = [{
    attr-dict `:` type(results)
  }];
  let builders = [
    OpBuilder<(ins), [{
      build($_builder, $_state, UnrankedTensorType::get($_builder.getF32Type()));
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// BroadcastOp
//===----------------------------------------------------------------------===//

def BroadcastOp : Sysy_Op<"broadcast", [
  DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
  // DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  DeclareOpInterfaceMethods<EffectOpInterface>
  ]> {
  let summary = "broadcast operation";
  let arguments = (ins F32Tensor:$output, F32Tensor:$init, F32:$input);
  let assemblyFormat = [{
    $output `,` $init `,` $input attr-dict `:` 
    type($output) `,` type($init) `,` type($input)
  }];
}

//===----------------------------------------------------------------------===//
// MulOp
//===----------------------------------------------------------------------===//

def MulOp : Sysy_Op<"mul", [
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
    // DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
    DeclareOpInterfaceMethods<EffectOpInterface>,
    // ElementwiseBinaryOpTrait
  ]> {
  let summary = "element-wise multiplication operation";
  let arguments = (ins F32Tensor:$dst, F32Tensor:$lhs, F32Tensor:$rhs);
  let assemblyFormat = [{
    $dst `,` $lhs `,` $rhs attr-dict `:` type($dst) `,` type($lhs) `,` type($rhs)
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

def AddOp : Sysy_Op<"add", [
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
    // DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
    DeclareOpInterfaceMethods<EffectOpInterface>,
    // ElementwiseBinaryOpTrait
  ]> {
  let summary = "element-wise addition operation";
  let arguments = (ins F32Tensor:$dst, F32Tensor:$lhs, F32Tensor:$rhs);
  let assemblyFormat = [{
    $dst `,` $lhs `,` $rhs attr-dict `:` type($dst) `,` type($lhs) `,` type($rhs)
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// SubOp
//===----------------------------------------------------------------------===//

def SubOp : Sysy_Op<"sub", [
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
    // DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
    DeclareOpInterfaceMethods<EffectOpInterface>,
    // ElementwiseBinaryOpTrait
  ]> {
  let summary = "element-wise subtraction operation";
  let arguments = (ins F32Tensor:$dst, F32Tensor:$lhs, F32Tensor:$rhs);
  let assemblyFormat = [{
    $dst `,` $lhs `,` $rhs attr-dict `:` type($dst) `,` type($lhs) `,` type($rhs)
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// DivOp
//===----------------------------------------------------------------------===//

def DivOp : Sysy_Op<"div", [
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
    // DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
    DeclareOpInterfaceMethods<EffectOpInterface>,
    // ElementwiseBinaryOpTrait
  ]> {
  let summary = "element-wise division operation";
  let arguments = (ins F32Tensor:$dst, F32Tensor:$lhs, F32Tensor:$rhs);
  let assemblyFormat = [{
    $dst `,` $lhs `,` $rhs attr-dict `:` type($dst) `,` type($lhs) `,` type($rhs)
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// MatMulOp
//===----------------------------------------------------------------------===//

def MatMulOp : Sysy_Op<"matmul", [
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
    // DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
    DeclareOpInterfaceMethods<EffectOpInterface>,
  ]>
  {
  let summary = "2D tensor multiplication";
  let description = [{
      Matmul operation will be converted into affine loops then be optimized.
  }];
  let arguments = (ins F32Tensor:$dst, F32Tensor:$lhs, F32Tensor:$rhs);
  let assemblyFormat = [{
    $dst `,` $lhs `,` $rhs attr-dict `:` type($dst) `,` type($lhs) `,` type($rhs)
  }];
  let hasVerifier = 1;
}


//===----------------------------------------------------------------------===//
// ValidConvOp
//===----------------------------------------------------------------------===//

def ValidConvOp : Sysy_Op<"conv_valid", [
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
    // DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
    DeclareOpInterfaceMethods<EffectOpInterface>,
  ]> {
  let summary = "valid mode convolution operation";
  let description = [{
    The "conv_valid" operation performs Valid Convolution between two tensors.
  }];
  let arguments = (ins F32Tensor:$output, F32Tensor:$input, F32Tensor:$kernel);
  let assemblyFormat = [{
    $output `,` $input `,` $kernel attr-dict `:` type($output) `,` type($input) `,` type($kernel)
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// SameConvOp
//===----------------------------------------------------------------------===//

def SameConvOp : Sysy_Op<"conv_same",[
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
    // DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
    DeclareOpInterfaceMethods<EffectOpInterface>,
  ]> {
  let summary = "same mode convolution operation";
  let description = [{
    The "conv_same" operation performs Same Convolution between two tensors.
  }];
  let arguments = (ins F32Tensor:$output, F32Tensor:$input, F32Tensor:$kernel);
  let assemblyFormat = [{
    $output `,` $input `,` $kernel attr-dict `:` type($output) `,` type($input) `,` type($kernel)
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// MaxPoolOp
//===----------------------------------------------------------------------===//

def MaxPoolOp : Sysy_Op<"max_pool",[
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
    DeclareOpInterfaceMethods<EffectOpInterface>,
  ]> {
  let summary = "max pool operation";
  let description = [{
    The "max_pool" operation performs max pool one tensor.
    input:4x4   kernel:2x2 stride:2  -> output:2x2
  }];
  let arguments = (ins F32Tensor:$output, F32Tensor:$input, F32Tensor:$kernel);
  let assemblyFormat = [{
    $output `,` $input `,` $kernel attr-dict `:` type($output) `,` type($input) `,` type($kernel)
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// TransposeOp
//===----------------------------------------------------------------------===//

def TransposeOp : Sysy_Op<"transpose", [
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
    // DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    DeclareOpInterfaceMethods<EffectOpInterface>,
  ]> {
  let summary = "transpose operation";
  let arguments = (ins F32Tensor:$output, F32Tensor:$input);
  let assemblyFormat = [{
    $output `,` $input attr-dict `:` type($output) `,` type($input)
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// CopyOp
//===----------------------------------------------------------------------===//

def CopyOp : Sysy_Op<"copy", [
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
    // DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    DeclareOpInterfaceMethods<EffectOpInterface>,
  ]> {
  let summary = "copy operation";
  let arguments = (ins F32Tensor:$output, F32Tensor:$input);
  let assemblyFormat = [{
    $output `,` $input attr-dict `:` type($output) `,` type($input)
  }];
  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// ReLUOp
//===----------------------------------------------------------------------===//

def ReLUOp : Sysy_Op<"relu",[
  DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
  DeclareOpInterfaceMethods<EffectOpInterface>
]>{
  let summary = "Applies the ReLU activation function element-wise";
  let arguments = (ins F32Tensor:$dst,F32Tensor:$src);
  let assemblyFormat = [{
    $dst `,` $src attr-dict `:` type($dst) `,` type($src) 
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// FlattenOp
//===----------------------------------------------------------------------===//

def FlattenOp : Sysy_Op<"flatten",[
  DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
  DeclareOpInterfaceMethods<EffectOpInterface>
]>{
  let arguments = (ins F32Tensor:$output, F32Tensor:$input);
  let assemblyFormat = [{
    $output `,` $input attr-dict `:` type($output) `,` type($input) 
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// MaxOp
//===----------------------------------------------------------------------===//

def MaxOp : Sysy_Op<"max",[
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    DeclareOpInterfaceMethods<EffectOpInterface>,
  ]> {
  let summary = "maximum reduction operation";
  let arguments = (ins F32Tensor:$input);
  let results = (outs F32:$output);
  let assemblyFormat = [{
    `(` $input `:` type($input) `)` attr-dict `:` type(results)
  }];
  let builders = [
    OpBuilder<(ins "Value":$input)>
  ];
}

//===----------------------------------------------------------------------===//
// MinOp
//===----------------------------------------------------------------------===//

def MinOp : Sysy_Op<"min",[
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    DeclareOpInterfaceMethods<EffectOpInterface>,
  ]> {
  let summary = "minimum reduction operation";
  let arguments = (ins F32Tensor:$input);
  let results = (outs F32:$output);
  let assemblyFormat = [{
    `(` $input `:` type($input) `)` attr-dict `:` type(results)
  }];
  let builders = [
    OpBuilder<(ins "Value":$input)>
  ];
}

//===----------------------------------------------------------------------===//
// SumOp
//===----------------------------------------------------------------------===//

def SumOp : Sysy_Op<"sum",[
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    DeclareOpInterfaceMethods<EffectOpInterface>,
  ]> {
  let summary = "sum reduction operation";
  let arguments = (ins F32Tensor:$input);
  let results = (outs F32:$output);
  let assemblyFormat = [{
    `(` $input `:` type($input) `)` attr-dict `:` type(results)
  }];
  let builders = [
    OpBuilder<(ins "Value":$input)>
  ];
}

//===----------------------------------------------------------------------===//
// CastOp
//===----------------------------------------------------------------------===//

def CastOp : Sysy_Op<"cast", [
     DeclareOpInterfaceMethods<CastOpInterface>,
     DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
     Pure,
     DeclareOpInterfaceMethods<EffectOpInterface>,
     SameOperandsAndResultShape
  ]> {
  let summary = "shape cast operation";
  let description = [{
    The "cast" operation converts a tensor from one type to an equivalent type
    without changing any data elements. The source and destination types must
    both be tensor types with the same element type. If both are ranked, then
    shape is required to match. The operation is invalid if converting to a
    mismatching constant dimension.
  }];

  let arguments = (ins F32Tensor:$input);
  let results = (outs F32Tensor:$output);

  let assemblyFormat = "$input attr-dict `:` type($input) `to` type($output)";

  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// FunctionOp
//===----------------------------------------------------------------------===//

def FunctionOp : Sysy_Op<"func", [
    DeclareOpInterfaceMethods<CallableOpInterface>,
    FunctionOpInterface,
    IsolatedFromAbove,
    DeclareOpInterfaceMethods<EffectOpInterface>,
  ]> {
  let summary = "user defined function operation";
  let description = [{
    ```mlir
    sysy.func @main() {
      %0 = sysy.constant dense<5.500000e+00> : tensor<f32>
      %1 = sysy.reshape(%0 : tensor<f32>) to tensor<2x2xf32>
      sysy.print %1 : tensor<2x2xf32>
      sysy.return
    }
    ```
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs
  );
  let regions = (region AnyRegion:$body);

  let builders = [OpBuilder<(ins
    "StringRef":$name, "FunctionType":$type,
    CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)
  >];
  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // FunctionOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }
  }];
  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

def RetOp : Sysy_Op<"return", [
    Pure, 
    DeclareOpInterfaceMethods<EffectOpInterface>,
    HasParent<"FunctionOp">,
    Terminator
  ]> {
  let summary = "return operation";
  let description = [{
    ```mlir
      sysy.func @foo() -> tensor<2xf32> {
        ...
        sysy.return %0 : tensor<2xf32>
      }
    ```
  }];

  // The return operation takes an optional input operand to return. This
  // value must match the return type of the enclosing function.
  let arguments = (ins Variadic<F32Tensor>:$input);

  // The return operation only emits the input in the format if it is present.
  let assemblyFormat = "($input^ `:` type($input))? attr-dict ";

  // Allow building a ReturnOp with no return operand.
  let builders = [
    OpBuilder<(ins), [{ build($_builder, $_state, std::nullopt); }]>
  ];

  // Provide extra utility definitions on the c++ operation class definition.
  let extraClassDeclaration = [{
    bool hasOperand() { return getNumOperands() != 0; }
  }];

  // Indicate that additional verification for this operation is necessary.
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// GenericCallOp
//===----------------------------------------------------------------------===//

def GenericCallOp : Sysy_Op<"call", [
    DeclareOpInterfaceMethods<CallOpInterface>,
    DeclareOpInterfaceMethods<EffectOpInterface>,
  ]> {
  let summary = "generic call operation";
  let description = [{
    ```mlir
     %4 = sysy.call @my_func(%1, %3)
           : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<*xf32>
    ```
  }];

  // The generic call operation takes a symbol reference attribute as the
  // callee, and inputs for the call.
  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<AnyType>:$inputs);

  // The generic call operation returns a single value of TensorType.
  let results = (outs Variadic<AnyType>);

  // Specialize assembly printing and parsing using a declarative format.
  let assemblyFormat = [{
    $callee `(` $inputs `)` attr-dict `:` functional-type($inputs, results)
  }];

  // Add custom build methods for the generic call operation.
  let builders = [
    OpBuilder<(ins "FunctionOp":$callee, CArg<"ValueRange", "{}">:$operands), [{
      $_state.addTypes(callee.getFunctionType().getResults());
      $_state.addOperands(operands);
      $_state.addAttribute("callee", SymbolRefAttr::get(callee));
    }]>,
  ];
}

//===----------------------------------------------------------------------===//
// PrintOp
//===----------------------------------------------------------------------===//

def PrintOp : Sysy_Op<"print", [
  // DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  DeclareOpInterfaceMethods<EffectOpInterface>,
]> {
  let summary = "print operation";
  let description = [{
    The "print" builtin operation prints a given input tensor, and produces
    no results.
  }];

  // The print operation takes an input tensor to print.
  // We also allow a F64MemRef to enable interop during partial lowering.
  let arguments = (ins F32Tensor:$input);

  let assemblyFormat = "$input attr-dict `:` type($input)";
}

//===----------------------------------------------------------------------===//
// ScanOp
//===----------------------------------------------------------------------===//

def ScanOp : Sysy_Op<"scan", [
  // DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  DeclareOpInterfaceMethods<EffectOpInterface>,
]> {
  let summary = "scan operation";
  let arguments = (ins F32Tensor:$input);
  let assemblyFormat = "$input attr-dict `:` type($input)";
}

//===----------------------------------------------------------------------===//
// ClearOp
//===----------------------------------------------------------------------===//

def ClearOp : Sysy_Op<"clear", [
  // DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  DeclareOpInterfaceMethods<EffectOpInterface>,
]> {
  let summary = "Clears all elements of a tensor by setting them to zero";
  let arguments = (ins F32Tensor:$input);
  let assemblyFormat = "$input attr-dict `:` type($input)";
}

#endif // SYSY_OPS