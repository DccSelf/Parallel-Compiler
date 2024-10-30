#include "dialect/Dialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/InliningUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "llvm/Support/Casting.h"
#include <cassert>

using namespace mlir;
using namespace mlir::sysy;

#include "dialect/Dialect.cpp.inc"

//===----------------------------------------------------------------------===//
// SysyInlinerInterface
//===----------------------------------------------------------------------===//

/// This class defines the interface for handling inlining with Sysy
/// operations.
struct SysyInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  //===--------------------------------------------------------------------===//
  // Analysis Hooks
  //===--------------------------------------------------------------------===//

  /// All call operations within sysy can be inlined.
  bool isLegalToInline(Operation *call, Operation *callable,
                       bool wouldBeCloned) const final {
    return true;
  }

  /// All operations within sysy can be inlined.
  bool isLegalToInline(Operation *, Region *, bool, IRMapping &) const final {
    return true;
  }

  // All functions within sysy can be inlined.
  bool isLegalToInline(Region *, Region *, bool, IRMapping &) const final {
    return true;
  }

  //===--------------------------------------------------------------------===//
  // Transformation Hooks
  //===--------------------------------------------------------------------===//

  /// Handle the given inlined terminator by replacing it with a new operation
  /// as necessary.
  void handleTerminator(Operation *op, Block *newDest) const final {
    // Only return needs to be handled here.
    auto returnOp = dyn_cast<RetOp>(op);
    if (!returnOp)
      return;

    // Replace the return with a branch to the dest.
    OpBuilder builder(op);
    builder.create<cf::BranchOp>(op->getLoc(), newDest, returnOp.getOperands());
    op->erase();
  }

  /// Handle the given inlined terminator(sysy.return) by replacing it with a new
  /// operation as necessary.
  void handleTerminator(Operation *op,
                        ArrayRef<Value> valuesToRepl) const final {
    // Only "sysy.return" needs to be handled here.
    auto returnOp = cast<RetOp>(op);

    // Replace the values directly with the return operands.
    assert(returnOp.getNumOperands() == valuesToRepl.size());
    for (const auto &it : llvm::enumerate(returnOp.getOperands()))
      valuesToRepl[it.index()].replaceAllUsesWith(it.value());
  }

  /// Attempts to materialize a conversion for a type mismatch between a call
  /// from this dialect, and a callable region. This method should generate an
  /// operation that takes 'input' as the only operand, and produces a single
  /// result of 'resultType'. If a conversion can not be generated, nullptr
  /// should be returned.
  Operation *materializeCallConversion(OpBuilder &builder, Value input,
                                       Type resultType,
                                       Location conversionLoc) const final {
    return builder.create<CastOp>(conversionLoc, resultType, input);
  }
};


//===----------------------------------------------------------------------===//
// SysyDialect
//===----------------------------------------------------------------------===//

void SysyDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "dialect/Ops.cpp.inc"
      >();
  addInterfaces<SysyInlinerInterface>();
}

static void materializeTensor(DeclTensorOp res, ArrayRef<int64_t> shape){
    OpBuilder builder(res);
    auto defTensorOp = builder.create<sysy::DefTensorOp>(builder.getUnknownLoc(), shape);
    // move to entry block
    auto* parentBlock = defTensorOp->getBlock();
    func::FuncOp funcOp = dyn_cast<func::FuncOp>(parentBlock->getParentOp());
    Block &entryBlock = funcOp.getBody().front();
    defTensorOp->moveBefore(&entryBlock.front());

    res.replaceAllUsesWith(defTensorOp.getResult());
}

static mlir::LogicalResult verifyBinaryElementwiseOp(mlir::Operation *op) {
  // Verify the shape of lhs and rhs
  auto lhsType = llvm::dyn_cast<RankedTensorType>(op->getOperand(1).getType());
  auto rhsType = llvm::dyn_cast<RankedTensorType>(op->getOperand(2).getType());
  if (!lhsType || !rhsType)
    return mlir::success();

  auto lhsShape = lhsType.getShape();
  auto rhsShape = rhsType.getShape();

  if (lhsShape.size() != rhsShape.size()) {
    return op->emitOpError() << "all operands must have the same number of dimensions";
  }

  for (size_t i = 0; i < lhsShape.size(); ++i) {
    if (lhsShape[i] != rhsShape[i]) {
      return op->emitOpError() << "dimension size mismatch at index " << i;
    }
  }

  // Verify the shape of res
  auto dstType = llvm::dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  if (!dstType)
    return mlir::success();

  auto dstShape = dstType.getShape();

  if (lhsShape.size() != dstShape.size()) {
    return op->emitOpError() << "all operands must have the same number of dimensions";
  }

  for (size_t i = 0; i < lhsShape.size(); ++i) {
    if (lhsShape[i] != dstShape[i]) {
      return op->emitOpError() << "dimension size mismatch at index " << i;
    }
  }

  return mlir::success();
}

static bool needsShapeInferenceForVoidOp(mlir::Operation *op){
  return op->getOperand(0).getType().isa<UnrankedTensorType>();
}

static bool allInputsInferredBinaryOp(mlir::Operation *op){
  return op->getOperand(1).getType().isa<RankedTensorType>()
      && op->getOperand(2).getType().isa<RankedTensorType>();
}

static bool allInputsInferredUnaryOp(mlir::Operation *op){
  return op->getOperand(1).getType().isa<RankedTensorType>();
}

static void inferShapesBinaryElementwiseOp(mlir::Operation *op){
  mlir::TensorType dstType = llvm::dyn_cast<TensorType>(op->getOperand(0).getType());
  mlir::TensorType lhsType = llvm::dyn_cast<TensorType>(op->getOperand(1).getType());
  if(dstType.isa<UnrankedTensorType>()){
    DeclTensorOp dst = llvm::dyn_cast<DeclTensorOp>(op->getOperand(0).getDefiningOp());
    materializeTensor(dst, lhsType.getShape());
  } else
    assert(dstType == lhsType);
}

static Value getWriteValue(mlir::Operation *op) {
  return op->getOperand(0);
}

static llvm::SmallVector<mlir::Value, 3> getReadUnaryOp(mlir::Operation *op) {
  return {op->getOperand(1)};
}

static llvm::SmallVector<mlir::Value, 3> getReadElementwiseBinaryOp(mlir::Operation *op) {
  return {op->getOperand(1), op->getOperand(2)};
}

static llvm::SmallVector<mlir::Value, 3> getReadBinaryOp(mlir::Operation *op) {
  return op->getOperands();
}

static void getEffectsRO(mlir::Operation *op, SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), op->getOperand(0),
                       SideEffects::DefaultResource::get());
}


//===----------------------------------------------------------------------===//
// Sysy Operations
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// InitTensorOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult InitTensorOp::verify() {
  if(!llvm::isa<sysy::DefTensorOp>(getDst().getDefiningOp()))
    return failure();
  // If the return type of the constant is not an unranked tensor, the shape
  // must match the shape of the attribute holding the data.
  auto dstType = llvm::dyn_cast<mlir::RankedTensorType>(getDst().getType());
  if (!dstType)
    return success();

  // Check that the rank of the attribute type matches the rank of the constant
  // result type.
  auto attrType = llvm::cast<mlir::RankedTensorType>(getValue().getType());
  if (attrType.getRank() != dstType.getRank()) {
    return emitOpError("return type must match the one of the attached value "
                       "attribute: ")
           << attrType.getRank() << " != " << dstType.getRank();
  }

  // Check that each of the dimensions match between the two types.
  for (int dim = 0, dimE = attrType.getRank(); dim < dimE; ++dim) {
    if (attrType.getShape()[dim] != dstType.getShape()[dim]) {
      return emitOpError(
                 "return type shape mismatches its attribute at dimension ")
             << dim << ": " << attrType.getShape()[dim]
             << " != " << dstType.getShape()[dim];
    }
  }
  return mlir::success();
}

// void InitTensorOp::getEffects(
//   SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
//   effects.emplace_back(MemoryEffects::Write::get(), getDst(),
//                        SideEffects::DefaultResource::get());
// }

mlir::Value InitTensorOp::getWriteValue() {
  return ::getWriteValue(*this);
}

llvm::SmallVector<mlir::Value, 3> InitTensorOp::getReadValues() {
  return {};
}

//===----------------------------------------------------------------------===//
// DefTensorOp
//===----------------------------------------------------------------------===//

mlir::Value DefTensorOp::getWriteValue() {
  return nullptr;
}

llvm::SmallVector<mlir::Value, 3> DefTensorOp::getReadValues() {
  return {};
}

//===----------------------------------------------------------------------===//
// DeclTensorOp
//===----------------------------------------------------------------------===//

mlir::Value DeclTensorOp::getWriteValue() {
  return nullptr;
}

llvm::SmallVector<mlir::Value, 3> DeclTensorOp::getReadValues() {
  return {};
}

//===----------------------------------------------------------------------===//
// BroadcastOp
//===----------------------------------------------------------------------===//

bool BroadcastOp::needsShapeInference() { 
  return needsShapeInferenceForVoidOp(*this); 
}

bool BroadcastOp::allInputsInferred() { 
  return getInit().getType().isa<RankedTensorType>(); 
}

void BroadcastOp::inferShapes() { 
  mlir::TensorType outputType = getOutput().getType();
  mlir::TensorType initType = getInit().getType();
  if(outputType.isa<UnrankedTensorType>()){
    DeclTensorOp output = llvm::dyn_cast<DeclTensorOp>(getOutput().getDefiningOp());
    materializeTensor(output, initType.getShape());
  } else
    assert(outputType == initType);
}

// void BroadcastOp::getEffects(
//   SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
//   getEffectsBinaryOpWO(*this, effects);
// }

mlir::Value BroadcastOp::getWriteValue() {
  return ::getWriteValue(*this);
}

llvm::SmallVector<mlir::Value, 3> BroadcastOp::getReadValues() {
  return getReadBinaryOp(*this);
}

//===----------------------------------------------------------------------===//
// MulOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult MulOp::verify() {
  return verifyBinaryElementwiseOp(*this);
}

bool MulOp::needsShapeInference() { 
  return needsShapeInferenceForVoidOp(*this); 
}

bool MulOp::allInputsInferred() { 
  return allInputsInferredBinaryOp(*this);
}

void MulOp::inferShapes() { 
  inferShapesBinaryElementwiseOp(*this);
}

// void MulOp::getEffects(
//   SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
//   getEffectsBinaryOpWO(*this, effects);
// }

mlir::Value MulOp::getWriteValue() {
  return ::getWriteValue(*this);
}

llvm::SmallVector<mlir::Value, 3> MulOp::getReadValues() {
  return getReadElementwiseBinaryOp(*this);
}

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult AddOp::verify() {
  return verifyBinaryElementwiseOp(*this);
}

bool AddOp::needsShapeInference() { 
  return needsShapeInferenceForVoidOp(*this); 
}

bool AddOp::allInputsInferred() { 
  return allInputsInferredBinaryOp(*this);
}

void AddOp::inferShapes() { 
  inferShapesBinaryElementwiseOp(*this);
}

// void AddOp::getEffects(
//   SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
//   getEffectsBinaryOpWO(*this, effects);
// }

mlir::Value AddOp::getWriteValue() {
  return ::getWriteValue(*this);
}

llvm::SmallVector<mlir::Value, 3> AddOp::getReadValues() {
  return getReadElementwiseBinaryOp(*this);
}

//===----------------------------------------------------------------------===//
// SubOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult SubOp::verify() {
  return verifyBinaryElementwiseOp(*this);
}

bool SubOp::needsShapeInference() { 
  return needsShapeInferenceForVoidOp(*this); 
}

bool SubOp::allInputsInferred() { 
  return allInputsInferredBinaryOp(*this);
}

void SubOp::inferShapes() { 
  inferShapesBinaryElementwiseOp(*this);
}

// void SubOp::getEffects(
//   SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
//   getEffectsBinaryOpWO(*this, effects);
// }

mlir::Value SubOp::getWriteValue() {
  return ::getWriteValue(*this);
}

llvm::SmallVector<mlir::Value, 3> SubOp::getReadValues() {
  return getReadElementwiseBinaryOp(*this);
}

//===----------------------------------------------------------------------===//
// DivOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult DivOp::verify() {
  return verifyBinaryElementwiseOp(*this);
}

bool DivOp::needsShapeInference() { 
  return needsShapeInferenceForVoidOp(*this); 
}

bool DivOp::allInputsInferred() { 
  return allInputsInferredBinaryOp(*this);
}

void DivOp::inferShapes() { 
  inferShapesBinaryElementwiseOp(*this);
}

// void DivOp::getEffects(
//   SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
//   getEffectsBinaryOpWO(*this, effects);
// }

mlir::Value DivOp::getWriteValue() {
  return ::getWriteValue(*this);
}

llvm::SmallVector<mlir::Value, 3> DivOp::getReadValues() {
  return getReadElementwiseBinaryOp(*this);
}

//===----------------------------------------------------------------------===//
// MatMulOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult MatMulOp::verify() {
  // Verify the shape of lhs and rhs
  auto lhsType = llvm::dyn_cast<RankedTensorType>(getLhs().getType());
  auto rhsType = llvm::dyn_cast<RankedTensorType>(getRhs().getType());
  if (!lhsType || !rhsType)
    return mlir::success();

  auto lhsShape = lhsType.getShape();
  auto rhsShape = rhsType.getShape();
  if (lhsShape.size() != 2 || rhsShape.size() != 2)
    return emitOpError() << "Both left-hand side and right-hand side operands must be 2D tensors";
  if (lhsShape[1] != rhsShape[0])
    return emitOpError() << "Incompatible matrix dimensions for matrix multiplication: "
                         << lhsShape[0] << "x" << lhsShape[1] << " * "
                         << rhsShape[0] << "x" << rhsShape[1];

  // Verify the shape of dst
  auto dstType = llvm::dyn_cast<RankedTensorType>(getDst().getType());
  if (!dstType)
    return mlir::success();

  auto dstShape = dstType.getShape();
  if (dstShape.size() != 2)
    return emitOpError() << "Result must be a 2D tensor";
  if(dstShape[0] != lhsShape[0] || dstShape[1] != rhsShape[1])
    return emitOpError() << "Result dimensions are incorrect. Expected: "
                         << lhsShape[0] << "x" << rhsShape[1] << ", but got: "
                         << dstShape[0] << "x" << dstShape[1];

  return mlir::success();
}

bool MatMulOp::needsShapeInference() { 
  return needsShapeInferenceForVoidOp(*this); 
}

bool MatMulOp::allInputsInferred() { 
  return allInputsInferredBinaryOp(*this);
}

void MatMulOp::inferShapes() {
  mlir::TensorType dstType = getDst().getType();
  mlir::TensorType lhsType = getLhs().getType();
  mlir::TensorType rhsType = getRhs().getType();
  auto lhsShape = lhsType.getShape();
  auto rhsShape = rhsType.getShape();
  if(dstType.isa<UnrankedTensorType>()){
    DeclTensorOp dst = llvm::dyn_cast<DeclTensorOp>(getDst().getDefiningOp());
    SmallVector<int64_t, 2> dstShape;
    dstShape.push_back(lhsShape[0]);
    dstShape.push_back(rhsShape[1]);
    materializeTensor(dst, dstShape);
  } else {
    auto dstShape = dstType.getShape();
    assert(dstShape[0] == lhsShape[0] && dstShape[1] == rhsShape[1]);
  }
}

// void MatMulOp::getEffects(
//   SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
//   getEffectsBinaryOpRW(*this, effects);
// }

mlir::Value MatMulOp::getWriteValue() {
  return ::getWriteValue(*this);
}

llvm::SmallVector<mlir::Value, 3> MatMulOp::getReadValues() {
  return getReadBinaryOp(*this);
}

//===----------------------------------------------------------------------===//
// ValidConvOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult ValidConvOp::verify() {
  auto inputType = getInput().getType().dyn_cast<RankedTensorType>();
  auto kernelType = getKernel().getType().dyn_cast<RankedTensorType>();
  if (!inputType || !kernelType)
    return mlir::success();

  auto inputShape = inputType.getShape();
  auto kernelShape = kernelType.getShape();
  if (inputShape.size() != 2 || kernelShape.size() != 2)
    return emitOpError() << "Both input and kernel must be 2D tensors, but got "
                         << "input shape: " << inputType << " and kernel shape: " << kernelType;
  if (inputShape[0] < kernelShape[0] || inputShape[1] < kernelShape[1])
    return emitOpError() << "Kernel size (" << kernelShape[0] << "x" << kernelShape[1] 
                         << ") is larger than input size (" << inputShape[0] << "x" << inputShape[1] 
                         << "), resulting in no valid output";

  auto outputType = getOutput().getType().dyn_cast<RankedTensorType>();
  if(!outputType)
    return mlir::success();

  auto outputShape = outputType.getShape();
  int64_t expectedHeight = inputShape[0] - kernelShape[0] + 1;
  int64_t expectedWidth = inputShape[1] - kernelShape[1] + 1;
  if (outputShape[0] != expectedHeight || outputShape[1] != expectedWidth)
      return emitError() << "Output dimensions are incorrect for valid convolution. "
                         << "Expected: " << expectedHeight << "x" << expectedWidth
                         << ", but got: " << outputShape[0] << "x" << outputShape[1];

  return mlir::success();
}

bool ValidConvOp::needsShapeInference() { 
  return needsShapeInferenceForVoidOp(*this); 
}

bool ValidConvOp::allInputsInferred() { 
  return allInputsInferredBinaryOp(*this);
}

void ValidConvOp::inferShapes() { 
  mlir::TensorType outputType = getOutput().getType();
  mlir::TensorType inputType = getInput().getType();
  mlir::TensorType kernelType = getKernel().getType();
  auto inputShape = inputType.getShape();
  auto kernelShape = kernelType.getShape();
  if(outputType.isa<UnrankedTensorType>()){
    DeclTensorOp output = llvm::dyn_cast<DeclTensorOp>(getOutput().getDefiningOp());
    SmallVector<int64_t, 4> outputShape;
    for (size_t i = 0; i < 2; ++i)
      outputShape.push_back(inputShape[i] - kernelShape[i] + 1);
    materializeTensor(output, outputShape);
  } else {
    auto outputShape = outputType.getShape();
    for (size_t i = 0; i < 2; ++i)
      assert(outputShape[i] == inputShape[i] - kernelShape[i] + 1);
  }
}

// void ValidConvOp::getEffects(
//   SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
//   getEffectsBinaryOpRW(*this, effects);
// }

mlir::Value ValidConvOp::getWriteValue() {
  return ::getWriteValue(*this);
}

llvm::SmallVector<mlir::Value, 3> ValidConvOp::getReadValues() {
  return getReadBinaryOp(*this);
}

//===----------------------------------------------------------------------===//
// SameConvOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult SameConvOp::verify() {
  auto inputType = getInput().getType().dyn_cast<RankedTensorType>();
  auto kernelType = getKernel().getType().dyn_cast<RankedTensorType>();
  if (!inputType || !kernelType)
    return mlir::success();

  auto inputShape = inputType.getShape();
  auto kernelShape = kernelType.getShape();
  if (inputShape.size() != 2 || kernelShape.size() != 2)
    return emitOpError() << "Both input and kernel must be 2D tensors, but got "
                         << "input shape: " << inputType << " and kernel shape: " << kernelType;

  auto outputType = getOutput().getType().dyn_cast<RankedTensorType>();
  if(!outputType)
    return mlir::success();

  auto outputShape = outputType.getShape();
  if (outputShape[0] != inputShape[0] || outputShape[1] != inputShape[1])
    return emitError() << "Output dimensions must be the same as input dimensions for same convolution. "
                       << "Expected: " << inputShape[0] << "x" << inputShape[1]
                       << ", but got: " << outputShape[0] << "x" << outputShape[1];

  return mlir::success();
}

bool SameConvOp::needsShapeInference() { 
  return needsShapeInferenceForVoidOp(*this); 
}

bool SameConvOp::allInputsInferred() { 
  return allInputsInferredBinaryOp(*this);
}

void SameConvOp::inferShapes() { 
  mlir::TensorType outputType = getOutput().getType();
  mlir::TensorType inputType = getInput().getType();
  if(outputType.isa<UnrankedTensorType>()){
    DeclTensorOp output = llvm::dyn_cast<DeclTensorOp>(getOutput().getDefiningOp());
    materializeTensor(output, inputType.getShape());
  } else {
    assert(outputType == inputType);
  }
}

// void SameConvOp::getEffects(
//   SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
//   getEffectsBinaryOpRW(*this, effects);
// }

mlir::Value SameConvOp::getWriteValue() {
  return ::getWriteValue(*this);
}

llvm::SmallVector<mlir::Value, 3> SameConvOp::getReadValues() {
  return getReadBinaryOp(*this);
}

//===----------------------------------------------------------------------===//
// MaxPoolOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult MaxPoolOp::verify() {
    auto inputType = getInput().getType().dyn_cast<RankedTensorType>();
    auto kernelType = getKernel().getType().dyn_cast<RankedTensorType>();

    if (!inputType || !kernelType)
        return mlir::success();

    auto inputShape = inputType.getShape();
    auto kernelShape = kernelType.getShape();
    if (inputShape.size() != 2 || kernelShape.size() != 2)
        return emitOpError() << "Both input and kernel must be 2D tensors, but got "
                             << "input shape: " << inputType << " and kernel shape: " << kernelType;
    if (inputShape[0] < kernelShape[0] || inputShape[1] < kernelShape[1])
        return emitOpError() << "Kernel size (" << kernelShape[0] << "x" << kernelShape[1]
                             << ") is larger than input size (" << inputShape[0] << "x" << inputShape[1]
                             << "), resulting in no valid output";

    auto outputType = getOutput().getType().dyn_cast<RankedTensorType>();
    if(!outputType)
        return mlir::success();

    auto outputShape = outputType.getShape();
    int64_t expectedHeight = (inputShape[0] - kernelShape[0])/2 + 1;
    int64_t expectedWidth = (inputShape[1] - kernelShape[1])/2 + 1;
    if (outputShape[0] != expectedHeight || outputShape[1] != expectedWidth)
        return emitError() << "Output dimensions are incorrect for valid convolution. "
                           << "Expected: " << expectedHeight << "x" << expectedWidth
                           << ", but got: " << outputShape[0] << "x" << outputShape[1];

    return mlir::success();
}

bool MaxPoolOp::needsShapeInference() {
    return needsShapeInferenceForVoidOp(*this);
}

bool MaxPoolOp::allInputsInferred() {
    return allInputsInferredBinaryOp(*this);
}

void MaxPoolOp::inferShapes() {
    mlir::TensorType outputType = getOutput().getType();
    mlir::TensorType inputType = getInput().getType();
    mlir::TensorType kernelType = getKernel().getType();

    //auto strideValue = getStride().dyn_cast<IntegerAttr>().getInt();

    auto inputShape = inputType.getShape();
    auto kernelShape = kernelType.getShape();
    if(outputType.isa<UnrankedTensorType>()){
        DeclTensorOp output = llvm::dyn_cast<DeclTensorOp>(getOutput().getDefiningOp());
        SmallVector<int64_t, 4> outputShape;
        for (size_t i = 0; i < 2; ++i)
            outputShape.push_back((inputShape[i] - kernelShape[i])/2 + 1);
        materializeTensor(output, outputShape);
    } else {
        auto outputShape = outputType.getShape();
        for (size_t i = 0; i < 2; ++i)
            assert(outputShape[i] == ((inputShape[i] - kernelShape[i])/2 + 1));
    }
}

mlir::Value MaxPoolOp::getWriteValue() {
  return ::getWriteValue(*this);
}

llvm::SmallVector<mlir::Value, 3> MaxPoolOp::getReadValues() {
  return getReadBinaryOp(*this);
}

//===----------------------------------------------------------------------===//
// TransposeOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult TransposeOp::verify() {
  auto outputType = llvm::dyn_cast<RankedTensorType>(getOperand(0).getType());
  auto inputType = llvm::dyn_cast<RankedTensorType>(getOperand(1).getType());
  if (!inputType || !outputType)
    return mlir::success();

  auto inputShape = inputType.getShape();
  if (!std::equal(inputShape.begin(), inputShape.end(),
                  outputType.getShape().rbegin())) {
    return emitError()
           << "expected output shape to be a transpose of the input";
  }
  return mlir::success();
}

bool TransposeOp::needsShapeInference() { 
  return needsShapeInferenceForVoidOp(*this);
}

bool TransposeOp::allInputsInferred() { 
  return allInputsInferredUnaryOp(*this);
}

void TransposeOp::inferShapes() { 
  mlir::TensorType outputType = getOutput().getType();
  mlir::TensorType inputType = getInput().getType();
  if(outputType.isa<UnrankedTensorType>()){
    DeclTensorOp output = llvm::dyn_cast<DeclTensorOp>(getOutput().getDefiningOp());
    auto inputShape = inputType.getShape();
    SmallVector<int64_t, 4> outputShape(inputShape.rbegin(), inputShape.rend());
    materializeTensor(output, outputShape);
  } else
    assert(outputType == inputType);
}

// void TransposeOp::getEffects(
//   SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
//   getEffectsUnaryOp(*this, effects);
// }

mlir::Value TransposeOp::getWriteValue() {
  return ::getWriteValue(*this);
}

llvm::SmallVector<mlir::Value, 3> TransposeOp::getReadValues() {
  return getReadUnaryOp(*this);
}

//===----------------------------------------------------------------------===//
// CopyOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult CopyOp::verify() {
  auto outputType = llvm::dyn_cast<RankedTensorType>(getOperand(0).getType());
  auto inputType = llvm::dyn_cast<RankedTensorType>(getOperand(1).getType());
  if (!inputType || !outputType)
    return mlir::success();

  auto inputShape = inputType.getShape();
  if (!std::equal(inputShape.begin(), inputShape.end(),
                  outputType.getShape().begin())) {
    return emitError()
           << "expected output shape to be the same as input";
  }
  return mlir::success();
}

bool CopyOp::needsShapeInference() { 
  return needsShapeInferenceForVoidOp(*this);
}

bool CopyOp::allInputsInferred() { 
  return allInputsInferredUnaryOp(*this);
}

void CopyOp::inferShapes() { 
  mlir::TensorType outputType = getOutput().getType();
  mlir::TensorType inputType = getInput().getType();
  if(outputType.isa<UnrankedTensorType>()){
    DeclTensorOp output = llvm::dyn_cast<DeclTensorOp>(getOutput().getDefiningOp());
    materializeTensor(output, inputType.getShape());
  } else
    assert(outputType == inputType);
}

// void CopyOp::getEffects(
//   SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
//   getEffectsUnaryOp(*this, effects);
// }

mlir::Value CopyOp::getWriteValue() {
  return ::getWriteValue(*this);
}

llvm::SmallVector<mlir::Value, 3> CopyOp::getReadValues() {
  return getReadUnaryOp(*this);
}

//===----------------------------------------------------------------------===//
// ReLUOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult ReLUOp::verify(){
  auto outputType = llvm::dyn_cast<RankedTensorType>(getOperand(0).getType());
  auto inputType = llvm::dyn_cast<RankedTensorType>(getOperand(1).getType());
  if (!inputType || !outputType)
    return mlir::success();

  auto inputShape = inputType.getShape();
  if (!std::equal(inputShape.begin(), inputShape.end(),
                  outputType.getShape().begin())) {
    return emitError()
           << "expected output shape to be the same as input";
  }
  return mlir::success();
}

bool ReLUOp::needsShapeInference() { 
  return needsShapeInferenceForVoidOp(*this); 
}

bool ReLUOp::allInputsInferred() { 
  return allInputsInferredUnaryOp(*this);
}

void ReLUOp::inferShapes() { 
  mlir::TensorType outputType = getDst().getType();
  mlir::TensorType inputType = getSrc().getType();
  if(outputType.isa<UnrankedTensorType>()){
    DeclTensorOp output = llvm::dyn_cast<DeclTensorOp>(getDst().getDefiningOp());
    materializeTensor(output, inputType.getShape());
  } else
    assert(outputType == inputType);
}

mlir::Value ReLUOp::getWriteValue() {
  return ::getWriteValue(*this);
}

llvm::SmallVector<mlir::Value, 3> ReLUOp::getReadValues() {
  return getReadUnaryOp(*this);
}

//===----------------------------------------------------------------------===//
// FlattenOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult FlattenOp::verify() {
  auto outputType = llvm::dyn_cast<RankedTensorType>(getOutput().getType());
  auto inputType = llvm::dyn_cast<RankedTensorType>(getInput().getType());
  if (!inputType || !outputType)
    return mlir::success();

  auto inputShape = inputType.getShape();
  auto outputShape = outputType.getShape();


  if (outputShape.size() != 2)
    return emitError() << "expected output to be 2-dimensional, but got "
                       << outputShape.size() << " dimensions";

  if (outputShape[0] != 1)
    return emitError() << "expected first dimension of output to be 1, but got "
                       << outputShape[0];

  int64_t inputSize = 1;
  for (auto dim : inputShape)
    inputSize *= dim;
  if (outputShape[1] != inputSize)
    return emitError() << "expected second dimension of output to be "
                       << inputSize << " (product of input dimensions), but got "
                       << outputShape[1];

  return mlir::success();
}

bool FlattenOp::needsShapeInference() { 
  return needsShapeInferenceForVoidOp(*this); 
}

bool FlattenOp::allInputsInferred() { 
  return allInputsInferredUnaryOp(*this);
}

void FlattenOp::inferShapes() {
  mlir::TensorType outputType = getOutput().getType();
  mlir::TensorType inputType = getInput().getType();

  if (outputType.isa<UnrankedTensorType>()) {
    if (auto rankedInputType = inputType.dyn_cast<mlir::RankedTensorType>()) {
      auto inputShape = rankedInputType.getShape();
      int64_t flattenedSize = 1;
      for (auto dim : inputShape) {
        flattenedSize *= dim;
      }
      llvm::SmallVector<int64_t, 2> newShape{1, flattenedSize};

      DeclTensorOp output = llvm::dyn_cast<DeclTensorOp>(getOutput().getDefiningOp());
      materializeTensor(output, newShape);
    }
  } else {
    auto rankedOutputType = outputType.cast<mlir::RankedTensorType>();
    auto outputShape = rankedOutputType.getShape();

    assert(outputShape.size() == 2 && "Output should be 2-dimensional");
    assert(outputShape[0] == 1 && "First dimension should be 1");

    auto rankedInputType = inputType.dyn_cast<mlir::RankedTensorType>();
    auto inputShape = rankedInputType.getShape();
    int64_t flattenedSize = 1;
    for (auto dim : inputShape)
      flattenedSize *= dim;
    assert(outputShape[1] == flattenedSize && "Second dimension should be product of input dimensions");
  }
}

mlir::Value FlattenOp::getWriteValue() {
  return ::getWriteValue(*this);
}

llvm::SmallVector<mlir::Value, 3> FlattenOp::getReadValues() {
  return getReadUnaryOp(*this);
}

//===----------------------------------------------------------------------===//
// MaxOp
//===----------------------------------------------------------------------===//

void MaxOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                        mlir::Value value) {
  state.addOperands(value);
  state.addTypes(builder.getF32Type());
}

void MaxOp::getEffects(
  SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  getEffectsRO(*this, effects);
}

mlir::Value MaxOp::getWriteValue() {
  return nullptr;
}

llvm::SmallVector<mlir::Value, 3> MaxOp::getReadValues() {
  return {getInput()};
}

//===----------------------------------------------------------------------===//
// MinOp
//===----------------------------------------------------------------------===//

void MinOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                        mlir::Value value) {
  state.addOperands(value);
  state.addTypes(builder.getF32Type());
}

void MinOp::getEffects(
  SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  getEffectsRO(*this, effects);
}

mlir::Value MinOp::getWriteValue() {
  return nullptr;
}

llvm::SmallVector<mlir::Value, 3> MinOp::getReadValues() {
  return {getInput()};
}

//===----------------------------------------------------------------------===//
// SumOp
//===----------------------------------------------------------------------===//

void SumOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                        mlir::Value value) {
  state.addOperands(value);
  state.addTypes(builder.getF32Type());
}

void SumOp::getEffects(
  SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  getEffectsRO(*this, effects);
}

mlir::Value SumOp::getWriteValue() {
  return nullptr;
}

llvm::SmallVector<mlir::Value, 3> SumOp::getReadValues() {
  return {getInput()};
}

//===----------------------------------------------------------------------===//
// CastOp
//===----------------------------------------------------------------------===//

/// Returns true if the given set of input and result types are compatible with
/// this cast operation. This is required by the `CastOpInterface` to verify
/// this operation and provide other additional utilities.
bool CastOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
  if (inputs.size() != 1 || outputs.size() != 1)
    return false;
  // The inputs must be Tensors with the same element type.
  TensorType input = llvm::dyn_cast<TensorType>(inputs.front());
  TensorType output = llvm::dyn_cast<TensorType>(outputs.front());
  if (!input || !output || input.getElementType() != output.getElementType())
    return false;
  // The shape is required to match if both types are ranked.
  return !input.hasRank() || !output.hasRank() || input == output;
}

bool CastOp::needsShapeInference() { return getResult().getType().isa<UnrankedTensorType>(); }

bool CastOp::allInputsInferred() { return getInput().getType().isa<RankedTensorType>(); }

void CastOp::inferShapes() { getResult().setType(getInput().getType()); }

mlir::Value CastOp::getWriteValue() {
  return nullptr;
}

llvm::SmallVector<mlir::Value, 3> CastOp::getReadValues() {
  return {getInput()};
}

//===----------------------------------------------------------------------===//
// FunctionOp
//===----------------------------------------------------------------------===//

void FunctionOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                   llvm::StringRef name, mlir::FunctionType type,
                   llvm::ArrayRef<mlir::NamedAttribute> attrs) {
  state.addAttribute(SymbolTable::getSymbolAttrName(),
                     builder.getStringAttr(name));
  state.addAttribute(getFunctionTypeAttrName(state.name), TypeAttr::get(type));
  state.attributes.append(attrs.begin(), attrs.end());
  state.addRegion();
}

mlir::ParseResult FunctionOp::parse(mlir::OpAsmParser &parser,
                                mlir::OperationState &result) {
  // Dispatch to the FunctionOpInterface provided utility method that parses the
  // function operation.
  auto buildFuncType =
      [](mlir::Builder &builder, llvm::ArrayRef<mlir::Type> argTypes,
         llvm::ArrayRef<mlir::Type> results,
         mlir::function_interface_impl::VariadicFlag,
         std::string &) { return builder.getFunctionType(argTypes, results); };

  return mlir::function_interface_impl::parseFunctionOp(
      parser, result, /*allowVariadic=*/false,
      getFunctionTypeAttrName(result.name), buildFuncType,
      getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));
}

void FunctionOp::print(mlir::OpAsmPrinter &p) {
  // Dispatch to the FunctionOpInterface provided utility method that prints the
  // function operation.
  mlir::function_interface_impl::printFunctionOp(
      p, *this, /*isVariadic=*/false, getFunctionTypeAttrName(),
      getArgAttrsAttrName(), getResAttrsAttrName());
}

/// Returns the region on the function operation that is callable.
mlir::Region *FunctionOp::getCallableRegion() { return &getBody(); }

/// Returns the results types that the callable region produces when
/// executed.
llvm::ArrayRef<mlir::Type> FunctionOp::getCallableResults() {
  return getFunctionType().getResults();
}

/// Returns the argument attributes for all callable region arguments or
/// null if there are none.
ArrayAttr FunctionOp::getCallableArgAttrs() {
  return getArgAttrs().value_or(nullptr);
}

/// Returns the result attributes for all callable region results or
/// null if there are none.
ArrayAttr FunctionOp::getCallableResAttrs() {
  return getResAttrs().value_or(nullptr);
}

mlir::Value FunctionOp::getWriteValue() {
  return nullptr;
}

llvm::SmallVector<mlir::Value, 3> FunctionOp::getReadValues() {
  return {};
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult RetOp::verify() {
  // We know that the parent operation is a function, because of the 'HasParent'
  // trait attached to the operation definition.
  auto function = cast<FunctionOp>((*this)->getParentOp());

  /// ReturnOps can only have a single optional operand.
  if (getNumOperands() > 1)
    return emitOpError() << "expects at most 1 return operand";

  // The operand number and types must match the function signature.
  const auto &results = function.getFunctionType().getResults();
  if (getNumOperands() != results.size())
    return emitOpError() << "does not return the same number of values ("
                         << getNumOperands() << ") as the enclosing function ("
                         << results.size() << ")";

  // If the operation does not have an input, we are done.
  if (!hasOperand())
    return mlir::success();

  auto inputType = *operand_type_begin();
  auto resultType = results.front();

  // Check that the result type of the function matches the operand type.
  if (inputType == resultType || llvm::isa<mlir::UnrankedTensorType>(inputType) ||
      llvm::isa<mlir::UnrankedTensorType>(resultType))
    return mlir::success();

  return emitError() << "type of return operand (" << inputType
                     << ") doesn't match function result type (" << resultType
                     << ")";
}

mlir::Value RetOp::getWriteValue() {
  return nullptr;
}

llvm::SmallVector<mlir::Value, 3> RetOp::getReadValues() {
  return getInput();
}

//===----------------------------------------------------------------------===//
// GenericCallOp
//===----------------------------------------------------------------------===//

/// Return the callee of the generic call operation, this is required by the
/// call interface.
CallInterfaceCallable GenericCallOp::getCallableForCallee() {
  return (*this)->getAttrOfType<SymbolRefAttr>("callee");
}

/// Set the callee for the generic call operation, this is required by the call
/// interface.
void GenericCallOp::setCalleeFromCallable(CallInterfaceCallable callee) {
  (*this)->setAttr("callee", callee.get<SymbolRefAttr>());
}

/// Get the argument operands to the called function, this is required by the
/// call interface.
Operation::operand_range GenericCallOp::getArgOperands() { return getInputs(); }

mlir::Value GenericCallOp::getWriteValue() {
  return nullptr;
}

llvm::SmallVector<mlir::Value, 3> GenericCallOp::getReadValues() {
  return getInputs();
}

//===----------------------------------------------------------------------===//
// PrintOp
//===----------------------------------------------------------------------===//

// void PrintOp::getEffects(
//   SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
//   getEffectsRO(*this, effects);
// }

mlir::Value PrintOp::getWriteValue() {
  return nullptr;
}

llvm::SmallVector<mlir::Value, 3> PrintOp::getReadValues() {
  return {getInput()};
}

//===----------------------------------------------------------------------===//
// ScanOp
//===----------------------------------------------------------------------===//

// void ScanOp::getEffects(
//   SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
//   getEffectsRO(*this, effects);
// }

mlir::Value ScanOp::getWriteValue() {
  return ::getWriteValue(*this);
}

llvm::SmallVector<mlir::Value, 3> ScanOp::getReadValues() {
  return {};
}

//===----------------------------------------------------------------------===//
// ClearOp
//===----------------------------------------------------------------------===//

// void ClearOp::getEffects(
//   SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
//   getEffectsRO(*this, effects);
// }

mlir::Value ClearOp::getWriteValue() {
  return ::getWriteValue(*this);
}

llvm::SmallVector<mlir::Value, 3> ClearOp::getReadValues() {
  return {};
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "dialect/Ops.cpp.inc"