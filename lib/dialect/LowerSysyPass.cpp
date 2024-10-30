#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/Support/raw_ostream.h"

#include "dialect/Dialect.h"
#include "dialect/Passes.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// SysyToAffine RewritePatterns
//===----------------------------------------------------------------------===//

/// Convert the given RankedTensorType into the corresponding MemRefType.
static MemRefType convertTensorToMemRef(RankedTensorType type) {
  return MemRefType::get(type.getShape(), type.getElementType());
}

static Value insertAllocAndDealloc(MemRefType type, Location loc,
                                   PatternRewriter &rewriter) {
  auto alloc = rewriter.create<memref::AllocOp>(loc, type);

  // Get the parent function
  auto* parentBlock = alloc->getBlock();
  func::FuncOp funcOp = dyn_cast<func::FuncOp>(parentBlock->getParentOp());
  // Move alloc to the beginning of the function's entry block
  Block &entryBlock = funcOp.getBody().front();
  alloc->moveBefore(&entryBlock.front());

  // Find the exit block(s) and insert dealloc before the return
  for (Block &block : funcOp.getBlocks()) {
    if (mlir::isa<func::ReturnOp>(block.getTerminator())) {
      auto dealloc = rewriter.create<memref::DeallocOp>(loc, alloc);
      dealloc->moveBefore(block.getTerminator());
      break;
    }
  }

  return alloc;
}

static bool isAllZero(ElementsAttr constantValue) {
  return llvm::all_of(constantValue.getValues<Attribute>(), [](Attribute attr) {
    if (auto floatAttr = attr.dyn_cast<FloatAttr>())
      return floatAttr.getValue().isZero();
    return false;
  });
}

static void fillZero(mlir::PatternRewriter& rewriter, mlir::Location loc, Value memref){
  Value memrefPtrAsIndex = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, rewriter.getIndexType(), memref);
  Value memrefPtrAsI64 = rewriter.create<arith::IndexCastOp>(loc, rewriter.getI64Type(), memrefPtrAsIndex);
  auto llvmPtrType = LLVM::LLVMPointerType::get(rewriter.getContext());
  Value memrefPtr = rewriter.create<LLVM::IntToPtrOp>(loc, llvmPtrType, memrefPtrAsI64);

  MemRefType memrefType = memref.getType().cast<MemRefType>();
  int64_t size = memrefType.getNumElements() * memrefType.getElementTypeBitWidth() / 8;
  Value sizeValue = rewriter.create<arith::ConstantIntOp>(loc, size, rewriter.getI64Type());
  
  rewriter.create<LLVM::MemsetOp>(
    loc, 
    memrefPtr,   // 目标内存
    rewriter.create<arith::ConstantIntOp>(loc, 0, rewriter.getI8Type()), // 设置值为 0
    sizeValue,   // 大小
    false
  );
}

static void copyMemref(mlir::PatternRewriter& rewriter, mlir::Location loc, Value srcMemref, Value dstMemref) {
  // 获取源memref的指针
  Value srcMemrefPtrAsIndex = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, rewriter.getIndexType(), srcMemref);
  Value srcMemrefPtrAsI64 = rewriter.create<arith::IndexCastOp>(loc, rewriter.getI64Type(), srcMemrefPtrAsIndex);
  auto llvmPtrType = LLVM::LLVMPointerType::get(rewriter.getContext());
  Value srcMemrefPtr = rewriter.create<LLVM::IntToPtrOp>(loc, llvmPtrType, srcMemrefPtrAsI64);

  // 获取目标memref的指针
  Value dstMemrefPtrAsIndex = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, rewriter.getIndexType(), dstMemref);
  Value dstMemrefPtrAsI64 = rewriter.create<arith::IndexCastOp>(loc, rewriter.getI64Type(), dstMemrefPtrAsIndex);
  Value dstMemrefPtr = rewriter.create<LLVM::IntToPtrOp>(loc, llvmPtrType, dstMemrefPtrAsI64);

  // 计算需要复制的字节数
  MemRefType srcMemrefType = srcMemref.getType().cast<MemRefType>();
  int64_t size = srcMemrefType.getNumElements() * srcMemrefType.getElementTypeBitWidth() / 8;
  Value sizeValue = rewriter.create<arith::ConstantIntOp>(loc, size, rewriter.getI64Type());

  // 使用LLVM::MemcpyOp进行内存复制
  rewriter.create<LLVM::MemcpyOp>(
    loc,
    dstMemrefPtr,  // 目标内存
    srcMemrefPtr,  // 源内存
    sizeValue,     // 复制的字节数
    false          // 是否易失性操作
  );
}

static void fill(PatternRewriter &rewriter, Location loc, Value alloc, Value value){
  MemRefType memRefType = alloc.getType().cast<MemRefType>();
  auto valueShape = memRefType.getShape();
  SmallVector<int64_t, 4> lowerBounds(valueShape.size(), 0);
  SmallVector<int64_t, 4> steps(valueShape.size(), 1);
  
  affine::buildAffineLoopNest(
  rewriter, loc, lowerBounds, valueShape, steps,
  [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {
    nestedBuilder.create<affine::AffineStoreOp>(
      loc, value, alloc, ivs);
  });
}

static void moveToBlockFront(mlir::Operation* op){
  auto *parentBlock = op->getBlock();
  op->moveBefore(&parentBlock->front());
}

static void initialWithGlobalMemory(Value alloc, PatternRewriter &rewriter, Location loc, MemRefType memRefType, ElementsAttr constantValue){
  static int counter = 0;
  std::string varName = "__const__" + std::to_string(counter++);

  ModuleOp moduleOp = alloc.getDefiningOp()->getParentOfType<ModuleOp>();
  OpBuilder::InsertPoint currentIP = rewriter.saveInsertionPoint();
  rewriter.setInsertionPointToStart(moduleOp.getBody());
  rewriter.create<memref::GlobalOp>(
    loc,
    varName,
    nullptr,
    memRefType,
    constantValue,
    true,
    nullptr
  );
  rewriter.restoreInsertionPoint(currentIP);

  auto globalAddr = rewriter.create<memref::GetGlobalOp>(loc, memRefType, varName);
  rewriter.create<memref::CopyOp>(loc, globalAddr, alloc);
}

namespace {
//===----------------------------------------------------------------------===//
// SysyToAffine RewritePatterns: InitTensor operations
//===----------------------------------------------------------------------===//

struct InitTensorOpLowering : public ConversionPattern {
  InitTensorOpLowering(MLIRContext *ctx) 
  : ConversionPattern(sysy::InitTensorOp::getOperationName(), 1, ctx) {}

  LogicalResult 
  matchAndRewrite(Operation *op, ArrayRef<Value> operands, 
                  ConversionPatternRewriter &rewriter) const final {
    Location loc = op->getLoc();

    sysy::InitTensorOpAdaptor initTensorAdaptor(operands);
    Value dst = initTensorAdaptor.getDst();

    auto initTensorOp = cast<sysy::InitTensorOp>(op);
    ElementsAttr constantValue = initTensorOp.getValue();

    if(isAllZero(constantValue))
      fillZero(rewriter, loc, dst);
    else
      initialWithGlobalMemory(dst, rewriter, loc, dst.getType().cast<MemRefType>(), constantValue);

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// SysyToAffine RewritePatterns: InitTensor operations
//===----------------------------------------------------------------------===//

struct DefTensorOpLowering : public OpRewritePattern<sysy::DefTensorOp> {
  using OpRewritePattern<sysy::DefTensorOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(sysy::DefTensorOp op,
                                PatternRewriter &rewriter) const final {
    ArrayAttr shapeAttr = op.getShape();
    Location loc = op.getLoc();

    llvm::SmallVector<int64_t, 4> shapeVector;
    for (auto attr : shapeAttr) {
      if (auto intAttr = attr.dyn_cast<IntegerAttr>()) {
        shapeVector.push_back(intAttr.getInt());
      } else {
        return failure();
      }
    }
    auto memRefType = MemRefType::get(shapeVector, rewriter.getF32Type());

    auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);
    rewriter.replaceOp(op, alloc);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// SysyToAffine RewritePatterns: Broadcast operations
//===----------------------------------------------------------------------===//

struct BroadcastOpLowering : public ConversionPattern {
  BroadcastOpLowering(MLIRContext *ctx) 
  : ConversionPattern(sysy::BroadcastOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();

    sysy::BroadcastOp::Adaptor broadcastAdaptor(operands);
    auto output = broadcastAdaptor.getOutput();
    auto input = broadcastAdaptor.getInput();

    if(llvm::isa<LLVM::ConstantOp>(input.getDefiningOp())){
      auto constantOp = llvm::dyn_cast<LLVM::ConstantOp>(input.getDefiningOp());
      auto attr = constantOp.getValue();
      assert(llvm::isa<FloatAttr>(attr));
      if(llvm::dyn_cast<FloatAttr>(attr).getValue().isZero()){
        fillZero(rewriter, loc, output);
        rewriter.eraseOp(op);
        return success();
      }
    }

    fill(rewriter, loc, output, input);
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// SysyToAffine RewritePatterns: Elementwise operations
//===----------------------------------------------------------------------===//

template <typename verifyBinaryElementwiseOp, typename LoweredBinaryOp>
struct BinaryElementwiseOpLowering : public ConversionPattern {
  BinaryElementwiseOpLowering(MLIRContext *ctx)
      : ConversionPattern(verifyBinaryElementwiseOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    auto dstType = llvm::cast<RankedTensorType>(op->getOperand(0).getType());
    SmallVector<int64_t, 4> lowerBounds(dstType.getRank(), /*Value=*/0);
    SmallVector<int64_t, 4> steps(dstType.getRank(), /*Value=*/1);
    affine::buildAffineLoopNest(
        rewriter, loc, lowerBounds, dstType.getShape(), steps,
        [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {
          typename verifyBinaryElementwiseOp::Adaptor binaryElementwiseAdaptor(operands);
          auto loadedLhs = rewriter.create<affine::AffineLoadOp>(loc, binaryElementwiseAdaptor.getLhs(), ivs);
          auto loadedRhs = rewriter.create<affine::AffineLoadOp>(loc, binaryElementwiseAdaptor.getRhs(), ivs);
          auto resVal = rewriter.create<LoweredBinaryOp>(loc, loadedLhs, loadedRhs);
          nestedBuilder.create<affine::AffineStoreOp>(loc, resVal, binaryElementwiseAdaptor.getDst(), ivs);
        });

    rewriter.eraseOp(op);
    return success();
  }
};
using MulOpLowering = BinaryElementwiseOpLowering<sysy::MulOp, arith::MulFOp>;
using AddOpLowering = BinaryElementwiseOpLowering<sysy::AddOp, arith::AddFOp>;
using SubOpLowering = BinaryElementwiseOpLowering<sysy::SubOp, arith::SubFOp>;
using DivOpLowering = BinaryElementwiseOpLowering<sysy::DivOp, arith::DivFOp>;

template <typename verifyBinaryElementwiseOp, typename LoweredBinaryOp>
struct BinaryElementwiseOpVecLowering : public ConversionPattern {
  BinaryElementwiseOpVecLowering(MLIRContext *ctx)
      : ConversionPattern(verifyBinaryElementwiseOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();

    typename verifyBinaryElementwiseOp::Adaptor binaryElementwiseAdaptor(operands);
    Value lhs = binaryElementwiseAdaptor.getLhs();
    Value rhs = binaryElementwiseAdaptor.getRhs();
    Value dst = binaryElementwiseAdaptor.getDst();

    auto dstType = llvm::cast<RankedTensorType>(op->getOperand(0).getType());

    constexpr uint64_t vecSize = 4;
    auto vType = VectorType::get({vecSize}, rewriter.getF32Type());
    
    SmallVector<int64_t, 2> lowerBounds(2, 0);
    SmallVector<int64_t, 2> upperBounds = {dstType.getShape()[0], dstType.getShape()[1]};
    SmallVector<int64_t, 2> steps{1, vecSize};

    affine::buildAffineLoopNest(
        rewriter, loc, lowerBounds, upperBounds, steps,
        [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {
          auto lhsVec = nestedBuilder.create<vector::LoadOp>(loc, vType, lhs, ValueRange{ivs[0], ivs[1]});
          auto rhsVec = nestedBuilder.create<vector::LoadOp>(loc, vType, rhs, ValueRange{ivs[0], ivs[1]});
          auto dstVec = nestedBuilder.create<LoweredBinaryOp>(loc, vType, lhsVec, rhsVec);
          nestedBuilder.create<vector::StoreOp>(loc, dstVec, dst, ValueRange{ivs[0], ivs[1]});
        });

    auto insertPoint = rewriter.getInsertionPoint();
    Operation *outerLoop = &*std::prev(insertPoint);
    // outerLoop->setAttr("tileSizes", rewriter.getI32ArrayAttr({16, 1}));
    outerLoop->setAttr("parallel", rewriter.getUnitAttr());

    rewriter.eraseOp(op);
    return success();
  }
};
using MulOpVecLowering = BinaryElementwiseOpVecLowering<sysy::MulOp, arith::MulFOp>;
using AddOpVecLowering = BinaryElementwiseOpVecLowering<sysy::AddOp, arith::AddFOp>;
using SubOpVecLowering = BinaryElementwiseOpVecLowering<sysy::SubOp, arith::SubFOp>;
using DivOpVecLowering = BinaryElementwiseOpVecLowering<sysy::DivOp, arith::DivFOp>;

//===----------------------------------------------------------------------===//
// SysyToAffine RewritePatterns: MatMul operations
//===----------------------------------------------------------------------===//

struct MatMulOpLowering : public ConversionPattern {
  MatMulOpLowering(MLIRContext *ctx) 
  : ConversionPattern(sysy::MatMulOp::getOperationName(), 1, ctx) {}

  LogicalResult 
  matchAndRewrite(Operation *op, ArrayRef<Value> operands, 
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    auto lhsType = llvm::cast<RankedTensorType>(op->getOperand(1).getType());
    auto rhsType = llvm::cast<RankedTensorType>(op->getOperand(2).getType());

    auto M = lhsType.getShape()[0];
    auto K = lhsType.getShape()[1];
    auto N = rhsType.getShape()[1];
    
    SmallVector<int64_t, 3> lowerBounds(3, 0);
    SmallVector<int64_t, 3> upperBounds = {M, N, K};
    SmallVector<int64_t, 3> steps(3, 1);

    affine::buildAffineLoopNest(
      rewriter, loc, lowerBounds, upperBounds, steps,
      [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {
        sysy::MatMulOpAdaptor matMulAdaptor(operands);
        Value lhs = matMulAdaptor.getLhs();
        Value rhs = matMulAdaptor.getRhs();
        Value dst = matMulAdaptor.getDst();

        auto lhsElement = nestedBuilder.create<affine::AffineLoadOp>(loc, lhs, ValueRange{ivs[0], ivs[2]});
        auto rhsElement = nestedBuilder.create<affine::AffineLoadOp>(loc, rhs, ValueRange{ivs[2], ivs[1]});
        auto dstElement = nestedBuilder.create<affine::AffineLoadOp>(loc, dst, ValueRange{ivs[0], ivs[1]});
        auto mulResult = nestedBuilder.create<arith::MulFOp>(loc, lhsElement, rhsElement);
        auto addResult = nestedBuilder.create<arith::AddFOp>(loc, dstElement, mulResult);
        nestedBuilder.create<affine::AffineStoreOp>(loc, addResult, dst, ValueRange{ivs[0], ivs[1]});
      });

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// SysyToAffine RewritePatterns: Matmul operations vector generation
//===----------------------------------------------------------------------===//

struct MatMulOpVecLowering : public ConversionPattern {
  MatMulOpVecLowering(MLIRContext *ctx)
  : ConversionPattern(sysy::MatMulOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    auto lhsType = llvm::cast<RankedTensorType>(op->getOperand(1).getType());
    auto rhsType = llvm::cast<RankedTensorType>(op->getOperand(2).getType());

    auto M = lhsType.getShape()[0];
    auto K = lhsType.getShape()[1];
    auto N = rhsType.getShape()[1];

    constexpr uint64_t vecSize = 4;
    auto vType = VectorType::get({vecSize}, rewriter.getF32Type());

    SmallVector<int64_t, 3> lowerBounds(3, 0);
    SmallVector<int64_t, 3> upperBounds = {M, N, K};
    SmallVector<int64_t, 3> steps { 1, vecSize, 1 };

    affine::buildAffineLoopNest(
      rewriter, loc, lowerBounds, upperBounds, steps,
      [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {
        sysy::MatMulOpAdaptor matMulAdaptor(operands);
        Value lhs = matMulAdaptor.getLhs();
        Value rhs = matMulAdaptor.getRhs();
        Value dst = matMulAdaptor.getDst();

        auto lhsElement = nestedBuilder.create<affine::AffineLoadOp>(loc, lhs, ValueRange{ivs[0], ivs[2]});
        auto lhsVec = nestedBuilder.create<vector::SplatOp>(loc, lhsElement, vType);
        auto rhsVec = nestedBuilder.create<vector::LoadOp>(loc, vType, rhs, ValueRange{ivs[2], ivs[1]});
        auto dstVec = nestedBuilder.create<vector::LoadOp>(loc, vType, dst, ValueRange{ivs[0], ivs[1]});
        auto resVec = nestedBuilder.create<vector::FMAOp>(loc, vType, lhsVec, rhsVec, dstVec);
        nestedBuilder.create<vector::StoreOp>(loc, resVec, dst, ValueRange{ivs[0], ivs[1]});
      });

    auto insertPoint = rewriter.getInsertionPoint();
    Operation *outerLoop = &*std::prev(insertPoint);
    outerLoop->setAttr("tileSizes", rewriter.getI32ArrayAttr({8, 4, 16}));
    outerLoop->setAttr("parallel", rewriter.getUnitAttr());

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// SysyToAffine RewritePatterns: Matmul operations vector generation
//===----------------------------------------------------------------------===//

struct MatMulOpVec2Lowering : public ConversionPattern {
  MatMulOpVec2Lowering(MLIRContext *ctx)
  : ConversionPattern(sysy::MatMulOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    auto lhsType = llvm::cast<RankedTensorType>(op->getOperand(1).getType());
    auto rhsType = llvm::cast<RankedTensorType>(op->getOperand(2).getType());

    auto M = lhsType.getShape()[0];
    auto K = lhsType.getShape()[1];
    auto N = rhsType.getShape()[1];

    constexpr uint64_t vecSize = 4;
    auto vType = VectorType::get({vecSize}, rewriter.getF32Type());

    SmallVector<int64_t, 3> lowerBounds(3, 0);
    SmallVector<int64_t, 3> upperBounds = {M, N, K};
    SmallVector<int64_t, 3> steps { 1, vecSize, 1 };

    affine::buildAffineLoopNest(
      rewriter, loc, lowerBounds, upperBounds, steps,
      [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {
        sysy::MatMulOpAdaptor matMulAdaptor(operands);
        Value lhs = matMulAdaptor.getLhs();
        Value rhs = matMulAdaptor.getRhs();
        Value dst = matMulAdaptor.getDst();

        auto lhsElement = nestedBuilder.create<affine::AffineLoadOp>(loc, lhs, ValueRange{ivs[0], ivs[2]});
        auto lhsVec = nestedBuilder.create<vector::SplatOp>(loc, lhsElement, vType);
        auto rhsVec = nestedBuilder.create<vector::LoadOp>(loc, vType, rhs, ValueRange{ivs[2], ivs[1]});
        auto dstVec = nestedBuilder.create<vector::LoadOp>(loc, vType, dst, ValueRange{ivs[0], ivs[1]});
        auto resVec = nestedBuilder.create<vector::FMAOp>(loc, vType, lhsVec, rhsVec, dstVec);
        nestedBuilder.create<vector::StoreOp>(loc, resVec, dst, ValueRange{ivs[0], ivs[1]});
      });

    auto insertPoint = rewriter.getInsertionPoint();
    Operation *outerLoop = &*std::prev(insertPoint);
    outerLoop->setAttr("tileSizes", rewriter.getI32ArrayAttr({8, 8, 32}));
    outerLoop->setAttr("interchange", rewriter.getI32ArrayAttr({4, 5}));
    outerLoop->setAttr("parallel", rewriter.getUnitAttr());

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// SysyToAffine RewritePatterns: ValidConv operations
//===----------------------------------------------------------------------===//

void performConvolution(OpBuilder &rewriter, Location loc, ValueRange operands, Value input) {
    Value output = operands[0];
    Value kernel = operands[2];
    std::vector<int64_t> outputShape = llvm::cast<MemRefType>(output.getType()).getShape();
    std::vector<int64_t> kernelShape = llvm::cast<MemRefType>(kernel.getType()).getShape();

    SmallVector<int64_t, 4> lowerBounds(4, 0);
    SmallVector<int64_t, 4> upperBounds;
    SmallVector<int64_t, 4> steps(4, 1);
    upperBounds.push_back(outputShape[0]);
    upperBounds.push_back(outputShape[1]);
    upperBounds.push_back(kernelShape[0]);
    upperBounds.push_back(kernelShape[1]);

    affine::buildAffineLoopNest(rewriter, loc, lowerBounds, upperBounds, steps,
                              [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {
      MLIRContext *context = nestedBuilder.getContext();
      AffineMap inputMap = AffineMap::get(4, 0, {
          getAffineDimExpr(0, context) + getAffineDimExpr(2, context),
          getAffineDimExpr(1, context) + getAffineDimExpr(3, context)}, context);
      auto LoadedInput = nestedBuilder.create<affine::AffineLoadOp>(loc, input, inputMap, ivs);
      auto LoadedKernel = nestedBuilder.create<affine::AffineLoadOp>(loc, kernel, ValueRange{ivs[2], ivs[3]});
      auto LoadedOutput = nestedBuilder.create<affine::AffineLoadOp>(loc, output, ValueRange{ivs[0], ivs[1]});
      auto mulResult = nestedBuilder.create<arith::MulFOp>(loc, LoadedInput, LoadedKernel);
      auto addResult = nestedBuilder.create<arith::AddFOp>(loc, mulResult, LoadedOutput);
      nestedBuilder.create<affine::AffineStoreOp>(loc, addResult, output, ValueRange{ivs[0], ivs[1]});
    });

    // 添加属性
    auto insertPoint = rewriter.getInsertionPoint();
    Operation *outerLoop = &*std::prev(insertPoint);
    outerLoop->setAttr("parallel", rewriter.getUnitAttr());
}

//Conv_Val：input 4x   kernel 4y  output 4x-4y+1
//Conv_same: input 任意  fill 4x(本身就填0)  kernel 4y  output 4z
//tile_size: 4
void performVec1Convolution(ConversionPatternRewriter &rewriter, Location loc, ValueRange operands, Value input) {
    Value output = operands[0];
    Value kernel = operands[2];
    std::vector<int64_t> outputShape = llvm::cast<MemRefType>(output.getType()).getShape();
    std::vector<int64_t> kernelShape = llvm::cast<MemRefType>(kernel.getType()).getShape();

    //向量化
    constexpr uint64_t vecSize = 4;
    auto vType = VectorType::get({vecSize}, rewriter.getF32Type());

    //临时buffer
    MemRefType bufferTy = MemRefType::get(1, vType);
    Value buffer = rewriter.create<memref::AllocOp>(loc, bufferTy);

    const AffineExpr d0 = rewriter.getAffineDimExpr(0);
    const AffineExpr d1 = rewriter.getAffineDimExpr(1);

    SmallVector<int64_t, 2> lowerBounds(2, 0);
    SmallVector<int64_t, 2> upperBounds;
    SmallVector<int64_t, 2> steps{1,1};
    upperBounds.push_back(outputShape[0]);
    upperBounds.push_back(outputShape[1]);
    const Value c0 = rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0));
    const Value cf0 = rewriter.create<arith::ConstantOp>(loc, rewriter.getF32FloatAttr(0.0));
    affine::buildAffineLoopNest(rewriter, loc, lowerBounds, upperBounds, steps,
        [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {

        SmallVector<Value, 2> Output_range{ivs[0],ivs[1]};
        Value t = nestedBuilder.create<vector::SplatOp>(loc, vType, cf0);
        nestedBuilder.create<memref::StoreOp>(loc, t, buffer, c0);//buffer置0.0

        SmallVector<int64_t, 2> lowerBounds1(2, 0);
        SmallVector<int64_t, 2> upperBounds1;
        SmallVector<int64_t, 2> steps1{1,vecSize};
        upperBounds1.push_back(kernelShape[0]);
        upperBounds1.push_back(kernelShape[1]);
        affine::buildAffineLoopNest(rewriter, loc, lowerBounds1, upperBounds1, steps1,
            [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {
            SmallVector<Value, 2> Kernel_range{ivs[0],ivs[1]};

            //affineApplyOp
            Value inputRangeX = nestedBuilder.create<affine::AffineApplyOp>(loc, AffineMap::get(2, 0, d0 + d1), ValueRange{Output_range[0], ivs[0]});
            Value inputRangeY = nestedBuilder.create<affine::AffineApplyOp>(loc, AffineMap::get(2, 0, d0 + d1), ValueRange{Output_range[1], ivs[1]});
            SmallVector<Value, 2> Input_range{inputRangeX, inputRangeY};

            auto LoadedInputVec = nestedBuilder.create<vector::LoadOp>(loc, vType, input, Input_range); //<4x4xf32xvector>

            auto LoadedKernelVec = nestedBuilder.create<vector::LoadOp>(loc, vType , kernel, Kernel_range); //<4x4xf32xvector>

            auto LoadedOutputVec = nestedBuilder.create<vector::LoadOp>(loc, vType, buffer,c0); //<4x4xf32xvector>
            auto ResultVec = nestedBuilder.create<vector::FMAOp>(loc, vType, LoadedInputVec, LoadedKernelVec, LoadedOutputVec); //<4x4xf32xvector>
            nestedBuilder.create<vector::StoreOp>(loc, ResultVec, buffer, c0); //<4x4xf32xmemref>
        });

        Value reduceVec = nestedBuilder.create<vector::LoadOp>(loc,vType, buffer, c0);
        Value reducedRes = nestedBuilder.create<vector::ReductionOp>(loc, vector::CombiningKind::ADD, reduceVec);
        Value bias = nestedBuilder.create<affine::AffineLoadOp>(loc, output, Output_range);
        Value addRes = nestedBuilder.create<arith::AddFOp>(loc, bias, reducedRes);
        nestedBuilder.create<affine::AffineStoreOp>(loc, addRes, output, Output_range);
        //bufferVec -> output
    });
    // 添加属性
    auto insertPoint = rewriter.getInsertionPoint();
    Operation *outerLoop = &*std::prev(insertPoint);
    outerLoop->setAttr("parallel", rewriter.getUnitAttr());

    rewriter.create<memref::DeallocOp>(loc, buffer);
}

//Conv_Val: input   kernel  output 任意
//Conv_Same: input  kernel  output 任意
//tile_size: 4  带填充
void performVec2Convolution(ConversionPatternRewriter &rewriter, Location loc, ValueRange operands, Value input) {
    Value output = operands[0];
    Value kernel = operands[2];
    std::vector<int64_t> outputShape = llvm::cast<MemRefType>(output.getType()).getShape();
    std::vector<int64_t> kernelShape = llvm::cast<MemRefType>(kernel.getType()).getShape();
    std::vector<int64_t> inputShape = llvm::cast<MemRefType>(input.getType()).getShape();

    //向量化
    constexpr uint64_t vecSize = 4;
    auto vType = VectorType::get({vecSize}, rewriter.getF32Type());

    //最右侧0填充Input、Kernel
    int64_t input_pad = vecSize - inputShape[1]%vecSize;
    int64_t kernel_pad = vecSize - kernelShape[1]%vecSize;

    // 计算填充后的缓冲区大小
    SmallVector<int64_t, 2> padInputShape;
    padInputShape.push_back(inputShape[0]);
    padInputShape.push_back(inputShape[1]+input_pad);
    // 计算填充后的缓冲区大小
    SmallVector<int64_t, 2> padKerneltShape;
    padKerneltShape.push_back(kernelShape[0]);
    padKerneltShape.push_back(kernelShape[1]+kernel_pad);

    // 申请临时缓冲区
    auto paddedInputType = MemRefType::get(padInputShape, rewriter.getF32Type());
    Value paddedInput = rewriter.create<memref::AllocOp>(loc, paddedInputType);
    // 申请临时缓冲区
    auto paddedKernelType = MemRefType::get(padKerneltShape, rewriter.getF32Type());
    Value paddedKernel = rewriter.create<memref::AllocOp>(loc, paddedKernelType);

    // 填充0
    fillZero(rewriter, loc, paddedInput);
    // 填充0
    fillZero(rewriter, loc, paddedKernel);

    // 填充input的数据
    SmallVector<int64_t, 2> lowerBoundsx(2, 0);
    SmallVector<int64_t, 2> upperBoundsx{inputShape[0], inputShape[1]};
    SmallVector<int64_t, 4> stepsx(2, 1);
    affine::buildAffineLoopNest(rewriter, loc, lowerBoundsx, upperBoundsx, stepsx,
        [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {
        auto LoadedInput = nestedBuilder.create<affine::AffineLoadOp>(loc, input, ivs);
        nestedBuilder.create<affine::AffineStoreOp>(loc, LoadedInput, paddedInput,ivs);
    });
    // 添加属性
    auto insertPointx = rewriter.getInsertionPoint();
    Operation *outerLoopx = &*std::prev(insertPointx);
    outerLoopx->setAttr("parallel", rewriter.getUnitAttr());

    //填充kernel
    SmallVector<int64_t, 2> lowerBoundsy(2, 0);
    SmallVector<int64_t, 2> upperBoundsy{kernelShape[0], kernelShape[1]};
    SmallVector<int64_t, 4> stepsy(2, 1);
    affine::buildAffineLoopNest(rewriter, loc, lowerBoundsy, upperBoundsy, stepsy,
             [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {
        auto LoadedKernel = nestedBuilder.create<affine::AffineLoadOp>(loc, kernel, ivs);
        nestedBuilder.create<affine::AffineStoreOp>(loc, LoadedKernel, paddedKernel, ivs);
    });
    // 添加属性
    auto insertPointy = rewriter.getInsertionPoint();
    Operation *outerLoopy = &*std::prev(insertPointy);
    outerLoopy->setAttr("parallel", rewriter.getUnitAttr());

    //临时buffer
    MemRefType bufferTy = MemRefType::get(1, vType);
    Value buffer = rewriter.create<memref::AllocOp>(loc, bufferTy);

    const AffineExpr d0 = rewriter.getAffineDimExpr(0);
    const AffineExpr d1 = rewriter.getAffineDimExpr(1);

    SmallVector<int64_t, 2> lowerBounds(2, 0);
    SmallVector<int64_t, 2> upperBounds;
    SmallVector<int64_t, 2> steps{1,1};
    upperBounds.push_back(outputShape[0]);
    upperBounds.push_back(outputShape[1]);
    const Value c0 = rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0));
    const Value cf0 = rewriter.create<arith::ConstantOp>(loc, rewriter.getF32FloatAttr(0.0));
    affine::buildAffineLoopNest(rewriter, loc, lowerBounds, upperBounds, steps,
       [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {
        SmallVector<Value, 2> Output_range{ivs[0],ivs[1]};
        Value t = nestedBuilder.create<vector::SplatOp>(loc, vType, cf0);
        nestedBuilder.create<memref::StoreOp>(loc, t, buffer, c0);//buffer置0.0

        SmallVector<int64_t, 2> lowerBounds1(2, 0);
        SmallVector<int64_t, 2> upperBounds1;
        SmallVector<int64_t, 2> steps1{1,vecSize};
        upperBounds1.push_back(kernelShape[0]);
        upperBounds1.push_back(kernelShape[1]);
        affine::buildAffineLoopNest(rewriter, loc, lowerBounds1, upperBounds1, steps1,
                                    [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {
            SmallVector<Value, 2> Kernel_range{ivs[0],ivs[1]};
            //affineApplyOp
            Value inputRangeX = nestedBuilder.create<affine::AffineApplyOp>(loc, AffineMap::get(2, 0, d0 + d1), ValueRange{Output_range[0], ivs[0]});
            Value inputRangeY = nestedBuilder.create<affine::AffineApplyOp>(loc, AffineMap::get(2, 0, d0 + d1), ValueRange{Output_range[1], ivs[1]});
            SmallVector<Value, 2> Input_range{inputRangeX, inputRangeY};

            auto LoadedInputVec = nestedBuilder.create<vector::LoadOp>(loc, vType, paddedInput, Input_range); //<4x4xf32xvector>
            auto LoadedKernelVec = nestedBuilder.create<vector::LoadOp>(loc, vType , paddedKernel, Kernel_range); //<4x4xf32xvector>
            auto LoadedOutputVec = nestedBuilder.create<vector::LoadOp>(loc, vType, buffer,c0); //<4x4xf32xvector>
            auto ResultVec = nestedBuilder.create<vector::FMAOp>(loc, vType, LoadedInputVec, LoadedKernelVec, LoadedOutputVec); //<4x4xf32xvector>
            nestedBuilder.create<vector::StoreOp>(loc, ResultVec, buffer, c0); //<4x4xf32xmemref>
        });

        Value reduceVec = nestedBuilder.create<vector::LoadOp>(loc,vType, buffer, c0);
        Value reducedRes = nestedBuilder.create<vector::ReductionOp>(loc, vector::CombiningKind::ADD, reduceVec);
        Value bias = nestedBuilder.create<affine::AffineLoadOp>(loc, output, Output_range);
        Value addRes = nestedBuilder.create<arith::AddFOp>(loc, bias, reducedRes);
        nestedBuilder.create<affine::AffineStoreOp>(loc, addRes, output, Output_range);
        //bufferVec -> output
    });
    // 添加属性
    auto insertPoint = rewriter.getInsertionPoint();
    Operation *outerLoop = &*std::prev(insertPoint);
    outerLoop->setAttr("parallel", rewriter.getUnitAttr());

    rewriter.create<memref::DeallocOp>(loc, buffer);
    rewriter.create<memref::DeallocOp>(loc, paddedInput);
    rewriter.create<memref::DeallocOp>(loc, paddedKernel);
}

struct ValidConvOpLowering : public ConversionPattern {
  ValidConvOpLowering(MLIRContext *ctx) 
  : ConversionPattern(sysy::ValidConvOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();

    // performConvolution(rewriter, loc, operands, operands[1]);
    performVec2Convolution(rewriter, loc, operands, operands[1]);

    //Replace this operation with the generated alloc.
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// SysyToAffine RewritePatterns: SameConv operations
//===----------------------------------------------------------------------===//

Value createPaddedInput(ConversionPatternRewriter &rewriter, Location loc, ValueRange operands) {
  Value input = operands[1];
  Value kernel = operands[2];
  std::vector<int64_t> inputShape = llvm::cast<MemRefType>(input.getType()).getShape();
  std::vector<int64_t> kernelShape = llvm::cast<MemRefType>(kernel.getType()).getShape();
  
  // 计算填充后的缓冲区大小
  SmallVector<int64_t, 2> padShape;
  padShape.push_back(inputShape[0] + kernelShape[0] - 1);
  padShape.push_back(inputShape[1] + kernelShape[1] - 1);

  // 申请临时缓冲区
  auto paddedType = MemRefType::get(padShape, rewriter.getF32Type());
  Value paddedInput = rewriter.create<memref::AllocOp>(loc, paddedType);

  // 填充0
  fillZero(rewriter, loc, paddedInput);

  // 填充input的数据
  SmallVector<int64_t, 2> lowerBounds(2, 0);
  SmallVector<int64_t, 2> upperBounds{inputShape[0], inputShape[1]};
  SmallVector<int64_t, 4> steps(2, 1);
  affine::buildAffineLoopNest(rewriter, loc, lowerBounds, upperBounds, steps,
    [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {
      auto LoadedInput = nestedBuilder.create<affine::AffineLoadOp>(loc, input, ivs);
      
      MLIRContext *context = nestedBuilder.getContext();
      AffineMap paddedInputMap = AffineMap::get(2, 0, {
        getAffineConstantExpr((kernelShape[0] - 1) / 2, context) + getAffineDimExpr(0, context),
        getAffineConstantExpr((kernelShape[1] - 1) / 2, context) + getAffineDimExpr(1, context)
      }, context);
      nestedBuilder.create<affine::AffineStoreOp>(loc, LoadedInput, paddedInput, paddedInputMap, ivs);
    }
  );
  
  // 添加属性
  auto insertPoint = rewriter.getInsertionPoint();
  Operation *outerLoop = &*std::prev(insertPoint);
  outerLoop->setAttr("parallel", rewriter.getUnitAttr());

  return paddedInput;
}

struct SameConvOpLowering : public ConversionPattern {
  SameConvOpLowering(MLIRContext *ctx) 
  : ConversionPattern(sysy::SameConvOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();

    // 创建填充后的input
    Value paddedInput = createPaddedInput(rewriter, loc, operands);

    // 执行卷积操作
    // performConvolution(rewriter, loc, operands, paddedInput);
    performVec2Convolution(rewriter, loc, operands, paddedInput);

    // 释放临时缓冲区
    rewriter.create<memref::DeallocOp>(loc, paddedInput);
    
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// SysyToAffine RewritePatterns: MaxPool operations
//===----------------------------------------------------------------------===//

struct MaxPoolOpLowering : public ConversionPattern {
    MaxPoolOpLowering(MLIRContext *ctx)
    : ConversionPattern(sysy::MaxPoolOp::getOperationName(), 1, ctx) {}

    LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,ConversionPatternRewriter &rewriter) const final {
        auto loc = op->getLoc();
        Value output = operands[0];
        Value input = operands[1];
        Value kernel = operands[2];

        std::vector<int64_t> outputShape = llvm::cast<MemRefType>(output.getType()).getShape();
        std::vector<int64_t> kernelShape = llvm::cast<MemRefType>(kernel.getType()).getShape();

        //分配临时缓冲区
        auto Max_ValueType = MemRefType::get({1},rewriter.getF32Type());
        Value Max_Value = rewriter.create<memref::AllocOp>(loc,Max_ValueType);

        const AffineExpr d0 = rewriter.getAffineDimExpr(0);
        const AffineExpr d1 = rewriter.getAffineDimExpr(1);

        SmallVector<int64_t, 2> lowerBounds(2, 0);
        SmallVector<int64_t, 2> upperBounds;
        SmallVector<int64_t, 2> steps{1,1};
        upperBounds.push_back(outputShape[0]);
        upperBounds.push_back(outputShape[1]);
        const Value c0 = rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0));
        const Value cf0 = rewriter.create<arith::ConstantOp>(loc, rewriter.getF32FloatAttr(0.0));
        affine::buildAffineLoopNest(rewriter, loc, lowerBounds, upperBounds, steps,
             [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {
            //affineApplyOp
            Value inputRangeX = nestedBuilder.create<affine::AffineApplyOp>(loc, AffineMap::get(1, 0, d0*2), ivs[0]);
            Value inputRangeY = nestedBuilder.create<affine::AffineApplyOp>(loc, AffineMap::get(1, 0, d0*2), ivs[1]);
            SmallVector<Value, 2> Input_range{inputRangeX, inputRangeY};
            auto t = nestedBuilder.create<affine::AffineLoadOp>(loc,input, Input_range); //max_value = input[i*stride,j*stride]
            nestedBuilder.create<affine::AffineStoreOp>(loc,t,Max_Value,c0);

            SmallVector<int64_t, 2> lowerBounds1(2, 0);
            SmallVector<int64_t, 2> upperBounds1;
            SmallVector<int64_t, 2> steps1{1,1};
            upperBounds1.push_back(kernelShape[0]);
            upperBounds1.push_back(kernelShape[1]);
            affine::buildAffineLoopNest(rewriter, loc, lowerBounds1, upperBounds1, steps1,
                 [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {
                MLIRContext *context = nestedBuilder.getContext();

                Value outputRangeX = nestedBuilder.create<affine::AffineApplyOp>(loc, AffineMap::get(2, 0, d0 + d1), ValueRange{inputRangeX,ivs[0]});
                Value outputRangeY = nestedBuilder.create<affine::AffineApplyOp>(loc, AffineMap::get(2, 0, d0 + d1), ValueRange{inputRangeY,ivs[1]});
                SmallVector<Value, 2> Output_range{outputRangeX, outputRangeY};
                auto LoadInput = nestedBuilder.create<affine::AffineLoadOp>(loc,input, Output_range); //max_value = input[i*stride,j*stride]
                auto MAX1 =  nestedBuilder.create<affine::AffineLoadOp>(loc,Max_Value,c0);
                auto MAX2 =  nestedBuilder.create<arith::MaxFOp>(loc,MAX1,LoadInput);
                nestedBuilder.create<affine::AffineStoreOp>(loc,MAX2,Max_Value,c0);
            });

            auto t1 = nestedBuilder.create<affine::AffineLoadOp>(loc,Max_Value,c0); //
            nestedBuilder.create<affine::AffineStoreOp>(loc, t1, output, ValueRange{ivs[0],ivs[1]});
        });

        // 添加属性
        auto insertPoint = rewriter.getInsertionPoint();
        Operation *outerLoop = &*std::prev(insertPoint);
        outerLoop->setAttr("parallel", rewriter.getUnitAttr());

        rewriter.create<memref::DeallocOp>(loc, Max_Value);
        rewriter.eraseOp(op);
        return success();
    }
};

//===----------------------------------------------------------------------===//
// SysyToAffine RewritePatterns: Transpose operations
//===----------------------------------------------------------------------===//

struct TransposeOpLowering : public ConversionPattern {
  TransposeOpLowering(MLIRContext *ctx)
      : ConversionPattern(sysy::TransposeOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    // sysy::TransposeOpAdaptor transposeAdaptor(operands);
    // rewriter.create<memref::TransposeOp>(loc, transposeAdaptor.getInput(), transposeAdaptor.getOutput());
    auto outputType = llvm::cast<RankedTensorType>(op->getOperand(0).getType());
    SmallVector<int64_t, 4> lowerBounds(outputType.getRank(), /*Value=*/0);
    SmallVector<int64_t, 4> steps(outputType.getRank(), /*Value=*/1);
    affine::buildAffineLoopNest(
        rewriter, loc, lowerBounds, outputType.getShape(), steps,
        [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {
          sysy::TransposeOpAdaptor transposeAdaptor(operands);
          SmallVector<Value, 2> reverseIvs(llvm::reverse(ivs));
          auto loadedInput = rewriter.create<affine::AffineLoadOp>(loc, transposeAdaptor.getInput(), reverseIvs);
          nestedBuilder.create<affine::AffineStoreOp>(loc, loadedInput, transposeAdaptor.getOutput(), ivs);
        });

    auto insertPoint = rewriter.getInsertionPoint();
    Operation *outerLoop = &*std::prev(insertPoint);
    outerLoop->setAttr("parallel", rewriter.getUnitAttr());

    rewriter.eraseOp(op);
    return success();
  }
};


//===----------------------------------------------------------------------===//
// SysyToAffine RewritePatterns: Copy operations
//===----------------------------------------------------------------------===//

struct CopyOpLowering : public ConversionPattern {
  CopyOpLowering(MLIRContext *ctx)
      : ConversionPattern(sysy::CopyOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    sysy::CopyOpAdaptor copyAdaptor(operands);
    rewriter.create<memref::CopyOp>(loc, copyAdaptor.getInput(), copyAdaptor.getOutput());
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// SysyToAffine RewritePatterns: ReLU operations
//===----------------------------------------------------------------------===//

struct ReLUOpLowering : public ConversionPattern {
  ReLUOpLowering(MLIRContext *ctx)
      : ConversionPattern(sysy::ReLUOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    //mem
    auto inputMem = operands[1];
    auto outputMem = operands[0];
    auto initialValue = rewriter.create<arith::ConstantOp>(loc, rewriter.getF32FloatAttr(0.0));
    //目前可以任意维度规约，但最内层循环必须是4的倍数
    constexpr int64_t vecSize = 4;
    auto vType = VectorType::get({vecSize}, rewriter.getF32Type());
    //verctor
    Value initVecValue = rewriter.create<vector::BroadcastOp>(loc, vType, initialValue);
    auto inputType = llvm::cast<RankedTensorType>(op->getOperand(1).getType());
    int64_t rank = inputType.getRank();
    SmallVector<int64_t, 4> lowerBounds(rank, 0);
    SmallVector<int64_t, 4> steps(rank, 1);
    // 将最内层循环的步长设置为 4
    steps[rank - 1] = vecSize;
    affine::buildAffineLoopNest(
      rewriter, loc, lowerBounds, inputType.getShape(), steps,
      [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs){
        //vector
        auto vecElements = nestedBuilder.create<vector::LoadOp>(loc, vType, inputMem, ValueRange(ivs.begin(), ivs.end()));
        //比较并赋值
        auto condition = nestedBuilder.create<arith::CmpFOp>(loc, arith::CmpFPredicate::UGE, vecElements, initVecValue); 
        auto newVec = nestedBuilder.create<arith::SelectOp>(loc, vType, condition, vecElements, initVecValue);
        //vec存到结果中
        nestedBuilder.create<vector::StoreOp>(loc, newVec, outputMem, ValueRange(ivs.begin(), ivs.end()));
      });
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// SysyToAffine RewritePatterns: Flatten operations
//===----------------------------------------------------------------------===//

struct FlattenOpLowering : public ConversionPattern {
  FlattenOpLowering(MLIRContext *ctx)
      : ConversionPattern(sysy::FlattenOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    sysy::FlattenOpAdaptor flattenAdaptor(operands);
    copyMemref(rewriter, loc, flattenAdaptor.getInput(), flattenAdaptor.getOutput());
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// SysyToAffine RewritePatterns: Reduction operations
//===----------------------------------------------------------------------===//

LogicalResult buildReductionLoop(Operation *op, ArrayRef<Value> operands,
                                 ConversionPatternRewriter &rewriter, Value initialValue,
                                 std::function<Value(OpBuilder&, Location, Value, Value)> reductionOp) {
  auto loc = op->getLoc();
  auto inputType = llvm::cast<RankedTensorType>(op->getOperand(0).getType());
  Value resultAddr = rewriter.create<memref::AllocaOp>(loc, MemRefType::get({}, rewriter.getF32Type()));
  moveToBlockFront(rewriter.create<memref::StoreOp>(loc, initialValue, resultAddr));
  moveToBlockFront(resultAddr.getDefiningOp());
  moveToBlockFront(initialValue.getDefiningOp());

  SmallVector<int64_t, 4> lowerBounds(inputType.getRank(), 0);
  SmallVector<int64_t, 4> steps(inputType.getRank(), 1);
  affine::buildAffineLoopNest(
    rewriter, loc, lowerBounds, inputType.getShape(), steps,
    [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs){
      auto input = operands[0];
      auto inputElement = nestedBuilder.create<affine::AffineLoadOp>(loc, input, ivs);
      auto currentValue = nestedBuilder.create<memref::LoadOp>(loc, resultAddr);
      auto newValue = reductionOp(nestedBuilder, loc, inputElement, currentValue);
      nestedBuilder.create<memref::StoreOp>(loc, newValue, resultAddr);
    });

  // auto result = rewriter.create<affine::AffineLoadOp>(loc, resultAddr, ValueRange());
  auto result = rewriter.create<memref::LoadOp>(loc, resultAddr);
  rewriter.replaceOp(op, result);

  return success();
}

LogicalResult buildVecReductionLoop(Operation *op, ArrayRef<Value> operands,
                                 ConversionPatternRewriter &rewriter, Value initialValue, vector::CombiningKind reduceType,
                                 std::function<Value(OpBuilder&, Location, Value, Value)> reductionOp) {

  //目前可以任意维度规约，但最内层循环必须是4的倍数
  auto loc = op->getLoc();
  auto inputType = llvm::cast<RankedTensorType>(op->getOperand(0).getType());

  constexpr int64_t vecSize = 4;
  auto vType = VectorType::get({vecSize}, rewriter.getF32Type());
  //verctor
  Value initVecValue = rewriter.create<vector::BroadcastOp>(loc, vType, initialValue);
  //memref
  Value resultAddr = rewriter.create<memref::AllocaOp>(loc, MemRefType::get({vecSize}, rewriter.getF32Type()));
   //memref
  Value index = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  moveToBlockFront(rewriter.create<vector::StoreOp>(loc, initVecValue, resultAddr, ValueRange{index}));
  moveToBlockFront(index.getDefiningOp());
  moveToBlockFront(resultAddr.getDefiningOp());
  moveToBlockFront(initVecValue.getDefiningOp());
  moveToBlockFront(initialValue.getDefiningOp());

  int64_t rank = inputType.getRank();
  SmallVector<int64_t, 4> lowerBounds(rank, 0);
  SmallVector<int64_t, 4> steps(rank, 1);
  // 将最内层循环的步长设置为 4
  if (rank > 0) {
    steps[rank - 1] = vecSize;
  }
  affine::buildAffineLoopNest(
    rewriter, loc, lowerBounds, inputType.getShape(), steps,
    [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs){
      auto input = operands[0];
      //vector
      auto vecElements = nestedBuilder.create<vector::LoadOp>(loc, vType, input, ValueRange(ivs.begin(), ivs.end()));
      //mem转vec
      auto currentVec = nestedBuilder.create<vector::LoadOp>(loc, vType, resultAddr, ValueRange(rewriter.create<arith::ConstantIndexOp>(loc, 0)));
      //回传vec
      auto newVec = reductionOp(nestedBuilder, loc, vecElements, currentVec);
      //vec转mem
      auto newValue = nestedBuilder.create<vector::StoreOp>(loc, newVec, resultAddr, ValueRange(rewriter.create<arith::ConstantIndexOp>(loc, 0)));
      
    });

    //生成的向量再规约
  auto resultVec = rewriter.create<vector::LoadOp>(loc, vType, resultAddr, ValueRange(rewriter.create<arith::ConstantIndexOp>(loc, 0)));
  // Value element = rewriter.create<vector::ExtractOp>(loc, resultVec, ValueRange(rewriter.create<arith::ConstantIndexOp>(loc, 3)));
  auto result =  rewriter.create<vector::ReductionOp>(loc, reduceType, resultVec);

  rewriter.replaceOp(op, result);

  return success();
}



//===----------------------------------------------------------------------===//
// SysyToAffine RewritePatterns: Max operations
//===----------------------------------------------------------------------===//

struct MaxOpLowering : public ConversionPattern {
  MaxOpLowering(MLIRContext *ctx)
      : ConversionPattern(sysy::MaxOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto initialValue = rewriter.create<arith::ConstantOp>(op->getLoc(), 
      rewriter.getFloatAttr(rewriter.getF32Type(), std::numeric_limits<float>::min()));

    // return buildReductionLoop(op, operands, rewriter, initialValue, 
    return buildVecReductionLoop(op, operands, rewriter, initialValue, vector::CombiningKind::MAXF,
      [](OpBuilder &nestedBuilder, Location loc, Value a, Value b) {
        return nestedBuilder.create<arith::MaxFOp>(loc, a, b);
      });
  }
};

//===----------------------------------------------------------------------===//
// SysyToAffine RewritePatterns: Min operations
//===----------------------------------------------------------------------===//

struct MinOpLowering : public ConversionPattern {
  MinOpLowering(MLIRContext *ctx)
      : ConversionPattern(sysy::MinOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto initialValue = rewriter.create<arith::ConstantOp>(op->getLoc(), 
        rewriter.getFloatAttr(rewriter.getF32Type(), std::numeric_limits<float>::max()));

    // return buildReductionLoop(op, operands, rewriter, initialValue, 
    return buildVecReductionLoop(op, operands, rewriter, initialValue, vector::CombiningKind::MINF,
      [](OpBuilder &nestedBuilder, Location loc, Value a, Value b) {
        return nestedBuilder.create<arith::MinFOp>(loc, a, b);
      });
  }
};

//===----------------------------------------------------------------------===//
// SysyToAffine RewritePatterns: Sum operations
//===----------------------------------------------------------------------===//

struct SumOpLowering : public ConversionPattern {
  SumOpLowering(MLIRContext *ctx)
      : ConversionPattern(sysy::SumOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto initialValue = rewriter.create<arith::ConstantOp>(op->getLoc(), 
        rewriter.getFloatAttr(rewriter.getF32Type(), 0.0f));

    return buildReductionLoop(op, operands, rewriter, initialValue,
    // return buildVecReductionLoop(op, operands, rewriter, initialValue, vector::CombiningKind::ADD, 
      [](OpBuilder &nestedBuilder, Location loc, Value a, Value b) {
        return nestedBuilder.create<arith::AddFOp>(loc, a, b);
      });
  }
};

//===----------------------------------------------------------------------===//
// SysyToAffine RewritePatterns: Print operations
//===----------------------------------------------------------------------===//

class PrintOpLowering : public OpConversionPattern<sysy::PrintOp> {
  using OpConversionPattern<sysy::PrintOp>::OpConversionPattern;
public:
  LogicalResult
  matchAndRewrite(sysy::PrintOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto memRefType = llvm::cast<MemRefType>(adaptor.getInput().getType());
    auto memRefShape = memRefType.getShape();
    auto loc = op->getLoc();

    ModuleOp parentModule = op->getParentOfType<ModuleOp>();

    // Get a symbol reference to the putfloat/putch function, inserting it if necessary.
    getOrInsertPutfloat(rewriter, parentModule);
    getOrInsertPutch(rewriter, parentModule);
    Value newLine = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI32IntegerAttr(10));
    Value blankSpace = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI32IntegerAttr(32));

    // Create a loop for each of the dimensions within the shape.
    SmallVector<Value, 4> loopIvs;
    for (unsigned i = 0, e = memRefShape.size(); i != e; ++i) {
      auto lowerBound = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      auto upperBound =
          rewriter.create<arith::ConstantIndexOp>(loc, memRefShape[i]);
      auto step = rewriter.create<arith::ConstantIndexOp>(loc, 1);
      auto loop =
          rewriter.create<scf::ForOp>(loc, lowerBound, upperBound, step);
      for (Operation &nested : *loop.getBody())
        rewriter.eraseOp(&nested);
      loopIvs.push_back(loop.getInductionVar());

      // Terminate the loop body.
      rewriter.setInsertionPointToEnd(loop.getBody());

      // Insert a newline after each of the inner dimensions of the shape.
      if (i != e - 1){
        rewriter.create<LLVM::CallOp>(loc,
                                      TypeRange({}), "putch", newLine);
      }
      rewriter.create<scf::YieldOp>(loc);
      rewriter.setInsertionPointToStart(loop.getBody());
    }

    // Generate a call to putfloat for the current element of the loop.
    auto elementLoad =
        rewriter.create<memref::LoadOp>(loc, adaptor.getInput(), loopIvs);
    LLVM::CallOp callOp = rewriter.create<LLVM::CallOp>(loc, 
                                  TypeRange({}), "putfloat", elementLoad->getResult(0));
    callOp = rewriter.create<LLVM::CallOp>(loc, 
                                  TypeRange({}), "putch", blankSpace);

    // Notify the rewriter that this operation has been removed.
    rewriter.eraseOp(op);
    return success();
  }

private:
  /// Return a symbol reference to the putfloat function, inserting it into the
  /// module if necessary.
  static FlatSymbolRefAttr getOrInsertPutfloat(PatternRewriter &rewriter,
                                             ModuleOp module) {
    auto *context = module.getContext();
    if (module.lookupSymbol<LLVM::LLVMFuncOp>("putfloat"))
      return SymbolRefAttr::get(context, "putfloat");

    // Create a function declaration for putfloat, the signature is:
    //   * `void (float)`
    
    auto llvmVoidTy = LLVM::LLVMVoidType::get(context);
    auto llvmF32Ty = Float32Type::get(context);
    auto llvmFnType = LLVM::LLVMFunctionType::get(llvmVoidTy, llvmF32Ty);

    // Insert the putfloat function into the body of the parent module.
    PatternRewriter::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    LLVM::LLVMFuncOp llvmFuncOp = rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), "putfloat", llvmFnType);
    llvmFuncOp.setPrivate();
    return SymbolRefAttr::get(context, "putfloat");
  }

  static FlatSymbolRefAttr getOrInsertPutch(PatternRewriter &rewriter,
                                             ModuleOp module) {
    auto *context = module.getContext();
    if (module.lookupSymbol<LLVM::LLVMFuncOp>("putch"))
      return SymbolRefAttr::get(context, "putch");

    // Create a function declaration for putch, the signature is:
    //   * `void (i32)`
    auto llvmVoidTy = LLVM::LLVMVoidType::get(context);
    auto llvmI32Ty = IntegerType::get(context, 32);
    auto llvmFnType = LLVM::LLVMFunctionType::get(llvmVoidTy, llvmI32Ty);

    // Insert the putch function into the body of the parent module.
    PatternRewriter::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    LLVM::LLVMFuncOp llvmFuncOp = rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), "putch", llvmFnType);
    llvmFuncOp.setPrivate();
    return SymbolRefAttr::get(context, "putch");
  }
};

//===----------------------------------------------------------------------===//
// SysyToAffine RewritePatterns: Scan operations
//===----------------------------------------------------------------------===//

class ScanOpLowering : public OpConversionPattern<sysy::ScanOp> {
  using OpConversionPattern<sysy::ScanOp>::OpConversionPattern;
public:
  LogicalResult
  matchAndRewrite(sysy::ScanOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto memRefType = llvm::cast<MemRefType>(adaptor.getInput().getType());
    auto memRefShape = memRefType.getShape();
    auto loc = op->getLoc();

    ModuleOp parentModule = op->getParentOfType<ModuleOp>();

    // Get a symbol reference to the getfloat function, inserting it if necessary.
    getOrInsertGetfloat(rewriter, parentModule);
    // Create a loop for each of the dimensions within the shape.
    SmallVector<Value, 4> loopIvs;
    for (unsigned i = 0, e = memRefShape.size(); i != e; ++i) {
      auto lowerBound = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      auto upperBound = rewriter.create<arith::ConstantIndexOp>(loc, memRefShape[i]);
      auto step = rewriter.create<arith::ConstantIndexOp>(loc, 1);
      auto loop = rewriter.create<scf::ForOp>(loc, lowerBound, upperBound, step);
      for (Operation &nested : *loop.getBody())
        rewriter.eraseOp(&nested);
      loopIvs.push_back(loop.getInductionVar());

      // Terminate the loop body.
      rewriter.setInsertionPointToEnd(loop.getBody());
      rewriter.create<scf::YieldOp>(loc);
      rewriter.setInsertionPointToStart(loop.getBody());
    }

    // Generate a call to putfloat for the current element of the loop.
    LLVM::CallOp callOp = rewriter.create<LLVM::CallOp>(loc, TypeRange({rewriter.getF32Type()}), "getfloat");
    rewriter.create<memref::StoreOp>(loc, callOp.getResult(), adaptor.getInput(), loopIvs);

    // Notify the rewriter that this operation has been removed.
    rewriter.eraseOp(op);
    return success();
  }
private:
  static FlatSymbolRefAttr getOrInsertGetfloat(PatternRewriter &rewriter,
                                             ModuleOp module) {
    auto *context = module.getContext();
    if (module.lookupSymbol<LLVM::LLVMFuncOp>("getfloat"))
      return SymbolRefAttr::get(context, "getfloat");

    // Create a function declaration for getfloat, the signature is:
    //   * `float ()`
    // auto llvmVoidTy = LLVM::LLVMVoidType::get(context);
    auto llvmF32Ty = Float32Type::get(context);
    auto llvmFnType = LLVM::LLVMFunctionType::get(llvmF32Ty, {});

    // Insert the getfloat function into the body of the parent module.
    PatternRewriter::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    LLVM::LLVMFuncOp llvmFuncOp = rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), "getfloat", llvmFnType);
    llvmFuncOp.setPrivate();
    return SymbolRefAttr::get(context, "getfloat");
  }
};

//===----------------------------------------------------------------------===//
// SysyToAffine RewritePatterns: Clear operations
//===----------------------------------------------------------------------===//

struct ClearOpLowering : public OpConversionPattern<sysy::ClearOp> {
  using OpConversionPattern<sysy::ClearOp>::OpConversionPattern;
public:
  LogicalResult
  matchAndRewrite(sysy::ClearOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    fillZero(rewriter, loc, adaptor.getInput());
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// LowerSysyPass
//===----------------------------------------------------------------------===//

/// This is a partial lowering to affine loops of the sysy operations that are
/// computationally intensive (like matmul for example...) while keeping the
/// rest of the code in the Sysy dialect.
namespace {
struct LowerSysyPass
    : public PassWrapper<LowerSysyPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerSysyPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, func::FuncDialect,
                    memref::MemRefDialect>();
  }
  void runOnOperation() final;
};
} // namespace

void LowerSysyPass::runOnOperation() {
  // The first thing to define is the conversion target. This will define the
  // final target for this lowering.
  ConversionTarget target(getContext());

  // We define the specific operations, or dialects, that are legal targets for
  // this lowering. In our case, we are lowering to a combination of the
  // `Affine`, `Arith`, `Func`, and `MemRef` dialects.
  target.addLegalDialect<affine::AffineDialect, BuiltinDialect,
                         arith::ArithDialect, func::FuncDialect,
                         memref::MemRefDialect, LLVM::LLVMDialect,
                         vector::VectorDialect, scf::SCFDialect>();

  // We also define the Sysy dialect as Illegal so that the conversion will fail
  // if any of these operations are *not* converted.
  target.addIllegalDialect<sysy::SysyDialect>();

  // Now that the conversion target has been defined, we just need to provide
  // the set of patterns that will lower the Sysy operations.
  RewritePatternSet patterns(&getContext());
  patterns.add<InitTensorOpLowering, DefTensorOpLowering, BroadcastOpLowering,
               TransposeOpLowering, CopyOpLowering, ReLUOpLowering, FlattenOpLowering,
              //  MulOpLowering, AddOpLowering, SubOpLowering, DivOpLowering,
               MulOpVecLowering, AddOpVecLowering, SubOpVecLowering, DivOpVecLowering,
              //  MatMulOpLowering,
               MatMulOpVecLowering,
              //  MatMulOpVec2Lowering,
               ValidConvOpLowering, SameConvOpLowering,
               MaxOpLowering, MinOpLowering, SumOpLowering, MaxPoolOpLowering,
               PrintOpLowering, ScanOpLowering, ClearOpLowering>(
      &getContext());

  // With the target and rewrite patterns defined, we can now attempt the
  // conversion. The conversion will signal failure if any of our `illegal`
  // operations were not converted successfully.
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<Pass> mlir::sysy::createLowerSysyPass() {
  return std::make_unique<LowerSysyPass>();
}