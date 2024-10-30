#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "dialect/Dialect.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/Casting.h"
#include <numeric>
using namespace mlir;
using namespace sysy;

namespace {
/// Include the patterns defined in the Declarative Rewrite framework.
#include "dialect/Combine.inc"
} // namespace

class RemoveRedundantCopy : public mlir::OpRewritePattern<sysy::CopyOp> {
public:
  RemoveRedundantCopy(mlir::MLIRContext *context)
      : OpRewritePattern<sysy::CopyOp>(context, /*benefit=*/1) {}

  mlir::LogicalResult matchAndRewrite(sysy::CopyOp copyOp,
                                      mlir::PatternRewriter &rewriter) const override {
    Value writeValue = copyOp.getOutput();
    Value readValue = copyOp.getInput();
    if(readValue == writeValue){
      rewriter.eraseOp(copyOp);
      return success();
    }

    // 查找唯一写入readValue的Op
    Operation* writeOp = nullptr;
    for (Operation *user : readValue.getUsers()) {
      if(user != copyOp){
        if (writeOp) // 只有两次使用
          return failure();
        auto userOp = dyn_cast<sysy::EffectInterface>(user);
        if (userOp.getWriteValue() != readValue) // 除copy读之外只有一次写
          return failure();
        else
          writeOp = user;
      }
    }

    if (writeOp){
      rewriter.updateRootInPlace(writeOp, [&]() {
        writeOp->setOperand(0, copyOp.getOperand(0));
      });
    }
    rewriter.eraseOp(copyOp);

    return success();
  }
};

// void TransposeOp::getCanonicalizationPatterns(RewritePatternSet &results,
//                                               MLIRContext *context) {
//   results.add<SimplifyRedundantTranspose>(context);
// }

void CastOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  results.add<RedundantCastOptPattern>(context);
}

void CopyOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  results.add<RemoveRedundantCopy>(context);
}