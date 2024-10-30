#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "mlir/Pass/Pass.h"
#include <vector>

#include "dialect/Passes.h"

namespace mlir {
namespace sysy {

using namespace mlir::affine;  

class SysyOptPass
    : public PassWrapper<SysyOptPass, OperationPass<mlir::func::FuncOp>> {
private:
  void runOnOperation() override;

  bool tileLoop(SmallVector<AffineForOp, 6>& band, 
                SmallVector<unsigned, 6>& tileSizes,
                SmallVector<AffineForOp, 6>& tiledNest);
  bool parallelizeLoop(AffineForOp forOp);
};

void SysyOptPass::runOnOperation() {
  auto funcOp = getOperation();
  funcOp.walk([&](Operation *op) {
    if (auto root = dyn_cast<AffineForOp>(op)) {
      SmallVector<AffineForOp, 6> nestedLoops;
      getPerfectlyNestedLoops(nestedLoops, root);

      if (auto tileSizesAttr = root->getAttrOfType<ArrayAttr>("tileSizes")){
        SmallVector<unsigned, 6> tileSizes;
        for (auto tileSize : tileSizesAttr)
          tileSizes.push_back(tileSize.dyn_cast<IntegerAttr>().getInt());
        SmallVector<AffineForOp, 6> tiledNest;
        if (tileLoop(nestedLoops, tileSizes, tiledNest)) {
          nestedLoops = tiledNest;
        }
      }

      if (auto interchangeAttr = root->getAttrOfType<ArrayAttr>("interchange")){
        SmallVector<unsigned, 2> interchange;
        for (auto tileSize : interchangeAttr)
          interchange.push_back(tileSize.dyn_cast<IntegerAttr>().getInt());
        AffineForOp a, b;
        for(unsigned i = 0; i < nestedLoops.size(); i++){
          if(std::find(interchange.begin(), interchange.end(), i) != interchange.end()){
            if(!a) a = nestedLoops[i];
            else b = nestedLoops[i];
          }
        }
        interchangeLoops(a, b);
      }

      if (root->hasAttr("parallel")){
        parallelizeLoop(nestedLoops[0]);
      }
    }

    return WalkResult::advance();
  });
}

bool SysyOptPass::tileLoop(SmallVector<AffineForOp, 6>& band, 
                           SmallVector<unsigned, 6>& tileSizes,
                           SmallVector<AffineForOp, 6>& tiledNest) {
  if (failed(tilePerfectlyNested(band, tileSizes, &tiledNest))) {
    band.front()->emitRemark("loop tiling failed\n");
    return false;
  }
  return true;
}

bool SysyOptPass::parallelizeLoop(AffineForOp forOp) {
  if (failed(affineParallelize(forOp))) {
    forOp.emitRemark("loop parallelization failed\n");
    return false;
  }
  return true;
}

std::unique_ptr<Pass> createSysyOptPass() {
  return std::make_unique<SysyOptPass>();
}

} // namespace sysy
} // namespace mlir