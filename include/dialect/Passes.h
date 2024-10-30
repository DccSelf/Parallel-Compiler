#ifndef SYSY_PASSES_H
#define SYSY_PASSES_H

#include <memory>

namespace mlir {
class Pass;

namespace sysy {
std::unique_ptr<Pass> createShapeInferencePass();

std::unique_ptr<Pass> createLowerSysyPass();

std::unique_ptr<Pass> createLowerToLLVMPass();

std::unique_ptr<Pass> createSysyOptPass();

std::unique_ptr<Pass> createSimplifyRedundantTransposePass();

std::unique_ptr<Pass> createDCEPass();
} // namespace sysy
} // namespace mlir

#endif // SYSY_PASSES_H