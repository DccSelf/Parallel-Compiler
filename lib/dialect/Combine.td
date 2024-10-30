#ifndef SYSY_COMBINE
#define SYSY_COMBINE

include "mlir/IR/PatternBase.td"
include "dialect/Ops.td"

// def SimplifyRedundantTranspose : Pat<(TransposeOp (TransposeOp $arg)), 
//     (replaceWithValue $arg)>;

def TypesAreIdentical : Constraint<CPred<"$0.getType() == $1.getType()">>;
def RedundantCastOptPattern : Pat<
    (CastOp:$res $arg), (replaceWithValue $arg),
    [(TypesAreIdentical $res, $arg)]>;

#endif // SYSY_COMBINE