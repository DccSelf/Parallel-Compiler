#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/IRBuilder.h"

#include "mlir/IR/Block.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "frontend/SysYBaseVisitor.h"
#include "frontend/SymbolTable.h"

struct LibFunc {
    std::string name;
    mlir::Type returnType;
    std::vector<mlir::Type> paramTypes;
};

class SysYVisitorImpl : public SysYBaseVisitor {
    mlir::MLIRContext& context;
    mlir::OpBuilder builder;
    mlir::ModuleOp module;

    FunctionTable *funcTable;
    GlobalVarTable *globalVarTable;
    LocalVarTable *localVarTable;

    mlir::Type basicType;
    bool tensorDef;
    bool tensorFuncDef;
    mlir::Value broadcastTensor;

    mlir::FunctionOpInterface curFunc;
    mlir::Block* entryBlock;
    mlir::Block::iterator allocaInsertPoint;
    mlir::Block* exitBlock;
    std::stack<mlir::Block*> breakTarget, continueTarget;

    llvm::LLVMContext llvmContext;
    llvm::IRBuilder<> llvmBuilder;
    llvm::Type* llvmI1Type;
    llvm::Type* llvmI32Type;
    llvm::Type* llvmFloatType;

    // 库函数
    std::vector<LibFunc> libFuncs;

public:
    SysYVisitorImpl(mlir::MLIRContext& context);

    mlir::ModuleOp getModule();

    virtual antlrcpp::Any visitChildren(antlr4::tree::ParseTree *ctx) override;

    mlir::FunctionOpInterface registerFunction(std::string funcName, mlir::Type retType, std::vector<mlir::Type> paramTypes, bool isLibFunc);

    void createEntryBlock(std::vector<std::pair<mlir::Type, std::string>>& params);

    void createExitBlock(mlir::Type retType);

    void connectBlockToExit();
    
    mlir::FunctionOpInterface resolveFunction(std::string funcName);

    bool resolveCoreTensorFunction(std::string funcName, std::vector<antlrcpp::Any> args, mlir::Value& ret);

    mlir::Value getOrCreateValue(const antlrcpp::Any& any);
    
    antlrcpp::Any convertType(const antlrcpp::Any& any, mlir::Type type);

    mlir::Type getMLIRType(const antlrcpp::Any& any);

    mlir::Type unifyArithRelTypes(antlrcpp::Any& lhs, antlrcpp::Any& rhs);

    std::vector<int64_t> getArrayDims(std::vector<SysYParser::ConstExpContext *> ctx);

    llvm::Constant* getZeroConst(mlir::Type type);

    void fillZeros(std::vector<antlrcpp::Any>& ret, int64_t count, int64_t size);

    antlrcpp::Any getScalarValue(SysYParser::ScalarConstInitValContext* node);

    antlrcpp::Any getScalarValue(SysYParser::ScalarInitValContext* node);

    template<typename ListInitValContext, typename ScalarInitValContext>
    void getListInitVal(ListInitValContext* node, std::vector<int64_t> arraySize, std::vector<antlrcpp::Any>& ret);

    template<typename InitValContext, typename ScalarInitValContext, typename ListInitValContext>
    std::vector<antlrcpp::Any> getInitVal(InitValContext* node, std::vector<int64_t> dims);

    mlir::Type getRealType(std::vector<int64_t> dims);

    void initLocalArrayAssign(mlir::Value var, int dimCount, const std::vector<antlrcpp::Any>& initVal);

    void initLocalArrayMemset(mlir::Value var, int64_t elementCount);

    mlir::Value resolveVariable(const std::string& varName);

    mlir::Value computeArrayElementAddress(mlir::Value ptr, const std::vector<SysYParser::ExpContext*>& indices);

    mlir::LLVM::AllocaOp createAllocaOp(mlir::Type type);

    mlir::LLVM::GEPOp createGEPOp(mlir::Value ptr, mlir::ValueRange indices);

    mlir::Attribute getInitialValue(const std::vector<int64_t>& dims, std::vector<antlrcpp::Any>& initVal);

    void processVarDef(const std::string& varName, const std::vector<int64_t>& dims, 
        mlir::Type type, std::vector<antlrcpp::Any>& initVal, bool isConst);

    virtual antlrcpp::Any visitCompUnit(SysYParser::CompUnitContext *ctx) override;

    virtual antlrcpp::Any visitDecl(SysYParser::DeclContext *ctx) override;

    virtual antlrcpp::Any visitConstDecl(SysYParser::ConstDeclContext *ctx) override;

    virtual antlrcpp::Any visitBType(SysYParser::BTypeContext *ctx) override;

    virtual antlrcpp::Any visitConstDef(SysYParser::ConstDefContext *ctx) override;

    virtual antlrcpp::Any visitScalarConstInitVal(SysYParser::ScalarConstInitValContext *ctx) override;

    virtual antlrcpp::Any visitListConstInitVal(SysYParser::ListConstInitValContext *ctx) override;

    virtual antlrcpp::Any visitVarDecl(SysYParser::VarDeclContext *ctx) override;

    virtual antlrcpp::Any visitUninitVarDef(SysYParser::UninitVarDefContext *ctx) override;

    virtual antlrcpp::Any visitInitVarDef(SysYParser::InitVarDefContext *ctx) override;

    virtual antlrcpp::Any visitScalarInitVal(SysYParser::ScalarInitValContext *ctx) override;

    virtual antlrcpp::Any visitListInitVal(SysYParser::ListInitValContext *ctx) override;

    virtual antlrcpp::Any visitFuncDef(SysYParser::FuncDefContext *ctx) override;

    virtual antlrcpp::Any visitFuncType(SysYParser::FuncTypeContext *ctx) override;

    virtual antlrcpp::Any visitFuncFParams(SysYParser::FuncFParamsContext *ctx) override;

    virtual antlrcpp::Any visitFuncFParam(SysYParser::FuncFParamContext *ctx) override;

    virtual antlrcpp::Any visitBlock(SysYParser::BlockContext *ctx) override;

    virtual antlrcpp::Any visitBlockItem(SysYParser::BlockItemContext *ctx) override;

    virtual antlrcpp::Any visitAssignment(SysYParser::AssignmentContext *ctx) override;

    virtual antlrcpp::Any visitExpStmt(SysYParser::ExpStmtContext *ctx) override;

    virtual antlrcpp::Any visitBlockStmt(SysYParser::BlockStmtContext *ctx) override;

    virtual antlrcpp::Any visitIfStmt1(SysYParser::IfStmt1Context *ctx) override;

    virtual antlrcpp::Any visitIfStmt2(SysYParser::IfStmt2Context *ctx) override;

    virtual antlrcpp::Any visitWhileStmt(SysYParser::WhileStmtContext *ctx) override;

    virtual antlrcpp::Any visitBreakStmt(SysYParser::BreakStmtContext *ctx) override;

    virtual antlrcpp::Any visitContinueStmt(SysYParser::ContinueStmtContext *ctx) override;

    virtual antlrcpp::Any visitReturnStmt(SysYParser::ReturnStmtContext *ctx) override;

    virtual antlrcpp::Any visitExp(SysYParser::ExpContext *ctx) override;

    virtual antlrcpp::Any visitCond(SysYParser::CondContext *ctx) override;

    virtual antlrcpp::Any visitLVal(SysYParser::LValContext *ctx) override;

    virtual antlrcpp::Any visitPrimaryExp1(SysYParser::PrimaryExp1Context *ctx) override;

    virtual antlrcpp::Any visitPrimaryExp2(SysYParser::PrimaryExp2Context *ctx) override;

    virtual antlrcpp::Any visitPrimaryExp3(SysYParser::PrimaryExp3Context *ctx) override;

    virtual antlrcpp::Any visitNumber(SysYParser::NumberContext *ctx) override;

    virtual antlrcpp::Any visitUnary1(SysYParser::Unary1Context *ctx) override;

    virtual antlrcpp::Any visitUnary2(SysYParser::Unary2Context *ctx) override;

    virtual antlrcpp::Any visitUnary3(SysYParser::Unary3Context *ctx) override;

    virtual antlrcpp::Any visitUnaryOp(SysYParser::UnaryOpContext *ctx) override;

    virtual antlrcpp::Any visitFuncRParams(SysYParser::FuncRParamsContext *ctx) override;

    virtual antlrcpp::Any visitExpAsRParam(SysYParser::ExpAsRParamContext *ctx) override;

    virtual antlrcpp::Any visitStringAsRParam(SysYParser::StringAsRParamContext *ctx) override;

    virtual antlrcpp::Any visitMul2(SysYParser::Mul2Context *ctx) override;

    virtual antlrcpp::Any visitMul1(SysYParser::Mul1Context *ctx) override;

    virtual antlrcpp::Any visitAdd2(SysYParser::Add2Context *ctx) override;

    virtual antlrcpp::Any visitAdd1(SysYParser::Add1Context *ctx) override;

    virtual antlrcpp::Any visitRel2(SysYParser::Rel2Context *ctx) override;

    virtual antlrcpp::Any visitRel1(SysYParser::Rel1Context *ctx) override;

    virtual antlrcpp::Any visitEq1(SysYParser::Eq1Context *ctx) override;

    virtual antlrcpp::Any visitEq2(SysYParser::Eq2Context *ctx) override;

    virtual antlrcpp::Any visitLAnd2(SysYParser::LAnd2Context *ctx) override;

    virtual antlrcpp::Any visitLAnd1(SysYParser::LAnd1Context *ctx) override;

    virtual antlrcpp::Any visitLOr1(SysYParser::LOr1Context *ctx) override;

    virtual antlrcpp::Any visitLOr2(SysYParser::LOr2Context *ctx) override;

    virtual antlrcpp::Any visitConstExp(SysYParser::ConstExpContext *ctx) override;
};