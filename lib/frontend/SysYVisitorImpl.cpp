#include <algorithm>
#include <cassert>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>
#include <numeric>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

#include "frontend/SysYVisitorImpl.h"
#include "frontend/SymbolTable.h"
#include "dialect/Dialect.h"


using namespace std;
using namespace mlir;


#define LOC builder.getUnknownLoc()

SysYVisitorImpl::SysYVisitorImpl(mlir::MLIRContext& context) 
    : context(context), builder(&context),
    module(ModuleOp::create(LOC)),
    llvmBuilder(llvmContext),
    llvmI1Type(llvm::Type::getInt1Ty(llvmContext)),
    llvmI32Type(llvm::Type::getInt32Ty(llvmContext)),
    llvmFloatType(llvm::Type::getFloatTy(llvmContext)){
        builder.setInsertionPointToEnd(module.getBody());
        // 库函数类型
        mlir::Type voidTy = builder.getNoneType();
        mlir::Type int32Ty = builder.getI32Type();
        mlir::Type floatTy = builder.getF32Type();
        mlir::Type int32PtrTy = LLVM::LLVMPointerType::get(int32Ty);
        mlir::Type floatPtrTy = LLVM::LLVMPointerType::get(floatTy);
        libFuncs = {
            {"getint", int32Ty, {}},
            {"getch", int32Ty, {}},
            {"getfloat", floatTy, {}},
            {"getarray", int32Ty, {int32PtrTy}},
            {"getfarray", int32Ty, {floatPtrTy}},
            {"putint", voidTy, {int32Ty}},
            {"putch", voidTy, {int32Ty}},
            {"putfloat", voidTy, {floatTy}},
            {"putarray", voidTy, {int32Ty, int32PtrTy}},
            {"putfarray", voidTy, {int32Ty, floatPtrTy}},
            {"_sysy_starttime", voidTy, {int32Ty}},
            {"_sysy_stoptime", voidTy, {int32Ty}}
        };
    }

ModuleOp SysYVisitorImpl::getModule(){
    return module;
}

antlrcpp::Any SysYVisitorImpl::visitChildren(antlr4::tree::ParseTree *ctx) {
    size_t n = ctx->children.size();
    for (size_t i = 0; i < n; ++i) ctx->children[i]->accept(this);
    return nullptr;
}

mlir::FunctionOpInterface SysYVisitorImpl::registerFunction(string funcName, Type retType, vector<Type> paramTypes, bool isLibFunc = false){
    vector<Type> retTypes;
    if(!retType.isa<NoneType>()){
        retTypes.push_back(retType);
    }
    FunctionType funcType = builder.getFunctionType(paramTypes, retTypes);
    // 根据参数和返回值类型，判断是否是张量函数
    mlir::FunctionOpInterface func;
    tensorFuncDef = false;
    // if(retType.isa<TensorType>()) tensorFuncDef = true;
    for(auto type : paramTypes)
        if(type.isa<TensorType>()) tensorFuncDef = true;
    if(tensorFuncDef){
        func = builder.create<sysy::FunctionOp>(LOC, funcName, funcType);
        func.setPrivate();
    } else {
        func = builder.create<func::FuncOp>(LOC, funcName, funcType);
        if (isLibFunc)
            func.setPrivate();
    }
    funcTable->registerFunc(funcName, func);
    return func;
}

void SysYVisitorImpl::createEntryBlock(vector<pair<mlir::Type, string>>& params){
    // 入口基本块
    entryBlock = curFunc.addEntryBlock();
    builder.setInsertionPointToEnd(entryBlock);
    allocaInsertPoint = entryBlock->begin();
    // 注册参数
    int64_t index = 0;
    for (auto param: params) {
        mlir::BlockArgument arg = entryBlock->getArgument(index++);
        mlir::Type argType = arg.getType();
        if(argType.isa<TensorType>()){
            localVarTable->registerVar(param.second, arg);
        } else {
            mlir::Value ptr = createAllocaOp(argType);
            builder.create<LLVM::StoreOp>(LOC, arg, ptr);
            localVarTable->registerVar(param.second, ptr);
        }
    }
}

void SysYVisitorImpl::createExitBlock(mlir::Type retType){
    // 创建出口基本块
    exitBlock = curFunc.addBlock();
    vector<mlir::Value> retVal;
    if(retType != builder.getNoneType()){
        exitBlock->addArgument(retType, LOC);
        retVal.push_back(exitBlock->getArgument(0));
    }
    builder.setInsertionPointToEnd(exitBlock);
    if(tensorFuncDef){
        builder.create<sysy::RetOp>(LOC, retVal);
    } else {
        builder.create<func::ReturnOp>(LOC, retVal);
    }
    builder.setInsertionPointToEnd(entryBlock);
}

void SysYVisitorImpl::connectBlockToExit(){
    // 检查当前基本块是否缺少终止操作
    mlir::Block* curBlock = builder.getInsertionBlock();
    if(curBlock->empty() || !curBlock->back().hasTrait<mlir::OpTrait::IsTerminator>()){
        mlir::FunctionType funcType = llvm::dyn_cast<mlir::FunctionType>(curFunc.getFunctionType());
        if(funcType.getResults().empty()){
            builder.create<cf::BranchOp>(LOC, exitBlock);
        } else {
            Value zero = getOrCreateValue(getZeroConst(funcType.getResult(0)));
            builder.create<cf::BranchOp>(LOC, exitBlock, zero);
        }
    }
}

mlir::FunctionOpInterface SysYVisitorImpl::resolveFunction(string funcName) {
    // 首先在符号表中查找
    mlir::FunctionOpInterface func = funcTable->resolve(funcName);
    if(func) return func;
    // 在库函数列表中查找
    auto it = std::find_if(libFuncs.begin(), libFuncs.end(),
                           [&funcName](const LibFunc& lf) { return lf.name == funcName; });
    if (it != libFuncs.end()) {
        // 找到了库函数，注册并返回
        mlir::OpBuilder::InsertionGuard insertGuard(builder);
        builder.setInsertionPointToStart(module.getBody());
        return registerFunction(it->name, it->returnType, it->paramTypes, true);
    }
    // 如果既不在符号表中，也不是库函数，则报错
    throw std::runtime_error("Use of undeclared identifier " + funcName);
}

bool SysYVisitorImpl::resolveCoreTensorFunction(string funcName, vector<antlrcpp::Any> args, mlir::Value& ret){
    if(funcName == "gettensor" || funcName == "puttensor" || funcName == "__clear"
        || funcName == "__max" || funcName == "__min" || funcName == "__sum"){
        assert(args.size() == 1 && args[0].as<mlir::Value>().getType().isa<TensorType>());
        if(funcName == "gettensor") 
            builder.create<sysy::ScanOp>(LOC, args[0].as<mlir::Value>());
        else if(funcName == "puttensor")
            builder.create<sysy::PrintOp>(LOC, args[0].as<mlir::Value>());
        else if(funcName == "__clear")
            builder.create<sysy::ClearOp>(LOC, args[0].as<mlir::Value>());
        else if(funcName == "__max")
            ret = builder.create<sysy::MaxOp>(LOC, args[0].as<mlir::Value>());
        else if(funcName == "__min")
            ret = builder.create<sysy::MinOp>(LOC, args[0].as<mlir::Value>());
        else if(funcName == "__sum")
            ret = builder.create<sysy::SumOp>(LOC, args[0].as<mlir::Value>());
        return true;
    } else if(funcName == "__transpose" || funcName == "__copy" || funcName == "__relu"
            ||funcName == "__flatten"){
        assert(args.size() == 2);
        mlir::Value output = args[0].as<mlir::Value>();
        mlir::Value input = args[1].as<mlir::Value>();
        assert(output.getType().isa<TensorType>()
                && input.getType().isa<TensorType>());
        if(funcName == "__transpose")
            builder.create<sysy::TransposeOp>(LOC, output, input);
        else if (funcName == "__copy")
            builder.create<sysy::CopyOp>(LOC, output, input);
        else if (funcName == "__relu")
            builder.create<sysy::ReLUOp>(LOC, output, input);
        else if (funcName == "__flatten")
            builder.create<sysy::FlattenOp>(LOC, output, input);
        return true;
    } else if(funcName == "__add" || funcName == "__sub" || funcName == "__mul" || funcName == "__div"
           || funcName == "__matmul" || funcName == "__conv_valid" || funcName == "__conv_same"
           || funcName == "__pool"){
        assert(args.size() == 3);
        mlir::Value dst = args[0].as<mlir::Value>();
        mlir::Value lhs = args[1].as<mlir::Value>();
        mlir::Value rhs = args[2].as<mlir::Value>();
        assert(dst.getType().isa<TensorType>()
                && rhs.getType().isa<TensorType>()
                && lhs.getType().isa<TensorType>());
        if(funcName == "__add")
            builder.create<sysy::AddOp>(LOC, dst, lhs, rhs);
        else if(funcName == "__sub")
            builder.create<sysy::SubOp>(LOC, dst, lhs, rhs);
        else if(funcName == "__mul")
            builder.create<sysy::MulOp>(LOC, dst, lhs, rhs);
        else if(funcName == "__div")
            builder.create<sysy::DivOp>(LOC, dst, lhs, rhs);
        else if(funcName == "__matmul")
            builder.create<sysy::MatMulOp>(LOC, dst, lhs, rhs);
        else if(funcName == "__conv_valid")
            builder.create<sysy::ValidConvOp>(LOC, dst, lhs, rhs);
        else if(funcName == "__conv_same")
            builder.create<sysy::SameConvOp>(LOC, dst, lhs, rhs);
        else if(funcName == "__pool")
            builder.create<sysy::MaxPoolOp>(LOC,dst,lhs,rhs);
        return true;
    }
    return false;
}

int s2i(const std::string& str){
    if(str[0] == '0'){
        if(str.length() > 1 && str[1] == 'x') return stoi(str, 0, 16);
        return stoi(str, 0, 8);
    }
    return stoi(str);
}

unsigned countNonZeros(const vector<antlrcpp::Any>& initVal){
    unsigned count = 0;
    for(auto value : initVal){
        if(value.is<llvm::Constant*>() && value.as<llvm::Constant*>()->isZeroValue())
            continue;
        count++;
    }
    return count;
}

int64_t getElementCount(const vector<int64_t>& dims){
    return std::accumulate(dims.begin(), dims.end(), 
                           static_cast<int64_t>(1), 
                           std::multiplies<int64_t>());
}

mlir::Value SysYVisitorImpl::getOrCreateValue(const antlrcpp::Any& any) {
    if (any.is<mlir::Value>()) {
        return any.as<mlir::Value>();
    } else {
        auto value = any.as<llvm::Constant*>();
        if (auto valueI32 = llvm::dyn_cast<llvm::ConstantInt>(value)) {
            return builder.create<LLVM::ConstantOp>(LOC, builder.getI32Type(), valueI32->getValue());
        } else if (auto valueF32 = llvm::dyn_cast<llvm::ConstantFP>(value)) {
            return builder.create<LLVM::ConstantOp>(LOC, builder.getF32Type(), valueF32->getValue());
        } else {
            throw std::runtime_error("Unsupported constant type");
        }
    }
}

antlrcpp::Any SysYVisitorImpl::convertType(const antlrcpp::Any& any, mlir::Type targetType){
    if(any.is<llvm::Constant*>()){
        llvm::Constant* constVal = any.as<llvm::Constant*>();
        llvm::Type* sourceType = constVal->getType();
        llvm::Constant* ret = nullptr;
        if(targetType.isInteger(1)){
            if(sourceType->isIntegerTy(1)){
                ret = constVal;
            } else if (llvm::ConstantInt* constInt = llvm::dyn_cast<llvm::ConstantInt>(constVal)){
                ret = llvm::ConstantInt::get(llvmI1Type, constInt->getSExtValue() != 0);
            } else if (llvm::ConstantFP* constFP = llvm::dyn_cast<llvm::ConstantFP>(constVal)) {
                ret = llvm::ConstantInt::get(llvmI1Type, constFP->getValueAPF().convertToDouble() != 0.0);
            } else {
                throw std::runtime_error("Unsupported sourceType");
            }
        } else if (targetType.isInteger(32)) {
            if(sourceType->isIntegerTy(32)){
                ret = constVal;
            } else if (llvm::ConstantInt* constInt = llvm::dyn_cast<llvm::ConstantInt>(constVal)) {
                ret = llvm::ConstantInt::get(llvmI32Type, constInt->getZExtValue());
            } else if (llvm::ConstantFP* constFP = llvm::dyn_cast<llvm::ConstantFP>(constVal)) {
                ret = llvm::ConstantInt::get(llvmI32Type, static_cast<int32_t>(constFP->getValueAPF().convertToDouble()));
            } else {
                throw std::runtime_error("Unsupported sourceType");
            }
        } else if (targetType.isF32()) {
            if(sourceType->isFloatTy()){
                ret = constVal;
            } else if (llvm::ConstantInt* constInt = llvm::dyn_cast<llvm::ConstantInt>(constVal)) {
                if (sourceType->isIntegerTy(1)) {
                    ret = llvm::ConstantFP::get(llvmFloatType, constInt->getZExtValue());
                } else if (sourceType->isIntegerTy(32)) {
                    APInt intValue = constInt->getValue();
                    APFloat floatValue(APFloat::IEEEsingle());
                    floatValue.convertFromAPInt(intValue, true, APFloat::rmNearestTiesToEven);
                    ret = llvm::ConstantFP::get(llvmFloatType, floatValue);
                } else {
                    throw std::runtime_error("Unsupported sourceType");
                }
            } else {
                throw std::runtime_error("Unsupported sourceType");
            }
        } else if(targetType.isa<TensorType>()) {
            Value source = getOrCreateValue(any);
            return convertType(source, targetType);
        } else {
            throw std::runtime_error("Unsupported targetType");
        }
        return ret;
    } else {
        mlir::Value value = any.as<mlir::Value>();
        mlir::Type sourceType = value.getType();
        if(sourceType == targetType)
            return value;

        mlir::Value ret = mlir::Value();
        if(targetType.isInteger(1)){
            if (sourceType.isInteger(32)) {
                mlir::Value zero = builder.create<LLVM::ConstantOp>(LOC, builder.getI32IntegerAttr(0));
                ret = builder.create<mlir::LLVM::ICmpOp>(LOC, mlir::LLVM::ICmpPredicate::ne, value, zero);
            } else if (sourceType.isF32()) {
                mlir::Value zero = builder.create<LLVM::ConstantOp>(LOC, builder.getF32FloatAttr(0.0));
                ret = builder.create<mlir::LLVM::FCmpOp>(LOC, mlir::LLVM::FCmpPredicate::une, value, zero);
            } else {
                throw std::runtime_error("Unsupported sourceType");
            }
        } else if (targetType.isInteger(32)) {
            if (sourceType.isInteger(1)){
                ret = builder.create<mlir::LLVM::ZExtOp>(LOC, builder.getI32Type(), value);
            } else if (sourceType.isF32()) {
                ret = builder.create<mlir::LLVM::FPToSIOp>(LOC, builder.getI32Type(), value);
            } else {
                throw std::runtime_error("Unsupported sourceType");
            }
        } else if (targetType.isF32()) {
            if (sourceType.isInteger(1)) {
                value = builder.create<mlir::LLVM::ZExtOp>(LOC, builder.getI32Type(), value);
                ret = builder.create<mlir::LLVM::SIToFPOp>(LOC, builder.getF32Type(), value);
            } else if (sourceType.isInteger(32)) {
                ret = builder.create<mlir::LLVM::SIToFPOp>(LOC, builder.getF32Type(), value);
            } else {
                throw std::runtime_error("Unsupported sourceType");
            }
        } else if(targetType.isa<TensorType>()){
            if(sourceType.isa<TensorType>())
                ret = builder.create<mlir::sysy::CastOp>(LOC, targetType, value);
            else if(sourceType.isInteger(1)){
                value = builder.create<mlir::LLVM::ZExtOp>(LOC, builder.getI32Type(), value);
                value = builder.create<mlir::LLVM::SIToFPOp>(LOC, builder.getF32Type(), value);
                ret = builder.create<sysy::DeclTensorOp>(LOC);
                builder.create<sysy::BroadcastOp>(LOC, ret, broadcastTensor, value);
            } else if(sourceType.isInteger(32)){
                value = builder.create<mlir::LLVM::SIToFPOp>(LOC, builder.getF32Type(), value);
                ret = builder.create<sysy::DeclTensorOp>(LOC);
                builder.create<sysy::BroadcastOp>(LOC, ret, broadcastTensor, value);
            } else if(sourceType.isF32()){
                ret = builder.create<sysy::DeclTensorOp>(LOC);
                builder.create<sysy::BroadcastOp>(LOC, ret, broadcastTensor, value);
            } else
                throw std::runtime_error("Unsupported sourceType");
        } else {
            llvm::outs() << "sourceType: " << sourceType << "\n";
            llvm::outs() << "targetType: " << targetType << "\n";
            throw std::runtime_error("Unsupported targetType");
        }
        return ret;
    }
}

mlir::Type SysYVisitorImpl::getMLIRType(const antlrcpp::Any& any){
    if(any.is<mlir::Value>()){
        return any.as<mlir::Value>().getType();
    } else {
        llvm::Constant* constVal = any.as<llvm::Constant*>();
        llvm::Type* type = constVal->getType();
        if(type->isIntegerTy(1))
            return builder.getIntegerType(1);
        else if (type->isIntegerTy(32))
            return builder.getIntegerType(32);
        else if (type->isFloatTy())
            return builder.getF32Type();
        throw std::runtime_error("Unsupported constant type");
    }
}

mlir::Type SysYVisitorImpl::unifyArithRelTypes(antlrcpp::Any& lhs, antlrcpp::Any& rhs) {
    mlir::Type lhsType = getMLIRType(lhs);
    mlir::Type rhsType = getMLIRType(rhs);

    mlir::Type targetType;
    if (llvm::isa<TensorType>(lhsType) || llvm::isa<TensorType>(rhsType)){
        if(llvm::isa<TensorType>(lhsType)){
            targetType = lhsType;
            broadcastTensor = lhs.as<mlir::Value>();
        } else {
            targetType = rhsType;
            broadcastTensor = rhs.as<mlir::Value>();
        }
    } else if (lhsType.isF32() || rhsType.isF32()) {
        targetType = builder.getF32Type();
    } else {
        targetType = builder.getIntegerType(32);
    }

    lhs = convertType(lhs, targetType);
    rhs = convertType(rhs, targetType);

    return targetType;
}

vector<int64_t> SysYVisitorImpl::getArrayDims(vector<SysYParser::ConstExpContext *> ctx) {
    vector<int64_t> dims;
    for (auto it = ctx.begin(); it != ctx.end(); ++it) {
        llvm::Constant* constExp = (*it)->accept(this).as<llvm::Constant*>();
        llvm::ConstantInt* constInt = llvm::dyn_cast<llvm::ConstantInt>(constExp);
        int64_t size = constInt->getZExtValue();
        dims.push_back(size);
    }
    return dims;
}

llvm::Constant* SysYVisitorImpl::getZeroConst(mlir::Type type) {
    if (type.isInteger(32)) {
        return llvm::ConstantInt::get(llvmI32Type, 0);
    } else if (type.isF32()) {
        return llvm::ConstantFP::get(llvmFloatType, 0.0);
    }
    throw std::runtime_error("Unsupported type for zero");
}

std::vector<int64_t> calculateArraySize(const vector<int64_t>& dims) {
    std::vector<int64_t> arraySize(dims.size() + 1);
    arraySize[0] = 1;
    for (size_t i = 0; i < dims.size(); ++i)
        arraySize[i + 1] = arraySize[i] * dims[dims.size() - 1 - i];
    return arraySize;
}

void SysYVisitorImpl::fillZeros(vector<antlrcpp::Any>& ret, int64_t count, int64_t size) {
    while(count < size) {
        ret.push_back(getZeroConst(basicType));
        ++count;
    }
}

auto getInitValChildren(SysYParser::ListConstInitValContext* node) {
    return node->constInitVal();
}

auto getInitValChildren(SysYParser::ListInitValContext* node) {
    return node->initVal();
}

antlrcpp::Any SysYVisitorImpl::getScalarValue(SysYParser::ScalarConstInitValContext* node) {
    return node->constExp()->accept(this);
}

antlrcpp::Any SysYVisitorImpl::getScalarValue(SysYParser::ScalarInitValContext* node) {
    return node->exp()->accept(this);
}

template<typename ListInitValContext, typename ScalarInitValContext>
void SysYVisitorImpl::getListInitVal(ListInitValContext* node, vector<int64_t> arraySize, vector<antlrcpp::Any>& ret) {
    int64_t count = 0;
    for (auto child : getInitValChildren(node)) {
        if (auto scalarChild = dynamic_cast<ScalarInitValContext*>(child)) {
            antlrcpp::Any value = getScalarValue(scalarChild);
            ret.push_back(convertType(value, basicType));
            ++count;
        }
        else {
            auto listChild = dynamic_cast<ListInitValContext*>(child);
            vector<int64_t> subArraySize(arraySize.begin(), arraySize.end() - 1); 
            while(count % subArraySize.back())
                subArraySize.pop_back();
            getListInitVal<ListInitValContext, ScalarInitValContext>(listChild, subArraySize, ret);
            count += subArraySize.back();
        }
    }
    fillZeros(ret, count, arraySize.back());
}

template<typename InitValContext, typename ScalarInitValContext, typename ListInitValContext>
std::vector<antlrcpp::Any> SysYVisitorImpl::getInitVal(InitValContext* node, vector<int64_t> dims) {
    vector<antlrcpp::Any> ret;
    if (auto scalarChild = dynamic_cast<ScalarInitValContext*>(node)) {
        assert(dims.empty());
        antlrcpp::Any value = getScalarValue(scalarChild);
        ret.push_back(convertType(value, basicType));
    } else {
        auto listChild = dynamic_cast<ListInitValContext*>(node);
        assert(!dims.empty());
        std::vector<int64_t> arraySize = calculateArraySize(dims);
        getListInitVal<ListInitValContext, ScalarInitValContext>(listChild, arraySize, ret);
    }
    return ret;
}

mlir::Type SysYVisitorImpl::getRealType(std::vector<int64_t> dims) {
    mlir::Type type = basicType;
    if(tensorDef){
        if(dims.empty())
            type = mlir::UnrankedTensorType::get(builder.getF32Type());
        else 
            type = mlir::RankedTensorType::get(dims, builder.getF32Type());
    } else {
        for (auto it = dims.rbegin(); it != dims.rend(); ++it)
            type = mlir::LLVM::LLVMArrayType::get(type, *it);
    }
    return type;
}

void SysYVisitorImpl::initLocalArrayAssign(mlir::Value var, int dimCount, const std::vector<antlrcpp::Any>& initVal){
    auto zero = builder.create<LLVM::ConstantOp>(LOC, builder.getI32IntegerAttr(0));
    auto one = builder.create<LLVM::ConstantOp>(LOC, builder.getI32IntegerAttr(1));
    std::vector<mlir::Value> indices(dimCount + 1, zero);

    // 对第一个元素进行赋值
    mlir::Value ptr = createGEPOp(var, indices);
    builder.create<LLVM::StoreOp>(LOC, getOrCreateValue(initVal[0]), ptr);
    // 对接下来的元素进行赋值
    for (uint64_t i = 1; i < initVal.size(); i++) {
        ptr = createGEPOp(ptr, llvm::ArrayRef<mlir::Value>{one});
        builder.create<LLVM::StoreOp>(LOC, getOrCreateValue(initVal[i]), ptr);
    }
}

void SysYVisitorImpl::initLocalArrayMemset(mlir::Value var, int64_t elementCount){
    auto zero = builder.create<LLVM::ConstantOp>(LOC, builder.getI8IntegerAttr(0));
    auto len = builder.create<LLVM::ConstantOp>(LOC, builder.getI64IntegerAttr(elementCount * 4));
    builder.create<LLVM::MemsetOp>(LOC, var, zero, len, false);
}

mlir::Value SysYVisitorImpl::resolveVariable(const string& varName) {
    if (localVarTable) {
        mlir::Value localVar = localVarTable->resolve(varName);
        if (localVar) return localVar;
    }
    LLVM::GlobalOp globalVar = globalVarTable->resolve(varName);
    assert(globalVar && "use of undeclared identifier");
    return builder.create<LLVM::AddressOfOp>(LOC, globalVar);
}

mlir::Value SysYVisitorImpl::computeArrayElementAddress(mlir::Value ptr, const std::vector<SysYParser::ExpContext*>& exps){
    auto zero = builder.create<LLVM::ConstantOp>(LOC, builder.getI32IntegerAttr(0));

    // Handle the first dimension
    std::vector<mlir::Value> indices;
    mlir::Type elementType = ptr.getType().dyn_cast<LLVM::LLVMPointerType>().getElementType();
    if (mlir::isa<LLVM::LLVMPointerType>(elementType)) // ptr is double pointer
        ptr = builder.create<LLVM::LoadOp>(LOC, ptr);
    else
        indices.push_back(zero);
    indices.push_back(getOrCreateValue(exps[0]->accept(this)));
    elementType = ptr.getType().dyn_cast<LLVM::LLVMPointerType>().getElementType();
    ptr = createGEPOp(ptr, indices);
    // Handle remaining dimensions
    for (size_t i = 1; i < exps.size(); i++) {
        indices.clear();
        indices.push_back(zero);
        indices.push_back(getOrCreateValue(exps[i]->accept(this)));
        ptr = createGEPOp(ptr, indices);
    }

    return ptr;
}

LLVM::AllocaOp SysYVisitorImpl::createAllocaOp(mlir::Type type) {
    mlir::OpBuilder::InsertionGuard insertGuard(builder);
    builder.setInsertionPoint(entryBlock, std::next(allocaInsertPoint));
    unsigned align = 0;
    if(type.isa<LLVM::LLVMArrayType>())
        align = 16;
    mlir::Type ptrType = LLVM::LLVMPointerType::get(type);
    mlir::Value one = builder.create<LLVM::ConstantOp>(LOC, builder.getI32IntegerAttr(1));
    LLVM::AllocaOp alloca = builder.create<LLVM::AllocaOp>(LOC, ptrType, one, align);
    allocaInsertPoint = alloca->getIterator();
    return alloca;
}

LLVM::GEPOp SysYVisitorImpl::createGEPOp(Value ptr, ValueRange indices) {
    assert(ptr.getType().isa<LLVM::LLVMPointerType>() && "GEP operation requires a pointer type");
    assert(indices.size() && "GEP operation requires at least one index");
    auto ptrType = ptr.getType().cast<LLVM::LLVMPointerType>();
    auto elementType = ptrType.getElementType();
    
    // Infer result type
    Type resultType;
    if (indices.size() == 1) {
        // If there's only one index, the result type is the same as the input pointer type
        resultType = ptrType;
    } else {
        // If there's more than one index, ptr must be a pointer to an array
        assert(elementType.isa<LLVM::LLVMArrayType>() && "Mismatch between number of indices and array dimensions");
        auto arrayType = elementType.dyn_cast<LLVM::LLVMArrayType>();
        // Progressively reduce dimensions
        Type currentType = arrayType;
        for (unsigned i = 0; i < indices.size() - 1; ++i) {
            assert(currentType.isa<LLVM::LLVMArrayType>() && "Mismatch between number of indices and array dimensions");
            currentType = currentType.cast<LLVM::LLVMArrayType>().getElementType();
        }
        // Create result pointer type
        resultType = LLVM::LLVMPointerType::get(currentType);
    }

    return builder.create<LLVM::GEPOp>(LOC, resultType, ptr, indices);
}

mlir::Attribute SysYVisitorImpl::getInitialValue(const vector<int64_t>& dims, vector<antlrcpp::Any>& initVal) {
    if (initVal.empty())
        fillZeros(initVal, 0, getElementCount(dims));
    if (dims.empty()) {
        // 标量
        if (basicType.isInteger(32)) {
            llvm::ConstantInt* valueI32 = llvm::dyn_cast<llvm::ConstantInt>(initVal[0].as<llvm::Constant*>());
            return builder.getI32IntegerAttr(valueI32->getSExtValue());
        } else {
            llvm::ConstantFP* valueF32 = llvm::dyn_cast<llvm::ConstantFP>(initVal[0].as<llvm::Constant*>());
            return builder.getF32FloatAttr(valueF32->getValue().convertToFloat());
        }
    } else {
        // 数组或张量
        auto tensorType = RankedTensorType::get(dims, basicType);
        if (basicType.isInteger(32)) {
            vector<APInt> initValAPInts;
            for (auto elementValue : initVal) {
                auto constI32 = llvm::dyn_cast<llvm::ConstantInt>(elementValue.as<llvm::Constant*>());
                initValAPInts.push_back(constI32->getValue());
            }
            return DenseElementsAttr::get(tensorType, initValAPInts);
        } else {
            vector<APFloat> initValAPFloats;
            for (auto elementValue : initVal) {
                auto constF32 = llvm::dyn_cast<llvm::ConstantFP>(elementValue.as<llvm::Constant*>());
                initValAPFloats.push_back(constF32->getValue());
            }
            return DenseElementsAttr::get(tensorType, initValAPFloats);
        }
    }
}

void SysYVisitorImpl::processVarDef(const string& varName, const vector<int64_t>& dims, 
    mlir::Type type, vector<antlrcpp::Any>& initVal, bool isConst) {
    if (type.isa<TensorType>())
        assert(curFunc && "Tensors can only be defined as local variables");
    if (curFunc) { // local var
        mlir::Value var;
        if(!type.isa<TensorType>()){
            var = createAllocaOp(type);
            // initialize
            if (!initVal.empty()) {
                if (dims.empty()) { // 标量
                    mlir::Value value = getOrCreateValue(initVal[0]);
                    builder.create<mlir::LLVM::StoreOp>(LOC, value, var);
                } else { // 数组
                    unsigned nonZeros = countNonZeros(initVal);
                    if(nonZeros == 0)
                        initLocalArrayMemset(var, getElementCount(dims));
                    else
                        initLocalArrayAssign(var, dims.size(), initVal);
                }
            }
        } else { // 张量
            if(type.isa<RankedTensorType>()){
                var = builder.create<sysy::DefTensorOp>(LOC, dims);
                if(!initVal.empty()){
                    mlir::DenseElementsAttr initialValue = dyn_cast<DenseElementsAttr>(getInitialValue(dims, initVal));
                    builder.create<sysy::InitTensorOp>(LOC, var, initialValue);
                }
            } else {
                var = builder.create<sysy::DeclTensorOp>(LOC);
            }
        }
        // register
        if (dims.empty() && isConst)
            localVarTable->registerConst(varName, var, initVal[0]);
        else
            localVarTable->registerVar(varName, var);
    } else { // global var
        mlir::Attribute initialValue = getInitialValue(dims, initVal);
        mlir::OpBuilder::InsertionGuard insertGuard(builder);
        builder.setInsertionPointToEnd(module.getBody());
        LLVM::GlobalOp var = builder.create<LLVM::GlobalOp>(
            LOC,
            type,
            isConst,
            mlir::LLVM::Linkage::Internal,
            varName,
            initialValue
        );
        // register
        if (dims.empty() && isConst)
            globalVarTable->registerConst(varName, var, initVal[0]);
        else
            globalVarTable->registerVar(varName, var);
    }
}

///////////////////////////////////编译单元/////////////////////////////////////
antlrcpp::Any SysYVisitorImpl::visitCompUnit(SysYParser::CompUnitContext *ctx){
    funcTable = new FunctionTable();
    globalVarTable = new GlobalVarTable();
    localVarTable = nullptr;

    basicType = mlir::Type();
    curFunc = mlir::func::FuncOp();
    entryBlock = nullptr;

    visitChildren(ctx);
    return nullptr;
}

///////////////////////////////////常量声明/////////////////////////////////////
antlrcpp::Any SysYVisitorImpl::visitDecl(SysYParser::DeclContext *ctx){
    visitChildren(ctx);
    return nullptr;
}

antlrcpp::Any SysYVisitorImpl::visitConstDecl(SysYParser::ConstDeclContext *ctx){
    visitChildren(ctx);
    return nullptr;
}

antlrcpp::Any SysYVisitorImpl::visitBType(SysYParser::BTypeContext *ctx){
    tensorDef = false;
    if(ctx->getText() == "int"){
        basicType = builder.getI32Type();
    } else if(ctx->getText() == "float") {
        basicType = builder.getF32Type();
    } else {
        basicType = builder.getF32Type();
        tensorDef = true;
    }
    return nullptr;
}

antlrcpp::Any SysYVisitorImpl::visitConstDef(SysYParser::ConstDefContext *ctx){
    string varName = ctx->Identifier()->getText();
#ifdef LLVM_DEBUG
    llvm::outs() << "visitConstDef:" << varName << "\n";
    llvm::outs().flush();
#endif
    vector<int64_t> dims = getArrayDims(ctx->constExp());
    mlir::Type type = getRealType(dims);
    vector<antlrcpp::Any> initVal = 
        getInitVal<SysYParser::ConstInitValContext, 
        SysYParser::ScalarConstInitValContext, 
        SysYParser::ListConstInitValContext>(ctx->constInitVal(), dims);
    processVarDef(varName, dims, type, initVal, true);
    return nullptr;
}

antlrcpp::Any SysYVisitorImpl::visitScalarConstInitVal(SysYParser::ScalarConstInitValContext *ctx){
    visitChildren(ctx);
    return nullptr;
}

antlrcpp::Any SysYVisitorImpl::visitListConstInitVal(SysYParser::ListConstInitValContext *ctx){
    visitChildren(ctx);
    return nullptr;
}

///////////////////////////////////变量声明/////////////////////////////////////
antlrcpp::Any SysYVisitorImpl::visitVarDecl(SysYParser::VarDeclContext *ctx){
    visitChildren(ctx);
    return nullptr;
}

antlrcpp::Any SysYVisitorImpl::visitUninitVarDef(SysYParser::UninitVarDefContext *ctx){
    string varName = ctx->Identifier()->getText();
#ifdef LLVM_DEBUG
    llvm::outs() << "visitUninitVarDef:" << varName << "\n";
    llvm::outs().flush();
#endif
    vector<int64_t> dims = getArrayDims(ctx->constExp());
    mlir::Type type = getRealType(dims);
    vector<antlrcpp::Any> initVal;
    processVarDef(varName, dims, type, initVal, false);
    return nullptr;
}

antlrcpp::Any SysYVisitorImpl::visitInitVarDef(SysYParser::InitVarDefContext *ctx){
    string varName = ctx->Identifier()->getText();
#ifdef LLVM_DEBUG
    llvm::outs() << "visitInitVarDef:" << varName << "\n";
    llvm::outs().flush();
#endif
    vector<int64_t> dims = getArrayDims(ctx->constExp());
    mlir::Type type = getRealType(dims);
    vector<antlrcpp::Any> initVal = 
        getInitVal<SysYParser::InitValContext, 
        SysYParser::ScalarInitValContext, 
        SysYParser::ListInitValContext>(ctx->initVal(), dims);
    processVarDef(varName, dims, type, initVal, false);
    return nullptr;
}

antlrcpp::Any SysYVisitorImpl::visitScalarInitVal(SysYParser::ScalarInitValContext *ctx){
    visitChildren(ctx);
    return nullptr;
}

antlrcpp::Any SysYVisitorImpl::visitListInitVal(SysYParser::ListInitValContext *ctx){
    visitChildren(ctx);
    return nullptr;
}

////////////////////////////////////函数//////////////////////////////////////

antlrcpp::Any SysYVisitorImpl::visitFuncDef(SysYParser::FuncDefContext *ctx){
    //函数名
    string funcName = ctx->Identifier()->getText();
#ifdef LLVM_DEBUG
    llvm::outs() << "visitFuncDef:" << funcName << "\n";
    llvm::outs().flush();
#endif
    builder.setInsertionPointToEnd(module.getBody());
    // 新建符号表
    LocalVarTable *parent = localVarTable;
    localVarTable = new LocalVarTable(parent);
    // 返回值类型
    Type retType = ctx->funcType()->accept(this);
    // 参数类型
    vector<pair<mlir::Type, string>> params;
    vector<mlir::Type> paramTypes;
    if(ctx->funcFParams()){
        params = ctx->funcFParams()->accept(this).as<vector<pair<mlir::Type, string>>>();
        for(auto param : params){
            paramTypes.push_back(param.first);
        }
    }
    // 注册函数
    curFunc = registerFunction(funcName, retType, paramTypes);
    // 创建入口基本块
    createEntryBlock(params);
    // 创建出口基本块
    createExitBlock(retType);
    // 处理语句块
    ctx->block()->accept(this);
    // 将当前基本块连接到出口基本块
    connectBlockToExit();
    // 清除函数状态
    curFunc = mlir::func::FuncOp();
    entryBlock = nullptr;
    exitBlock = nullptr;
    // 恢复符号表
    delete localVarTable;
    localVarTable = parent;
    return nullptr;
}

antlrcpp::Any SysYVisitorImpl::visitFuncType(SysYParser::FuncTypeContext *ctx){
    Type ret;
    if(ctx->getText() == "int"){
        ret = builder.getI32Type();
    } else if(ctx->getText() == "float"){
        ret = builder.getF32Type();
    } else if(ctx->getText() == "tensor"){
        throw std::runtime_error("Function return value type is not allowed to be tensor");
        // ret = mlir::UnrankedTensorType::get(builder.getF32Type());
    } else{
        ret = builder.getNoneType();
    }
    return ret;
}

antlrcpp::Any SysYVisitorImpl::visitFuncFParams(SysYParser::FuncFParamsContext *ctx){
    vector<pair<mlir::Type, string>> ret;
    for (auto i : ctx->funcFParam()){
        ret.push_back(i->accept(this));
    }
    return ret;
}

antlrcpp::Any SysYVisitorImpl::visitFuncFParam(SysYParser::FuncFParamContext *ctx){
    ctx->bType()->accept(this); // 设置参数basicType
    string name = ctx->Identifier()->getText(); // 参数名

    vector<int64_t> dims = getArrayDims(ctx->constExp());
    if(tensorDef)
        assert(ctx->getText().find("[") == string::npos && dims.empty());
    mlir::Type type = getRealType(dims);
    if(ctx->getText().find("[") != string::npos)
        type = LLVM::LLVMPointerType::get(type);
    return pair<mlir::Type, string>{type, name};
}

////////////////////////////////////代码块和语句/////////////////////////////////////

antlrcpp::Any SysYVisitorImpl::visitBlock(SysYParser::BlockContext *ctx){
    LocalVarTable *parent = localVarTable;
    localVarTable = new LocalVarTable(parent);
    for(auto blockItem : ctx->blockItem()){
        mlir::Block* curBlock = builder.getInsertionBlock();
        if(!curBlock->empty() && curBlock->back().hasTrait<mlir::OpTrait::IsTerminator>()) 
            break;
        blockItem->accept(this);
    }
    delete localVarTable;
    localVarTable = parent;
    return nullptr;
}

antlrcpp::Any SysYVisitorImpl::visitBlockItem(SysYParser::BlockItemContext *ctx){
    visitChildren(ctx);
    return nullptr;
}

antlrcpp::Any SysYVisitorImpl::visitAssignment(SysYParser::AssignmentContext *ctx){
    mlir::Value lVal = ctx->lVal()->accept(this).as<mlir::Value>();
    antlrcpp::Any expAny = ctx->exp()->accept(this);
    // 类型转换
    mlir::Type lValType = lVal.getType();
    mlir::Type elementType;
    if(lValType.isa<TensorType>()){
        elementType = lValType;
        broadcastTensor = lVal;
    } else {
        elementType = lVal.getType().dyn_cast<LLVM::LLVMPointerType>().getElementType();
    }
    expAny = convertType(expAny, elementType);
    // 转换为Value类型
    mlir::Value exp = getOrCreateValue(expAny);
    if(lValType.isa<TensorType>()){
        builder.create<sysy::CopyOp>(LOC, lVal, exp);
    } else {
        builder.create<LLVM::StoreOp>(LOC, exp, lVal);
    }
    return nullptr;
}

antlrcpp::Any SysYVisitorImpl::visitExpStmt(SysYParser::ExpStmtContext *ctx){
    visitChildren(ctx);
    return nullptr;
}

antlrcpp::Any SysYVisitorImpl::visitBlockStmt(SysYParser::BlockStmtContext *ctx){
    visitChildren(ctx);
    return nullptr;
}

antlrcpp::Any SysYVisitorImpl::visitIfStmt1(SysYParser::IfStmt1Context *ctx){
    auto condAny = ctx->cond()->accept(this);
    if(condAny.is<llvm::Constant*>()){ // condAny is Constant*
        auto condVal = condAny.as<llvm::Constant*>();
        if(dyn_cast<llvm::ConstantInt>(condVal)->isOne()) // cond == true
            ctx->stmt()->accept(this);
        return nullptr;
    }
    // condAny is Value
    auto condVal = condAny.as<mlir::Value>();
    // insert 2 basic blocks.
    mlir::Block* originBB = builder.getBlock();
    mlir::Block* thenBB = curFunc.addBlock();
    mlir::Block* mergeBB = curFunc.addBlock();
    builder.setInsertionPointToEnd(originBB);
    builder.create<mlir::cf::CondBranchOp>(LOC, condVal, thenBB, mergeBB);
    // codeGen for thenBB
    builder.setInsertionPointToEnd(thenBB);
    ctx->stmt()->accept(this);
    mlir::Block* thenEndBB = builder.getInsertionBlock();
    if(thenEndBB->empty() || !thenEndBB->back().hasTrait<mlir::OpTrait::IsTerminator>())
        builder.create<mlir::cf::BranchOp>(LOC, mergeBB);
    builder.setInsertionPointToEnd(mergeBB);
    return nullptr;
}

antlrcpp::Any SysYVisitorImpl::visitIfStmt2(SysYParser::IfStmt2Context *ctx){
    auto condAny = ctx->cond()->accept(this);
    if(condAny.is<llvm::Constant*>()){ // condAny is Constant*
        auto condVal = condAny.as<llvm::Constant*>();
        if(dyn_cast<llvm::ConstantInt>(condVal)->isOne()) // cond == true
            ctx->stmt()[0]->accept(this);
        else // cond == false
            ctx->stmt()[1]->accept(this);
        return nullptr;
    }
    // condAny is Value
    auto condVal = condAny.as<mlir::Value>();
    // Create basic blocks of Then, Else, Merge
    mlir::Block* originBB = builder.getBlock();
    mlir::Block* thenBB = curFunc.addBlock();
    mlir::Block* elseBB = curFunc.addBlock();
    mlir::Block* mergeBB = curFunc.addBlock();
    builder.setInsertionPointToEnd(originBB);
    builder.create<mlir::cf::CondBranchOp>(LOC, condVal, thenBB, elseBB);
    // codeGen for thenBB
    builder.setInsertionPointToEnd(thenBB);
    ctx->stmt()[0]->accept(this);
    mlir::Block* thenEndBB = builder.getInsertionBlock();
    if(thenEndBB->empty() || !thenEndBB->back().hasTrait<mlir::OpTrait::IsTerminator>())
        builder.create<mlir::cf::BranchOp>(LOC, mergeBB);
    // CodeGen for elseBB
    builder.setInsertionPointToEnd(elseBB);
    ctx->stmt()[1]->accept(this);
    mlir::Block* elseEndBB = builder.getInsertionBlock();
    if(elseEndBB->empty() || !elseEndBB->back().hasTrait<mlir::OpTrait::IsTerminator>())
        builder.create<mlir::cf::BranchOp>(LOC, mergeBB);
    builder.setInsertionPointToEnd(mergeBB);
    return nullptr;
}

antlrcpp::Any SysYVisitorImpl::visitWhileStmt(SysYParser::WhileStmtContext *ctx){
    mlir::Block* originBB = builder.getBlock();
    mlir::Block* condBB = curFunc.addBlock();
    builder.setInsertionPointToEnd(condBB);
    auto condAny = ctx->cond()->accept(this); // llvm::Constant* or mlir::Value 
    mlir::Block* condEndBB = builder.getBlock();
    if(condAny.is<llvm::Constant*>()){
        // delete condBB
        condBB->erase();
        auto condVal = condAny.as<llvm::Constant*>();
        if(dyn_cast<llvm::ConstantInt>(condVal)->isOne()){
            mlir::Block* bodyBB = curFunc.addBlock();
            mlir::Block* exitBB = curFunc.addBlock();
            // originBB -> bodyBB
            builder.setInsertionPointToEnd(originBB);
            builder.create<cf::BranchOp>(LOC, bodyBB);
            // handle bodyBB
            builder.setInsertionPointToEnd(bodyBB);
            continueTarget.push(bodyBB);
            breakTarget.push(exitBB);
            ctx->stmt()->accept(this);
            continueTarget.pop();
            breakTarget.pop();
            mlir::Block* bodyEndBB = builder.getBlock();
            // bodyEndBB -> bodyBB
            if(bodyEndBB->empty() || !bodyEndBB->back().hasTrait<mlir::OpTrait::IsTerminator>())
                builder.create<cf::BranchOp>(LOC, bodyBB);
            // move InsertionPoint to exitBB
            builder.setInsertionPointToEnd(exitBB);
        } else {
            builder.setInsertionPointToEnd(originBB);
        }
    } else {
        auto condVal = condAny.as<mlir::Value>();
        mlir::Block* bodyBB = curFunc.addBlock();
        mlir::Block* exitBB = curFunc.addBlock();
        // originBB -> condBB
        builder.setInsertionPointToEnd(originBB);
        builder.create<cf::BranchOp>(LOC, condBB);
        // condEndBB -> bodyBB/exitBB
        builder.setInsertionPointToEnd(condEndBB);
        builder.create<cf::CondBranchOp>(LOC, condVal, bodyBB, exitBB);
        // handle bodyBB
        builder.setInsertionPointToEnd(bodyBB);
        continueTarget.push(condBB);
        breakTarget.push(exitBB);
        ctx->stmt()->accept(this);
        continueTarget.pop();
        breakTarget.pop();
        mlir::Block* bodyEndBB = builder.getBlock();
        // bodyEndBB -> condBB
        if(bodyEndBB->empty() || !bodyEndBB->back().hasTrait<mlir::OpTrait::IsTerminator>())
            builder.create<cf::BranchOp>(LOC, condBB);
        // move InsertionPoint to exitBB
        builder.setInsertionPointToEnd(exitBB);
    }
    return nullptr;
}

antlrcpp::Any SysYVisitorImpl::visitBreakStmt(SysYParser::BreakStmtContext *ctx){
    builder.create<cf::BranchOp>(LOC, breakTarget.top());
    return nullptr;
}

antlrcpp::Any SysYVisitorImpl::visitContinueStmt(SysYParser::ContinueStmtContext *ctx){
    builder.create<cf::BranchOp>(LOC, continueTarget.top());
    return nullptr;
}

antlrcpp::Any SysYVisitorImpl::visitReturnStmt(SysYParser::ReturnStmtContext *ctx){
    mlir::FunctionType funcType = llvm::dyn_cast<mlir::FunctionType>(curFunc.getFunctionType());
    if(funcType.getResults().empty()){
        assert(!ctx->exp() && "void function should not return a value");
        builder.create<cf::BranchOp>(LOC, exitBlock);
    } else {
        assert(ctx->exp() && "non-void function should return a value");
        antlrcpp::Any exp = ctx->exp()->accept(this);
        exp = convertType(exp, funcType.getResult(0));
        mlir::Value retVal = getOrCreateValue(exp);
        builder.create<cf::BranchOp>(LOC, exitBlock, retVal);
    }
    return nullptr;
}

//////////////////////////////////////表达式///////////////////////////////////////

antlrcpp::Any SysYVisitorImpl::visitExp(SysYParser::ExpContext *ctx){
    return ctx->addExp()->accept(this); // mlir::Value or llvm::Constant*
}

antlrcpp::Any SysYVisitorImpl::visitCond(SysYParser::CondContext *ctx){
    // mlir::Value or llvm::Constant*
    return convertType(ctx->lOrExp()->accept(this), builder.getI1Type());
}

antlrcpp::Any SysYVisitorImpl::visitLVal(SysYParser::LValContext *ctx){
    // return mlir::Value
    string varName = ctx->Identifier()->getText();
#ifdef LLVM_DEBUG
    llvm::outs() << "visitLVal:" << varName << "\n";
    llvm::outs().flush();
#endif
    mlir::Value ptr = resolveVariable(varName);
    if (!ctx->exp().size()) return ptr;
    // 数组
    return computeArrayElementAddress(ptr, ctx->exp());
}

antlrcpp::Any SysYVisitorImpl::visitPrimaryExp1(SysYParser::PrimaryExp1Context *ctx){
    return ctx->exp()->accept(this); // mlir::Value or llvm::Constant*
}

antlrcpp::Any SysYVisitorImpl::visitPrimaryExp2(SysYParser::PrimaryExp2Context *ctx){
    // return mlir::Value or llvm::Constant*
    mlir::Value lVal = ctx->lVal()->accept(this).as<mlir::Value>();
    // 如果是张量
    if(lVal.getType().isa<TensorType>())
        return lVal;
    // 如果是常量，返回常量值
    if (auto addressOfOp = lVal.getDefiningOp<LLVM::AddressOfOp>()) {
        SymbolTableCollection symbolTable;
        LLVM::GlobalOp globalOp = addressOfOp.getGlobal(symbolTable);
        llvm::Constant* constVal = globalVarTable->loadValue(globalOp);
        if(constVal) return constVal;
    } else {
        llvm::Constant* constVal = localVarTable->loadValue(lVal);
        if(constVal) return constVal;
    }
    mlir::Value ret;
    mlir::Type elementType = lVal.getType().dyn_cast<LLVM::LLVMPointerType>().getElementType();
    if(mlir::isa<LLVM::LLVMArrayType>(elementType)){
        // 数组作为参数传递给函数调用，计算地址而不是加载值
        auto zero = builder.create<LLVM::ConstantOp>(LOC, builder.getI32IntegerAttr(0));
        ret = createGEPOp(lVal, llvm::ArrayRef<mlir::Value>{zero, zero});
    } else
        ret = builder.create<LLVM::LoadOp>(LOC, lVal);
    return ret;
}

antlrcpp::Any SysYVisitorImpl::visitPrimaryExp3(SysYParser::PrimaryExp3Context *ctx){
    llvm::Constant* ret = ctx->number()->accept(this);
    return ret;
}

antlrcpp::Any SysYVisitorImpl::visitNumber(SysYParser::NumberContext *ctx){
    llvm::Constant* num;
    if(ctx->IntConst())
        num = llvm::ConstantInt::get(llvmI32Type, s2i(ctx->getText()));
    else
        num = llvm::ConstantFP::get(llvmFloatType, stof(ctx->getText()));
#ifdef LLVM_DEBUG
    llvm::outs() << "visitNumber:" << ctx->getText() << "\n";
    llvm::outs().flush();
#endif
    return num;
}

antlrcpp::Any SysYVisitorImpl::visitUnary1(SysYParser::Unary1Context *ctx){
    return ctx->primaryExp()->accept(this); // mlir::Value or llvm::Constant*
}

antlrcpp::Any SysYVisitorImpl::visitUnary2(SysYParser::Unary2Context *ctx){
    mlir::Value ret = mlir::Value();
    string funcName = ctx->Identifier()->getText();
    if(funcName == "starttime") funcName = "_sysy_starttime";
    else if(funcName == "stoptime") funcName = "_sysy_stoptime";
    // 获取实参
    vector<antlrcpp::Any> args;
    if(funcName == "_sysy_starttime" || funcName == "_sysy_stoptime"){
        mlir::Value line = builder.create<LLVM::ConstantOp>(LOC, builder.getI32IntegerAttr(ctx->start->getLine()));
        args.push_back(line);
    }
    else if (ctx->funcRParams()){
        args = ctx->funcRParams()->accept(this).as<vector<antlrcpp::Any>>();
    }
    // 判断是否是核心张量函数调用
    if(resolveCoreTensorFunction(funcName, args, ret))
        return ret;
    mlir::FunctionOpInterface funcOpInterface = resolveFunction(funcName);
    mlir::FunctionType funcType = llvm::dyn_cast<mlir::FunctionType>(funcOpInterface.getFunctionType());
    assert(args.size() == funcType.getNumInputs());
    // 对参数进行类型转换
    vector<mlir::Value> argss;
    for(uint64_t i = 0; i < args.size(); i++){
        mlir::Type paramType = funcType.getInput(i);
        argss.push_back(getOrCreateValue(convertType(args[i], paramType)));
    }
    // 创建调用操作
    mlir::CallOpInterface call;
    if(llvm::isa<sysy::FunctionOp>(funcOpInterface)){
        auto tensorFunc = llvm::cast<sysy::FunctionOp>(funcOpInterface);
        call = builder.create<sysy::GenericCallOp>(LOC, tensorFunc, argss);
    } else {
        auto func = llvm::cast<func::FuncOp>(funcOpInterface);
        call = builder.create<func::CallOp>(LOC, func, argss);
    }
    if(!funcOpInterface.getResultTypes().empty())
        ret = call->getResult(0);
    return ret;
}

antlrcpp::Any SysYVisitorImpl::visitUnary3(SysYParser::Unary3Context *ctx){
    char op = ctx->unaryOp()->getText()[0];
#ifdef LLVM_DEBUG
    llvm::outs() << "visitUnary3:" << op << "\n";
    llvm::outs().flush();
#endif
    antlrcpp::Any operand = ctx->unaryExp()->accept(this); // mlir::Value or llvm::Constant*
    if(op == '+')
        return operand;
    if(operand.is<mlir::Value>()){
        auto oprVal = operand.as<mlir::Value>();
        if(op == '-'){
            mlir::Value retVal;
            mlir::Type type = oprVal.getType();
            if(type.isInteger(1) || type.isInteger(32)){
                oprVal = convertType(oprVal, builder.getI32Type());
                auto zero = builder.create<LLVM::ConstantOp>(LOC, builder.getI32IntegerAttr(0));
                retVal = builder.create<LLVM::SubOp>(LOC, builder.getI32Type(), zero, oprVal);
            }else if(type.isF32()){
                retVal = builder.create<LLVM::FNegOp>(LOC, oprVal);
            }
            return retVal;
        } else if(op == '!'){
            oprVal = convertType(oprVal, builder.getI1Type());
            auto truee = builder.create<LLVM::ConstantOp>(LOC, builder.getIntegerAttr(builder.getI1Type(), 1));
            mlir::Value retVal = builder.create<LLVM::XOrOp>(LOC, builder.getI1Type(), oprVal, truee);
            return retVal;
        }
    } else if(operand.is<llvm::Constant*>()){
        llvm::Constant* oprConst = operand.as<llvm::Constant*>();
        if(op == '-'){
            llvm::Constant* retConst;
            llvm::Type* type = oprConst->getType();
            if(type->isIntegerTy(1) || type->isIntegerTy(32)){
                oprConst = convertType(oprConst, builder.getI32Type());
                retConst = llvm::ConstantExpr::getNeg(oprConst);
            } else if (auto fpConst = llvm::dyn_cast<llvm::ConstantFP>(oprConst)) {
                retConst = llvm::ConstantFP::get(llvmFloatType, -fpConst->getValue().convertToDouble());
            }
            return retConst;
        } else if (op == '!') {
            // 转换为int1再取非
            oprConst = convertType(oprConst, builder.getI1Type());
            llvm::Constant* retConst = llvm::ConstantExpr::getNot(oprConst);
            return retConst;
        }
    }
    throw std::runtime_error("Unsupported unary operation or operand type in visitUnary3");
}

antlrcpp::Any SysYVisitorImpl::visitUnaryOp(SysYParser::UnaryOpContext *ctx){
    visitChildren(ctx);
    return nullptr;
}

antlrcpp::Any SysYVisitorImpl::visitFuncRParams(SysYParser::FuncRParamsContext *ctx){
    vector<antlrcpp::Any> args;
    for(auto i : ctx->funcRParam()){
        antlrcpp::Any arg = i->accept(this);
        args.emplace_back(arg);
    }
    return args;
}

antlrcpp::Any SysYVisitorImpl::visitExpAsRParam(SysYParser::ExpAsRParamContext *ctx){
    return ctx->exp()->accept(this); // mlir::Value or llvm::Constant*
}

antlrcpp::Any SysYVisitorImpl::visitStringAsRParam(SysYParser::StringAsRParamContext *ctx){
    visitChildren(ctx);
    return nullptr;
}

antlrcpp::Any SysYVisitorImpl::visitMul1(SysYParser::Mul1Context *ctx){
    return ctx->unaryExp()->accept(this); // mlir::Value or llvm::Constant*
}

antlrcpp::Any SysYVisitorImpl::visitMul2(SysYParser::Mul2Context *ctx){
    // return mlir::Value or llvm::Constant*
    // 操作符
    char op = ctx->children[1]->getText()[0];
    // 操作数
    antlrcpp::Any lhsAny = ctx->mulExp()->accept(this);
    antlrcpp::Any rhsAny = ctx->unaryExp()->accept(this);
    // 类型转换
    Type type = unifyArithRelTypes(lhsAny, rhsAny);
    // 只要其中一个是Value类型（变量），就都要转换为Value类型
    if(lhsAny.is<mlir::Value>() || rhsAny.is<mlir::Value>()){
        mlir::Value ret = mlir::Value();
        // 创建常量
        mlir::Value lhs = getOrCreateValue(lhsAny);
        mlir::Value rhs = getOrCreateValue(rhsAny);
        switch (op) {
            case '*':
                if(type.isInteger(32))
                    ret = builder.create<LLVM::MulOp>(LOC, lhs, rhs);
                else if(type.isF32())
                    ret = builder.create<LLVM::FMulOp>(LOC, lhs, rhs);
                else if(llvm::isa<TensorType>(type)){
                    ret = builder.create<sysy::DeclTensorOp>(LOC);
                    builder.create<sysy::MulOp>(LOC, ret, lhs, rhs);
                }
                break;
            case '/':
                if(type.isInteger(32))
                    ret = builder.create<LLVM::SDivOp>(LOC, lhs, rhs);
                else if(type.isF32())
                    ret = builder.create<LLVM::FDivOp>(LOC, lhs, rhs);
                else if(llvm::isa<TensorType>(type)){
                    ret = builder.create<sysy::DeclTensorOp>(LOC);
                    builder.create<sysy::DivOp>(LOC, ret, lhs, rhs);
                }
                break;
            case '%':
                if(type.isInteger(32))
                    ret = builder.create<LLVM::SRemOp>(LOC, lhs, rhs);
                else if(type.isF32())
                    throw std::runtime_error("invalid operands to modulo operation ('%')");
                break;
        }
        return ret;
    } else {
        llvm::Constant* ret;
        auto lhs = lhsAny.as<llvm::Constant*>();
        auto rhs = rhsAny.as<llvm::Constant*>();
        switch (op) {
            case '*':
                if(type.isInteger(32))
                    ret = llvm::dyn_cast<llvm::Constant>(llvmBuilder.CreateMul(lhs, rhs));
                else if(type.isF32())
                    ret = llvm::dyn_cast<llvm::Constant>(llvmBuilder.CreateFMul(lhs, rhs));
                break;
            case '/':
                if(type.isInteger(32))
                    ret = llvm::dyn_cast<llvm::Constant>(llvmBuilder.CreateSDiv(lhs, rhs));
                else if(type.isF32())
                    ret = llvm::dyn_cast<llvm::Constant>(llvmBuilder.CreateFDiv(lhs, rhs));
                break;
            case '%':
                if(type.isInteger(32))
                    ret = llvm::dyn_cast<llvm::Constant>(llvmBuilder.CreateSRem(lhs, rhs));
                else
                    throw std::runtime_error("invalid operands to modulo operation ('%')");
                break;
        }
        return ret;
    }
}

antlrcpp::Any SysYVisitorImpl::visitAdd1(SysYParser::Add1Context *ctx){
    return ctx->mulExp()->accept(this); // mlir::Value or llvm::Constant*
}

antlrcpp::Any SysYVisitorImpl::visitAdd2(SysYParser::Add2Context *ctx){
    // return mlir::Value or llvm::Constant*
    // 操作符
    char op = ctx->children[1]->getText()[0];
    // 操作数
    antlrcpp::Any lhsAny = ctx->addExp()->accept(this);
    antlrcpp::Any rhsAny = ctx->mulExp()->accept(this);
    // 类型转换
    mlir::Type type = unifyArithRelTypes(lhsAny, rhsAny);
    // 只要其中一个是Value类型（变量），就都要转换为Value类型
    if(lhsAny.is<mlir::Value>() || rhsAny.is<mlir::Value>()){
        mlir::Value ret = mlir::Value();
        // 创建常量
        mlir::Value lhs = getOrCreateValue(lhsAny);
        mlir::Value rhs = getOrCreateValue(rhsAny);
        switch (op) {
            case '+':
                if(type.isInteger(32))
                    ret = builder.create<LLVM::AddOp>(LOC, lhs, rhs);
                else if(type.isF32())
                    ret = builder.create<LLVM::FAddOp>(LOC, lhs, rhs);
                else if(llvm::isa<TensorType>(type)){
                    ret = builder.create<sysy::DeclTensorOp>(LOC);
                    builder.create<sysy::AddOp>(LOC, ret, lhs, rhs);
                }
                break;
            case '-':
                if(type.isInteger(32))
                    ret = builder.create<LLVM::SubOp>(LOC, lhs, rhs);
                else if(type.isF32())
                    ret = builder.create<LLVM::FSubOp>(LOC, lhs, rhs);
                else if(llvm::isa<TensorType>(type)){
                    ret = builder.create<sysy::DeclTensorOp>(LOC);
                    builder.create<sysy::SubOp>(LOC, ret, lhs, rhs);
                }
                break;
        }
        return ret;
    } else {
        llvm::Constant* ret;
        llvm::Constant* lhs = lhsAny.as<llvm::Constant*>();
        llvm::Constant* rhs = rhsAny.as<llvm::Constant*>();
        switch (op) {
            case '+':
                if(type.isInteger(32))
                    ret = llvm::dyn_cast<llvm::Constant>(llvmBuilder.CreateAdd(lhs, rhs));
                else if(type.isF32())
                    ret = llvm::dyn_cast<llvm::Constant>(llvmBuilder.CreateFAdd(lhs, rhs));
                break;
            case '-':
                if(type.isInteger(32))
                    ret = llvm::dyn_cast<llvm::Constant>(llvmBuilder.CreateSub(lhs, rhs));
                else if(type.isF32())
                    ret = llvm::dyn_cast<llvm::Constant>(llvmBuilder.CreateFSub(lhs, rhs));
                break;
        }
        return ret;
    }
}

antlrcpp::Any SysYVisitorImpl::visitRel1(SysYParser::Rel1Context *ctx){
    return ctx->addExp()->accept(this); // mlir::Value or llvm::Constant*
}

antlrcpp::Any SysYVisitorImpl::visitRel2(SysYParser::Rel2Context *ctx) {
    // 操作符
    string op = ctx->children[1]->getText();
    // 操作数
    antlrcpp::Any lhsAny = ctx->relExp()->accept(this);
    antlrcpp::Any rhsAny = ctx->addExp()->accept(this);
    // 统一类型
    mlir::Type type = unifyArithRelTypes(lhsAny, rhsAny);
    if(lhsAny.is<mlir::Value>() || rhsAny.is<mlir::Value>()){
        mlir::Value lhs = getOrCreateValue(lhsAny);
        mlir::Value rhs = getOrCreateValue(rhsAny);
        // 对操作符进行映射
        const std::map<std::string, std::pair<LLVM::ICmpPredicate, LLVM::FCmpPredicate>> opMap = {
            {"<", {LLVM::ICmpPredicate::slt, LLVM::FCmpPredicate::olt}},
            {">", {LLVM::ICmpPredicate::sgt, LLVM::FCmpPredicate::ogt}},
            {"<=", {LLVM::ICmpPredicate::sle, LLVM::FCmpPredicate::ole}},
            {">=", {LLVM::ICmpPredicate::sge, LLVM::FCmpPredicate::oge}}
        };
        auto opPair = opMap.at(op);
        mlir::Value ret;
        if (type.isa<mlir::IntegerType>()) 
            ret = builder.create<LLVM::ICmpOp>(builder.getUnknownLoc(), opPair.first, lhs, rhs);
        else if (type.isa<mlir::FloatType>()) 
            ret = builder.create<LLVM::FCmpOp>(builder.getUnknownLoc(), opPair.second, lhs, rhs);
        return ret;
    } else {
        llvm::Constant* lhs = lhsAny.as<llvm::Constant*>();
        llvm::Constant* rhs = rhsAny.as<llvm::Constant*>();
        // 对操作符进行映射
        const std::map<std::string, std::pair<llvm::CmpInst::Predicate, llvm::CmpInst::Predicate>> opMap = {
            {"<", {llvm::CmpInst::ICMP_SLT, llvm::CmpInst::FCMP_OLT}},
            {">", {llvm::CmpInst::ICMP_SGT, llvm::CmpInst::FCMP_OGT}},
            {"<=", {llvm::CmpInst::ICMP_SLE, llvm::CmpInst::FCMP_OLE}},
            {">=", {llvm::CmpInst::ICMP_SGE, llvm::CmpInst::FCMP_OGE}}
        };
        auto opPair = opMap.at(op);
        llvm::CmpInst::Predicate predicate = type.isF32() ? opPair.second : opPair.first;
        llvm::Constant *ret = llvm::dyn_cast<llvm::Constant>(llvmBuilder.CreateCmp(predicate, lhs, rhs));
        return ret;
    }
}

antlrcpp::Any SysYVisitorImpl::visitEq1(SysYParser::Eq1Context *ctx){
    return ctx->relExp()->accept(this); // mlir::Value or llvm::Constant*
}

antlrcpp::Any SysYVisitorImpl::visitEq2(SysYParser::Eq2Context *ctx){
    // return mlir::Value or llvm::Constant*
    // 操作符
    string op = ctx->children[1]->getText();
    // 操作数
    antlrcpp::Any lhsAny = ctx->eqExp()->accept(this);
    antlrcpp::Any rhsAny = ctx->relExp()->accept(this);
    // 统一类型
    mlir::Type type = unifyArithRelTypes(lhsAny, rhsAny);
    if (lhsAny.is<mlir::Value>() || rhsAny.is<mlir::Value>()) {
        mlir::Value lhs = getOrCreateValue(lhsAny);
        mlir::Value rhs = getOrCreateValue(rhsAny);
        // 对操作符进行映射
        const std::map<std::string, std::pair<LLVM::ICmpPredicate, LLVM::FCmpPredicate>> opMap = {
            {"==", {LLVM::ICmpPredicate::eq, LLVM::FCmpPredicate::oeq}},
            {"!=", {LLVM::ICmpPredicate::ne, LLVM::FCmpPredicate::one}}
        };
        auto opPair = opMap.at(op);
        mlir::Value ret;
        if (type.isa<mlir::IntegerType>())
            ret = builder.create<LLVM::ICmpOp>(builder.getUnknownLoc(), opPair.first, lhs, rhs);
        else if (type.isa<mlir::FloatType>())
            ret = builder.create<LLVM::FCmpOp>(builder.getUnknownLoc(), opPair.second, lhs, rhs);
        return ret;
    } else {
        llvm::Constant* lhs = lhsAny.as<llvm::Constant*>();
        llvm::Constant* rhs = rhsAny.as<llvm::Constant*>();
        // 对操作符进行映射
        const std::map<std::string, std::pair<llvm::CmpInst::Predicate, llvm::CmpInst::Predicate>> opMap = {
            {"==", {llvm::CmpInst::ICMP_EQ, llvm::CmpInst::FCMP_OEQ}},
            {"!=", {llvm::CmpInst::ICMP_NE, llvm::CmpInst::FCMP_ONE}}
        };
        auto opPair = opMap.at(op);
        llvm::CmpInst::Predicate predicate = type.isF32() ? opPair.second : opPair.first;
        llvm::Constant *ret = llvm::dyn_cast<llvm::Constant>(llvmBuilder.CreateCmp(predicate, lhs, rhs));
        return ret;
    }
}

antlrcpp::Any SysYVisitorImpl::visitLAnd1(SysYParser::LAnd1Context *ctx){
    return ctx->eqExp()->accept(this); // mlir::Value or llvm::Constant*
}

antlrcpp::Any SysYVisitorImpl::visitLAnd2(SysYParser::LAnd2Context *ctx){
    // return mlir::Value or llvm::Constant*
    antlrcpp::Any lhsAny = ctx->lAndExp()->accept(this);
    antlrcpp::Any rhsAny;
    lhsAny = convertType(lhsAny, builder.getI1Type());
    if(lhsAny.is<llvm::Constant*>()){ // lhs is llvm::Constant*
        llvm::Constant* lhs = lhsAny.as<llvm::Constant*>();
        if(dyn_cast<llvm::ConstantInt>(lhs)->isZero()) // lhs is false
            return lhs;
        // lhs is true
        rhsAny = ctx->eqExp()->accept(this);
        rhsAny = convertType(rhsAny, builder.getI1Type());
        return rhsAny;
    } else { // lhs is mlir::Value
        mlir::Block* originBB = builder.getBlock();
        mlir::Block* rValueBB = curFunc.addBlock();
        builder.setInsertionPointToEnd(rValueBB);
        rhsAny = ctx->eqExp()->accept(this);
        mlir::Block* rValEndBB = builder.getInsertionBlock();
        rhsAny = convertType(rhsAny, builder.getI1Type());
        if(rhsAny.is<llvm::Constant*>()){ // rhs is llvm::Constant*
            llvm::Constant* rhs = rhsAny.as<llvm::Constant*>();
            rValueBB->erase();
            builder.setInsertionPointToEnd(originBB);
            if(dyn_cast<llvm::ConstantInt>(rhs)->isZero()) // rhs is false
                return rhs;
            else // rhs is true
                return lhsAny;
        } else { // rhs is mlir::Value
            mlir::Value lhs = lhsAny.as<mlir::Value>();
            mlir::Value rhs = rhsAny.as<mlir::Value>();
            mlir::Block* lValueBB = curFunc.addBlock();
            mlir::Block* mergeBB = curFunc.addBlock();
            // originBB -> rValueBB/lValueBB
            builder.setInsertionPointToEnd(originBB);
            mlir::Value condAddr = createAllocaOp(builder.getI1Type());
            builder.create<mlir::cf::CondBranchOp>(LOC, lhs, rValueBB, lValueBB);
            // lValueBB -> mergeBB
            builder.setInsertionPointToEnd(lValueBB);
            mlir::Value falseVal = builder.create<LLVM::ConstantOp>(LOC, 
                builder.getIntegerAttr(builder.getI1Type(), 0));
            builder.create<LLVM::StoreOp>(LOC, falseVal, condAddr);
            builder.create<mlir::cf::BranchOp>(LOC, mergeBB);
            // rValEndBB -> mergeBB
            builder.setInsertionPointToEnd(rValEndBB);
            builder.create<LLVM::StoreOp>(LOC, rhs, condAddr);
            builder.create<mlir::cf::BranchOp>(LOC, mergeBB);
            // mergeBB
            builder.setInsertionPointToEnd(mergeBB);
            mlir::Value condVal = builder.create<LLVM::LoadOp>(LOC, condAddr);
            return condVal;
        }
    }
}

antlrcpp::Any SysYVisitorImpl::visitLOr1(SysYParser::LOr1Context *ctx){
    return ctx->lAndExp()->accept(this); // mlir::Value or llvm::Constant*
}

antlrcpp::Any SysYVisitorImpl::visitLOr2(SysYParser::LOr2Context *ctx){
    // return mlir::Value or llvm::Constant*
    antlrcpp::Any lhsAny = ctx->lOrExp()->accept(this);
    antlrcpp::Any rhsAny;
    lhsAny = convertType(lhsAny, builder.getI1Type());
    if(lhsAny.is<llvm::Constant*>()){ // lhs is llvm::Constant*
        llvm::Constant* lhs = lhsAny.as<llvm::Constant*>();
        if(dyn_cast<llvm::ConstantInt>(lhs)->isOne()) // lhs is true
            return lhs;
        // lhs is false
        rhsAny = ctx->lAndExp()->accept(this);
        rhsAny = convertType(rhsAny, builder.getI1Type());
        return rhsAny;
    } else { // lhs is mlir::Value
        mlir::Block* originBB = builder.getBlock();
        mlir::Block* rValueBB = curFunc.addBlock();
        builder.setInsertionPointToEnd(rValueBB);
        rhsAny = ctx->lAndExp()->accept(this);
        mlir::Block* rValEndBB = builder.getInsertionBlock();
        rhsAny = convertType(rhsAny, builder.getI1Type());
        if(rhsAny.is<llvm::Constant*>()){ // rhs is llvm::Constant*
            llvm::Constant* rhs = rhsAny.as<llvm::Constant*>();
            rValueBB->erase();
            builder.setInsertionPointToEnd(originBB);
            if(dyn_cast<llvm::ConstantInt>(rhs)->isOne()) // rhs is true
                return rhs;
            else // rhs is false
                return lhsAny;
        } else { // rhs is mlir::Value
            mlir::Value lhs = lhsAny.as<mlir::Value>();
            mlir::Value rhs = rhsAny.as<mlir::Value>();
            mlir::Block* lValueBB = curFunc.addBlock();
            mlir::Block* mergeBB = curFunc.addBlock();
            // originBB -> lValueBB/rValueBB
            builder.setInsertionPointToEnd(originBB);
            mlir::Value condAddr = createAllocaOp(builder.getI1Type());
            builder.create<mlir::cf::CondBranchOp>(LOC, lhs, lValueBB, rValueBB);
            // lValueBB -> mergeBB
            builder.setInsertionPointToEnd(lValueBB);
            mlir::Value trueVal = builder.create<LLVM::ConstantOp>(LOC, 
                builder.getIntegerAttr(builder.getI1Type(), 1));
            builder.create<LLVM::StoreOp>(LOC, trueVal, condAddr);
            builder.create<mlir::cf::BranchOp>(LOC, mergeBB);
            // rValEndBB -> mergeBB
            builder.setInsertionPointToEnd(rValEndBB);
            builder.create<LLVM::StoreOp>(LOC, rhs, condAddr);
            builder.create<mlir::cf::BranchOp>(LOC, mergeBB);
            // mergeBB
            builder.setInsertionPointToEnd(mergeBB);
            mlir::Value condVal = builder.create<LLVM::LoadOp>(LOC, condAddr);
            return condVal;
        }
    }
}

antlrcpp::Any SysYVisitorImpl::visitConstExp(SysYParser::ConstExpContext *ctx){
    llvm::Constant* ret = ctx->addExp()->accept(this);
    return ret;
}
