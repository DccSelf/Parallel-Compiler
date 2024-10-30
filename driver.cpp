#include <iostream>
#include <fstream>
#include <string>
#include <memory>
#include <getopt.h>

#include "llvm/Support/CodeGen.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/Host.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/IR/LegacyPassManager.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/OpenMPToLLVM/ConvertOpenMPToLLVM.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/OpenMP/OpenMPToLLVMIRTranslation.h"

#include "frontend/SysYLexer.h"
#include "frontend/SysYParser.h"
#include "frontend/SysYVisitorImpl.h"
#include "dialect/Dialect.h"
#include "dialect/test.h"
#include "dialect/Passes.h"
#include "dialect/InlinerExtension.h"
#include "dialect/AffinePasses.h"

enum class OutputLevel {
    ObjectFile, // 默认级别，输出目标文件
    LLVMIR,     // 输出 LLVM IR
    LLVMDialect,// 输出 LLVM 方言
    MixedMLIR   // 输出混合的 MLIR
};

enum class TargetArch {
    Default,
    ARMv7
};

void printUsage(const char* programName) {
    std::cerr << "Usage: " << programName << " [options] <input_file>" << std::endl;
    std::cerr << "Options:" << std::endl;
    std::cerr << "  -o <file>    Specify output file (default: output to console)" << std::endl;
    std::cerr << "  -m           Output mixed MLIR instead of object file" << std::endl;
    std::cerr << "  -l           Output LLVM dialect instead of object file" << std::endl;
    std::cerr << "  -L           Output LLVM IR instead of object file" << std::endl;
    std::cerr << "  -O<level>    Set llvm optimization level (0, 1, 2, 3, default: 0)" << std::endl;
    std::cerr << "  -a           Compile for ARMv7 architecture" << std::endl;
    std::cerr << "  -h           Display this help message" << std::endl;
}

bool parseCommandLineArgs(int argc, char* argv[], std::string& inputFilename, 
                          std::string& outputFilename, OutputLevel& outputLevel,
                          llvm::OptimizationLevel& optLevel, TargetArch& targetArch) {
    int opt;
    outputLevel = OutputLevel::ObjectFile;
    optLevel = llvm::OptimizationLevel::O0;
    targetArch = TargetArch::Default;
    while ((opt = getopt(argc, argv, "o:mlLO:ah")) != -1) {
        switch (opt) {
            case 'o':
                outputFilename = optarg;
                break;
            case 'm':
                outputLevel = OutputLevel::MixedMLIR;
                break;
            case 'l':
                outputLevel = OutputLevel::LLVMDialect;
                break;
            case 'L':
                outputLevel = OutputLevel::LLVMIR;
                break;
            case 'O':
                switch (optarg[0]) {
                    case '0': optLevel = llvm::OptimizationLevel::O0; break;
                    case '1': optLevel = llvm::OptimizationLevel::O1; break;
                    case '2': optLevel = llvm::OptimizationLevel::O2; break;
                    case '3': optLevel = llvm::OptimizationLevel::O3; break;
                    default:
                        std::cerr << "Invalid optimization level. Using default (O0)." << std::endl;
                        break;
                }
                break;
            case 'a':
                targetArch = TargetArch::ARMv7;
                break;
            case 'h':
                printUsage(argv[0]);
                return false;
            default:
                printUsage(argv[0]);
                return false;
        }
    }
    // 检查是否还有剩余的非选项参数（输入文件）
    if (optind < argc) {
        inputFilename = argv[optind];
    } else {
        std::cerr << "Error: Input file is required." << std::endl;
        printUsage(argv[0]);
        return false;
    }

    return true;
}

SysYParser::CompUnitContext* parseInputFile(const std::string& inputFilename) {
    std::ifstream inputFile(inputFilename);
    if (!inputFile.is_open()) {
        std::cerr << "Error: Cannot open input file: " << inputFilename << std::endl;
        return nullptr;
    }

    static antlr4::ANTLRInputStream input(inputFile);
    static SysYLexer lexer(&input);
    static antlr4::CommonTokenStream tokens(&lexer);
    static SysYParser parser(&tokens);

    return parser.compUnit();
}

void outputToFileOrConsole(const std::string& content, const std::string& filename) {
    if (!filename.empty()) {
        std::ofstream outFile(filename);
        if (outFile.is_open()) {
            outFile << content;
            outFile.close();
        } else {
            std::cerr << "Error: Unable to open output file: " << filename << std::endl;
        }
    } else {
        std::cout << content;
    }
}

mlir::LogicalResult lowerToLLVM(mlir::ModuleOp& module) {
    mlir::MLIRContext* context = module->getContext();
    mlir::PassManager pm(context);
    module->dump();

    pm.addPass(mlir::createCanonicalizerPass());
    pm.run(module);
    std::cout << std::endl << "After CanonicalizerPass1:" << std::endl;
    module->dump();
    pm.clear();

    pm.addPass(mlir::createSymbolDCEPass());
    pm.run(module);
    std::cout << std::endl << "After SymbolDCEPass1:" << std::endl;
    module->dump();
    pm.clear();

    pm.addPass(mlir::createInlinerPass());
    pm.run(module);
    std::cout << std::endl << "After InlinerPass:" << std::endl;
    module->dump();
    pm.clear();

    pm.addNestedPass<mlir::func::FuncOp>(mlir::sysy::createShapeInferencePass());
    pm.run(module);
    std::cout << std::endl << "After ShapeInferencePass:" << std::endl;
    module->dump();
    pm.clear();

    pm.addNestedPass<mlir::func::FuncOp>(mlir::sysy::createSimplifyRedundantTransposePass());
    pm.run(module);
    std::cout << std::endl << "After SimplifyRedundantTransposePass:" << std::endl;
    module->dump();
    pm.clear();

    pm.addNestedPass<mlir::func::FuncOp>(mlir::sysy::createDCEPass());
    pm.run(module);
    std::cout << std::endl << "After DCEPass:" << std::endl;
    module->dump();
    pm.clear();

    pm.addPass(mlir::createCanonicalizerPass());
    pm.run(module);
    std::cout << std::endl << "After CanonicalizerPass2:" << std::endl;
    module->dump();
    pm.clear();

    pm.addPass(mlir::sysy::createLowerSysyPass());
    pm.run(module);
    std::cout << std::endl << "After LowerSysyPass:" << std::endl;
    module->dump();
    pm.clear();

    pm.addNestedPass<mlir::func::FuncOp>(mlir::sysy::createSysyOptPass());
    pm.run(module);
    std::cout << std::endl << "After SysyOptPass:" << std::endl;
    module->dump();
    pm.clear();

    pm.addPass(mlir::createLowerAffinePass());
    // pm.run(module);
    // std::cout << std::endl << "After LowerAffinePass:" << std::endl;
    // module->dump();
    // pm.clear();

    pm.addPass(mlir::createConvertSCFToOpenMPPass());
    // pm.run(module);
    // std::cout << std::endl << "After ConvertSCFToOpenMPPass:" << std::endl;
    // module->dump();
    // pm.clear();

    pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
    pm.addPass(mlir::createArithToLLVMConversionPass());
    pm.addPass(mlir::createConvertVectorToLLVMPass());
    pm.addPass(mlir::createConvertSCFToCFPass());
    pm.addPass(mlir::createConvertOpenMPToLLVMPass());
    pm.addPass(mlir::createConvertControlFlowToLLVMPass());
    pm.addPass(mlir::createConvertFuncToLLVMPass());
    pm.addPass(mlir::createConvertIndexToLLVMPass());
    pm.addPass(mlir::createReconcileUnrealizedCastsPass());
    // pm.run(module);
    // std::cout << std::endl << "After LowerToLLVM:" << std::endl;
    // module->dump();
    // pm.clear();

    // pm.addNestedPass<mlir::func::FuncOp>(mlir::affine::createLoopFusionPass(
    //     0, 0, false, mlir::affine::FusionMode::Greedy));
    // pm.addNestedPass<mlir::func::FuncOp>(mlir::affine::createAffineScalarReplacementPass());
    // pm.run(module);
    // // std::cout << std::endl << "After LoopFusionPass:" << std::endl;
    // // module->dump();

    return pm.run(module);
}

std::unique_ptr<llvm::Module> translateToLLVMIR(mlir::ModuleOp& mlirModule, llvm::LLVMContext& llvmContext) {
    mlir::MLIRContext* context = mlirModule->getContext();
    mlir::registerBuiltinDialectTranslation(*context);
    mlir::registerLLVMDialectTranslation(*context);
    mlir::registerOpenMPDialectTranslation(*context);
    return mlir::translateModuleToLLVMIR(mlirModule, llvmContext);
}

void applyOptimization(llvm::Module &module, llvm::OptimizationLevel optLevel) {
    // 创建 pass pipeline
    llvm::LoopAnalysisManager LAM;
    llvm::FunctionAnalysisManager FAM;
    llvm::CGSCCAnalysisManager CGAM;
    llvm::ModuleAnalysisManager MAM;

    llvm::PassBuilder PB;

    // 注册所有需要的分析
    PB.registerModuleAnalyses(MAM);
    PB.registerCGSCCAnalyses(CGAM);
    PB.registerFunctionAnalyses(FAM);
    PB.registerLoopAnalyses(LAM);
    PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

    // 创建优化 pipeline
    llvm::ModulePassManager MPM = PB.buildPerModuleDefaultPipeline(optLevel);

    // 运行优化
    MPM.run(module, MAM);
}

bool compileToObjectFile(llvm::Module& module, const std::string& outputFilename, TargetArch targetArch) {
    // 初始化LLVM目标
    llvm::InitializeAllTargetInfos();
    llvm::InitializeAllTargets();
    llvm::InitializeAllTargetMCs();
    llvm::InitializeAllAsmParsers();
    llvm::InitializeAllAsmPrinters();

    std::string targetTriple;
    if (targetArch == TargetArch::ARMv7){
        targetTriple = "armv7l-linux-gnueabihf";
        // targetTriple = "aarch64-linux-gnu";
        module.setDataLayout("e-m:e-p:32:32-i32:32-v128:64:128-a:0:32-n32-S32");
    }
    else
        targetTriple = llvm::sys::getDefaultTargetTriple();
    module.setTargetTriple(targetTriple);

    std::string error;
    const llvm::Target *target = llvm::TargetRegistry::lookupTarget(targetTriple, error);

    if (!target) {
        llvm::errs() << error;
        return false;
    }

    llvm::TargetOptions opt;
    opt.FloatABIType = llvm::FloatABI::Hard;  // 设置硬浮点 ABI
    auto RM = std::optional<llvm::Reloc::Model>(llvm::Reloc::PIC_);
    auto CM = std::optional<llvm::CodeModel::Model>(llvm::CodeModel::Small);
    llvm::TargetMachine* targetMachine;
    if (targetArch == TargetArch::ARMv7) {
        targetMachine = target->createTargetMachine(
            targetTriple, 
            "cortex-a72", 
            "+neon,+vfp4,+thumb2",  // armv7 特性
            // "+neon,+crypto,+crc",  // aarch64 特性
            opt, 
            RM, 
            CM
        );
    } else {
        targetMachine = target->createTargetMachine(
            targetTriple, 
            "generic", 
            "", 
            opt, 
            RM, 
            CM
        );
    }

    module.setDataLayout(targetMachine->createDataLayout());

    std::error_code EC;
    llvm::raw_fd_ostream dest(outputFilename, EC, llvm::sys::fs::OF_None);

    if (EC) {
        llvm::errs() << "Could not open file: " << EC.message();
        return false;
    }

    llvm::legacy::PassManager pass;
    auto FileType = llvm::CGFT_ObjectFile;

    if (targetMachine->addPassesToEmitFile(pass, dest, nullptr, FileType)) {
        llvm::errs() << "TargetMachine can't emit a file of this type";
        return false;
    }

    pass.run(module);
    dest.flush();

    return true;
}

void convertCallocCallsToI32(llvm::Module &M) {
    llvm::LLVMContext &Ctx = M.getContext();
    llvm::Type *I32Ty = llvm::Type::getInt32Ty(Ctx);
    
    // 找到原始的 calloc 函数
    llvm::Function *OrigCalloc = M.getFunction("calloc");
    if (!OrigCalloc)
        return;

    // 创建新的 calloc 函数类型
    llvm::FunctionType *NewCallocTy = llvm::FunctionType::get(
        OrigCalloc->getReturnType(),
        {I32Ty, I32Ty},
        false);

    // 临时重命名原始 calloc 函数
    OrigCalloc->setName("calloc.old");

    // 创建新的 calloc 函数，使用原始名称
    llvm::Function *NewCalloc = llvm::Function::Create(
        NewCallocTy,
        OrigCalloc->getLinkage(),
        "calloc",
        &M);

    // 复制原始 calloc 的属性到新的 calloc
    NewCalloc->copyAttributesFrom(OrigCalloc);

    // 替换所有 calloc 调用
    for (llvm::Function &F : M) {
        for (llvm::BasicBlock &BB : F) {
            for (auto I = BB.begin(); I != BB.end(); ) {
                if (auto *CI = llvm::dyn_cast<llvm::CallInst>(&*I)) {
                    if (CI->getCalledFunction() == OrigCalloc) {
                        llvm::IRBuilder<> Builder(CI);
                        llvm::Value *Arg1 = CI->getArgOperand(0);
                        llvm::Value *Arg2 = CI->getArgOperand(1);
                        
                        llvm::Value *NewArg1 = Builder.CreateTrunc(Arg1, I32Ty);
                        llvm::Value *NewArg2 = Builder.CreateTrunc(Arg2, I32Ty);
                        
                        llvm::CallInst *NewCI = Builder.CreateCall(NewCalloc, {NewArg1, NewArg2});
                        NewCI->setTailCall(CI->isTailCall());
                        NewCI->setAttributes(CI->getAttributes());
                        
                        CI->replaceAllUsesWith(NewCI);
                        I = CI->eraseFromParent();
                        continue;
                    }
                }
                ++I;
            }
        }
    }

    // 删除原始的 calloc 函数
    OrigCalloc->eraseFromParent();
}

int main(int argc, char* argv[]) {
    std::string inputFilename, outputFilename;
    OutputLevel outputLevel;
    llvm::OptimizationLevel optLevel;
    TargetArch targetArch;

    if (!parseCommandLineArgs(argc, argv, inputFilename, outputFilename, 
    outputLevel, optLevel, targetArch)) {
        return 1;
    }

    // 词法和语法分析
    auto root = parseInputFile(inputFilename);
    if (!root) {
        return 1;
    }

    // 创建 MLIR 上下文并加载方言
    mlir::MLIRContext context;
    mlir::DialectRegistry registry;
    context.loadDialect<mlir::sysy::SysyDialect>();
    context.loadDialect<mlir::func::FuncDialect>();
    context.loadDialect<mlir::memref::MemRefDialect>();
    context.loadDialect<mlir::arith::ArithDialect>();
    context.loadDialect<mlir::scf::SCFDialect>();
    context.loadDialect<mlir::vector::VectorDialect>();
    context.loadDialect<mlir::omp::OpenMPDialect>();
    context.loadDialect<mlir::cf::ControlFlowDialect>();
    context.loadDialect<mlir::LLVM::LLVMDialect>();
    mlir::func::registerInlinerExtension(registry);
    context.appendDialectRegistry(registry);

    std::string output;
    // 生成混合的 MLIR
    SysYVisitorImpl visitor(context);
    visitor.visitCompUnit(root);
    mlir::ModuleOp module = visitor.getModule();
    if (outputLevel == OutputLevel::MixedMLIR) {
        llvm::raw_string_ostream os(output);
        module.print(os);
        outputToFileOrConsole(output, outputFilename);
        return 0;
    }

    // 降级到 LLVM 方言
    if (mlir::failed(lowerToLLVM(module))) {
        llvm::errs() << "Lowering to LLVM dialect failed.\n";
        return 1;
    }
    if (outputLevel == OutputLevel::LLVMDialect) {
        llvm::raw_string_ostream os(output);
        module.print(os);
        outputToFileOrConsole(output, outputFilename);
        return 0;
    }

    // 翻译到 LLVM IR
    llvm::LLVMContext llvmContext;
    auto llvmModule = translateToLLVMIR(module, llvmContext);
    if (!llvmModule) {
        llvm::errs() << "Translation to LLVM IR failed.\n";
        return 1;
    }
    // 应用优化
    applyOptimization(*llvmModule, optLevel);
    // 修改calloc的参数类型
    convertCallocCallsToI32(*llvmModule);
    if (outputLevel == OutputLevel::LLVMIR) {
        llvm::raw_string_ostream os(output);
        llvmModule->print(os, nullptr);
        outputToFileOrConsole(output, outputFilename);
        return 0;
    }

    // 输出目标文件
    if (!compileToObjectFile(*llvmModule, outputFilename, targetArch)) {
        llvm::errs() << "Failed to compile to object file.\n";
        return 1;
    }
    std::cout << "Object file generated: " << outputFilename << std::endl;

    return 0;
}