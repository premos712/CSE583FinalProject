// BuildBLTablePass.cpp
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"

using namespace llvm;

namespace {

    struct BuildBLTablePass : public PassInfoMixin<BuildBLTablePass> {
        PreservedAnalyses run(Module& M, ModuleAnalysisManager&) {
            LLVMContext& Ctx = M.getContext();

            Type* I64 = Type::getInt64Ty(Ctx);
            Type* I32 = Type::getInt32Ty(Ctx);
            Type* I8Ty = Type::getInt8Ty(Ctx);
            Type* I8PtrTy = PointerType::getUnqual(I8Ty);

            StructType* RecTy =
                StructType::get(I8PtrTy, PointerType::getUnqual(I64), I32);

            SmallVector<Constant*, 256> Recs;

            for (GlobalVariable& GV : M.globals()) {
                if (!GV.hasName()) continue;
                if (!GV.getName().starts_with("__bl_rec_")) continue;
                if (!GV.hasInitializer()) continue;

                Constant* Init = GV.getInitializer();
                if (Init->getType() != RecTy) continue;

                Recs.push_back(Init);
            }

            ArrayType* ArrTy = ArrayType::get(RecTy, Recs.size());
            Constant* ArrInit = ConstantArray::get(ArrTy, Recs);

            if (auto* Old = M.getGlobalVariable("__bl_table"))
                Old->eraseFromParent();
            auto* TableGV = new GlobalVariable(
                M, ArrTy, /*isConstant=*/false, GlobalValue::ExternalLinkage,
                ArrInit, "__bl_table");
            TableGV->setDSOLocal(false);

            if (auto* Old = M.getGlobalVariable("__bl_table_size"))
                Old->eraseFromParent();
            auto* SizeGV = new GlobalVariable(
                M, I32, /*isConstant=*/true, GlobalValue::ExternalLinkage,
                ConstantInt::get(I32, (uint32_t)Recs.size()), "__bl_table_size");
            SizeGV->setDSOLocal(false);

            return PreservedAnalyses::none();
        }
    };

} // namespace

// THIS is what Plugin.cpp expects.
void registerBuildBLTablePass(PassBuilder& PB) {
    PB.registerPipelineParsingCallback(
        [](StringRef Name, ModulePassManager& MPM,
            ArrayRef<PassBuilder::PipelineElement>) {
                if (Name == "bl-build-table") {
                    MPM.addPass(BuildBLTablePass());
                    return true;
                }
                return false;
        });
}
