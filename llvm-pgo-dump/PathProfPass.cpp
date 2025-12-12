// PathProfPass.cpp
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <string>
using namespace llvm;


namespace {

    static bool isBackEdge(const DominatorTree& DT,
        const BasicBlock* Src,
        const BasicBlock* Dst) {
        // backedge if Dst dominates Src
        return DT.dominates(Dst, Src);
    }

    static void sortSuccsDeterministic(SmallVectorImpl<BasicBlock*>& Succs) {
        llvm::sort(Succs, [](BasicBlock* A, BasicBlock* B) {
            if (A->hasName() && B->hasName())
                return A->getName() < B->getName();
            if (A->hasName() != B->hasName())
                return A->hasName();
            return A < B;
            });
    }

    struct PathProfPass : public PassInfoMixin<PathProfPass> {
        PreservedAnalyses run(Function& F, FunctionAnalysisManager& FAM) {
            if (F.isDeclaration())
                return PreservedAnalyses::all();

            Module* M = F.getParent();
            LLVMContext& Ctx = M->getContext();
            auto& DT = FAM.getResult<DominatorTreeAnalysis>(F);

            // ------------------------------------------------------------
            // 1) Build DAG successors ignoring backedges
            // ------------------------------------------------------------
            DenseMap<BasicBlock*, SmallVector<BasicBlock*, 4>> DAGSuccs;
            SmallVector<BasicBlock*, 32> Nodes;
            Nodes.reserve(F.size());

            for (BasicBlock& BB : F)
                Nodes.push_back(&BB);

            for (BasicBlock* BB : Nodes) {
                SmallVector<BasicBlock*, 4> Succs;
                for (BasicBlock* S : successors(BB)) {
                    if (!isBackEdge(DT, BB, S))
                        Succs.push_back(S);
                }
                sortSuccsDeterministic(Succs);
                DAGSuccs[BB] = std::move(Succs);
            }

            // ------------------------------------------------------------
            // 2) Get a reachable topological-ish order from entry (DFS on DAG)
            //    We'll store reverse-postorder then reverse it => entry->exit.
            // ------------------------------------------------------------
            SmallVector<BasicBlock*, 64> Topo;
            DenseMap<BasicBlock*, uint8_t> Color; // 0=unseen,1=visiting,2=done

            std::function<void(BasicBlock*)> dfs = [&](BasicBlock* BB) {
                auto It = Color.find(BB);
                if (It != Color.end() && It->second != 0)
                    return;

                Color[BB] = 1;
                for (BasicBlock* S : DAGSuccs[BB])
                    dfs(S);
                Color[BB] = 2;

                Topo.push_back(BB);
                };

            dfs(&F.getEntryBlock());
            std::reverse(Topo.begin(), Topo.end()); // now entry->exit

            if (Topo.empty()) {
                // Shouldn't happen, but be safe.
                return PreservedAnalyses::all();
            }

            // ------------------------------------------------------------
            // 3) DP: NumPaths[BB] = sum NumPaths[Succ] + (return?1:0)
            //    IMPORTANT: compute in exit->entry direction.
            // ------------------------------------------------------------
            DenseMap<BasicBlock*, uint64_t> NumPaths;

            for (auto It = Topo.rbegin(); It != Topo.rend(); ++It) {
                BasicBlock* BB = *It;

                uint64_t Sum = 0;
                for (BasicBlock* S : DAGSuccs[BB]) {
                    auto ItNP = NumPaths.find(S);
                    if (ItNP != NumPaths.end())
                        Sum += ItNP->second;
                }

                if (isa<ReturnInst>(BB->getTerminator()))
                    Sum += 1;

                // Robustness: if dead-end in DAG (no succ) and not return, still give 1.
                if (Sum == 0)
                    Sum = 1;

                NumPaths[BB] = Sum;
            }

            uint64_t TotalPaths = 1;
            {
                auto ItE = NumPaths.find(&F.getEntryBlock());
                if (ItE != NumPaths.end() && ItE->second != 0)
                    TotalPaths = ItE->second;
            }

            // guard path explosion
            if (TotalPaths > 5'000'000ULL) {
                errs() << "[PathProfPass] Too many paths in " << F.getName() << ": "
                    << TotalPaths << " (skip)\n";
                return PreservedAnalyses::all();
            }

            // ------------------------------------------------------------
            // 4) Ball–Larus weights for DAG edges + exit edge
            // ------------------------------------------------------------
            struct EdgeKey {
                BasicBlock* Src;
                BasicBlock* Dst; // nullptr means exit
                bool operator==(const EdgeKey& O) const { return Src == O.Src && Dst == O.Dst; }
            };
            struct EdgeKeyInfo {
                static inline EdgeKey getEmptyKey() { return { nullptr, (BasicBlock*)1 }; }
                static inline EdgeKey getTombstoneKey() { return { nullptr, (BasicBlock*)2 }; }
                static unsigned getHashValue(const EdgeKey& K) {
                    return (unsigned)((uintptr_t)K.Src * 1315423911u) ^
                        (unsigned)((uintptr_t)K.Dst * 2654435761u);
                }
                static bool isEqual(const EdgeKey& A, const EdgeKey& B) { return A == B; }
            };

            DenseMap<EdgeKey, uint64_t, EdgeKeyInfo> Weight;

            for (BasicBlock* BB : Topo) {
                uint64_t Acc = 0;
                for (BasicBlock* S : DAGSuccs[BB]) {
                    Weight[{BB, S}] = Acc;

                    // NumPaths[S] must exist because S is reachable (but still be safe)
                    uint64_t NP = 0;
                    auto ItNP = NumPaths.find(S);
                    if (ItNP != NumPaths.end())
                        NP = ItNP->second;

                    Acc += NP;
                }

                if (isa<ReturnInst>(BB->getTerminator())) {
                    Weight[{BB, nullptr}] = Acc; // exit weight
                }
            }

            // ------------------------------------------------------------
            // 5) Create per-function counts global: [TotalPaths x i64]
            // ------------------------------------------------------------
            Type* I64 = Type::getInt64Ty(Ctx);
            Type* I32 = Type::getInt32Ty(Ctx);

            ArrayType* CountsArrTy = ArrayType::get(I64, (uint64_t)TotalPaths);
            std::string CountsName = "__bl_counts_" + F.getName().str();

            auto* CountsGV = new GlobalVariable(
                *M, CountsArrTy, /*isConstant=*/false, GlobalValue::InternalLinkage,
                ConstantAggregateZero::get(CountsArrTy), CountsName);

            // ------------------------------------------------------------
            // 6) Path register at entry: i64 __bl_path_id = 0
            // ------------------------------------------------------------
            IRBuilder<> EntryB(&*F.getEntryBlock().getFirstInsertionPt());
            AllocaInst* PathReg = EntryB.CreateAlloca(I64, nullptr, "__bl_path_id");
            EntryB.CreateStore(ConstantInt::get(I64, 0), PathReg);

            auto addToPathReg = [&](IRBuilder<>& B, uint64_t W) {
                if (W == 0)
                    return;
                Value* Cur = B.CreateLoad(I64, PathReg);
                Value* Add = B.CreateAdd(Cur, ConstantInt::get(I64, W));
                B.CreateStore(Add, PathReg);
                };

            auto bumpCountAndReset = [&](IRBuilder<>& B) {
                Value* Pid64 = B.CreateLoad(I64, PathReg);
                Value* Pid32 = B.CreateTrunc(Pid64, I32);

                Value* Idxs[2] = { ConstantInt::get(I32, 0), Pid32 };
                Value* Ptr = B.CreateInBoundsGEP(CountsArrTy, CountsGV, Idxs);

                Value* Old = B.CreateLoad(I64, Ptr);
                Value* NewV = B.CreateAdd(Old, ConstantInt::get(I64, 1));
                B.CreateStore(NewV, Ptr);

                B.CreateStore(ConstantInt::get(I64, 0), PathReg);
                };

            // ------------------------------------------------------------
            // 7) Instrument edges:
            //    - forward edges: SplitEdge + add weight
            //    - backedges: SplitEdge + bump+reset (treat loop as path boundary)
            // ------------------------------------------------------------
            for (BasicBlock* BB : Nodes) {
                for (BasicBlock* S : successors(BB)) {
                    BasicBlock* NewBB = SplitEdge(BB, S);
                    IRBuilder<> B(&*NewBB->getFirstInsertionPt());

                    if (isBackEdge(DT, BB, S)) {
                        bumpCountAndReset(B);
                        continue;
                    }

                    uint64_t W = 0;
                    auto ItW = Weight.find({ BB, S });
                    if (ItW != Weight.end())
                        W = ItW->second;

                    addToPathReg(B, W);
                }
            }

            // return edges: add exit weight then bump
            for (BasicBlock& BB : F) {
                if (auto* RI = dyn_cast<ReturnInst>(BB.getTerminator())) {
                    uint64_t ExitW = 0;
                    auto ItW = Weight.find({ &BB, nullptr });
                    if (ItW != Weight.end())
                        ExitW = ItW->second;

                    IRBuilder<> B(RI);
                    addToPathReg(B, ExitW);
                    bumpCountAndReset(B);
                }
            }

            // ------------------------------------------------------------
            // 8) Emit per-function record: __bl_rec_<func>
            //    layout: { i8* name, i64* counts, i32 npaths }
            // ------------------------------------------------------------
            Type* I8Ty = Type::getInt8Ty(Ctx);
            Type* I8PtrTy = PointerType::getUnqual(I8Ty);

            StructType* RecTy = StructType::get(I8PtrTy, PointerType::getUnqual(I64), I32);

            Constant* NameC = ConstantDataArray::getString(Ctx, F.getName(), true);
            auto* NameGV = new GlobalVariable(
                *M, NameC->getType(), /*isConstant=*/true, GlobalValue::PrivateLinkage,
                NameC, "__bl_name_" + F.getName().str());
            NameGV->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);

            Constant* NamePtr = ConstantExpr::getBitCast(NameGV, I8PtrTy);
            Constant* CountsPtr = ConstantExpr::getBitCast(CountsGV, PointerType::getUnqual(I64));
            Constant* NPathsC = ConstantInt::get(I32, (uint32_t)TotalPaths);

            Constant* RecC = ConstantStruct::get(RecTy, NamePtr, CountsPtr, NPathsC);

            (void)new GlobalVariable(*M, RecTy, /*isConstant=*/true,
                GlobalValue::InternalLinkage, RecC,
                "__bl_rec_" + F.getName().str());

            return PreservedAnalyses::none();
        }
    };

} // namespace

void registerPathProfPass(PassBuilder& PB) {
    PB.registerPipelineParsingCallback(
        [](StringRef Name, FunctionPassManager& FPM,
            ArrayRef<PassBuilder::PipelineElement>) {
                if (Name == "ball-larus-prof") {
                    FPM.addPass(PathProfPass());
                    return true;
                }
                return false;
        }
    );
}
