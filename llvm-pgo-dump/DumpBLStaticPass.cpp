// DumpBLStaticPass.cpp
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <queue>
#include <sstream>
#include <string>
#include <functional>
#include <algorithm>
#include <cstdint>
#include <vector>

using namespace llvm;

namespace {

    // Remove metadata tokens (starting with '!') from IR text.
    static std::string stripMetadataTokens(const std::string& IRText) {
        std::stringstream ss(IRText);
        std::string token;
        std::string result;

        while (ss >> token) {
            if (!token.empty() && token[0] == '!') {
                continue;
            }
            if (!result.empty())
                result.push_back(' ');
            result += token;
        }
        return result;
    }

    static bool isBackEdge(const DominatorTree& DT,
        const BasicBlock* Src,
        const BasicBlock* Dst) {
        // Backedge if Dst dominates Src
        return DT.dominates(Dst, Src);
    }

    // MUST match PathProfPass successor ordering
    static void sortSuccsDeterministic(SmallVectorImpl<const BasicBlock*>& Succs) {
        llvm::sort(Succs, [](const BasicBlock* A, const BasicBlock* B) {
            if (A->hasName() && B->hasName())
                return A->getName() < B->getName();
            if (A->hasName() != B->hasName())
                return A->hasName();
            return A < B;
            });
    }

    struct DumpBLStaticPass : public PassInfoMixin<DumpBLStaticPass> {

        PreservedAnalyses run(Function& F, FunctionAnalysisManager& FAM) {
            if (F.isDeclaration())
                return PreservedAnalyses::all();

            auto& LI = FAM.getResult<LoopAnalysis>(F);
            auto& DT = FAM.getResult<DominatorTreeAnalysis>(F);

            Module* M = F.getParent();
            StringRef SourceName = M->getSourceFileName();
            StringRef FuncName = F.getName();

            // --- Precompute distance from entry block (BFS on CFG) ---
            DenseMap<const BasicBlock*, unsigned> BBDistance;
            {
                std::queue<const BasicBlock*> Q;
                const BasicBlock* Entry = &F.getEntryBlock();
                BBDistance[Entry] = 0;
                Q.push(Entry);

                while (!Q.empty()) {
                    const BasicBlock* BB = Q.front();
                    Q.pop();
                    unsigned CurDist = BBDistance[BB];

                    const auto* TI = BB->getTerminator();
                    if (!TI) continue;

                    for (unsigned i = 0; i < TI->getNumSuccessors(); ++i) {
                        const BasicBlock* Succ = TI->getSuccessor(i);
                        auto It = BBDistance.find(Succ);
                        if (It == BBDistance.end() || It->second > CurDist + 1) {
                            BBDistance[Succ] = CurDist + 1;
                            Q.push(Succ);
                        }
                    }
                }
            }

            // Helper to compute dominator-tree depth of a block.
            auto computeDomDepth = [&DT](BasicBlock* BB) -> unsigned {
                const DomTreeNode* Node = DT.getNode(BB);
                unsigned Depth = 0;
                while (Node) {
                    ++Depth;
                    Node = Node->getIDom();
                }
                return Depth;
                };

            // --- Per-BB static info (NO dynamic count here) ---
            struct BBData {
                int Index = -1;

                int InstCount = 0;
                int BranchCount = 0;
                int CallCount = 0;

                unsigned LoopDepth = 0;
                unsigned InLoop = 0;
                unsigned NumSucc = 0;
                unsigned NumPreds = 0;
                unsigned DistFromEntry = 0;
                unsigned DomDepth = 0;

                unsigned IntOperands = 0;
                unsigned FPOperands = 0;
                unsigned PtrOperands = 0;
                unsigned VectorOperands = 0;
                unsigned PhiIncoming = 0;

                std::array<unsigned, Instruction::OtherOpsEnd> OpcodeCounts{};
                std::string IR;
            };

            DenseMap<const BasicBlock*, BBData> Info;

            int bbIndex = 0;
            for (BasicBlock& BB : F) {
                auto* BBPtr = &BB;
                BBData Data;
                Data.Index = bbIndex++;
                Data.OpcodeCounts.fill(0);

                for (Instruction& I : BB) {
                    ++Data.InstCount;

                    unsigned Opc = I.getOpcode();
                    if (Opc < Instruction::OtherOpsEnd)
                        ++Data.OpcodeCounts[Opc];

                    if (isa<BranchInst>(I))
                        ++Data.BranchCount;
                    if (isa<CallBase>(I))
                        ++Data.CallCount;

                    for (Use& U : I.operands()) {
                        Type* Ty = U->getType();
                        if (!Ty) continue;

                        if (Ty->isIntegerTy()) ++Data.IntOperands;
                        else if (Ty->isFloatingPointTy()) ++Data.FPOperands;
                        else if (Ty->isPointerTy()) ++Data.PtrOperands;
                        else if (Ty->isVectorTy()) ++Data.VectorOperands;
                    }

                    if (auto* Phi = dyn_cast<PHINode>(&I))
                        Data.PhiIncoming += Phi->getNumIncomingValues();
                }

                if (auto* L = LI.getLoopFor(&BB)) {
                    Data.InLoop = 1;
                    Data.LoopDepth = L->getLoopDepth();
                }

                if (auto* TI = BB.getTerminator())
                    Data.NumSucc = TI->getNumSuccessors();

                Data.NumPreds = pred_size(&BB);

                if (auto It = BBDistance.find(&BB); It != BBDistance.end())
                    Data.DistFromEntry = It->second;

                Data.DomDepth = computeDomDepth(&BB);

                // IR text of BB
                {
                    std::string IRText;
                    raw_string_ostream RSO(IRText);
                    for (Instruction& I : BB) {
                        I.print(RSO, /*IsForDebug=*/false);
                        RSO << " ";
                    }
                    RSO.flush();

                    IRText = stripMetadataTokens(IRText);
                    for (char& c : IRText) {
                        if (c == '"') c = '\'';
                        if (c == '\n' || c == '\r') c = ' ';
                    }
                    Data.IR = std::move(IRText);
                }

                Info[BBPtr] = std::move(Data);
            }

            const BasicBlock* Entry = &F.getEntryBlock();
            if (!Info.count(Entry))
                return PreservedAnalyses::all();

            // ------------------------------------------------------------
            // Rebuild Ball–Larus DAG (ignore backedges), match PathProfPass
            // ------------------------------------------------------------
            DenseMap<const BasicBlock*, SmallVector<const BasicBlock*, 4>> DAGSuccs;
            SmallVector<const BasicBlock*, 64> Nodes;
            Nodes.reserve(F.size());
            for (BasicBlock& BB : F)
                Nodes.push_back(&BB);

            // Also collect loop headers (backedge destinations) as possible "segment starts"
            SmallPtrSet<const BasicBlock*, 32> SegmentStarts;
            SegmentStarts.insert(Entry);

            // Track which blocks have outgoing backedges (these become "segment terminals" too)
            SmallPtrSet<const BasicBlock*, 32> HasOutBackEdge;

            for (const BasicBlock* BB : Nodes) {
                SmallVector<const BasicBlock*, 4> Succs;
                for (const BasicBlock* S : successors(BB)) {
                    if (isBackEdge(DT, BB, S)) {
                        SegmentStarts.insert(S);     // after reset, next segment starts at header
                        HasOutBackEdge.insert(BB);   // taking backedge ends a segment
                        continue;
                    }
                    Succs.push_back(S);
                }
                sortSuccsDeterministic(Succs);
                DAGSuccs[BB] = std::move(Succs);
            }

            // ------------------------------------------------------------
            // Topological order of DAG (DFS from entry only is OK for weights;
            // but for safety, we compute topo from entry and reachable nodes).
            // ------------------------------------------------------------
            SmallVector<const BasicBlock*, 128> Topo;
            DenseMap<const BasicBlock*, uint8_t> Color;

            std::function<void(const BasicBlock*)> dfsTopo = [&](const BasicBlock* BB) {
                auto It = Color.find(BB);
                if (It != Color.end() && It->second != 0) return;
                Color[BB] = 1;
                for (const BasicBlock* S : DAGSuccs[BB]) dfsTopo(S);
                Color[BB] = 2;
                Topo.push_back(BB);
                };

            dfsTopo(Entry);
            std::reverse(Topo.begin(), Topo.end());

            // ------------------------------------------------------------
            // DP: NumPaths[BB] = sum NumPaths[Succ] (+1 if return)
            // MUST match PathProfPass logic
            // ------------------------------------------------------------
            DenseMap<const BasicBlock*, uint64_t> NumPaths;
            for (const BasicBlock* BB : Topo) {
                uint64_t Sum = 0;
                for (const BasicBlock* S : DAGSuccs[BB]) {
                    auto It = NumPaths.find(S);
                    if (It != NumPaths.end()) Sum += It->second;
                }
                if (isa<ReturnInst>(BB->getTerminator()))
                    Sum += 1;
                if (Sum == 0)
                    Sum = 1;
                NumPaths[BB] = Sum;
            }

            // Edge weights (BB->succ[i]) = sum_{j<i} NumPaths[succ[j]]
            // Exit weight at return BB = sum_{all succ} NumPaths[succ]
            struct EdgeKey {
                const BasicBlock* Src;
                const BasicBlock* Dst; // nullptr means exit at Src
                bool operator==(const EdgeKey& O) const { return Src == O.Src && Dst == O.Dst; }
            };
            struct EdgeKeyInfo {
                static inline EdgeKey getEmptyKey() { return { nullptr, (const BasicBlock*)1 }; }
                static inline EdgeKey getTombstoneKey() { return { nullptr, (const BasicBlock*)2 }; }
                static unsigned getHashValue(const EdgeKey& K) {
                    return (unsigned)((uintptr_t)K.Src * 1315423911u) ^
                        (unsigned)((uintptr_t)K.Dst * 2654435761u);
                }
                static bool isEqual(const EdgeKey& A, const EdgeKey& B) { return A == B; }
            };

            DenseMap<EdgeKey, uint64_t, EdgeKeyInfo> Weight;

            for (const BasicBlock* BB : Topo) {
                uint64_t Acc = 0;
                for (const BasicBlock* S : DAGSuccs[BB]) {
                    Weight[{BB, S}] = Acc;
                    Acc += NumPaths[S];
                }
                if (isa<ReturnInst>(BB->getTerminator()))
                    Weight[{BB, nullptr}] = Acc;
            }

            // ------------------------------------------------------------
            // CSV header (printed once globally)
            // Output key: program,function,path_id
            // ------------------------------------------------------------
            static bool PrintedHeader = false;
            if (!PrintedHeader) {
                errs()
                    << "program,function,path_id,path_len,path_ir"
                    << ",inst_count"
                    << ",branch_count"
                    << ",call_count"
                    << ",loop_depth"
                    << ",in_loop"
                    << ",num_succ"
                    << ",num_preds"
                    << ",dist_from_entry"
                    << ",dom_depth"
                    << ",int_operands"
                    << ",fp_operands"
                    << ",ptr_operands"
                    << ",vector_operands"
                    << ",phi_incoming";

                for (unsigned op = 0; op < Instruction::OtherOpsEnd; ++op) {
                    errs() << ",op_" << Instruction::getOpcodeName(op);
                }
                errs() << "\n";
                PrintedHeader = true;
            }

            // ------------------------------------------------------------
            // Enumerate Ball–Larus segment paths:
            // Segment starts: entry + each backedge destination (loop headers)
            // Segment ends:
            //   - Return: pid = cur + exitWeight(BB)
            //   - Backedge-out block: pid = cur (because your runtime bumps+reset on backedge edge)
            // ------------------------------------------------------------
            const unsigned MaxPathsPerStart = 4000;
            const unsigned MaxPathLen = 128;

            SmallVector<const BasicBlock*, 32> Path;
            DenseMap<const BasicBlock*, uint32_t> LocalPathCountPerStart;

            auto buildArrayString = [&](auto Extractor) -> std::string {
                std::string S;
                raw_string_ostream RSO(S);
                RSO << "[";
                bool First = true;
                for (const BasicBlock* BB : Path) {
                    auto It = Info.find(BB);
                    if (It == Info.end()) continue;
                    const BBData& D = It->second;
                    if (!First) RSO << ";";
                    First = false;
                    RSO << Extractor(D);
                }
                RSO << "]";
                RSO.flush();
                return S;
                };

            auto emitRow = [&](uint64_t PathId) {
                if (Path.empty()) return;

                std::string PathIR;
                {
                    raw_string_ostream RSO(PathIR);
                    for (const BasicBlock* BB : Path) {
                        auto It = Info.find(BB);
                        if (It == Info.end()) continue;
                        RSO << It->second.IR << " ";
                    }
                    RSO.flush();
                }
                for (char& c : PathIR) {
                    if (c == '"') c = '\'';
                    if (c == '\n' || c == '\r') c = ' ';
                }

                std::string InstCountArr = buildArrayString([](const BBData& D) { return D.InstCount; });
                std::string BranchCountArr = buildArrayString([](const BBData& D) { return D.BranchCount; });
                std::string CallCountArr = buildArrayString([](const BBData& D) { return D.CallCount; });
                std::string LoopDepthArr = buildArrayString([](const BBData& D) { return D.LoopDepth; });
                std::string InLoopArr = buildArrayString([](const BBData& D) { return D.InLoop; });
                std::string NumSuccArr = buildArrayString([](const BBData& D) { return D.NumSucc; });
                std::string NumPredsArr = buildArrayString([](const BBData& D) { return D.NumPreds; });
                std::string DistFromEntryArr = buildArrayString([](const BBData& D) { return D.DistFromEntry; });
                std::string DomDepthArr = buildArrayString([](const BBData& D) { return D.DomDepth; });
                std::string IntOperandsArr = buildArrayString([](const BBData& D) { return D.IntOperands; });
                std::string FPOperandsArr = buildArrayString([](const BBData& D) { return D.FPOperands; });
                std::string PtrOperandsArr = buildArrayString([](const BBData& D) { return D.PtrOperands; });
                std::string VectorOperandsArr = buildArrayString([](const BBData& D) { return D.VectorOperands; });
                std::string PhiIncomingArr = buildArrayString([](const BBData& D) { return D.PhiIncoming; });

                std::vector<std::string> OpcodeArrays(Instruction::OtherOpsEnd);
                for (unsigned op = 0; op < Instruction::OtherOpsEnd; ++op) {
                    OpcodeArrays[op] = buildArrayString([op](const BBData& D) { return D.OpcodeCounts[op]; });
                }

                errs()
                    << SourceName << ","
                    << FuncName << ","
                    << PathId << ","
                    << (unsigned)Path.size() << ","
                    << "\"" << PathIR << "\""
                    << ",\"" << InstCountArr << "\""
                    << ",\"" << BranchCountArr << "\""
                    << ",\"" << CallCountArr << "\""
                    << ",\"" << LoopDepthArr << "\""
                    << ",\"" << InLoopArr << "\""
                    << ",\"" << NumSuccArr << "\""
                    << ",\"" << NumPredsArr << "\""
                    << ",\"" << DistFromEntryArr << "\""
                    << ",\"" << DomDepthArr << "\""
                    << ",\"" << IntOperandsArr << "\""
                    << ",\"" << FPOperandsArr << "\""
                    << ",\"" << PtrOperandsArr << "\""
                    << ",\"" << VectorOperandsArr << "\""
                    << ",\"" << PhiIncomingArr << "\"";

                for (unsigned op = 0; op < Instruction::OtherOpsEnd; ++op) {
                    errs() << ",\"" << OpcodeArrays[op] << "\"";
                }
                errs() << "\n";
                };

            // DFS over DAG from a given start, tracking current BL id
            std::function<void(const BasicBlock*, uint64_t)> dfsPaths =
                [&](const BasicBlock* BB, uint64_t CurId) {

                const BasicBlock* CurBB = BB;

                // hard limits
                if (Path.size() >= MaxPathLen) {
                    emitRow(CurId);
                    return;
                }

                // Terminal 1: return -> add exit weight then emit
                if (isa<ReturnInst>(CurBB->getTerminator())) {
                    uint64_t ExitW = 0;
                    auto ItW = Weight.find({ CurBB, nullptr });
                    if (ItW != Weight.end()) ExitW = ItW->second;
                    emitRow(CurId + ExitW);
                    return;
                }

                // Terminal 2: this block has an outgoing backedge in original CFG
                // (your runtime bumps+resets when traversing that backedge edge)
                if (HasOutBackEdge.contains(CurBB)) {
                    emitRow(CurId);
                    // NOTE: we do NOT "continue after reset" here; that will be covered by
                    // starting DFS from the backedge destination in SegmentStarts.
                    return;
                }

                // Otherwise follow DAG successors
                auto ItS = DAGSuccs.find(CurBB);
                if (ItS == DAGSuccs.end() || ItS->second.empty()) {
                    // dead-end (should be rare)
                    emitRow(CurId);
                    return;
                }

                for (const BasicBlock* S : ItS->second) {
                    uint64_t W = 0;
                    auto ItW = Weight.find({ CurBB, S });
                    if (ItW != Weight.end()) W = ItW->second;

                    Path.push_back(S);
                    dfsPaths(S, CurId + W);
                    Path.pop_back();
                }
                };

            // enumerate from each segment start
            for (const BasicBlock* Start : SegmentStarts) {
                // avoid exploding too much: only starts that are reachable from entry
                // (BBDistance exists only for reachable)
                if (!BBDistance.count(Start)) continue;

                // start path
                Path.clear();
                Path.push_back(Start);

                // in your runtime, a segment after reset starts with id=0
                dfsPaths(Start, /*CurId=*/0);
            }

            return PreservedAnalyses::all();
        }
    };

} // namespace


void registerDumpBLStaticPass(PassBuilder& PB) {
    PB.registerPipelineParsingCallback(
        [](StringRef Name, FunctionPassManager& FPM,
            ArrayRef<PassBuilder::PipelineElement>) {
                if (Name == "dump-bl-static") {
                    FPM.addPass(DumpBLStaticPass());
                    return true;
                }
                return false;
        }
    );
}
