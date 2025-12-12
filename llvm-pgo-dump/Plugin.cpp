// Plugin.cpp
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"

using namespace llvm;

// These are implemented in other .cpp files:
void registerPathProfPass(PassBuilder& PB);
void registerBuildBLTablePass(PassBuilder& PB);
void registerDumpBLStaticPass(PassBuilder& PB);

extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo llvmGetPassPluginInfo() {
    return {
      LLVM_PLUGIN_API_VERSION,
      "ManualBallLarusPlugin",
      "1.0",
      [](PassBuilder& PB) {
        registerPathProfPass(PB);
        registerBuildBLTablePass(PB);
        registerDumpBLStaticPass(PB);
      }
    };
}
