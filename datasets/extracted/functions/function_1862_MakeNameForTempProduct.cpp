#include "xla/service/gpu/llvm_gpu_backend/gpu_backend_lib.h"
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <ios>
#include <memory>
#include <mutex>  
#include <optional>
#include <string>
#include <string_view>
#include <system_error>  
#include <utility>
#include <variant>
#include <vector>
#include "absl/base/call_once.h"
#include "absl/base/const_init.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "llvm/ADT/Any.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/LazyCallGraph.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/InitializePasses.h"
#include "llvm/Linker/Linker.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/PassRegistry.h"
#include "llvm/Passes/OptimizationLevel.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/StandardInstrumentations.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#include "llvm/Transforms/IPO/Internalize.h"
#include "llvm/Transforms/Scalar.h"
#include "xla/service/gpu/llvm_gpu_backend/utils.h"
#include "xla/service/gpu/metrics.h"
#include "xla/service/llvm_ir/llvm_command_line_options.h"
#include "xla/service/llvm_ir/llvm_type_conversion_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/tsl/util/env_var.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "tsl/platform/cuda_root_path.h"
#include "tsl/platform/env.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/path.h"
#include "tsl/platform/random.h"
#include "tsl/platform/rocm_rocdl_path.h"
#include "tsl/platform/status.h"
#include "tsl/platform/statusor.h"
#include "tsl/profiler/lib/scoped_annotation.h"
#include "tsl/profiler/lib/traceme.h"
#if !defined(PLATFORM_GOOGLE) && TENSORFLOW_USE_ROCM
#include "rocm/rocm_config.h"
#endif
#if GOOGLE_CUDA
#include "third_party/gpus/cuda/include/cuda.h"
#include "xla/stream_executor/cuda/cuda_asm_compiler.h"
#endif
#if TENSORFLOW_USE_SYCL
#include "LLVMSPIRVLib.h"
#include "LLVMSPIRVOpts.h"
#endif  
namespace xla {
namespace gpu {
namespace {
static llvm::codegen::RegisterCodeGenFlags CGF;
const int kAMDGPUInlineThreshold = 0x100000;
const int kDefaultInlineThreshold = 1100;
std::string MakeNameForTempProduct(absl::string_view input_filename,
                                   absl::string_view extension) {
  return ReplaceFilenameExtension(tsl::io::Basename(input_filename), extension);
}
void InitializePasses(llvm::PassRegistry* pass_registry) {
  llvm::initializeCore(*pass_registry);
  llvm::initializeCodeGen(*pass_registry);
  llvm::initializeScalarOpts(*pass_registry);
  llvm::initializeVectorization(*pass_registry);
  llvm::initializeIPO(*pass_registry);
  llvm::initializeAnalysis(*pass_registry);
  llvm::initializeTransformUtils(*pass_registry);
  llvm::initializeInstCombine(*pass_registry);
  llvm::initializeTarget(*pass_registry);
  llvm::initializeCodeGenPrepareLegacyPassPass(*pass_registry);
}
std::unique_ptr<llvm::TargetMachine> GetTargetMachine(
    llvm::Triple triple, absl::string_view cpu_name,
    const DebugOptions& debug_options, absl::string_view feature_str) {
  std::string error;
  const llvm::Target* target =
      llvm::TargetRegistry::lookupTarget("", triple, error);
  if (target == nullptr) {
    LOG(FATAL) << "Unable to find Target for triple '" << triple.str() << "'"
               << " -- " << error;
    return nullptr;
  }
  llvm::TargetOptions target_options =
      llvm::codegen::InitTargetOptionsFromCodeGenFlags(llvm::Triple());
  target_options.MCOptions.AsmVerbose = false;
  llvm::CodeGenOptLevel codegen_opt_level;
  switch (debug_options.xla_backend_optimization_level()) {
    case 1:
      codegen_opt_level = llvm::CodeGenOptLevel::Less;
      break;
    case 2:
      codegen_opt_level = llvm::CodeGenOptLevel::Default;
      break;
    case 3:
      codegen_opt_level = llvm::CodeGenOptLevel::Aggressive;
      break;
    default:
      codegen_opt_level = llvm::CodeGenOptLevel::None;
  }
  return absl::WrapUnique(target->createTargetMachine(
      triple.str(), llvm_ir::AsStringRef(cpu_name),
      llvm_ir::AsStringRef(feature_str), target_options,
      llvm::codegen::getExplicitRelocModel(),
      llvm::codegen::getExplicitCodeModel(), codegen_opt_level));
}
std::string EmitModuleToPTX(llvm::Module* module,
                            llvm::TargetMachine* target_machine) {
  tsl::profiler::ScopedAnnotation annotation([&] {
    return absl::StrFormat("XlaEmitGpuAsm:#module=%s#",
                           module->getName().str());
  });
  std::string ptx;
  llvm::raw_string_ostream stream(ptx);
  llvm::buffer_ostream pstream(stream);
  llvm::legacy::PassManager pm;
  pm.add(new llvm::TargetLibraryInfoWrapperPass(
      llvm::Triple(module->getTargetTriple())));
  target_machine->addPassesToEmitFile(pm, pstream, nullptr,
                                      llvm::CodeGenFileType::AssemblyFile);
  pm.run(*module);
  return ptx;
}
void FeedLLVMWithFlags(const std::vector<std::string>& cl_opts) {
  std::vector<const char*> fake_argv = {""};
  for (const std::string& cl_opt : cl_opts) {
    fake_argv.push_back(cl_opt.c_str());
  }
  llvm::cl::ParseCommandLineOptions(fake_argv.size(), fake_argv.data());
}
bool CouldNeedDeviceBitcode(const llvm::Module& module) {
  for (const llvm::Function& function : module.functions()) {
    if (!function.isIntrinsic() && function.isDeclaration() &&
        (function.getName().starts_with("__nv_") ||
         function.getName().starts_with("__ocml_") ||
         function.getName().starts_with("__ockl_"))) {
      return true;
    }
  }
  return false;
}
absl::Status LinkWithBitcodeVector(
    llvm::Module* module, const std::vector<std::string>& bitcode_path_vector) {
  llvm::Linker linker(*module);
  for (auto& bitcode_path : bitcode_path_vector) {
    if (!tsl::Env::Default()->FileExists(bitcode_path).ok()) {
      LOG(ERROR) << "bitcode module is required by this HLO module but was "
                    "not found at "
                 << bitcode_path;
      return xla::Internal("bitcode module not found at %s", bitcode_path);
    }
    std::unique_ptr<llvm::Module> bitcode_module =
        LoadIRModule(bitcode_path, &module->getContext());
    bitcode_module->setDataLayout(module->getDataLayout());
    if (linker.linkInModule(
            std::move(bitcode_module), llvm::Linker::Flags::LinkOnlyNeeded,
            [](llvm::Module& M, const llvm::StringSet<>& GVS) {
              internalizeModule(M, [&GVS](const llvm::GlobalValue& GV) {
                return !GV.hasName() || (GVS.count(GV.getName()) == 0);
              });
            })) {
      return xla::Internal("Error linking bitcode module from %s",
                           bitcode_path);
    }
  }
  return absl::OkStatus();
}
absl::Status NVPTXTargetModuleLinker(llvm::Module* module,
                                     se::GpuComputeCapability gpu_version,
                                     const DebugOptions& debug_options,
                                     const std::string& device_bitcode_path) {
  TF_RETURN_IF_ERROR(
      nvptx::LinkLibdeviceIfNecessary(module, device_bitcode_path));
  module->addModuleFlag(llvm::Module::Override, "nvvm-reflect-ftz",
                        debug_options.xla_gpu_ftz());
  if (debug_options.xla_gpu_ftz()) {
    for (llvm::Function& fn : *module) {
      fn.addFnAttr("denormal-fp-math-f32", "preserve-sign");
    }
  }
  return absl::OkStatus();
}
std::unique_ptr<llvm::TargetMachine> NVPTXGetTargetMachine(
    llvm::Triple target_triple, se::CudaComputeCapability compute_capability,
    const DebugOptions& debug_options) {
#ifdef GOOGLE_CUDA
  absl::StatusOr<stream_executor::SemanticVersion> runtime_cuda_version =
      stream_executor::GetAsmCompilerVersion(
          debug_options.xla_gpu_cuda_data_dir());
  constexpr stream_executor::SemanticVersion kCompileTimeCudaVersion{
      CUDA_VERSION / 1000, (CUDA_VERSION / 10) % 100, CUDA_VERSION % 10};
  auto highest_supported_cuda_version = [&] {
    if (runtime_cuda_version.ok()) {
      return std::min(runtime_cuda_version.value(), kCompileTimeCudaVersion);
    }
    return kCompileTimeCudaVersion;
  }();
  auto ptx_version = nvptx::DetermineHighestSupportedPtxVersionFromCudaVersion(
      highest_supported_cuda_version);
  int highest_supported_ptx_version =
      ptx_version.major() * 10 + ptx_version.minor();
  VLOG(1) << "Targeting PTX version: " << highest_supported_ptx_version;
  std::string feature_str =
      absl::StrFormat("+ptx%d", highest_supported_ptx_version);
#else
  std::string feature_str;
#endif  
  return GetTargetMachine(target_triple, nvptx::GetSmName(compute_capability),
                          debug_options, feature_str);
}
using TargetModuleLinker =
    std::function<absl::Status(llvm::Module*, se::GpuComputeCapability,
                               const DebugOptions&, const std::string&)>;
void DumpModule(const std::string output_filename, const llvm::Module* module) {
  std::error_code ec;
  auto out = std::make_unique<llvm::raw_fd_ostream>(
      llvm::StringRef(output_filename), ec, llvm::sys::fs::OF_None);
  if (ec) {
    LOG(FATAL) << "Unable to open " << output_filename
               << " to dump LLVM IR: " << ec.message();
    return;
  }
  module->print(*out, nullptr);
  out->close();
}
const llvm::Module* GetModule(llvm::Any IR) {
  if (const auto** M = llvm::any_cast<const llvm::Module*>(&IR)) return *M;
  if (const auto** F = llvm::any_cast<const llvm::Function*>(&IR)) {
    return (*F)->getParent();
  }
  if (const auto** C = llvm::any_cast<const llvm::LazyCallGraph::SCC*>(&IR)) {
    return (*C)->begin()->getFunction().getParent();
  }
  if (const auto** L = llvm::any_cast<const llvm::Loop*>(&IR)) {
    const llvm::Function* F = (*L)->getHeader()->getParent();
    return F->getParent();
  }
  return nullptr;
}
auto DumpCallbackForModule(std::string module_identifier,
                           std::string outputs_dir) {
  int i = 0;
  return [=](llvm::StringRef pass, llvm::Any ir) mutable {
    const llvm::Module* module = GetModule(ir);
    if (!module) {
      return;
    }
    const std::string basename = ReplaceFilenameExtension(
        absl::string_view(tsl::io::Basename(module_identifier)),
        absl::StrFormat("pass-%02d.before.%s.ll", i++,
                        absl::string_view(pass.str())));
    DumpModule(tsl::io::JoinPath(outputs_dir, basename), module);
  };
}
absl::Status LinkAndOptimizeModule(
    llvm::Module* module, se::GpuComputeCapability gpu_version,
    const DebugOptions& debug_options, const std::string& device_bitcode_path,
    TargetModuleLinker module_linker, llvm::Triple default_target_triple,
    llvm::TargetMachine* target_machine, int inline_threshold) {
  tsl::profiler::ScopedAnnotation annotation([&] {
    return absl::StrFormat("XlaOptimizeLlvmIr:#module=%s#",
                           module->getName().str());
  });
  TF_RETURN_IF_ERROR(
      module_linker(module, gpu_version, debug_options, device_bitcode_path));
  llvm::LoopAnalysisManager lam;
  llvm::FunctionAnalysisManager fam;
  llvm::CGSCCAnalysisManager cgam;
  llvm::ModuleAnalysisManager mam;
  if (target_machine) {
    fam.registerPass([&] { return target_machine->getTargetIRAnalysis(); });
  }
  llvm::PipelineTuningOptions pto;
  pto.SLPVectorization = true;
  pto.InlinerThreshold = inline_threshold;
  llvm::PassInstrumentationCallbacks pic;
  llvm::StandardInstrumentations si(module->getContext(), false);
  si.registerCallbacks(pic, &mam);
  llvm::PassBuilder pb(target_machine, pto, std::nullopt, &pic);
  pb.registerModuleAnalyses(mam);
  pb.registerCGSCCAnalyses(cgam);
  pb.registerFunctionAnalyses(fam);
  pb.registerLoopAnalyses(lam);
  pb.crossRegisterProxies(lam, fam, cgam, mam);
  if (debug_options.xla_gpu_dump_llvmir()) {
    std::string outputs_dir;
    if (!tsl::io::GetTestUndeclaredOutputsDir(&outputs_dir)) {
      outputs_dir = debug_options.xla_dump_to();
    }
    if (!outputs_dir.empty()) {
      pic.registerBeforeNonSkippedPassCallback(
          DumpCallbackForModule(module->getModuleIdentifier(), outputs_dir));
    } else {
      LOG(ERROR) << "--xla_gpu_dump_llvmir is set, but neither the environment "
                 << "variable TEST_UNDECLARED_OUTPUTS_DIR nor the flag "
                 << "--xla_dump_to is set, so the llvm dumps are disabled.";
    }
  }
  llvm::OptimizationLevel ol;
  switch (debug_options.xla_backend_optimization_level()) {
    case 0:
      ol = llvm::OptimizationLevel::O0;
      break;
    case 1:
      ol = llvm::OptimizationLevel::O1;
      break;
    case 2:
      ol = llvm::OptimizationLevel::O2;
      break;
    case 3:
      ol = llvm::OptimizationLevel::O3;
      break;
  }
  llvm::ModulePassManager mpm;
  mpm.addPass(llvm::VerifierPass());
  if (ol == llvm::OptimizationLevel::O0) {
    mpm.addPass(pb.buildO0DefaultPipeline(ol));
  } else {
    mpm.addPass(pb.buildPerModuleDefaultPipeline(ol));
  }
  mpm.addPass(llvm::VerifierPass());
  mpm.run(*module, mam);
  return absl::OkStatus();
}
void NVPTXBackendInit(const DebugOptions& debug_options) {
  FeedLLVMWithFlags({"-bonus-inst-threshold=2"});
  FeedLLVMWithFlags({"-nvptx-prec-divf32=1"});
  FeedLLVMWithFlags({
      "-slp-vectorize-hor=false",
      "-slp-max-reg-size=32",
  });
  llvm_ir::InitializeLLVMCommandLineOptions(
      debug_options.xla_backend_extra_options());
  LLVMInitializeNVPTXTarget();
  LLVMInitializeNVPTXTargetInfo();
  LLVMInitializeNVPTXTargetMC();
  LLVMInitializeNVPTXAsmPrinter();
  llvm::PassRegistry* registry = llvm::PassRegistry::getPassRegistry();
  InitializePasses(registry);
}
}  
namespace nvptx {
std::string GetSmName(se::CudaComputeCapability compute_capability) {
  int compute_capability_version =
      compute_capability.major * 10 + compute_capability.minor;
  int sm_version = 30;
  int supported_versions[] = {90, 89, 87, 86, 80, 75, 72, 70, 62,
                              61, 60, 53, 52, 50, 37, 35, 32, 30};
  for (int v : supported_versions) {
    if (v <= compute_capability_version) {
      sm_version = v;
      break;
    }
  }
  if (sm_version != compute_capability_version &&
      compute_capability_version < supported_versions[0]) {
    LOG(WARNING) << "Unknown compute capability "
                 << compute_capability.ToString()
                 << ". Defaulting to telling LLVM that we're compiling for sm_"
                 << sm_version;
  }
  std::string_view extension =
      (compute_capability.major == 9 && sm_version == 90) ? "a" : "";
  return absl::StrCat("sm_", sm_version, extension);
}
std::string CantFindCudaMessage(absl::string_view msg,
                                absl::string_view xla_gpu_cuda_data_dir) {
  return absl::StrCat(
      msg, "\nSearched for CUDA in the following directories:\n  ",
      absl::StrJoin(tsl::CandidateCudaRoots(std::string{xla_gpu_cuda_data_dir}),
                    "\n  "),
      "\nYou can choose the search directory by setting xla_gpu_cuda_data_dir "
      "in HloModule's DebugOptions.  For most apps, setting the environment "
      "variable XLA_FLAGS=--xla_gpu_cuda_data_dir=/path/to/cuda will work.");
}
static std::string GetLibdeviceDir(absl::string_view xla_gpu_cuda_data_dir) {
  for (const std::string& cuda_root :
       tsl::CandidateCudaRoots(std::string{xla_gpu_cuda_data_dir})) {
    std::string libdevice_dir =
        tsl::io::JoinPath(cuda_root, "nvvm", "libdevice");
    VLOG(2) << "Looking for libdevice at " << libdevice_dir;
    if (tsl::Env::Default()->IsDirectory(libdevice_dir).ok()) {
      VLOG(2) << "Found libdevice dir " << libdevice_dir;
      return libdevice_dir;
    }
  }
  LOG(WARNING) << CantFindCudaMessage(
      "Can't find libdevice directory ${CUDA_DIR}/nvvm/libdevice. This may "
      "result in compilation or runtime failures, if the program we try to run "
      "uses routines from libdevice.",
      xla_gpu_cuda_data_dir);
  return ".";
}
std::string LibDevicePath(absl::string_view xla_gpu_cuda_data_dir) {
  static absl::Mutex libdevice_cache_mu(absl::kConstInit);
  static auto& libdevice_dir_path_cache ABSL_GUARDED_BY(libdevice_cache_mu) =
      *new absl::flat_hash_map<std::string, std::string>();
  std::string libdevice_dir_path = [&] {
    absl::MutexLock l(&libdevice_cache_mu);
    auto it = libdevice_dir_path_cache.find(xla_gpu_cuda_data_dir);
    if (it != libdevice_dir_path_cache.end()) {
      return it->second;
    }
    auto [it2, inserted] = libdevice_dir_path_cache.emplace(
        xla_gpu_cuda_data_dir, GetLibdeviceDir(xla_gpu_cuda_data_dir));
    return it2->second;
  }();
  return tsl::io::JoinPath(libdevice_dir_path, "libdevice.10.bc");
}
absl::Status LinkLibdeviceIfNecessary(llvm::Module* module,
                                      const std::string& libdevice_path) {
  if (!CouldNeedDeviceBitcode(*module)) {
    return absl::OkStatus();
  }
  if (!tsl::Env::Default()->FileExists(libdevice_path).ok()) {
    LOG(WARNING)
        << "libdevice is required by this HLO module but was not found at "
        << libdevice_path;
    return xla::Internal("libdevice not found at %s", libdevice_path);
  }
  VLOG(1) << "Linking with libdevice from: " << libdevice_path;
  return LinkWithBitcodeVector(module, {libdevice_path});
}
absl::StatusOr<std::string> CompileToPtx(
    llvm::Module* module, se::GpuComputeCapability gpu_version,
    const DebugOptions& debug_options,
    std::function<void(llvm::TargetMachine*)> configure_target) {
  static absl::once_flag backend_init_flag;
  absl::call_once(backend_init_flag, NVPTXBackendInit, debug_options);
  std::string ptx;
  std::unique_ptr<llvm::TargetMachine> target_machine;
  {
    tsl::profiler::TraceMe activity(
        [&] { return absl::StrCat("Compiling IR:", module->getName().str()); },
        tsl::profiler::TraceMeLevel::kInfo);
    XLA_SCOPED_LOGGING_TIMER("Compile module " + module->getName().str());
    if (module->empty() && module->global_empty()) {
      VLOG(2) << "Module '" << module->getName().str()
              << "' is empty. Skipping compilation.";
      return std::string();
    }
    auto compute_capability =
        std::get_if<se::CudaComputeCapability>(&gpu_version);
    if (!compute_capability) {
      return xla::Internal("Incompatible compute capability was specified.");
    }
    llvm::Triple default_target_triple("nvptx64-unknown-unknown");
    std::unique_ptr<llvm::TargetMachine> target_machine = NVPTXGetTargetMachine(
        default_target_triple, *compute_capability, debug_options);
    if (configure_target) {
      configure_target(target_machine.get());
    }
    uint64_t start_usecs = tsl::Env::Default()->NowMicros();
    TF_RETURN_IF_ERROR(LinkAndOptimizeModule(
        module, gpu_version, debug_options,
        LibDevicePath(debug_options.xla_gpu_cuda_data_dir()),
        NVPTXTargetModuleLinker, default_target_triple, target_machine.get(),
        kDefaultInlineThreshold));
    uint64_t end_usecs = tsl::Env::Default()->NowMicros();
    RecordLlvmPassesDuration(end_usecs - start_usecs);
    start_usecs = tsl::Env::Default()->NowMicros();
    ptx = EmitModuleToPTX(module, target_machine.get());
    end_usecs = tsl::Env::Default()->NowMicros();
    RecordLlvmToPtxDuration(end_usecs - start_usecs);
  }
  return ptx;
}
namespace {
constexpr stream_executor::SemanticVersion kFallbackPtxVersion{6, 5, 0};
constexpr stream_executor::SemanticVersion kMaxPtxVersion{8, 5, 0};
}  
stream_executor::SemanticVersion
DetermineHighestSupportedPtxVersionFromCudaVersion(
    stream_executor::SemanticVersion cuda_version) {
  if (cuda_version < stream_executor::SemanticVersion{11, 0, 0}) {
    return kFallbackPtxVersion;
  }
  if (cuda_version < stream_executor::SemanticVersion{12, 6, 0}) {
    return {cuda_version.major() - 4, cuda_version.minor(), 0};
  }
  return kMaxPtxVersion;
}
}  
namespace {
std::vector<std::string> GetROCDLPaths(std::string gcn_arch_name,
                                       const std::string& rocdl_dir_path) {
  static std::vector<std::string>* rocdl_filenames =
      new std::vector<std::string>(
          {"opencl.bc", "ocml.bc", "ockl.bc", "oclc_finite_only_off.bc",
           "oclc_daz_opt_off.bc", "oclc_correctly_rounded_sqrt_on.bc",
           "oclc_unsafe_math_off.bc", "oclc_wavefrontsize64_on.bc",
           "oclc_abi_version_500.bc"});
  std::vector<std::string> result;
  result.reserve(rocdl_filenames->size() + 1);
  for (auto& filename : *rocdl_filenames) {
    result.push_back(tsl::io::JoinPath(rocdl_dir_path, filename));
  }
  std::vector<std::string> tokens = absl::StrSplit(gcn_arch_name, ':');
  std::string amdgpu_version = gcn_arch_name;
  if (!tokens.empty() && tokens[0].size() >= 3) {
    amdgpu_version = tokens[0].substr(3);
  }
  result.push_back(tsl::io::JoinPath(
      rocdl_dir_path,
      absl::StrCat("oclc_isa_version_", amdgpu_version, ".bc")));
  return result;
}
struct HsacoCacheEntry {
  uint64_t hash;
  std::string ir;
  std::string gfx;
  std::vector<uint8_t> hsaco;
};
struct HsacoCache {
 protected:
  std::vector<HsacoCacheEntry> cache;
  std::mutex m_mutex;
  int request_count = 0;
  int hit_count = 0;
 public:
  static bool Find(const std::string& ir, uint64_t& hash,
                   const std::string& gfx, std::vector<uint8_t>& hsaco);
  static void Add(const std::string& ir, uint64_t hash, const std::string& gfx,
                  const std::vector<uint8_t>& hsaco);
};
static HsacoCache g_hsacoCache;  
bool HsacoCache::Find(const std::string& ir, uint64_t& hash,
                      const std::string& gfx, std::vector<uint8_t>& hsaco) {
  std::lock_guard<std::mutex> lg(g_hsacoCache.m_mutex);
  hash = std::hash<std::string>{}(ir);
  bool hit = false;
  for (auto& x : g_hsacoCache.cache) {
    if (x.hash != hash) continue;
    if (x.gfx != gfx) continue;
    if (x.ir != ir) continue;
    hsaco = x.hsaco;
    hit = true;
    break;
  }
  g_hsacoCache.request_count++;
  if (hit) g_hsacoCache.hit_count++;
  if (!(g_hsacoCache.request_count % 50))
    VLOG(1) << "HSACO cache: " << g_hsacoCache.request_count << " requests, "
            << g_hsacoCache.hit_count << " hits";
  return hit;
}
void HsacoCache::Add(const std::string& ir, uint64_t hash,
                     const std::string& gfx,
                     const std::vector<uint8_t>& hsaco) {
  std::lock_guard<std::mutex> lg(g_hsacoCache.m_mutex);
  g_hsacoCache.cache.resize(g_hsacoCache.cache.size() + 1);
  g_hsacoCache.cache.back().ir = ir;
  g_hsacoCache.cache.back().hash = hash;
  g_hsacoCache.cache.back().gfx = gfx;
  g_hsacoCache.cache.back().hsaco = hsaco;
}
absl::StatusOr<std::vector<uint8_t>> EmitModuleToHsaco(
    llvm::Module* module, llvm::TargetMachine* target_machine) {
  auto* env = tsl::Env::Default();
  std::vector<std::string> tempdir_vector;
  env->GetLocalTempDirectories(&tempdir_vector);
  if (tempdir_vector.empty()) {
    return xla::Internal(
        "Unable to locate a temporary directory for compile-time artifacts.");
  }
  std::string tempdir_name = tempdir_vector.front();
  VLOG(1) << "Compile-time artifacts located at: " << tempdir_name;
  bool keep_tempfiles = false;
  TF_CHECK_OK(tsl::ReadBoolFromEnvVar("TF_ROCM_KEEP_XLA_TEMPFILES",
                                      false, &keep_tempfiles));
  std::string random_number = std::to_string(tsl::random::New64());
  std::string ir_filename =
      absl::StrCat(module->getModuleIdentifier(), random_number + ".ll");
  std::string ir_path = tsl::io::JoinPath(tempdir_name, ir_filename);
  std::string ir_opt_filename =
      absl::StrCat(module->getModuleIdentifier(), random_number + "_opt.ll");
  std::string ir_opt_path = tsl::io::JoinPath(tempdir_name, ir_opt_filename);
  std::string isabin_filename =
      absl::StrCat(module->getModuleIdentifier(), random_number + ".o");
  std::string isabin_path = tsl::io::JoinPath(tempdir_name, isabin_filename);
  std::string hsaco_filename =
      absl::StrCat(module->getModuleIdentifier(), random_number + ".hsaco");
  std::string hsaco_path = tsl::io::JoinPath(tempdir_name, hsaco_filename);
  std::error_code ec;
  std::unique_ptr<llvm::raw_fd_ostream> ir_fs(
      new llvm::raw_fd_ostream(ir_path, ec, llvm::sys::fs::OF_None));
  module->print(*ir_fs, nullptr);
  ir_fs->flush();
  llvm::legacy::PassManager pm;
  pm.add(new llvm::TargetLibraryInfoWrapperPass(
      llvm::Triple(module->getTargetTriple())));
  llvm::SmallVector<char, 0> stream;
  llvm::raw_svector_ostream pstream(stream);
  std::unique_ptr<llvm::raw_fd_ostream> isabin_fs(
      new llvm::raw_fd_ostream(isabin_path, ec, llvm::sys::fs::OF_Text));
  module->setDataLayout(target_machine->createDataLayout());
  target_machine->addPassesToEmitFile(pm, *isabin_fs, nullptr,
                                      llvm::CodeGenFileType::ObjectFile);
  pm.run(*module);
  isabin_fs->flush();
  if (keep_tempfiles) {
    std::unique_ptr<llvm::raw_fd_ostream> ir_fs(
        new llvm::raw_fd_ostream(ir_opt_path, ec, llvm::sys::fs::OF_None));
    module->print(*ir_fs, nullptr);
    ir_fs->flush();
  }
  std::string lld_path;
  if (std::getenv("LLVM_PATH")) {
    lld_path = tsl::io::JoinPath(std::getenv("LLVM_PATH"), "bin");
  } else {
    lld_path = tsl::io::JoinPath(tsl::RocmRoot(), "llvm/bin");
  }
  auto lld_program = llvm::sys::findProgramByName("ld.lld", {lld_path});
  if (!lld_program) {
    return xla::Internal("unable to find ld.lld in PATH: %s",
                         lld_program.getError().message());
  }
  std::vector<llvm::StringRef> lld_args{
      llvm_ir::AsStringRef("ld.lld"),    llvm_ir::AsStringRef("-flavor"),
      llvm_ir::AsStringRef("gnu"),       llvm_ir::AsStringRef("-shared"),
      llvm_ir::AsStringRef(isabin_path), llvm_ir::AsStringRef("-o"),
      llvm_ir::AsStringRef(hsaco_path),
  };
  std::string error_message;
  int lld_result =
      llvm::sys::ExecuteAndWait(*lld_program, llvm_ir::AsArrayRef(lld_args),
                                std::nullopt, {}, 0, 0, &error_message);
  if (lld_result) {
    return xla::Internal("ld.lld execute fail: %s, error code %d",
                         error_message, lld_result);
  }
  std::ifstream hsaco_file(hsaco_path, std::ios::binary | std::ios::ate);
  std::ifstream::pos_type hsaco_file_size = hsaco_file.tellg();
  std::vector<uint8_t> hsaco(hsaco_file_size);
  hsaco_file.seekg(0, std::ios::beg);
  hsaco_file.read(reinterpret_cast<char*>(hsaco.data()), hsaco_file_size);
  hsaco_file.close();
  if (!keep_tempfiles) {
    remove(ir_path.c_str());
    remove(isabin_path.c_str());
    remove(hsaco_path.c_str());
  }
  return hsaco;
}
absl::Status LinkROCDLIfNecessary(llvm::Module* module,
                                  std::string gcn_arch_name,
                                  const std::string& rocdl_dir_path) {
  if (!CouldNeedDeviceBitcode(*module)) {
    return absl::OkStatus();
  }
  return LinkWithBitcodeVector(module,
                               GetROCDLPaths(gcn_arch_name, rocdl_dir_path));
}
absl::Status AMDGPUTargetModuleLinker(
    llvm::Module* module, se::GpuComputeCapability gpu_version,
    const DebugOptions& debug_options,
    const std::string& device_bitcode_dir_path) {
  auto compute_capability =
      std::get_if<se::RocmComputeCapability>(&gpu_version);
  if (!compute_capability) {
    return xla::Internal("Incompatible compute capability was specified.");
  }
  std::string gcn_arch_name = compute_capability->gcn_arch_name();
  TF_RETURN_IF_ERROR(
      LinkROCDLIfNecessary(module, gcn_arch_name, device_bitcode_dir_path));
  if (debug_options.xla_gpu_ftz()) {
    for (llvm::Function& fn : *module) {
      fn.addFnAttr("denormal-fp-math-f32", "preserve-sign");
    }
  }
  return absl::OkStatus();
}
std::string MapGCNArchNameTokenToFeatureStr(const std::string& token,
                                            const std::string& gfx) {
  if (token == "sramecc+") {
    return "+sramecc";
  } else if (token == "sramecc-") {
    if (gfx == "gfx90a" || gfx == "gfx940" || gfx == "gfx941" ||
        gfx == "gfx942")
      return "";
    return "-sramecc";
  } else if (token == "xnack+") {
    return "+xnack";
  } else if (token == "xnack-") {
    return "-xnack";
  }
  return "";
}
std::pair<std::string, std::string> GetFeatureStrFromGCNArchName(
    const std::string& gcn_arch_name) {
  std::string feature_str;
  std::string gfx = gcn_arch_name;
  std::vector<std::string> tokens = absl::StrSplit(gcn_arch_name, ':');
  std::vector<std::string> mapped_tokens;
  if (!tokens.empty()) gfx = tokens[0];
  for (auto it = tokens.begin(); it != tokens.end(); it++) {
    if (it != tokens.begin()) {
      std::string token(*it);
      std::string mapped_token = MapGCNArchNameTokenToFeatureStr(token, gfx);
      mapped_tokens.push_back(mapped_token);
    }
  }
  feature_str = absl::StrJoin(mapped_tokens, ",");
  return std::make_pair(gfx, feature_str);
}
std::unique_ptr<llvm::TargetMachine> AMDGPUGetTargetMachine(
    llvm::Triple target_triple, se::GpuComputeCapability gpu_version,
    const DebugOptions& debug_options) {
  auto compute_capability =
      std::get_if<se::RocmComputeCapability>(&gpu_version);
  std::string gcn_arch_name = compute_capability->gcn_arch_name();
  auto arch = GetFeatureStrFromGCNArchName(gcn_arch_name);
  return GetTargetMachine(std::move(target_triple), arch.first, debug_options,
                          arch.second);
}
std::string GetROCDLDir(const DebugOptions& debug_options) {
  std::vector<std::string> potential_rocdl_dirs;
  const std::string& datadir = debug_options.xla_gpu_cuda_data_dir();
  if (!datadir.empty()) {
    potential_rocdl_dirs.push_back(datadir);
  }
  potential_rocdl_dirs.push_back(tsl::RocdlRoot());
  for (const std::string& potential_rocdl_dir : potential_rocdl_dirs) {
    if (tsl::Env::Default()->IsDirectory(potential_rocdl_dir).ok()) {
      VLOG(2) << "Found ROCm-Device-Libs dir " << potential_rocdl_dir;
      return potential_rocdl_dir;
    }
    VLOG(2) << "Unable to find potential ROCm-Device-Libs dir "
            << potential_rocdl_dir;
  }
  return ".";
}
void AMDGPUBackendInit(const DebugOptions& debug_options,
                       std::string& rocdl_dir_path) {
  llvm_ir::InitializeLLVMCommandLineOptions(
      debug_options.xla_backend_extra_options());
#if TENSORFLOW_USE_ROCM
  LLVMInitializeAMDGPUTarget();
  LLVMInitializeAMDGPUTargetInfo();
  LLVMInitializeAMDGPUTargetMC();
  LLVMInitializeAMDGPUAsmParser();
  LLVMInitializeAMDGPUAsmPrinter();
#endif
  rocdl_dir_path = GetROCDLDir(debug_options);
  llvm::PassRegistry* registry = llvm::PassRegistry::getPassRegistry();
  InitializePasses(registry);
}
}  
namespace amdgpu {
std::string LibDevicePath(std::string gcn_arch_name,
                          const std::string& rocdl_dir_path) {
  auto libdevice_dir_paths = GetROCDLPaths(gcn_arch_name, rocdl_dir_path);
  for (auto libdevice_dir_path : libdevice_dir_paths) {
    if (libdevice_dir_path.find("ocml.bc")) {
      return libdevice_dir_path;
    }
  }
  return "";
}
absl::StatusOr<std::vector<uint8_t>> CompileToHsaco(
    llvm::Module* module, se::GpuComputeCapability gpu_version,
    const DebugOptions& debug_options,
    const std::string& module_config_cache_key) {
  static absl::once_flag backend_init_flag;
  static std::string rocdl_dir_path;  
  absl::call_once(backend_init_flag, AMDGPUBackendInit, debug_options,
                  rocdl_dir_path);
  std::vector<uint8_t> hsaco;
  std::unique_ptr<llvm::TargetMachine> target_machine;
  std::string str;
  llvm::raw_string_ostream stream(str);
  stream << *module;
  if (str.size() >= 13 && str.substr(0, 13) == "; ModuleID = ") {
    auto pos = str.find('\n');
    if (pos != std::string::npos) str = str.substr(pos + 1);
  }
  if (str.size() >= 18 && str.substr(0, 18) == "source_filename = ") {
    auto pos = str.find('\n');
    if (pos != std::string::npos) str = str.substr(pos + 1);
  }
  str += module_config_cache_key;
  {
    tsl::profiler::TraceMe activity(
        [&] { return absl::StrCat("Compiling IR", module->getName().str()); },
        tsl::profiler::TraceMeLevel::kInfo);
    XLA_SCOPED_LOGGING_TIMER("Compile module " + module->getName().str());
    auto compute_capability =
        std::get_if<se::RocmComputeCapability>(&gpu_version);
    if (!compute_capability) {
      return xla::Internal("Incompatible compute capability was specified.");
    }
    std::string gcn_arch_name = compute_capability->gcn_arch_name();
    uint64_t hash;
    if (HsacoCache::Find(str, hash, gcn_arch_name, hsaco)) {
      VLOG(1) << "HSACO cache hit";
      return hsaco;
    }
    VLOG(1) << "HSACO cache miss";
    bool dump_lls = false;
    if (dump_lls) {
      static int hsaco_count = 0;
      std::string name = "/tmp/" + std::to_string(hsaco_count) + ".ll";
      hsaco_count++;
      std::ofstream ofs(name);
      ofs << str;
      ofs.close();
    }
    llvm::Triple default_target_triple("amdgcn--amdhsa-amdgiz");
    std::unique_ptr<llvm::TargetMachine> target_machine =
        AMDGPUGetTargetMachine(default_target_triple, gpu_version,
                               debug_options);
    TF_RETURN_IF_ERROR(LinkAndOptimizeModule(
        module, gpu_version, debug_options, rocdl_dir_path,
        AMDGPUTargetModuleLinker, default_target_triple, target_machine.get(),
        kAMDGPUInlineThreshold));
    TF_ASSIGN_OR_RETURN(hsaco, EmitModuleToHsaco(module, target_machine.get()));
    HsacoCache::Add(str, hash, gcn_arch_name, hsaco);
  }
  return hsaco;
}
}  
namespace {
std::unique_ptr<llvm::TargetMachine> SPIRGetTargetMachine(
    llvm::Triple target_triple, se::GpuComputeCapability gpu_version,
    const DebugOptions& debug_options) {
  return nullptr;
}
absl::Status SPIRTargetModuleLinker(
    llvm::Module* module, se::GpuComputeCapability gpu_version,
    const DebugOptions& debug_options,
    const std::string& device_bitcode_dir_path) {
  return absl::OkStatus();
}
absl::StatusOr<std::string> EmitModuleToSpir(
    llvm::Module* module, se::GpuComputeCapability gpu_version,
    const DebugOptions& debug_options) {
#if TENSORFLOW_USE_SYCL
  SPIRV::TranslatorOpts::ExtensionsStatusMap ExtensionsStatus;
  SPIRV::TranslatorOpts opts(SPIRV::VersionNumber::MaximumVersion,
                             ExtensionsStatus);
  opts.enableAllExtensions();  
  std::ostringstream oss;
  std::string err;
  bool success = llvm::writeSpirv(module, opts, oss, err);
  if (!success) {
    return xla::Internal("Fails to convert LLVM as SPIR-V: %s", err);
  }
  return oss.str();
#else
  return absl::UnimplementedError("Not implemented for SYCL");
#endif
}
void SPIRBackendInit(const DebugOptions& debug_options) {
  FeedLLVMWithFlags({
      "-slp-vectorize-hor=false",
      "-slp-min-reg-size=64",
      "-slp-max-reg-size=64",
  });
  llvm_ir::InitializeLLVMCommandLineOptions(
      debug_options.xla_backend_extra_options());
  llvm::PassRegistry* registry = llvm::PassRegistry::getPassRegistry();
  InitializePasses(registry);
}
}  
namespace spir {
absl::StatusOr<std::vector<uint8_t>> CompileToSpir(
    llvm::Module* module, se::GpuComputeCapability gpu_version,
    const DebugOptions& debug_options) {
  std::string libdevice_dir_path;
  static absl::once_flag backend_init_flag;
  absl::call_once(backend_init_flag, SPIRBackendInit, debug_options);
  std::string spir;
  {
    XLA_SCOPED_LOGGING_TIMER("Compile module " + module->getName().str());
    if (module->empty() && module->global_empty()) {
      VLOG(2) << "Module '" << module->getName().str()
              << "' is empty. Skipping compilation.";
      return std::vector<uint8_t>();
    }
    llvm::Triple default_target_triple("spir64-unknown-unknown");
    std::unique_ptr<llvm::TargetMachine> target_machine =
        SPIRGetTargetMachine(default_target_triple, gpu_version, debug_options);
    TF_RETURN_IF_ERROR(LinkAndOptimizeModule(
        module, gpu_version, debug_options, libdevice_dir_path,
        SPIRTargetModuleLinker, default_target_triple, target_machine.get(),
        kDefaultInlineThreshold));
    TF_ASSIGN_OR_RETURN(spir,
                        EmitModuleToSpir(module, gpu_version, debug_options));
  }
  return std::vector<uint8_t>(spir.begin(), spir.end());
}
}  
}  
}  