#include "tensorflow/compiler/mlir/quantization/tensorflow/cc/convert_asset_args.h"
#include "absl/algorithm/container.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  
#include "mlir/IR/Attributes.h"  
#include "mlir/IR/Builders.h"  
#include "mlir/IR/BuiltinAttributes.h"  
#include "mlir/IR/BuiltinOps.h"  
#include "mlir/IR/SymbolTable.h"  
#include "mlir/IR/Value.h"  
#include "mlir/Support/LLVM.h"  
#include "mlir/Support/LogicalResult.h"  
#include "tensorflow/compiler/mlir/quantization/common/func.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/import_model.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
namespace mlir::quant {
namespace {
using ::mlir::tf_saved_model::AssetOp;
using ::mlir::tf_saved_model::kTfSavedModelIndexPathAttr;
using ::mlir::tf_saved_model::LookupBoundInputOfType;
using ::tensorflow::AssetFileDef;
SmallVector<NamedAttribute> ReplaceBoundInputAttrWithIndexPathAttr(
    const ArrayRef<NamedAttribute> arg_attrs, const StringRef index_path,
    Builder& builder) {
  SmallVector<NamedAttribute> new_arg_attrs;
  for (auto arg_attr : arg_attrs) {
    if (arg_attr.getName() == "tf_saved_model.bound_input") continue;
    new_arg_attrs.emplace_back(arg_attr);
  }
  const NamedAttribute index_path_attr(
      builder.getStringAttr(kTfSavedModelIndexPathAttr),
      builder.getStrArrayAttr({index_path}));
  new_arg_attrs.emplace_back(index_path_attr);
  return new_arg_attrs;
}
StringRef MaybeStripAssetDirectoryPrefix(const StringRef filename) {
  if (filename.find("assets/") == 0) {
    return filename.drop_front(7);
  } else {
    return filename;
  }
}
AssetFileDef CreateAssetFileDef(const StringRef filename,
                                const StringRef tensor_name) {
  AssetFileDef asset_file_def{};
  asset_file_def.set_filename(MaybeStripAssetDirectoryPrefix(filename).str());
  tensorflow::TensorInfo tensor_info{};
  tensor_info.set_name(tensor_name.str());
  *asset_file_def.mutable_tensor_info() = tensor_info;
  return asset_file_def;
}
SmallVector<StringRef> GetEntryFunctionInputs(func::FuncOp func_op) {
  auto entry_function_attr =
      func_op->getAttrOfType<DictionaryAttr>("tf.entry_function");
  SmallVector<StringRef> inputs;
  mlir::dyn_cast_or_null<StringAttr>(entry_function_attr.get("inputs"))
      .strref()
      .split(inputs, ",");
  return inputs;
}
void ConvertMainArgAttrs(func::FuncOp main_func_op, const int arg_idx,
                         const StringRef index_path) {
  const ArrayRef<NamedAttribute> arg_attrs =
      main_func_op.getArgAttrDict(arg_idx).getValue();
  Builder builder(main_func_op.getContext());
  SmallVector<NamedAttribute> new_arg_attrs =
      ReplaceBoundInputAttrWithIndexPathAttr(arg_attrs, index_path, builder);
  main_func_op.setArgAttrs(arg_idx, new_arg_attrs);
}
}  
FailureOr<SmallVector<AssetFileDef>> ConvertAssetArgs(ModuleOp module_op) {
  func::FuncOp main_func_op = FindMainFuncOp(module_op);
  if (!main_func_op) return failure();
  SmallVector<StringRef> input_names = GetEntryFunctionInputs(main_func_op);
  SymbolTable symbol_table(module_op);
  SmallVector<AssetFileDef> asset_file_defs;
  for (BlockArgument argument : main_func_op.getArguments()) {
    const int arg_idx = argument.getArgNumber();
    auto asset_op =
        LookupBoundInputOfType<AssetOp>(main_func_op, arg_idx, symbol_table);
    if (!asset_op) continue;
    const StringRef input_name = input_names[arg_idx];
    ConvertMainArgAttrs(main_func_op, arg_idx, input_name);
    asset_file_defs.emplace_back(CreateAssetFileDef(
        asset_op.getFilenameAttr(), input_name));
  }
  return asset_file_defs;
}
}  