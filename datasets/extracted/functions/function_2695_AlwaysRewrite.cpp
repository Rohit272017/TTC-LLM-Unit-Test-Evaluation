#ifdef INTEL_MKL
#include <string>
#include <unordered_map>
#include "tensorflow/core/common_runtime/eager/eager_op_rewrite_registry.h"
#include "tensorflow/core/graph/mkl_graph_util.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/util/mkl_util.h"
#include "tensorflow/core/util/util.h"
namespace tensorflow {
class MklEagerOpRewrite : public EagerOpRewrite {
 public:
  MklEagerOpRewrite(string name, string file, string line);
  struct MklEagerOp {
    string op_name;
    std::function<bool(EagerOperation*)> RewriteRule;
    std::function<Status(EagerOperation*, std::unique_ptr<EagerOperation>*)>
        CreateMklOp;
  };
 private:
  std::unordered_map<std::string, MklEagerOp> mkl_eager_ops_;
  Status Run(EagerOperation* orig_op,
             std::unique_ptr<tensorflow::EagerOperation>* out_op);
  static Status SetupNewOp(EagerOperation* orig_op, const string mkl_op_name,
                           std::unique_ptr<EagerOperation>* new_mkl_op);
  static Status CreateGenericMklOp(EagerOperation* orig_op,
                                   std::unique_ptr<EagerOperation>* mkl_op);
  static bool RewriteConv2D(EagerOperation* op);
  static bool RewriteSparseMatrixMatMul(EagerOperation* op);
  static bool RewriteFusedBatchNormV3(EagerOperation* op);
  Status RewriteToMklOp(EagerOperation* orig_op,
                        std::unique_ptr<EagerOperation>* mkl_op);
  bool ShouldRewriteOp(EagerOperation* op);
  static bool AlwaysRewrite(EagerOperation* op) { return true; }
  bool IsKernelRegistered(string op_name, DataType dt);
  void InsertMKLEagerOps(MklEagerOp op);
};
REGISTER_REWRITE(EagerOpRewriteRegistry::POST_PLACEMENT, 10000,
                 MklEagerOpRewrite);
MklEagerOpRewrite::MklEagerOpRewrite(string name, string file, string line)
    : EagerOpRewrite(name, file, line) {
  InsertMKLEagerOps({"AvgPool", AlwaysRewrite, CreateGenericMklOp});
  InsertMKLEagerOps({"AvgPoolGrad", AlwaysRewrite, CreateGenericMklOp});
  InsertMKLEagerOps({"AvgPool3D", AlwaysRewrite, CreateGenericMklOp});
  InsertMKLEagerOps({"AvgPool3DGrad", AlwaysRewrite, CreateGenericMklOp});
  InsertMKLEagerOps({"BatchMatMul", AlwaysRewrite, CreateGenericMklOp});
  InsertMKLEagerOps({"BatchMatMulV2", AlwaysRewrite, CreateGenericMklOp});
  InsertMKLEagerOps({"Conv2D", AlwaysRewrite, CreateGenericMklOp});
  InsertMKLEagerOps(
      {"Conv2DBackpropFilter", RewriteConv2D, CreateGenericMklOp});
  InsertMKLEagerOps({"Conv2DBackpropInput", RewriteConv2D, CreateGenericMklOp});
  InsertMKLEagerOps({"Conv3D", AlwaysRewrite, CreateGenericMklOp});
  InsertMKLEagerOps(
      {"Conv3DBackpropFilterV2", RewriteConv2D, CreateGenericMklOp});
  InsertMKLEagerOps(
      {"Conv3DBackpropInputV2", RewriteConv2D, CreateGenericMklOp});
  InsertMKLEagerOps(
      {"DepthwiseConv2dNative", AlwaysRewrite, CreateGenericMklOp});
  InsertMKLEagerOps({"DepthwiseConv2dNativeBackpropFilter", RewriteConv2D,
                     CreateGenericMklOp});
  InsertMKLEagerOps({"DepthwiseConv2dNativeBackpropInput", RewriteConv2D,
                     CreateGenericMklOp});
  InsertMKLEagerOps({"Einsum", AlwaysRewrite, CreateGenericMklOp});
  InsertMKLEagerOps({"FusedBatchNorm", AlwaysRewrite, CreateGenericMklOp});
  InsertMKLEagerOps({"FusedBatchNormGrad", AlwaysRewrite, CreateGenericMklOp});
  InsertMKLEagerOps(
      {"FusedBatchNormGradV2", AlwaysRewrite, CreateGenericMklOp});
  InsertMKLEagerOps(
      {"FusedBatchNormGradV3", RewriteFusedBatchNormV3, CreateGenericMklOp});
  InsertMKLEagerOps({"FusedBatchNormV2", AlwaysRewrite, CreateGenericMklOp});
  InsertMKLEagerOps(
      {"FusedBatchNormV3", RewriteFusedBatchNormV3, CreateGenericMklOp});
  InsertMKLEagerOps({"MatMul", AlwaysRewrite, CreateGenericMklOp});
#ifdef ENABLE_ONEDNN_V3
  InsertMKLEagerOps(
      {"SparseMatrixMatMul", RewriteSparseMatrixMatMul, CreateGenericMklOp});
#endif  
};
void MklEagerOpRewrite::InsertMKLEagerOps(MklEagerOp op) {
  mkl_eager_ops_.insert(std::make_pair(op.op_name, op));
}
Status MklEagerOpRewrite::Run(
    EagerOperation* orig_op,
    std::unique_ptr<tensorflow::EagerOperation>* out_op) {
  if (ShouldRewriteOp(orig_op)) {
    TF_CHECK_OK(RewriteToMklOp(orig_op, out_op));
  }
  return OkStatus();
}
Status MklEagerOpRewrite::SetupNewOp(
    EagerOperation* orig_op, const string mkl_op_name,
    std::unique_ptr<EagerOperation>* new_mkl_op) {
  bool is_remote = false;
  new_mkl_op->reset(new tensorflow::EagerOperation(&orig_op->EagerContext()));
  TF_RETURN_IF_ERROR(new_mkl_op->get()->Reset(mkl_op_name.c_str(), nullptr,
                                              is_remote, nullptr));
  int num_inputs = orig_op->Inputs().size();
  for (int i = 0; i < num_inputs; ++i) {
    TF_RETURN_IF_ERROR((*new_mkl_op)->AddInput(orig_op->Inputs()[i]));
  }
  const NodeDef& orig_ndef = orig_op->MutableAttrs()->BuildNodeDef();
  AttrSlice attr_list(orig_ndef);
  for (const auto& attr : attr_list) {
    (*new_mkl_op)->MutableAttrs()->Set(attr.first, attr.second);
  }
  if (!orig_op->EagerContext().RunEagerOpAsFunction()) {
    (*new_mkl_op)
        ->MutableAttrs()
        ->Set("_kernel", mkl_op_registry::kMklNameChangeOpLabel);
  }
  string device_name = orig_op->DeviceName();
  return (*new_mkl_op)->SetDeviceName(device_name.c_str());
}
Status MklEagerOpRewrite::CreateGenericMklOp(
    EagerOperation* orig_op, std::unique_ptr<EagerOperation>* mkl_op) {
  const string mkl_op_name =
      mkl_op_registry::GetMklNativeOpName(orig_op->Name());
  TF_CHECK_OK(SetupNewOp(orig_op, mkl_op_name, mkl_op));
  return OkStatus();
}
bool MklEagerOpRewrite::ShouldRewriteOp(EagerOperation* op) {
  if (!IsMKLEnabled()) {
    return false;
  }
  DataType data_type;
  if (op->Attrs().Get("T", &data_type) != OkStatus()) {
    return false;
  }
  if (op->GetDeviceParsedName().type != "CPU") {
    return false;
  }
  bool kernel_found = IsKernelRegistered(op->Name(), data_type);
  if (!kernel_found) {
    return false;
  }
  auto it = mkl_eager_ops_.find(op->Name());
  if (it != mkl_eager_ops_.end()) {
    if (it->second.RewriteRule(op)) {
      return true;
    }
  }
  return false;
}
bool MklEagerOpRewrite::IsKernelRegistered(string op_name, DataType dt) {
  auto element = mkl_eager_ops_.find(op_name);
  if (element != mkl_eager_ops_.end()) {
    return (mkl_op_registry::IsMklOp(
                mkl_op_registry::GetMklNativeOpName(op_name), dt, true) ||
            mkl_op_registry::IsMklOp(mkl_op_registry::GetMklOpName(op_name), dt,
                                     true));
  } else {
    return false;
  }
}
Status MklEagerOpRewrite::RewriteToMklOp(
    EagerOperation* orig_op, std::unique_ptr<EagerOperation>* mkl_op) {
  TF_RETURN_IF_ERROR(
      mkl_eager_ops_[orig_op->Name()].CreateMklOp(orig_op, mkl_op));
  return OkStatus();
}
bool MklEagerOpRewrite::RewriteConv2D(EagerOperation* op) {
  const NodeDef& ndef = op->MutableAttrs()->BuildNodeDef();
  string padding;
  TF_CHECK_OK(GetNodeAttr(ndef, "padding", &padding));
  return (padding != "EXPLICIT");
}
bool MklEagerOpRewrite::RewriteSparseMatrixMatMul(EagerOperation* op) {
  const NodeDef& ndef = op->MutableAttrs()->BuildNodeDef();
  DataType T;
  Tensor tensor;
  bool adjoint_a, adjoint_b, transpose_a, transpose_b, transpose_out;
  TF_CHECK_OK(GetNodeAttr(ndef, "T", &T));
  if (T != DT_FLOAT) {
    VLOG(1) << "_MklSparseMatrixMatMul only supports DT_FLOAT";
    return false;
  }
  TF_CHECK_OK(GetNodeAttr(ndef, "adjoint_a", &adjoint_a));
  TF_CHECK_OK(GetNodeAttr(ndef, "adjoint_b", &adjoint_b));
  if (adjoint_a || adjoint_b) {
    VLOG(1)
        << "_MklNativeSparseMatrixMatMul doesn't support adjointing matrices";
    return false;
  }
  TF_CHECK_OK(GetNodeAttr(ndef, "transpose_a", &transpose_a));
  TF_CHECK_OK(GetNodeAttr(ndef, "transpose_b", &transpose_b));
  TF_CHECK_OK(GetNodeAttr(ndef, "transpose_output", &transpose_out));
  if (transpose_a || transpose_b || transpose_out) {
    VLOG(1)
        << "_MklNativeSparseMatrixMatMul doesn't support transposing matrices";
    return false;
  }
  return true;
}
bool MklEagerOpRewrite::RewriteFusedBatchNormV3(EagerOperation* op) {
  const NodeDef& ndef = op->MutableAttrs()->BuildNodeDef();
  if (Check5DFormat(ndef)) {
    VLOG(1) << "Eager Op Rewrite: FusedBatchNorm(Grad)V3 op currently does not "
            << "support 5D tensors.";
    return false;
  }
  return true;
}
}  
#endif  