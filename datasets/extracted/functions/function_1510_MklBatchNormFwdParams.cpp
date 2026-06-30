#ifdef INTEL_MKL
#include "unsupported/Eigen/CXX11/Tensor"  
#include "dnnl.hpp"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/fused_batch_norm_op.h"
#include "tensorflow/core/kernels/no_op.h"
#include "tensorflow/core/util/mkl_util.h"
#include "tensorflow/core/util/tensor_format.h"
#if defined(DNNL_AARCH64_USE_ACL) && defined(ENABLE_ONEDNN_OPENMP)
#include "tensorflow/core/platform/mutex.h"
#endif
#define GET_FLAG(bn_flag) static_cast<int>(dnnl::normalization_flags::bn_flag)
#define IS_SET(cflag) (context_.flags & GET_FLAG(cflag))
using dnnl::batch_normalization_backward;
using dnnl::batch_normalization_forward;
using dnnl::prop_kind;
using dnnl::stream;
using BatchNormFwdPd = dnnl::batch_normalization_forward::primitive_desc;
using BatchNormBwdPd = dnnl::batch_normalization_backward::primitive_desc;
namespace tensorflow {
#ifndef ENABLE_ONEDNN_V3
#define FORWARD_INFERENCE prop_kind::forward_scoring
#define GET_DIFF_SCALE_DATA_BUFFER diff_scale_shift_data
#define GET_DIFF_SCALE_SHIFT_DATA_BUFFERS diff_scale_shift_data
#define GET_DIFF_SHIFT_DATA_BUFFER diff_scale_shift_data + depth_
#define GET_SCALE_AND_SHIFT_FLAGS GET_FLAG(use_scale_shift)
#define GET_SCALE_DATA_BUFFER scale_shift_data
#define IS_SCALE_AND_SHIFT_FLAG_SET IS_SET(use_scale_shift)
#define SCALE_SHIFT_NET_ARGS \
  { DNNL_ARG_SCALE_SHIFT, *context_.scale_shift_mem }
#define SET_MKL_LAYOUT(md) SetMklLayout(&md)
#else
#define FORWARD_INFERENCE prop_kind::forward_inference
#define GET_DIFF_SCALE_DATA_BUFFER diff_scale_data
#define GET_DIFF_SCALE_SHIFT_DATA_BUFFERS diff_scale_data, diff_shift_data
#define GET_DIFF_SHIFT_DATA_BUFFER diff_shift_data
#define GET_SCALE_AND_SHIFT_FLAGS GET_FLAG(use_scale) | GET_FLAG(use_shift)
#define GET_SCALE_DATA_BUFFER scale_data
#define IS_SCALE_AND_SHIFT_FLAG_SET IS_SET(use_scale) && IS_SET(use_shift)
#define SCALE_SHIFT_NET_ARGS \
  {DNNL_ARG_SCALE, *context_.scale_mem}, { DNNL_ARG_SHIFT, *context_.shift_mem }
#define SET_MKL_LAYOUT(md) SetMklLayout(md)
#endif  
using CPUDevice = Eigen::ThreadPoolDevice;
using FusedBNActivationMode = functor::FusedBatchNormActivationMode;
struct MklBatchNormFwdParams {
  memory::dims src_dims;
  int depth;
  float eps;
  bool training;
  TensorFormat data_format;
  FusedBNActivationMode activation_mode;
  memory::desc src_md;
#ifdef ENABLE_ONEDNN_V3
  memory::desc dst_md;
#endif  
  MklBatchNormFwdParams(const memory::dims& src_dims, int depth, float eps,
                        bool training, TensorFormat data_format,
                        memory::desc src_md,
#ifdef ENABLE_ONEDNN_V3
                        memory::desc dst_md,
#endif  
                        FusedBNActivationMode activation_mode)
      : src_dims(src_dims),
        depth(depth),
        eps(eps),
        training(training),
        data_format(data_format),
        activation_mode(activation_mode),
#ifndef ENABLE_ONEDNN_V3
        src_md(src_md) {
  }
#else
        src_md(src_md),
        dst_md(dst_md) {
  }
#endif  
};
template <typename T, typename U>
class MklFusedBatchNormFwdPrimitive : public MklPrimitive {
 public:
  explicit MklFusedBatchNormFwdPrimitive(const MklBatchNormFwdParams& fwdParams)
      : MklPrimitive(engine(engine::kind::cpu, 0)) {
    if (context_.bn_fwd == nullptr) Setup(fwdParams);
  }
  ~MklFusedBatchNormFwdPrimitive() {}
#ifndef ENABLE_ONEDNN_V3
  void Execute(const T* src_data, const U* scale_shift_data, T* dst_data,
#else
  void Execute(const T* src_data, const U* scale_data, const U* shift_data,
               T* dst_data,
#endif  
               U* mean_data, U* variance_data,
               std::shared_ptr<stream> fwd_stream, U* workspace_data) {
#if defined(DNNL_AARCH64_USE_ACL) && defined(ENABLE_ONEDNN_OPENMP)
    mutex_lock lock(primitive_execution_mu_);
#endif
#if !defined(ENABLE_ONEDNN_OPENMP) && !defined(ENABLE_ONEDNN_V3)
    context_.src_mem->set_data_handle(
        static_cast<void*>(const_cast<T*>(src_data)), *fwd_stream);
    context_.dst_mem->set_data_handle(static_cast<void*>(dst_data),
                                      *fwd_stream);
    if (IS_SET(use_scale_shift))
      context_.scale_shift_mem->set_data_handle(
          static_cast<void*>(const_cast<U*>(scale_shift_data)), *fwd_stream);
    if ((context_.pkind == prop_kind::forward_training) ||
        (IS_SET(use_global_stats))) {
      context_.mean_mem->set_data_handle(static_cast<void*>(mean_data),
                                         *fwd_stream);
      context_.variance_mem->set_data_handle(static_cast<void*>(variance_data),
                                             *fwd_stream);
    }
    if (workspace_data != nullptr) {
      context_.ws_mem->set_data_handle(workspace_data, *fwd_stream);
    }
#else
    context_.src_mem->set_data_handle(
        static_cast<void*>(const_cast<T*>(src_data)));
    context_.dst_mem->set_data_handle(static_cast<void*>(dst_data));
    if (IS_SCALE_AND_SHIFT_FLAG_SET) {
#ifndef ENABLE_ONEDNN_V3
      context_.scale_shift_mem->set_data_handle(
          static_cast<void*>(const_cast<U*>(scale_shift_data)));
#else
      context_.scale_mem->set_data_handle(
          static_cast<void*>(const_cast<U*>(scale_data)));
      context_.shift_mem->set_data_handle(
          static_cast<void*>(const_cast<U*>(shift_data)));
#endif  
    }
    if ((context_.pkind == prop_kind::forward_training) ||
        (IS_SET(use_global_stats))) {
      context_.mean_mem->set_data_handle(static_cast<void*>(mean_data));
      context_.variance_mem->set_data_handle(static_cast<void*>(variance_data));
    }
    if (workspace_data != nullptr) {
      context_.ws_mem->set_data_handle(workspace_data);
    }
#endif  
    execute_primitives(context_.fwd_primitives, fwd_stream, context_.net_args);
    context_.src_mem->set_data_handle(DummyData);
    context_.dst_mem->set_data_handle(DummyData);
    if (IS_SCALE_AND_SHIFT_FLAG_SET) {
#ifndef ENABLE_ONEDNN_V3
      context_.scale_shift_mem->set_data_handle(DummyData);
#else
      context_.scale_mem->set_data_handle(DummyData);
      context_.shift_mem->set_data_handle(DummyData);
#endif  
    }
    if ((context_.pkind == prop_kind::forward_training) ||
        (IS_SET(use_global_stats))) {
      context_.mean_mem->set_data_handle(DummyData);
      context_.variance_mem->set_data_handle(DummyData);
    }
    if (workspace_data != nullptr) {
      context_.ws_mem->set_data_handle(DummyData);
    }
  }
  memory::desc GetDstPd() const { return context_.dst_mem->get_desc(); }
  std::shared_ptr<BatchNormFwdPd> GetBatchNormFwdPd() const {
    return context_.fwd_pd;
  }
 private:
  struct BatchNormFwdContext {
    int64 flags;
    dnnl::prop_kind pkind;
    std::shared_ptr<dnnl::memory> src_mem;
#ifndef ENABLE_ONEDNN_V3
    std::shared_ptr<dnnl::memory> scale_shift_mem;
#else
    std::shared_ptr<dnnl::memory> scale_mem;
    std::shared_ptr<dnnl::memory> shift_mem;
#endif  
    std::shared_ptr<dnnl::memory> dst_mem;
    std::shared_ptr<dnnl::memory> mean_mem;
    std::shared_ptr<dnnl::memory> variance_mem;
    std::shared_ptr<dnnl::memory> ws_mem;
    std::shared_ptr<BatchNormFwdPd> fwd_pd;
    std::shared_ptr<dnnl::primitive> bn_fwd;
    std::vector<dnnl::primitive> fwd_primitives;
    std::vector<std::unordered_map<int, memory>> net_args;
    BatchNormFwdContext()
        : flags(0),
          pkind(prop_kind::forward_training),
          src_mem(nullptr),
#ifndef ENABLE_ONEDNN_V3
          scale_shift_mem(nullptr),
#else
          scale_mem(nullptr),
          shift_mem(nullptr),
#endif  
          dst_mem(nullptr),
          mean_mem(nullptr),
          variance_mem(nullptr),
          ws_mem(nullptr),
          bn_fwd(nullptr) {
    }
  };
  void Setup(const MklBatchNormFwdParams& fwdParams) {
    context_.flags = GET_SCALE_AND_SHIFT_FLAGS |
                     (fwdParams.training ? false : GET_FLAG(use_global_stats));
    context_.pkind =
        fwdParams.training ? prop_kind::forward_training : FORWARD_INFERENCE;
    if (fwdParams.activation_mode == FusedBNActivationMode::kRelu) {
      context_.flags |= GET_FLAG(fuse_norm_relu);
    }
    auto src_md = fwdParams.src_md;
#ifndef ENABLE_ONEDNN_V3
    auto fwd_desc = batch_normalization_forward::desc(
        context_.pkind, src_md, fwdParams.eps,
        static_cast<dnnl::normalization_flags>(context_.flags));
    context_.fwd_pd.reset(new BatchNormFwdPd(fwd_desc, cpu_engine_));
#else
    auto dst_md = fwdParams.dst_md;
    context_.fwd_pd.reset(new BatchNormFwdPd(
        cpu_engine_, context_.pkind, src_md, dst_md, fwdParams.eps,
        static_cast<dnnl::normalization_flags>(context_.flags)));
#endif  
    context_.src_mem.reset(
        new memory(context_.fwd_pd->src_desc(), cpu_engine_, DummyData));
    context_.dst_mem.reset(
        new memory(context_.fwd_pd->dst_desc(), cpu_engine_, DummyData));
    memory::dims m_dims = {1, fwdParams.depth};
    if (IS_SCALE_AND_SHIFT_FLAG_SET) {
#ifndef ENABLE_ONEDNN_V3
      memory::dims s_dims = {2, fwdParams.depth};
      context_.scale_shift_mem.reset(
          new memory({{s_dims}, MklDnnType<U>(), memory::format_tag::nc},
                     cpu_engine_, DummyData));
#else
      memory::dims s_dims = {fwdParams.depth};
      context_.scale_mem.reset(
          new memory({{s_dims}, MklDnnType<U>(), memory::format_tag::x},
                     cpu_engine_, DummyData));
      context_.shift_mem.reset(
          new memory({{s_dims}, MklDnnType<U>(), memory::format_tag::x},
                     cpu_engine_, DummyData));
#endif  
    }
    if (fwdParams.training || (IS_SET(use_global_stats))) {
      context_.mean_mem.reset(
          new memory({{m_dims}, MklDnnType<U>(), memory::format_tag::nc},
                     cpu_engine_, DummyData));
      context_.variance_mem.reset(
          new memory({{m_dims}, MklDnnType<U>(), memory::format_tag::nc},
                     cpu_engine_, DummyData));
    }
    if (IS_SET(fuse_norm_relu)) {
      context_.ws_mem.reset(new memory(context_.fwd_pd->workspace_desc(),
                                       cpu_engine_, DummyData));
    }
    if (!fwdParams.training && !(IS_SET(use_global_stats))) {
      if (IS_SCALE_AND_SHIFT_FLAG_SET) {
        context_.net_args.push_back({{DNNL_ARG_SRC, *context_.src_mem},
                                     SCALE_SHIFT_NET_ARGS,
                                     {DNNL_ARG_DST, *context_.dst_mem}});
      } else {
        context_.net_args.push_back({{DNNL_ARG_SRC, *context_.src_mem},
                                     {DNNL_ARG_DST, *context_.dst_mem}});
      }
      context_.bn_fwd.reset(new batch_normalization_forward(*context_.fwd_pd));
    } else if (IS_SET(use_global_stats)) {
      if (IS_SCALE_AND_SHIFT_FLAG_SET) {
        if (IS_SET(fuse_norm_relu)) {
          context_.net_args.push_back(
              {{DNNL_ARG_SRC, *context_.src_mem},
               {DNNL_ARG_MEAN, *context_.mean_mem},
               {DNNL_ARG_VARIANCE, *context_.variance_mem},
               SCALE_SHIFT_NET_ARGS,
               {DNNL_ARG_DST, *context_.dst_mem},
               {DNNL_ARG_WORKSPACE, *context_.ws_mem}});
        } else {
          context_.net_args.push_back(
              {{DNNL_ARG_SRC, *context_.src_mem},
               {DNNL_ARG_MEAN, *context_.mean_mem},
               {DNNL_ARG_VARIANCE, *context_.variance_mem},
               SCALE_SHIFT_NET_ARGS,
               {DNNL_ARG_DST, *context_.dst_mem}});
        }
      } else {
        if (IS_SET(fuse_norm_relu)) {
          context_.net_args.push_back(
              {{DNNL_ARG_SRC, *context_.src_mem},
               {DNNL_ARG_MEAN, *context_.mean_mem},
               {DNNL_ARG_VARIANCE, *context_.variance_mem},
               {DNNL_ARG_DST, *context_.dst_mem},
               {DNNL_ARG_WORKSPACE, *context_.ws_mem}});
        } else {
          context_.net_args.push_back(
              {{DNNL_ARG_SRC, *context_.src_mem},
               {DNNL_ARG_MEAN, *context_.mean_mem},
               {DNNL_ARG_VARIANCE, *context_.variance_mem},
               {DNNL_ARG_DST, *context_.dst_mem}});
        }
      }
      context_.bn_fwd.reset(new batch_normalization_forward(*context_.fwd_pd));
    } else {
      if (IS_SCALE_AND_SHIFT_FLAG_SET) {
        if (IS_SET(fuse_norm_relu)) {
          context_.net_args.push_back(
              {{DNNL_ARG_SRC, *context_.src_mem},
               SCALE_SHIFT_NET_ARGS,
               {DNNL_ARG_DST, *context_.dst_mem},
               {DNNL_ARG_MEAN, *context_.mean_mem},
               {DNNL_ARG_VARIANCE, *context_.variance_mem},
               {DNNL_ARG_WORKSPACE, *context_.ws_mem}});
        } else {
          context_.net_args.push_back(
              {{DNNL_ARG_SRC, *context_.src_mem},
               SCALE_SHIFT_NET_ARGS,
               {DNNL_ARG_DST, *context_.dst_mem},
               {DNNL_ARG_MEAN, *context_.mean_mem},
               {DNNL_ARG_VARIANCE, *context_.variance_mem}});
        }
      } else {
        if (IS_SET(fuse_norm_relu)) {
          context_.net_args.push_back(
              {{DNNL_ARG_SRC, *context_.src_mem},
               {DNNL_ARG_DST, *context_.dst_mem},
               {DNNL_ARG_MEAN, *context_.mean_mem},
               {DNNL_ARG_VARIANCE, *context_.variance_mem},
               {DNNL_ARG_WORKSPACE, *context_.ws_mem}});
        } else {
          context_.net_args.push_back(
              {{DNNL_ARG_SRC, *context_.src_mem},
               {DNNL_ARG_DST, *context_.dst_mem},
               {DNNL_ARG_MEAN, *context_.mean_mem},
               {DNNL_ARG_VARIANCE, *context_.variance_mem}});
        }
      }
      context_.bn_fwd.reset(new batch_normalization_forward(*context_.fwd_pd));
    }
    context_.fwd_primitives.push_back(*context_.bn_fwd);
  }
  struct BatchNormFwdContext context_;
#if defined(DNNL_AARCH64_USE_ACL) && defined(ENABLE_ONEDNN_OPENMP)
  mutex primitive_execution_mu_;
#endif
};
template <typename T, typename U>
class MklFusedBatchNormFwdPrimitiveFactory : public MklPrimitiveFactory<T> {
 public:
  static MklFusedBatchNormFwdPrimitive<T, U>* Get(
      const MklBatchNormFwdParams& fwdParams) {
    auto bn_fwd = static_cast<MklFusedBatchNormFwdPrimitive<T, U>*>(
        MklFusedBatchNormFwdPrimitiveFactory<T, U>::GetInstance()
            .GetBatchNormFwd(fwdParams));
    if (bn_fwd == nullptr) {
      bn_fwd = new MklFusedBatchNormFwdPrimitive<T, U>(fwdParams);
      MklFusedBatchNormFwdPrimitiveFactory<T, U>::GetInstance().SetBatchNormFwd(
          fwdParams, bn_fwd);
    }
    return bn_fwd;
  }
  static MklFusedBatchNormFwdPrimitiveFactory& GetInstance() {
    static MklFusedBatchNormFwdPrimitiveFactory instance_;
    return instance_;
  }
 private:
  MklFusedBatchNormFwdPrimitiveFactory() {}
  ~MklFusedBatchNormFwdPrimitiveFactory() {}
  static string CreateKey(const MklBatchNormFwdParams& fwdParams) {
    string prefix = "bn_fwd";
    FactoryKeyCreator key_creator;
    key_creator.AddAsKey(prefix);
    key_creator.AddAsKey(fwdParams.src_dims);
    key_creator.AddAsKey<int>(fwdParams.depth);
    key_creator.AddAsKey<float>(fwdParams.eps);
    key_creator.AddAsKey<bool>(fwdParams.training);
    key_creator.AddAsKey<TensorFormat>(fwdParams.data_format);
    key_creator.AddAsKey<FusedBNActivationMode>(fwdParams.activation_mode);
    key_creator.AddAsKey(typeid(T).name());
    key_creator.AddAsKey(typeid(U).name());
    return key_creator.GetKey();
  }
  MklPrimitive* GetBatchNormFwd(const MklBatchNormFwdParams& fwdParams) {
    string key = CreateKey(fwdParams);
    return this->GetOp(key);
  }
  void SetBatchNormFwd(const MklBatchNormFwdParams& fwdParams,
                       MklPrimitive* op) {
    string key = CreateKey(fwdParams);
    this->SetOp(key, op);
  }
};
struct MklBatchNormBwdParams {
  memory::dims src_dims;
  memory::dims diff_dst_dims;
  int depth;
  float eps;
  bool training;
  TensorFormat data_format;
  memory::desc src_md;
#ifdef ENABLE_ONEDNN_V3
  memory::desc dst_md;
  memory::desc diff_src_md;
#endif  
  memory::desc diff_dst_md;
  MklBatchNormBwdParams(memory::dims src_dims, memory::dims diff_dst_dims,
                        int depth, float eps, bool training,
                        TensorFormat data_format, memory::desc src_md,
#ifdef ENABLE_ONEDNN_V3
                        memory::desc dst_md, memory::desc diff_src_md,
#endif  
                        memory::desc diff_dst_md)
      : src_dims(src_dims),
        diff_dst_dims(diff_dst_dims),
        depth(depth),
        eps(eps),
        training(training),
        data_format(data_format),
        src_md(src_md),
#ifdef ENABLE_ONEDNN_V3
        dst_md(dst_md),
        diff_src_md(diff_src_md),
#endif  
        diff_dst_md(diff_dst_md) {
  }
};
template <typename T, typename U>
class MklFusedBatchNormBwdPrimitive : public MklPrimitive {
 public:
  explicit MklFusedBatchNormBwdPrimitive(const MklBatchNormBwdParams& bwdParams)
      : MklPrimitive(engine(engine::kind::cpu, 0)) {
    if (context_.bn_bwd == nullptr) Setup(bwdParams);
  }
  ~MklFusedBatchNormBwdPrimitive() {}
  void Execute(const T* src_data, const U* mean_data, const U* variance_data,
#ifndef ENABLE_ONEDNN_V3
               const T* diff_dst_data, const U* scale_shift_data,
               T* diff_src_data, U* diff_scale_shift_data, U* res_space_data,
#else
               const T* diff_dst_data, const U* scale_data, T* diff_src_data,
               U* diff_scale_data, U* diff_shift_data, U* res_space_data,
#endif  
               std::shared_ptr<stream> bwd_stream) {
#if defined(DNNL_AARCH64_USE_ACL) && defined(ENABLE_ONEDNN_OPENMP)
    mutex_lock lock(primitive_execution_mu_);
#endif
#if !defined(ENABLE_ONEDNN_OPENMP) && !defined(ENABLE_ONEDNN_V3)
    context_.src_mem->set_data_handle(
        static_cast<void*>(const_cast<T*>(src_data)), *bwd_stream);
    context_.mean_mem->set_data_handle(
        static_cast<void*>(const_cast<U*>(mean_data)), *bwd_stream);
    context_.variance_mem->set_data_handle(
        static_cast<void*>(const_cast<U*>(variance_data)), *bwd_stream);
    context_.diff_dst_mem->set_data_handle(
        static_cast<void*>(const_cast<T*>(diff_dst_data)), *bwd_stream);
    if (IS_SET(use_scale_shift)) {
      context_.scale_shift_mem->set_data_handle(
          static_cast<void*>(const_cast<U*>(scale_shift_data)), *bwd_stream);
      context_.diff_scale_shift_mem->set_data_handle(
          static_cast<void*>(diff_scale_shift_data), *bwd_stream);
    }
    context_.diff_src_mem->set_data_handle(static_cast<void*>(diff_src_data),
                                           *bwd_stream);
#else
    context_.src_mem->set_data_handle(
        static_cast<void*>(const_cast<T*>(src_data)));
    context_.mean_mem->set_data_handle(
        static_cast<void*>(const_cast<U*>(mean_data)));
    context_.variance_mem->set_data_handle(
        static_cast<void*>(const_cast<U*>(variance_data)));
    context_.diff_dst_mem->set_data_handle(
        static_cast<void*>(const_cast<T*>(diff_dst_data)));
    if (IS_SCALE_AND_SHIFT_FLAG_SET) {
#ifndef ENABLE_ONEDNN_V3
      context_.scale_shift_mem->set_data_handle(
          static_cast<void*>(const_cast<U*>(scale_shift_data)));
      context_.diff_scale_shift_mem->set_data_handle(
          static_cast<void*>(diff_scale_shift_data));
#else
      context_.scale_mem->set_data_handle(
          static_cast<void*>(const_cast<U*>(scale_data)));
      context_.diff_scale_mem->set_data_handle(
          static_cast<void*>(diff_scale_data));
      context_.diff_shift_mem->set_data_handle(
          static_cast<void*>(diff_shift_data));
#endif  
    }
    context_.diff_src_mem->set_data_handle(static_cast<void*>(diff_src_data));
#endif  
    DCHECK_EQ(context_.bwd_primitives.size(), context_.net_args.size());
    execute_primitives(context_.bwd_primitives, bwd_stream, context_.net_args);
    context_.src_mem->set_data_handle(DummyData);
    context_.mean_mem->set_data_handle(DummyData);
    context_.variance_mem->set_data_handle(DummyData);
    context_.diff_dst_mem->set_data_handle(DummyData);
    if (IS_SCALE_AND_SHIFT_FLAG_SET) {
#ifndef ENABLE_ONEDNN_V3
      context_.scale_shift_mem->set_data_handle(DummyData);
      context_.diff_scale_shift_mem->set_data_handle(DummyData);
#else
      context_.scale_mem->set_data_handle(DummyData);
      context_.diff_scale_mem->set_data_handle(DummyData);
      context_.diff_shift_mem->set_data_handle(DummyData);
#endif  
    }
    context_.diff_src_mem->set_data_handle(DummyData);
  }
  std::shared_ptr<BatchNormBwdPd> GetBatchNormBwdPd() const {
    return context_.bwd_pd;
  }
  memory::desc GetDiffSrcPd() { return context_.diff_src_mem->get_desc(); }
 private:
  struct BatchNormBwdContext {
    int64 flags;
    std::shared_ptr<dnnl::memory> src_mem;
    std::shared_ptr<dnnl::memory> mean_mem;
    std::shared_ptr<dnnl::memory> variance_mem;
    std::shared_ptr<dnnl::memory> diff_dst_mem;
#ifndef ENABLE_ONEDNN_V3
    std::shared_ptr<dnnl::memory> scale_shift_mem;
    std::shared_ptr<dnnl::memory> diff_scale_shift_mem;
#else
    std::shared_ptr<dnnl::memory> scale_mem;
    std::shared_ptr<dnnl::memory> diff_scale_mem;
    std::shared_ptr<dnnl::memory> diff_shift_mem;
#endif  
    std::shared_ptr<dnnl::memory> diff_src_mem;
    std::shared_ptr<BatchNormBwdPd> bwd_pd;
    std::shared_ptr<dnnl::primitive> bn_bwd;
    std::vector<dnnl::primitive> bwd_primitives;
    std::vector<std::unordered_map<int, memory>> net_args;
    BatchNormBwdContext()
        : flags(0),
          src_mem(nullptr),
          mean_mem(nullptr),
          variance_mem(nullptr),
          diff_dst_mem(nullptr),
#ifndef ENABLE_ONEDNN_V3
          scale_shift_mem(nullptr),
          diff_scale_shift_mem(nullptr),
#else
          scale_mem(nullptr),
          diff_scale_mem(nullptr),
          diff_shift_mem(nullptr),
#endif  
          diff_src_mem(nullptr) {
    }
  };
  inline int64 GetBatchNormFlags(const MklBatchNormBwdParams& bwdParams) const {
    return GET_SCALE_AND_SHIFT_FLAGS |
           (bwdParams.training ? false : GET_FLAG(use_global_stats));
  }
  void Setup(const MklBatchNormBwdParams& bwdParams) {
    context_.flags = GetBatchNormFlags(bwdParams);
    auto src_md = bwdParams.src_md;
    auto diff_dst_md = bwdParams.diff_dst_md;
    auto variance_desc = memory::desc({1, bwdParams.depth}, MklDnnType<U>(),
                                      memory::format_tag::nc);
    auto mean_desc = memory::desc({1, bwdParams.depth}, MklDnnType<U>(),
                                  memory::format_tag::nc);
#ifndef ENABLE_ONEDNN_V3
    auto scale_shift_desc = memory::desc({2, bwdParams.depth}, MklDnnType<U>(),
                                         memory::format_tag::nc);
#else
    auto scale_shift_desc =
        memory::desc({bwdParams.depth}, MklDnnType<U>(), memory::format_tag::x);
#endif  
    auto diff_scale_shift_desc = scale_shift_desc;
    auto bn_flags = GetBatchNormFlags(bwdParams);
#ifndef ENABLE_ONEDNN_V3
    auto fwd_desc = batch_normalization_forward::desc(
        prop_kind::forward_training, src_md, bwdParams.eps,
        static_cast<dnnl::normalization_flags>(bn_flags));
    auto fwd_pd = BatchNormFwdPd(fwd_desc, cpu_engine_);
    auto bwd_desc = batch_normalization_backward::desc(
        prop_kind::backward, diff_dst_md, src_md, bwdParams.eps,
        static_cast<dnnl::normalization_flags>(bn_flags));
    context_.bwd_pd.reset(new BatchNormBwdPd(bwd_desc, cpu_engine_, fwd_pd));
#else
    auto dst_md = bwdParams.dst_md;
    auto diff_src_md = bwdParams.diff_src_md;
    auto fwd_pd = BatchNormFwdPd(
        cpu_engine_, prop_kind::forward_training, src_md, dst_md, bwdParams.eps,
        static_cast<dnnl::normalization_flags>(bn_flags));
    context_.bwd_pd.reset(new BatchNormBwdPd(
        cpu_engine_, prop_kind::backward, diff_src_md, diff_dst_md, src_md,
        bwdParams.eps, static_cast<dnnl::normalization_flags>(bn_flags),
        fwd_pd));
#endif  
    context_.src_mem.reset(new memory(src_md, cpu_engine_, DummyData));
    context_.diff_dst_mem.reset(
        new memory(diff_dst_md, cpu_engine_, DummyData));
    context_.variance_mem.reset(
        new memory(variance_desc, cpu_engine_, DummyData));
    context_.mean_mem.reset(new memory(mean_desc, cpu_engine_, DummyData));
#ifndef ENABLE_ONEDNN_V3
    context_.scale_shift_mem.reset(
        new memory(scale_shift_desc, cpu_engine_, DummyData));
    context_.diff_scale_shift_mem.reset(
        new memory(diff_scale_shift_desc, cpu_engine_, DummyData));
#else
    context_.scale_mem.reset(
        new memory(scale_shift_desc, cpu_engine_, DummyData));
    context_.diff_scale_mem.reset(
        new memory(diff_scale_shift_desc, cpu_engine_, DummyData));
    context_.diff_shift_mem.reset(
        new memory(diff_scale_shift_desc, cpu_engine_, DummyData));
#endif  
    context_.diff_src_mem.reset(new memory(src_md, cpu_engine_, DummyData));
    context_.bn_bwd.reset(new batch_normalization_backward(*context_.bwd_pd));
    context_.net_args.push_back(
        {{DNNL_ARG_SRC, *context_.src_mem},
         {DNNL_ARG_MEAN, *context_.mean_mem},
         {DNNL_ARG_VARIANCE, *context_.variance_mem},
         {DNNL_ARG_DIFF_DST, *context_.diff_dst_mem},
         {DNNL_ARG_DIFF_SRC, *context_.diff_src_mem},
#ifndef ENABLE_ONEDNN_V3
         {DNNL_ARG_SCALE_SHIFT, *context_.scale_shift_mem},
         { DNNL_ARG_DIFF_SCALE_SHIFT,
           *context_.diff_scale_shift_mem }});
#else
         {DNNL_ARG_SCALE, *context_.scale_mem},
         {DNNL_ARG_DIFF_SCALE, *context_.diff_scale_mem},
         {DNNL_ARG_DIFF_SHIFT, *context_.diff_shift_mem}});
#endif  
    context_.bwd_primitives.push_back(*context_.bn_bwd);
  }
  struct BatchNormBwdContext context_;
#if defined(DNNL_AARCH64_USE_ACL) && defined(ENABLE_ONEDNN_OPENMP)
  mutex primitive_execution_mu_;
#endif
};
template <typename T, typename U>
class MklFusedBatchNormBwdPrimitiveFactory : public MklPrimitiveFactory<T> {
 public:
  static MklFusedBatchNormBwdPrimitive<T, U>* Get(
      const MklBatchNormBwdParams& bwdParams) {
    auto bn_bwd = static_cast<MklFusedBatchNormBwdPrimitive<T, U>*>(
        MklFusedBatchNormBwdPrimitiveFactory<T, U>::GetInstance()
            .GetBatchNormBwd(bwdParams));
    if (bn_bwd == nullptr) {
      bn_bwd = new MklFusedBatchNormBwdPrimitive<T, U>(bwdParams);
      MklFusedBatchNormBwdPrimitiveFactory<T, U>::GetInstance().SetBatchNormBwd(
          bwdParams, bn_bwd);
    }
    return bn_bwd;
  }
  static MklFusedBatchNormBwdPrimitiveFactory& GetInstance() {
    static MklFusedBatchNormBwdPrimitiveFactory instance_;
    return instance_;
  }
 private:
  MklFusedBatchNormBwdPrimitiveFactory() {}
  ~MklFusedBatchNormBwdPrimitiveFactory() {}
  static string CreateKey(const MklBatchNormBwdParams& bwdParams) {
    string prefix = "bn_bwd";
    FactoryKeyCreator key_creator;
    key_creator.AddAsKey(prefix);
    key_creator.AddAsKey(bwdParams.src_dims);
    key_creator.AddAsKey(bwdParams.diff_dst_dims);
    key_creator.AddAsKey<int>(bwdParams.depth);
    key_creator.AddAsKey<float>(bwdParams.eps);
    key_creator.AddAsKey<bool>(bwdParams.training);
    key_creator.AddAsKey<TensorFormat>(bwdParams.data_format);
    key_creator.AddAsKey(typeid(T).name());
    key_creator.AddAsKey(typeid(U).name());
    return key_creator.GetKey();
  }
  MklPrimitive* GetBatchNormBwd(const MklBatchNormBwdParams& bwdParams) {
    string key = CreateKey(bwdParams);
    return this->GetOp(key);
  }
  void SetBatchNormBwd(const MklBatchNormBwdParams& bwdParams,
                       MklPrimitive* op) {
    string key = CreateKey(bwdParams);
    this->SetOp(key, op);
  }
};
template <typename Device, typename T, typename U, bool reserved_space,
          bool is_batch_norm_ex = false, bool native_format = false>
class MklFusedBatchNormOp : public OpKernel {
 public:
  explicit MklFusedBatchNormOp(OpKernelConstruction* context)
      : OpKernel(context) {
    float epsilon;
    OP_REQUIRES_OK(context, context->GetAttr("epsilon", &epsilon));
    epsilon_ = epsilon;
    float exponential_avg_factor;
    OP_REQUIRES_OK(context, context->GetAttr("exponential_avg_factor",
                                             &exponential_avg_factor));
    exponential_avg_factor_ = static_cast<U>(exponential_avg_factor);
    string tensor_format;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &tensor_format));
    OP_REQUIRES(context, FormatFromString(tensor_format, &tensor_format_),
                absl::InvalidArgumentError("Invalid data format"));
    OP_REQUIRES_OK(context, context->GetAttr("is_training", &is_training_));
    depth_ = 0;
    mean_values_ = nullptr;
    variance_values_ = nullptr;
    if (!is_batch_norm_ex) {
      activation_mode_ = FusedBNActivationMode::kIdentity;
    } else {
      int num_side_inputs;
      OP_REQUIRES_OK(context,
                     context->GetAttr("num_side_inputs", &num_side_inputs));
      OP_REQUIRES(context, num_side_inputs == 0,
                  absl::InvalidArgumentError(
                      "_MKLFusedBatchNorm do not support side input now."));
      OP_REQUIRES_OK(context, ParseActivationMode(context, &activation_mode_));
      OP_REQUIRES(context, activation_mode_ == FusedBNActivationMode::kRelu,
                  absl::InvalidArgumentError(
                      "_MKLFusedBatchNorm only support Relu activation"));
    }
  }
  void Compute(OpKernelContext* context) override {
    try {
      const size_t kSrcIndex = 0;       
      const size_t kScaleIndex = 1;     
      const size_t kShiftIndex = 2;     
      const size_t kMeanIndex = 3;      
      const size_t kVarianceIndex = 4;  
      const Tensor& src_tensor = MklGetInput(context, kSrcIndex);
      const Tensor& scale_tensor = MklGetInput(context, kScaleIndex);
      const Tensor& shift_tensor = MklGetInput(context, kShiftIndex);
      const Tensor& est_mean_tensor = MklGetInput(context, kMeanIndex);
      const Tensor& est_variance_tensor = MklGetInput(context, kVarianceIndex);
      TensorShape tf_shape_src;
      MklDnnShape dnn_shape_src;
      GetMklShape(context, kSrcIndex, &dnn_shape_src, native_format);
      if (dnn_shape_src.IsMklTensor()) {
        tf_shape_src = dnn_shape_src.GetTfShape();
        OP_REQUIRES(context, dnn_shape_src.GetDimension() == 4,
                    absl::InvalidArgumentError(
                        absl::StrCat("input must be 4-dimensional",
                                     src_tensor.shape().DebugString())));
      } else {
        tf_shape_src = src_tensor.shape();
        OP_REQUIRES(context, src_tensor.dims() == 4,
                    absl::InvalidArgumentError(
                        absl::StrCat("input must be 4-dimensional",
                                     src_tensor.shape().DebugString())));
      }
      OP_REQUIRES(context, scale_tensor.dims() == 1,
                  absl::InvalidArgumentError(
                      absl::StrCat("scale must be 1-dimensional",
                                   scale_tensor.shape().DebugString())));
      OP_REQUIRES(context, shift_tensor.dims() == 1,
                  absl::InvalidArgumentError(
                      absl::StrCat("offset must be 1-dimensional",
                                   shift_tensor.shape().DebugString())));
      OP_REQUIRES(context, est_mean_tensor.dims() == 1,
                  absl::InvalidArgumentError(
                      absl::StrCat("estimated_mean must be 1-dimensional",
                                   est_mean_tensor.shape().DebugString())));
      OP_REQUIRES(context, est_variance_tensor.dims() == 1,
                  absl::InvalidArgumentError(
                      absl::StrCat("estimated_variance must be 1-dimensional",
                                   est_variance_tensor.shape().DebugString())));
      int num_channels;
      if (dnn_shape_src.IsMklTensor()) {
        num_channels = dnn_shape_src.DimSize(MklDnnDims::Dim_C);
      } else {
        num_channels = GetTensorDim(src_tensor, tensor_format_, 'C');
      }
      OP_REQUIRES(context, scale_tensor.NumElements() == num_channels,
                  absl::InvalidArgumentError(absl::StrCat(
                      "scale must have the same number of elements "
                      "as the channels of x, got ",
                      scale_tensor.NumElements(), " and ", num_channels)));
      OP_REQUIRES(context, shift_tensor.NumElements() == num_channels,
                  absl::InvalidArgumentError(absl::StrCat(
                      "offset must have the same number of elements "
                      "as the channels of x, got ",
                      shift_tensor.NumElements(), " and ", num_channels)));
      if (!is_training_ || exponential_avg_factor_ != 1.) {
        std::string prefix_msg = is_training_
                                     ? "When exponential_avg_factor != 1"
                                     : "When is_training=false";
        OP_REQUIRES(context, est_mean_tensor.NumElements() == num_channels,
                    absl::InvalidArgumentError(absl::StrCat(
                        prefix_msg,
                        ", mean must have the same number "
                        "of elements as the channels of x, got ",
                        est_mean_tensor.NumElements(), " and ", num_channels)));
        OP_REQUIRES(
            context, est_variance_tensor.NumElements() == num_channels,
            absl::InvalidArgumentError(absl::StrCat(
                prefix_msg,
                ", variance must have the same "
                "number of elements as the channels of x, got ",
                est_variance_tensor.NumElements(), " and ", num_channels)));
      }
      Tensor* dst_tensor = nullptr;
      TensorShape workspace_tf_shape;
      if (tf_shape_src.num_elements() == 0) {
        size_t workspace_bytes = 0;
        workspace_tf_shape.AddDim(workspace_bytes);
        HandleEmptyInput(context, tf_shape_src, workspace_tf_shape,
                         scale_tensor.shape(), &dst_tensor);
        return;
      }
      if (dnn_shape_src.IsMklTensor())
        depth_ = dnn_shape_src.DimSize(MklDnnDims::Dim_C);
      else
        ExtractParams(context);
      const size_t kDstIndex = 0;
      Tensor* batch_mean_tensor = nullptr;
      Tensor* batch_variance_tensor = nullptr;
      Tensor* saved_mean_tensor = nullptr;
      Tensor* saved_variance_tensor = nullptr;
      Tensor* reserved_space_tensor = nullptr;
      MklDnnData<T> src(&cpu_engine_);
#ifndef ENABLE_ONEDNN_V3
      MklDnnData<U> scale_shift(&cpu_engine_);
#else
      MklDnnData<U> scale(&cpu_engine_);
      MklDnnData<U> shift(&cpu_engine_);
#endif  
      MklDnnData<U> wksp(&cpu_engine_);
      memory::format_tag dnn_fmt;
      MklTensorFormat mkl_tensor_fmt;
      if (dnn_shape_src.IsMklTensor()) {
        if (dnn_shape_src.IsTensorInNCHWFormat()) {
          dnn_fmt = memory::format_tag::nchw;
          mkl_tensor_fmt = MklTensorFormat::FORMAT_NCHW;
        } else {
          dnn_fmt = memory::format_tag::nhwc;
          mkl_tensor_fmt = MklTensorFormat::FORMAT_NHWC;
        }
      } else {
        mkl_tensor_fmt = TFDataFormatToMklDnnDataFormat(tensor_format_);
        dnn_fmt = MklTensorFormatToMklDnnDataFormat(mkl_tensor_fmt);
      }
      memory::dims src_dims =
          dnn_shape_src.IsMklTensor()
              ? dnn_shape_src.GetSizesAsMklDnnDims()
              : TFShapeToMklDnnDimsInNCHW(src_tensor.shape(), tensor_format_);
      auto src_md = dnn_shape_src.IsMklTensor()
                        ? dnn_shape_src.GetMklLayout()
                        : memory::desc(src_dims, MklDnnType<T>(), dnn_fmt);
#ifdef ENABLE_ONEDNN_V3
      auto dst_md = memory::desc(src_dims, MklDnnType<T>(), dnn_fmt);
#endif  
      MklBatchNormFwdParams fwdParams(src_dims, depth_, epsilon_, is_training_,
#ifndef ENABLE_ONEDNN_V3
                                      tensor_format_, src_md, activation_mode_);
#else
                                      tensor_format_, src_md, dst_md,
                                      activation_mode_);
#endif  
      Eigen::ThreadPoolInterface* eigen_interface =
          EigenThreadPoolFromTfContext(context);
      tsl::OneDnnThreadPool eigen_tp(eigen_interface,
                                     ThreadPoolUseCallerThread());
      MklFusedBatchNormFwdPrimitive<T, U>* bn_fwd =
          MklFusedBatchNormFwdPrimitiveFactory<T, U>::Get(fwdParams);
      U* ws_data = nullptr;
      if (fwdParams.activation_mode == FusedBNActivationMode::kRelu) {
        memory::desc workspace_md =
            bn_fwd->GetBatchNormFwdPd()->workspace_desc();
        size_t workspace_bytes = workspace_md.get_size();
        workspace_tf_shape.AddDim(workspace_bytes);
        AllocateTFOutputs(context, scale_tensor.shape(), workspace_tf_shape,
                          &batch_mean_tensor, &batch_variance_tensor,
                          &saved_mean_tensor, &saved_variance_tensor,
                          &reserved_space_tensor);
        if (reserved_space) {
          wksp.SetUsrMem(workspace_md, reserved_space_tensor);
          ws_data = static_cast<U*>(wksp.GetOpMem().get_data_handle());
        }
      } else {
        size_t workspace_bytes = 0;
        workspace_tf_shape.AddDim(workspace_bytes);
        AllocateTFOutputs(context, scale_tensor.shape(), workspace_tf_shape,
                          &batch_mean_tensor, &batch_variance_tensor,
                          &saved_mean_tensor, &saved_variance_tensor,
                          &reserved_space_tensor);
      }
      if (is_training_)
        SetMeanVariance(*batch_mean_tensor, *batch_variance_tensor);
      else
        SetMeanVariance(est_mean_tensor, est_variance_tensor);
#ifndef ENABLE_ONEDNN_V3
      scale_shift.AllocateBuffer(2 * depth_ * sizeof(U));
      U* scale_shift_data =
          reinterpret_cast<U*>(scale_shift.GetAllocatedBuffer());
      const U* scale_tf = scale_tensor.flat<U>().data();
      const U* shift_tf = shift_tensor.flat<U>().data();
      std::memcpy(scale_shift_data, scale_tf, depth_ * sizeof(U));
      std::memcpy(scale_shift_data + depth_, shift_tf, depth_ * sizeof(U));
#else
      scale.AllocateBuffer(depth_ * sizeof(U));
      U* scale_data = reinterpret_cast<U*>(scale.GetAllocatedBuffer());
      shift.AllocateBuffer(depth_ * sizeof(U));
      U* shift_data = reinterpret_cast<U*>(shift.GetAllocatedBuffer());
      const U* scale_tf = scale_tensor.flat<U>().data();
      const U* shift_tf = shift_tensor.flat<U>().data();
      std::memcpy(scale_data, scale_tf, depth_ * sizeof(U));
      std::memcpy(shift_data, shift_tf, depth_ * sizeof(U));
#endif  
      char* saved_mean_data_tf =
          reinterpret_cast<char*>(saved_mean_tensor->flat<U>().data());
      std::memcpy(saved_mean_data_tf, reinterpret_cast<char*>(mean_values_),
                  depth_ * sizeof(U));
      char* saved_variance_data_tf =
          reinterpret_cast<char*>(saved_variance_tensor->flat<U>().data());
      std::memcpy(saved_variance_data_tf,
                  reinterpret_cast<char*>(variance_values_),
                  depth_ * sizeof(U));
      const T* src_data = nullptr;
      std::shared_ptr<BatchNormFwdPd> bn_fwd_pd = bn_fwd->GetBatchNormFwdPd();
      if (!native_format && src_md != bn_fwd_pd->src_desc()) {
        src.SetUsrMem(src_md, &src_tensor);
        src.CheckReorderToOpMem(bn_fwd_pd->src_desc(), cpu_engine_, context);
        src_data = static_cast<T*>(src.GetOpMem().get_data_handle());
      } else {
        src_data = static_cast<T*>(const_cast<T*>(src_tensor.flat<T>().data()));
      }
      MklDnnShape dnn_shape_dst;
      TensorShape tf_shape_dst;
      dnn_shape_dst.SetMklTensor(true);
      auto dst_pd = bn_fwd->GetDstPd();
      dnn_shape_dst.SET_MKL_LAYOUT(dst_pd);
      dnn_shape_dst.SetElemType(MklDnnType<T>());
      auto ndims = dnn_shape_src.IsMklTensor() ? dnn_shape_src.GetDimension()
                                               : src_tensor.shape().dims();
      dnn_shape_dst.SetTfLayout(ndims, src_dims, mkl_tensor_fmt);
      tf_shape_dst.AddDim(dst_pd.get_size() / sizeof(T));
      if (native_format) {
        tf_shape_dst = dnn_shape_dst.GetTfShape();
      }
      AllocateOutputSetMklShape(context, kDstIndex, &dst_tensor, tf_shape_dst,
                                dnn_shape_dst, native_format);
#ifndef ENABLE_ONEDNN_V3
      U* scale_shift_op_data = scale_shift_data;
#else
      U* scale_op_data = scale_data;
      U* shift_op_data = shift_data;
#endif  
      U* mean_op_data = saved_mean_tensor->flat<U>().data();
      U* variance_op_data = saved_variance_tensor->flat<U>().data();
      T* dst_data = dst_tensor->flat<T>().data();
      std::shared_ptr<stream> fwd_cpu_stream;
      fwd_cpu_stream.reset(CreateStream(&eigen_tp, bn_fwd->GetEngine()));
#ifndef ENABLE_ONEDNN_V3
      bn_fwd->Execute(src_data, scale_shift_op_data, dst_data, mean_op_data,
#else
      bn_fwd->Execute(src_data, scale_op_data, shift_op_data, dst_data,
                      mean_op_data,
#endif  
                      variance_op_data, fwd_cpu_stream, ws_data);
      float adjust_factor = 1.0;
      if (is_training_) {
        size_t orig_size = src_dims[0] * src_dims[2] * src_dims[3];
        size_t adjust_size = (orig_size > 1) ? (orig_size - 1) : 1;
        adjust_factor = (static_cast<float>(orig_size)) / adjust_size;
      }
      auto mean_data = reinterpret_cast<U*>(saved_mean_data_tf);
      auto variance_data = reinterpret_cast<U*>(saved_variance_data_tf);
      auto batch_mean_data = batch_mean_tensor->flat<U>().data();
      auto batch_variance_data = batch_variance_tensor->flat<U>().data();
      auto est_mean_data = est_mean_tensor.flat<U>().data();
      auto est_variance_data = est_variance_tensor.flat<U>().data();
      if (is_training_) {
        if (exponential_avg_factor_ == U(1.0)) {
          for (int k = 0; k < depth_; k++) {
            batch_mean_data[k] = mean_data[k];
            batch_variance_data[k] =
                static_cast<U>(adjust_factor) * variance_data[k];
          }
        } else {
          U one_minus_factor = U(1.0) - exponential_avg_factor_;
          for (int k = 0; k < depth_; k++) {
            batch_mean_data[k] = one_minus_factor * est_mean_data[k] +
                                 exponential_avg_factor_ * mean_data[k];
            batch_variance_data[k] = one_minus_factor * est_variance_data[k] +
                                     exponential_avg_factor_ *
                                         static_cast<U>(adjust_factor) *
                                         variance_data[k];
          }
        }
      } else {
        std::memcpy(batch_mean_data, mean_data, depth_ * sizeof(U));
        std::memcpy(batch_variance_data, variance_data, depth_ * sizeof(U));
      }
    } catch (dnnl::error& e) {
      string error_msg = "Status: " + std::to_string(e.status) +
                         ", message: " + string(e.message) + ", in file " +
                         string(__FILE__) + ":" + std::to_string(__LINE__);
      OP_REQUIRES_OK(context,
                     absl::AbortedError(absl::StrCat(
                         "Operation received an exception:", error_msg)));
    }
  }
 private:
  float epsilon_;
  U exponential_avg_factor_;
  TensorFormat tensor_format_;
  bool is_training_;
  U* mean_values_;
  U* variance_values_;
  size_t depth_;  
  FusedBNActivationMode activation_mode_;
  engine cpu_engine_ = engine(engine::kind::cpu, 0);
  void ExtractParams(OpKernelContext* context) {
    const Tensor& input = MklGetInput(context, 0);
    depth_ = static_cast<int>(GetTensorDim(input, tensor_format_, 'C'));
  }
  void SetMeanVariance(const Tensor& mean, const Tensor& variance) {
    mean_values_ = reinterpret_cast<U*>(const_cast<U*>(mean.flat<U>().data()));
    variance_values_ =
        reinterpret_cast<U*>(const_cast<U*>(variance.flat<U>().data()));
  }
  void HandleEmptyInput(OpKernelContext* context, TensorShape tf_shape_src,
                        TensorShape workspace_tf_shape,
                        TensorShape tf_shape_scale, Tensor** dst_tensor) {
    DCHECK(dst_tensor);
    const size_t kDstIndex = 0;
    MklDnnShape dnn_shape_dst;
    dnn_shape_dst.SetMklTensor(false);
    AllocateOutputSetMklShape(context, kDstIndex, dst_tensor, tf_shape_src,
                              dnn_shape_dst, native_format);
    DCHECK(*dst_tensor);
    memset(const_cast<char*>((*dst_tensor)->tensor_data().data()), 0,
           (*dst_tensor)->tensor_data().size());
    Tensor* batch_mean_tensor = nullptr;
    Tensor* batch_variance_tensor = nullptr;
    Tensor* saved_mean_tensor = nullptr;
    Tensor* saved_variance_tensor = nullptr;
    Tensor* reserved_space_tensor = nullptr;
    AllocateTFOutputs(context, tf_shape_scale, workspace_tf_shape,
                      &batch_mean_tensor, &batch_variance_tensor,
                      &saved_mean_tensor, &saved_variance_tensor,
                      &reserved_space_tensor);
  }
  void AllocateTFOutputs(OpKernelContext* context, TensorShape tf_shape_scale,
                         TensorShape workspace_tf_shape,
                         Tensor** batch_mean_tensor,
                         Tensor** batch_variance_tensor,
                         Tensor** saved_mean_tensor,
                         Tensor** saved_variance_tensor,
                         Tensor** reserved_space_tensor) {
    DCHECK(batch_mean_tensor);
    DCHECK(batch_variance_tensor);
    DCHECK(saved_mean_tensor);
    DCHECK(saved_variance_tensor);
    const size_t kBatchMeanIndex = 1;
    const size_t kBatchVarianceIndex = 2;
    const size_t kSavedMeanIndex = 3;
    const size_t kSavedVarianceIndex = 4;
    const size_t kReservedSpaceIndex = 5;
    MklDnnShape mkl_shape_batch_mean;
    mkl_shape_batch_mean.SetMklTensor(false);
    AllocateOutputSetMklShape(context, kBatchMeanIndex, batch_mean_tensor,
                              tf_shape_scale, mkl_shape_batch_mean,
                              native_format);
    DCHECK(*batch_mean_tensor);
    int num_elements = tf_shape_scale.num_elements();
    auto batch_mean_data = (*batch_mean_tensor)->flat<U>().data();
    std::fill_n(batch_mean_data, num_elements, static_cast<U>(NAN));
    MklDnnShape mkl_shape_batch_variance;
    mkl_shape_batch_variance.SetMklTensor(false);
    AllocateOutputSetMklShape(context, kBatchVarianceIndex,
                              batch_variance_tensor, tf_shape_scale,
                              mkl_shape_batch_variance, native_format);
    DCHECK(*batch_variance_tensor);
    auto batch_variance_data = (*batch_variance_tensor)->flat<U>().data();
    std::fill_n(batch_variance_data, num_elements, static_cast<U>(NAN));
    MklDnnShape mkl_shape_saved_mean;
    mkl_shape_saved_mean.SetMklTensor(false);
    AllocateOutputSetMklShape(context, kSavedMeanIndex, saved_mean_tensor,
                              tf_shape_scale, mkl_shape_saved_mean,
                              native_format);
    DCHECK(*saved_mean_tensor);
    auto saved_mean_data = (*saved_mean_tensor)->flat<U>().data();
    std::fill_n(saved_mean_data, num_elements, static_cast<U>(0));
    MklDnnShape mkl_shape_saved_variance;
    mkl_shape_saved_variance.SetMklTensor(false);
    AllocateOutputSetMklShape(context, kSavedVarianceIndex,
                              saved_variance_tensor, tf_shape_scale,
                              mkl_shape_saved_variance, native_format);
    DCHECK(*saved_variance_tensor);
    auto saved_variance_data = (*saved_variance_tensor)->flat<U>().data();
    std::fill_n(saved_variance_data, num_elements, static_cast<U>(0));
    if (reserved_space) {
      DCHECK(reserved_space_tensor != nullptr);
      MklDnnShape mkl_shape_reserved_space;
      mkl_shape_reserved_space.SetMklTensor(false);
      AllocateOutputSetMklShape(context, kReservedSpaceIndex,
                                reserved_space_tensor, workspace_tf_shape,
                                mkl_shape_reserved_space, native_format);
      DCHECK((*reserved_space_tensor) != nullptr);
    }
  }
};
template <typename Device, typename T, typename U, bool reserved_space,
          bool native_format = false>
class MklFusedBatchNormGradOp : public OpKernel {
 public:
  explicit MklFusedBatchNormGradOp(OpKernelConstruction* context)
      : OpKernel(context) {
    float epsilon;
    OP_REQUIRES_OK(context, context->GetAttr("epsilon", &epsilon));
    epsilon_ = epsilon;
    string tensor_format;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &tensor_format));
    OP_REQUIRES(context, FormatFromString(tensor_format, &tensor_format_),
                absl::InvalidArgumentError("Invalid data format"));
    OP_REQUIRES_OK(context, context->GetAttr("is_training", &is_training_));
    depth_ = 0;
  }
  void Compute(OpKernelContext* context) override {
    try {
      const size_t kDiffDstIndex = 0;        
      const size_t kSrcIndex = 1;            
      const size_t kScaleIndex = 2;          
      const size_t kMeanIndex = 3;           
      const size_t kVarianceIndex = 4;       
      const size_t kReservedSpaceIndex = 5;  
      const Tensor& diff_dst_tensor = MklGetInput(context, kDiffDstIndex);
      const Tensor& src_tensor = MklGetInput(context, kSrcIndex);
      const Tensor& scale_tensor = MklGetInput(context, kScaleIndex);
      const Tensor& saved_mean_tensor = MklGetInput(context, kMeanIndex);
      const Tensor& saved_variance_tensor =
          MklGetInput(context, kVarianceIndex);
      const Tensor& reserved_space_tensor =
          (reserved_space) ? MklGetInput(context, kReservedSpaceIndex)
                           : Tensor();
      MklDnnShape dnn_shape_src, dnn_shape_diff_dst;
      GetMklShape(context, kSrcIndex, &dnn_shape_src, native_format);
      GetMklShape(context, kDiffDstIndex, &dnn_shape_diff_dst, native_format);
      TensorShape tf_shape_src, tf_shape_diff_dst;
      if (dnn_shape_diff_dst.IsMklTensor()) {
        tf_shape_diff_dst = dnn_shape_diff_dst.GetTfShape();
        OP_REQUIRES(context, dnn_shape_diff_dst.GetDimension() == 4,
                    absl::InvalidArgumentError(
                        absl::StrCat("input must be 4-dimensional",
                                     diff_dst_tensor.shape().DebugString())));
      } else {
        tf_shape_diff_dst = diff_dst_tensor.shape();
        OP_REQUIRES(context, diff_dst_tensor.dims() == 4,
                    absl::InvalidArgumentError(
                        absl::StrCat("input must be 4-dimensional",
                                     diff_dst_tensor.shape().DebugString())));
      }
      if (dnn_shape_src.IsMklTensor()) {
        tf_shape_src = dnn_shape_src.GetTfShape();
        OP_REQUIRES(context, dnn_shape_src.GetDimension() == 4,
                    absl::InvalidArgumentError(
                        absl::StrCat("input must be 4-dimensional",
                                     src_tensor.shape().DebugString())));
      } else {
        tf_shape_src = src_tensor.shape();
        OP_REQUIRES(context, src_tensor.dims() == 4,
                    absl::InvalidArgumentError(
                        absl::StrCat("input must be 4-dimensional",
                                     src_tensor.shape().DebugString())));
      }
      OP_REQUIRES(context, scale_tensor.dims() == 1,
                  absl::InvalidArgumentError(
                      absl::StrCat("scale must be 1-dimensional",
                                   scale_tensor.shape().DebugString())));
      OP_REQUIRES(context, saved_mean_tensor.dims() == 1,
                  absl::InvalidArgumentError(
                      absl::StrCat("saved mean must be 1-dimensional",
                                   saved_mean_tensor.shape().DebugString())));
      OP_REQUIRES(context, saved_variance_tensor.dims() == 1,
                  absl::InvalidArgumentError(absl::StrCat(
                      "saved variance must be 1-dimensional",
                      saved_variance_tensor.shape().DebugString())));
      OP_REQUIRES(
          context, tf_shape_src == tf_shape_diff_dst,
          absl::InvalidArgumentError(absl::StrCat(
              "x and y_backprop must have same shape, but x has shape ",
              src_tensor.shape().DebugString(), " and y_backprop has shape ",
              diff_dst_tensor.shape().DebugString())));
      int num_channels;
      if (dnn_shape_src.IsMklTensor()) {
        num_channels = dnn_shape_src.DimSize(MklDnnDims::Dim_C);
      } else {
        num_channels = GetTensorDim(src_tensor, tensor_format_, 'C');
      }
      OP_REQUIRES(context, scale_tensor.NumElements() == num_channels,
                  absl::InvalidArgumentError(absl::StrCat(
                      "scale must have the same number of elements "
                      "as the channels of x, got ",
                      scale_tensor.NumElements(), " and ", num_channels)));
      OP_REQUIRES(context, saved_mean_tensor.NumElements() == num_channels,
                  absl::InvalidArgumentError(absl::StrCat(
                      "reserve_space_1 must have the same number of "
                      "elements as the channels of x, got ",
                      saved_mean_tensor.NumElements(), " and ", num_channels)));
      OP_REQUIRES(
          context, saved_variance_tensor.NumElements() == num_channels,
          absl::InvalidArgumentError(absl::StrCat(
              "reserve_space_2 must have the same number of "
              "elements as the channels of x, got ",
              saved_variance_tensor.NumElements(), " and ", num_channels)));
      Tensor* diff_src_tensor = nullptr;
      if (tf_shape_src.num_elements() == 0 ||
          tf_shape_diff_dst.num_elements() == 0) {
        HandleEmptyInput(context, tf_shape_src, scale_tensor.shape(),
                         &diff_src_tensor);
        return;
      }
      if (dnn_shape_src.IsMklTensor()) {
        depth_ = dnn_shape_src.DimSize(MklDnnDims::Dim_C);
      } else if (dnn_shape_diff_dst.IsMklTensor()) {
        depth_ = dnn_shape_diff_dst.DimSize(MklDnnDims::Dim_C);
      } else {
        ExtractParams(context);
      }
      memory::format_tag dnn_fmt;
      MklTensorFormat mkl_tensor_fmt;
      if (dnn_shape_src.IsMklTensor()) {
        if (dnn_shape_src.IsTensorInNCHWFormat()) {
          dnn_fmt = memory::format_tag::nchw;
          mkl_tensor_fmt = MklTensorFormat::FORMAT_NCHW;
        } else {
          dnn_fmt = memory::format_tag::nhwc;
          mkl_tensor_fmt = MklTensorFormat::FORMAT_NHWC;
        }
      } else {
        mkl_tensor_fmt = TFDataFormatToMklDnnDataFormat(tensor_format_);
        dnn_fmt = MklTensorFormatToMklDnnDataFormat(mkl_tensor_fmt);
      }
      MklDnnData<T> src(&cpu_engine_);
      MklDnnData<T> diff_dst(&cpu_engine_);
#ifndef ENABLE_ONEDNN_V3
      MklDnnData<U> scale_shift(&cpu_engine_);
      MklDnnData<U> diff_scale_shift(&cpu_engine_);
#else
      MklDnnData<U> scale(&cpu_engine_);
      MklDnnData<U> diff_scale(&cpu_engine_);
      MklDnnData<U> diff_shift(&cpu_engine_);
#endif  
      memory::dims src_dims =
          dnn_shape_src.IsMklTensor()
              ? dnn_shape_src.GetSizesAsMklDnnDims()
              : TFShapeToMklDnnDimsInNCHW(src_tensor.shape(), tensor_format_);
      memory::dims diff_dst_dims =
          dnn_shape_diff_dst.IsMklTensor()
              ? dnn_shape_diff_dst.GetSizesAsMklDnnDims()
              : TFShapeToMklDnnDimsInNCHW(diff_dst_tensor.shape(),
                                          tensor_format_);
      memory::desc src_md =
          dnn_shape_src.IsMklTensor()
              ? dnn_shape_src.GetMklLayout()
              : memory::desc(src_dims, MklDnnType<T>(), dnn_fmt);
      memory::desc diff_dst_md =
          dnn_shape_diff_dst.IsMklTensor()
              ? dnn_shape_diff_dst.GetMklLayout()
              : memory::desc(diff_dst_dims, MklDnnType<T>(), dnn_fmt);
#ifdef ENABLE_ONEDNN_V3
      memory::desc dst_md = memory::desc(src_dims, MklDnnType<T>(), dnn_fmt);
      memory::desc diff_src_md =
          memory::desc(diff_dst_dims, MklDnnType<T>(), dnn_fmt);
#endif  
      MklDnnData<T> reorder_src(&cpu_engine_);
      MklDnnData<T> reorder_diff_dst(&cpu_engine_);
      T* diff_dst_data =
          static_cast<T*>(const_cast<T*>(diff_dst_tensor.flat<T>().data()));
      T* src_data =
          static_cast<T*>(const_cast<T*>(src_tensor.flat<T>().data()));
      if (!native_format) {
        if (dnn_shape_src.IsMklTensor() && !dnn_shape_diff_dst.IsMklTensor()) {
          reorder_diff_dst.SetUsrMem(diff_dst_md, &diff_dst_tensor);
          reorder_diff_dst.CheckReorderToOpMem(src_md, cpu_engine_, context);
          diff_dst_md = src_md;
          diff_dst_data =
              static_cast<T*>(reorder_diff_dst.GetOpMem().get_data_handle());
        } else if (!dnn_shape_src.IsMklTensor() &&
                   dnn_shape_diff_dst.IsMklTensor()) {
          reorder_src.SetUsrMem(src_md, &src_tensor);
          reorder_src.CheckReorderToOpMem(diff_dst_md, cpu_engine_, context);
          src_md = diff_dst_md;
          src_data = static_cast<T*>(reorder_src.GetOpMem().get_data_handle());
        }
      }
#ifndef ENABLE_ONEDNN_V3
      scale_shift.AllocateBuffer(2 * depth_ * sizeof(U));
      U* scale_shift_data_tf =
          reinterpret_cast<U*>(scale_shift.GetAllocatedBuffer());
      const U* scale_tf = scale_tensor.flat<U>().data();
      for (int k = 0; k < depth_; k++) {
        scale_shift_data_tf[k] = scale_tf[k];
        scale_shift_data_tf[k + depth_] = static_cast<U>(0);
      }
      diff_scale_shift.AllocateBuffer(2 * depth_ * sizeof(U));
#else
      scale.AllocateBuffer(depth_ * sizeof(U));
      U* scale_data_tf = reinterpret_cast<U*>(scale.GetAllocatedBuffer());
      const U* scale_tf = scale_tensor.flat<U>().data();
      std::memcpy(scale_data_tf, scale_tf, depth_ * sizeof(U));
      diff_scale.AllocateBuffer(depth_ * sizeof(U));
      diff_shift.AllocateBuffer(depth_ * sizeof(U));
#endif  
      MklBatchNormBwdParams bwdParams(src_dims, diff_dst_dims, depth_, epsilon_,
                                      is_training_, tensor_format_, src_md,
#ifdef ENABLE_ONEDNN_V3
                                      dst_md, diff_src_md,
#endif  
                                      diff_dst_md);
      Eigen::ThreadPoolInterface* eigen_interface =
          EigenThreadPoolFromTfContext(context);
      tsl::OneDnnThreadPool eigen_tp(eigen_interface,
                                     ThreadPoolUseCallerThread());
      MklFusedBatchNormBwdPrimitive<T, U>* bn_bwd =
          MklFusedBatchNormBwdPrimitiveFactory<T, U>::Get(bwdParams);
      std::shared_ptr<BatchNormBwdPd> bn_bwd_pd = bn_bwd->GetBatchNormBwdPd();
      if (!native_format && diff_dst_md != bn_bwd_pd->diff_dst_desc()) {
        diff_dst.SetUsrMem(diff_dst_md, diff_dst_data);
        diff_dst.CheckReorderToOpMem(bn_bwd_pd->diff_dst_desc(), cpu_engine_,
                                     context);
        diff_dst_data = static_cast<T*>(diff_dst.GetOpMem().get_data_handle());
      }
      if (!native_format && (src_md != bn_bwd_pd->src_desc())) {
        src.SetUsrMem(src_md, src_data);
        src.CheckReorderToOpMem(bn_bwd_pd->src_desc(), cpu_engine_, context);
        src_data = static_cast<T*>(src.GetOpMem().get_data_handle());
      }
      const size_t kDiffSrcIndex = 0;
      MklDnnShape dnn_shape_diff_src;
      TensorShape tf_shape_diff_src;
      dnn_shape_diff_src.SetMklTensor(true);
      auto diff_src_pd = bn_bwd->GetDiffSrcPd();
      dnn_shape_diff_src.SET_MKL_LAYOUT(diff_src_pd);
      dnn_shape_diff_src.SetElemType(MklDnnType<T>());
      dnn_shape_diff_src.SetTfLayout(src_dims.size(), src_dims, mkl_tensor_fmt);
      dnn_shape_diff_src.SetTfDimOrder(src_dims.size(), tensor_format_);
      tf_shape_diff_src.AddDim(diff_src_pd.get_size() / sizeof(T));
      if (native_format) {
        tf_shape_diff_src = dnn_shape_diff_src.GetTfShape();
      }
      AllocateOutputSetMklShape(context, kDiffSrcIndex, &diff_src_tensor,
                                tf_shape_diff_src, dnn_shape_diff_src,
                                native_format);
      U* mean_data =
          static_cast<U*>(const_cast<U*>(saved_mean_tensor.flat<U>().data()));
      U* variance_data = static_cast<U*>(
          const_cast<U*>(saved_variance_tensor.flat<U>().data()));
#ifndef ENABLE_ONEDNN_V3
      U* scale_shift_data = scale_shift_data_tf;
      U* diff_scale_shift_data =
          static_cast<U*>(diff_scale_shift.GetAllocatedBuffer());
#else
      U* scale_data = scale_data_tf;
      U* diff_scale_data = static_cast<U*>(diff_scale.GetAllocatedBuffer());
      U* diff_shift_data = static_cast<U*>(diff_shift.GetAllocatedBuffer());
#endif  
      T* diff_src_data = static_cast<T*>(diff_src_tensor->flat<T>().data());
      U* res_space_data =
          ((reserved_space) ? static_cast<U*>(const_cast<U*>(
                                  reserved_space_tensor.flat<U>().data()))
                            : nullptr);
      std::shared_ptr<stream> bwd_cpu_stream;
      bwd_cpu_stream.reset(CreateStream(&eigen_tp, bn_bwd->GetEngine()));
      bn_bwd->Execute(src_data, mean_data, variance_data, diff_dst_data,
                      GET_SCALE_DATA_BUFFER, diff_src_data,
                      GET_DIFF_SCALE_SHIFT_DATA_BUFFERS, res_space_data,
                      bwd_cpu_stream);
      Tensor* diff_scale_tensor = nullptr;
      Tensor* diff_shift_tensor = nullptr;
      AllocateTFOutputs(context, scale_tensor.shape(), &diff_scale_tensor,
                        &diff_shift_tensor);
      auto diff_scale_data_out = diff_scale_tensor->flat<U>().data();
      auto diff_shift_data_out = diff_shift_tensor->flat<U>().data();
      std::memcpy(reinterpret_cast<char*>(diff_scale_data_out),
                  reinterpret_cast<char*>(GET_DIFF_SCALE_DATA_BUFFER),
                  depth_ * sizeof(U));
      std::memcpy(reinterpret_cast<char*>(diff_shift_data_out),
                  reinterpret_cast<char*>(GET_DIFF_SHIFT_DATA_BUFFER),
                  depth_ * sizeof(U));
    } catch (dnnl::error& e) {
      string error_msg = "Status: " + std::to_string(e.status) +
                         ", message: " + string(e.message) + ", in file " +
                         string(__FILE__) + ":" + std::to_string(__LINE__);
      OP_REQUIRES_OK(context,
                     absl::AbortedError(absl::StrCat(
                         "Operation received an exception:", error_msg)));
    }
  }
 private:
  float epsilon_;
  TensorFormat tensor_format_;
  size_t depth_;  
  bool is_training_;
  engine cpu_engine_ = engine(engine::kind::cpu, 0);
  void ExtractParams(OpKernelContext* context) {
    const Tensor& input = MklGetInput(context, 0);
    depth_ = static_cast<int>(GetTensorDim(input, tensor_format_, 'C'));
  }
  void HandleEmptyInput(OpKernelContext* context, TensorShape tf_shape_src,
                        TensorShape tf_shape_scale_shift,
                        Tensor** diff_src_tensor) {
    const size_t kDiffSrcIndex = 0;
    MklDnnShape dnn_shape_diff_src;
    dnn_shape_diff_src.SetMklTensor(false);
    AllocateOutputSetMklShape(context, kDiffSrcIndex, diff_src_tensor,
                              tf_shape_src, dnn_shape_diff_src, native_format);
    auto diff_src_data = (*diff_src_tensor)->flat<T>().data();
    std::fill_n(diff_src_data, (*diff_src_tensor)->shape().num_elements(),
                static_cast<T>(0));
    Tensor* diff_scale_tensor = nullptr;
    Tensor* diff_shift_tensor = nullptr;
    AllocateTFOutputs(context, tf_shape_scale_shift, &diff_scale_tensor,
                      &diff_shift_tensor);
  }
  void AllocateTFOutputs(OpKernelContext* context,
                         TensorShape tf_shape_scale_shift,
                         Tensor** diff_scale_tensor,
                         Tensor** diff_shift_tensor) {
    DCHECK(diff_scale_tensor);
    DCHECK(diff_shift_tensor);
    const size_t kDiffScaleIndex = 1;
    const size_t kDiffShiftIndex = 2;
    const size_t kP1Index = 3;
    const size_t kP2Index = 4;
    MklDnnShape mkl_shape_diff_scale;
    mkl_shape_diff_scale.SetMklTensor(false);
    AllocateOutputSetMklShape(context, kDiffScaleIndex, diff_scale_tensor,
                              tf_shape_scale_shift, mkl_shape_diff_scale,
                              native_format);
    DCHECK(*diff_scale_tensor);
    auto diff_scale_data = (*diff_scale_tensor)->flat<U>().data();
    std::fill_n(diff_scale_data, (*diff_scale_tensor)->shape().num_elements(),
                static_cast<U>(0));
    MklDnnShape mkl_shape_diff_shift;
    mkl_shape_diff_shift.SetMklTensor(false);
    AllocateOutputSetMklShape(context, kDiffShiftIndex, diff_shift_tensor,
                              tf_shape_scale_shift, mkl_shape_diff_shift,
                              native_format);
    DCHECK(*diff_shift_tensor);
    auto diff_shift_data = (*diff_shift_tensor)->flat<U>().data();
    std::fill_n(diff_shift_data, (*diff_shift_tensor)->shape().num_elements(),
                static_cast<U>(0));
    Tensor *p1_tensor = nullptr, *p2_tensor = nullptr;
    MklDnnShape mkl_shape_p;
    mkl_shape_p.SetMklTensor(false);
    AllocateOutputSetMklShape(context, kP1Index, &p1_tensor, TensorShape({}),
                              mkl_shape_p, native_format);
    std::fill_n(p1_tensor->flat<U>().data(), p1_tensor->shape().num_elements(),
                static_cast<U>(0));
    AllocateOutputSetMklShape(context, kP2Index, &p2_tensor, TensorShape({}),
                              mkl_shape_p, native_format);
    std::fill_n(p2_tensor->flat<U>().data(), p2_tensor->shape().num_elements(),
                static_cast<U>(0));
  }
  memory::dims GetMeanVarianceDims() { return memory::dims({1, depth_}); }
};
#define REGISTER_MKL_FUSED_BATCHNORM_CPU(T)                    \
  REGISTER_KERNEL_BUILDER(                                     \
      Name("_MklFusedBatchNorm")                               \
          .Device(DEVICE_CPU)                                  \
          .TypeConstraint<T>("T")                              \
          .Label(mkl_op_registry::kMklLayoutDependentOpLabel), \
      MklFusedBatchNormOp<CPUDevice, T, T, false, false>);     \
  REGISTER_KERNEL_BUILDER(                                     \
      Name("_MklNativeFusedBatchNorm")                         \
          .Device(DEVICE_CPU)                                  \
          .TypeConstraint<T>("T")                              \
          .Label(mkl_op_registry::kMklNameChangeOpLabel),      \
      MklFusedBatchNormOp<CPUDevice, T, T, false, false, true>);
TF_CALL_float(REGISTER_MKL_FUSED_BATCHNORM_CPU);
TF_CALL_bfloat16(REGISTER_MKL_FUSED_BATCHNORM_CPU);
TF_CALL_half(REGISTER_MKL_FUSED_BATCHNORM_CPU);
#undef REGISTER_MKL_FUSED_BATCHNORM_CPU
#define REGISTER_MKL_FUSED_BATCHNORM_V2_CPU(T, U)              \
  REGISTER_KERNEL_BUILDER(                                     \
      Name("_MklFusedBatchNormV2")                             \
          .Device(DEVICE_CPU)                                  \
          .TypeConstraint<T>("T")                              \
          .TypeConstraint<U>("U")                              \
          .Label(mkl_op_registry::kMklLayoutDependentOpLabel), \
      MklFusedBatchNormOp<CPUDevice, T, U, false, false>);     \
  REGISTER_KERNEL_BUILDER(                                     \
      Name("_MklNativeFusedBatchNormV2")                       \
          .Device(DEVICE_CPU)                                  \
          .TypeConstraint<T>("T")                              \
          .TypeConstraint<U>("U")                              \
          .Label(mkl_op_registry::kMklNameChangeOpLabel),      \
      MklFusedBatchNormOp<CPUDevice, T, U, false, false, true>);
REGISTER_MKL_FUSED_BATCHNORM_V2_CPU(float, float);
REGISTER_MKL_FUSED_BATCHNORM_V2_CPU(bfloat16, float);
REGISTER_MKL_FUSED_BATCHNORM_V2_CPU(Eigen::half, float);
#undef REGISTER_MKL_FUSED_BATCHNORM_V2_CPU
#define REGISTER_MKL_FUSED_BATCHNORM_GRAD_CPU(T)               \
  REGISTER_KERNEL_BUILDER(                                     \
      Name("_MklFusedBatchNormGrad")                           \
          .Device(DEVICE_CPU)                                  \
          .TypeConstraint<T>("T")                              \
          .Label(mkl_op_registry::kMklLayoutDependentOpLabel), \
      MklFusedBatchNormGradOp<CPUDevice, T, T, false>);        \
  REGISTER_KERNEL_BUILDER(                                     \
      Name("_MklNativeFusedBatchNormGrad")                     \
          .Device(DEVICE_CPU)                                  \
          .TypeConstraint<T>("T")                              \
          .Label(mkl_op_registry::kMklNameChangeOpLabel),      \
      MklFusedBatchNormGradOp<CPUDevice, T, T, false, true>);
TF_CALL_float(REGISTER_MKL_FUSED_BATCHNORM_GRAD_CPU);
TF_CALL_bfloat16(REGISTER_MKL_FUSED_BATCHNORM_GRAD_CPU);
#undef REGISTER_MKL_FUSED_BATCHNORM_GRAD_CPU
#define REGISTER_MKL_FUSED_BATCHNORM_GRAD_V2_CPU(T, U)         \
  REGISTER_KERNEL_BUILDER(                                     \
      Name("_MklFusedBatchNormGradV2")                         \
          .Device(DEVICE_CPU)                                  \
          .TypeConstraint<T>("T")                              \
          .TypeConstraint<U>("U")                              \
          .Label(mkl_op_registry::kMklLayoutDependentOpLabel), \
      MklFusedBatchNormGradOp<CPUDevice, T, U, false>);        \
  REGISTER_KERNEL_BUILDER(                                     \
      Name("_MklNativeFusedBatchNormGradV2")                   \
          .Device(DEVICE_CPU)                                  \
          .TypeConstraint<T>("T")                              \
          .TypeConstraint<U>("U")                              \
          .Label(mkl_op_registry::kMklNameChangeOpLabel),      \
      MklFusedBatchNormGradOp<CPUDevice, T, U, false, true>);
REGISTER_MKL_FUSED_BATCHNORM_GRAD_V2_CPU(float, float);
REGISTER_MKL_FUSED_BATCHNORM_GRAD_V2_CPU(bfloat16, float);
REGISTER_MKL_FUSED_BATCHNORM_GRAD_V2_CPU(Eigen::half, float);
#undef REGISTER_MKL_FUSED_BATCHNORM_GRAD_V2_CPU
#define REGISTER_MKL_FUSED_BATCHNORM_V3_CPU(T, U)               \
  REGISTER_KERNEL_BUILDER(                                      \
      Name("_MklFusedBatchNormV3")                              \
          .Device(DEVICE_CPU)                                   \
          .TypeConstraint<T>("T")                               \
          .TypeConstraint<U>("U")                               \
          .Label(mkl_op_registry::kMklLayoutDependentOpLabel),  \
      MklFusedBatchNormOp<CPUDevice, T, U, true, false>);       \
  REGISTER_KERNEL_BUILDER(                                      \
      Name("_MklFusedBatchNormEx")                              \
          .Device(DEVICE_CPU)                                   \
          .TypeConstraint<T>("T")                               \
          .TypeConstraint<U>("U")                               \
          .Label(mkl_op_registry::kMklLayoutDependentOpLabel),  \
      MklFusedBatchNormOp<CPUDevice, T, U, true, true>);        \
  REGISTER_KERNEL_BUILDER(                                      \
      Name("_MklNativeFusedBatchNormV3")                        \
          .Device(DEVICE_CPU)                                   \
          .TypeConstraint<T>("T")                               \
          .TypeConstraint<U>("U")                               \
          .Label(mkl_op_registry::kMklNameChangeOpLabel),       \
      MklFusedBatchNormOp<CPUDevice, T, U, true, false, true>); \
  REGISTER_KERNEL_BUILDER(                                      \
      Name("_MklNativeFusedBatchNormEx")                        \
          .Device(DEVICE_CPU)                                   \
          .TypeConstraint<T>("T")                               \
          .TypeConstraint<U>("U")                               \
          .Label(mkl_op_registry::kMklNameChangeOpLabel),       \
      MklFusedBatchNormOp<CPUDevice, T, U, true, true, true>);
REGISTER_MKL_FUSED_BATCHNORM_V3_CPU(float, float);
REGISTER_MKL_FUSED_BATCHNORM_V3_CPU(bfloat16, float);
REGISTER_MKL_FUSED_BATCHNORM_V3_CPU(Eigen::half, float);
#undef REGISTER_MKL_FUSED_BATCHNORM_V3_CPU
REGISTER_KERNEL_BUILDER(Name("_FusedBatchNormEx")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<float>("T")
                            .TypeConstraint<float>("U"),
                        NoOp);
REGISTER_KERNEL_BUILDER(Name("_FusedBatchNormEx")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<bfloat16>("T")
                            .TypeConstraint<float>("U"),
                        NoOp);
#define REGISTER_MKL_FUSED_BATCHNORM_GRAD_V3_CPU(T, U)         \
  REGISTER_KERNEL_BUILDER(                                     \
      Name("_MklFusedBatchNormGradV3")                         \
          .Device(DEVICE_CPU)                                  \
          .TypeConstraint<T>("T")                              \
          .TypeConstraint<U>("U")                              \
          .Label(mkl_op_registry::kMklLayoutDependentOpLabel), \
      MklFusedBatchNormGradOp<CPUDevice, T, U, true>);         \
  REGISTER_KERNEL_BUILDER(                                     \
      Name("_MklNativeFusedBatchNormGradV3")                   \
          .Device(DEVICE_CPU)                                  \
          .TypeConstraint<T>("T")                              \
          .TypeConstraint<U>("U")                              \
          .Label(mkl_op_registry::kMklNameChangeOpLabel),      \
      MklFusedBatchNormGradOp<CPUDevice, T, U, true, true>);
REGISTER_MKL_FUSED_BATCHNORM_GRAD_V3_CPU(float, float);
REGISTER_MKL_FUSED_BATCHNORM_GRAD_V3_CPU(bfloat16, float);
#undef REGISTER_MKL_FUSED_BATCHNORM_GRAD_V3_CPU
}  
#undef FORWARD_INFERENCE
#undef GET_DIFF_SCALE_DATA_BUFFER
#undef GET_DIFF_SCALE_SHIFT_DATA_BUFFERS
#undef GET_DIFF_SHIFT_DATA_BUFFER
#undef GET_SCALE_AND_SHIFT_FLAGS
#undef GET_SCALE_DATA_BUFFER
#undef IS_SCALE_AND_SHIFT_FLAG_SET
#undef SCALE_SHIFT_NET_ARGS
#undef SET_MKL_LAYOUT
#endif  