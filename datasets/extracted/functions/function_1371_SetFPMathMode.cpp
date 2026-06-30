#ifndef TENSORFLOW_CORE_UTIL_MKL_UTIL_H_
#define TENSORFLOW_CORE_UTIL_MKL_UTIL_H_
#ifdef INTEL_MKL
#include <list>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "dnnl.hpp"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/graph/mkl_graph_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/util/env_var.h"
#include "tensorflow/core/util/onednn_env_vars.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"
#if defined(DNNL_AARCH64_USE_ACL) && defined(ENABLE_ONEDNN_OPENMP)
#include "tensorflow/core/platform/mutex.h"
#endif
#include "xla/tsl/util/onednn_threadpool.h"
using dnnl::engine;
using dnnl::memory;
using dnnl::primitive;
using dnnl::reorder;
using dnnl::stream;
using CPUDevice = Eigen::ThreadPoolDevice;
using MemoryArgsMap = std::unordered_map<int, memory>;
using ReorderPd = dnnl::reorder::primitive_desc;
#ifdef _WIN32
typedef unsigned int uint;
#endif
namespace tensorflow {
typedef enum {
  Dim_N = 0,
  Dim_C = 1,
  Dim_H = 2,
  Dim_W = 3,
  Dim_O = 0,
  Dim_I = 1
} MklDnnDims;
typedef enum {
  Dim3d_N = 0,
  Dim3d_C = 1,
  Dim3d_D = 2,
  Dim3d_H = 3,
  Dim3d_W = 4,
  Dim3d_O = 0,
  Dim3d_I = 1
} MklDnnDims3D;
typedef enum {
  TF_2DFILTER_DIM_H = 0,
  TF_2DFILTER_DIM_W = 1,
  TF_2DFILTER_DIM_I = 2,
  TF_2DFILTER_DIM_O = 3
} TFFilterDims2d;
typedef enum {
  TF_3DFILTER_DIM_P = 0,
  TF_3DFILTER_DIM_H = 1,
  TF_3DFILTER_DIM_W = 2,
  TF_3DFILTER_DIM_I = 3,
  TF_3DFILTER_DIM_O = 4
} TFFilterDims3d;
typedef enum {
  MKL_GROUP_FILTER_DIM_G = 0,
  MKL_GROUP_FILTER_DIM_O = 1,
  MKL_GROUP_FILTER_DIM_I = 2,
  MKL_GROUP_FILTER_DIM_H = 3,
  MKL_GROUP_FILTER_DIM_W = 4
} MklDnnFilterGroupDims;
enum class MklQuantization {
  QUANTIZED_VERSION,
  FP_VERSION,
};
static const int kSmallBatchSize = 32;
enum class OneDNNMathModeSetting {
  kNone = 0,
  kBF16,
};
inline OneDNNMathModeSetting SetFPMathMode() {
  static OneDNNMathModeSetting math_mode = [] {
    OneDNNMathModeSetting mode = OneDNNMathModeSetting::kNone;
    if (FPMathModeSetting() == "BF16") {
      if (dnnl::set_default_fpmath_mode(dnnl::fpmath_mode::bf16) ==
          dnnl::status::success) {
        mode = OneDNNMathModeSetting::kBF16;
      }
    }
    return mode;
  }();
  return math_mode;
}
inline void execute_primitives(
    std::vector<dnnl::primitive>& primitives, std::shared_ptr<stream> stream,
    std::vector<std::unordered_map<int, memory>>& net_args) {
  DCHECK_EQ(primitives.size(), net_args.size());
  for (size_t i = 0; i < primitives.size(); ++i) {
    primitives.at(i).execute(*stream, net_args.at(i));
  }
}
#ifndef ENABLE_ONEDNN_V3
#define ARE_MEMORY_DESCS_EQUAL(md1, md2) dnnl_memory_desc_equal(&md1, &md2)
#define CREATE_MEMORY_DESC_USING_STRIDES dnnl_memory_desc_init_by_strides
#define GET_DATA_TYPE data_type
#define GET_DIMS dims
#define GET_INNER_BLKS format_desc.blocking.inner_blks
#define GET_INNER_DIMS(dims, dims_1) dims_1
#define GET_INNER_IDXS format_desc.blocking.inner_idxs
#define GET_INNER_NBLKS format_desc.blocking.inner_nblks
#define GET_MEMORY_DESC get_desc().data
#define GET_MEMORY_DESC_FLAGS extra.flags
#define GET_MEMORY_DESC_USING_MKLDNN_SHAPE_PTR GetMklLayout().data
#define GET_NDIMS ndims
#define GET_STRIDES format_desc.blocking.strides
#define GET_STRIDES_DIMS(dims, dims_outer_blocks) dims_outer_blocks
#define INIT_DIMS_FROM_DESC(in_dims, md) in_dims(md.dims, &md.dims[md.ndims])
#define MEMORY_DESC dnnl_memory_desc_t
#else
#define ARE_MEMORY_DESCS_EQUAL(md1, md2) md1 == md2
#define CREATE_MEMORY_DESC_USING_STRIDES dnnl_memory_desc_create_with_strides
#define GET_DATA_TYPE get_data_type()
#define GET_DIMS get_dims()
#define GET_INNER_BLKS get_inner_blks()
#define GET_INNER_DIMS(dims, dims_1) dims
#define GET_INNER_IDXS get_inner_idxs()
#define GET_INNER_NBLKS get_inner_nblks()
#define GET_MEMORY_DESC get_desc()
#define GET_MEMORY_DESC_FLAGS get_size()
#define GET_MEMORY_DESC_USING_MKLDNN_SHAPE_PTR GetMklLayout()
#define GET_NDIMS get_ndims()
#define GET_STRIDES get_strides()
#define GET_STRIDES_DIMS(dims, dims_outer_blocks) dims
#define INIT_DIMS_FROM_DESC(in_dims, md) in_dims = md.get_dims()
#define MEMORY_DESC memory::desc
#endif  
enum class MklTensorFormat {
  FORMAT_NHWC = 0,
  FORMAT_NCHW = 1,
  FORMAT_NDHWC = 2,
  FORMAT_NCDHW = 3,
  FORMAT_X = 4,
  FORMAT_NC = 5,
  FORMAT_TNC = 6,
  FORMAT_BLOCKED = 7,
  FORMAT_INVALID = 8,
};
memory::format_tag MklTensorFormatToMklDnnDataFormat(MklTensorFormat format);
TensorFormat MklDnn3DDataFormatToTFDataFormat(MklTensorFormat format);
TensorFormat MklDnnDataFormatToTFDataFormat(MklTensorFormat format);
memory::dims CalculateTFStrides(const memory::dims& dims_tf_order);
Status CreateBlockedMemDescHelper(const memory::dims& dim,
                                  const memory::dims& strides,
                                  memory::data_type dtype,
                                  dnnl_memory_desc_t* blocked_md);
inline std::ostream& operator<<(std::ostream& os,
                                const memory::format_tag& tag) {
  if (tag == memory::format_tag::undef) {
    os << "undef";
  } else if (tag == memory::format_tag::any) {
    os << "any";
  } else {
    os << "invalid";
  }
  return os;
}
inline void operator<<(std::ostream& os, const MklTensorFormat& format) {
  if (format == MklTensorFormat::FORMAT_NHWC) {
    os << "FORMAT_NHWC";
  } else if (format == MklTensorFormat::FORMAT_NCHW) {
    os << "FORMAT_NCHW";
  } else if (format == MklTensorFormat::FORMAT_NDHWC) {
    os << "FORMAT_NDHWC";
  } else if (format == MklTensorFormat::FORMAT_NCDHW) {
    os << "FORMAT_NCDHW";
  } else if (format == MklTensorFormat::FORMAT_X) {
    os << "FORMAT_X";
  } else if (format == MklTensorFormat::FORMAT_NC) {
    os << "FORMAT_NC";
  } else if (format == MklTensorFormat::FORMAT_TNC) {
    os << "FORMAT_TNC";
  } else if (format == MklTensorFormat::FORMAT_BLOCKED) {
    os << "FORMAT_BLOCKED";
  } else {
    os << "INVALID FORMAT";
  }
}
template <typename T>
inline bool array_cmp(const T* a1, const T* a2, size_t size) {
  for (size_t i = 0; i < size; ++i)
    if (a1[i] != a2[i]) return false;
  return true;
}
inline dnnl::stream* CreateStream(tsl::OneDnnThreadPool* eigen_tp,
                                  const engine& engine) {
#ifndef ENABLE_ONEDNN_OPENMP
  if (eigen_tp != nullptr) {
    stream* tp_stream =
        new stream(dnnl::threadpool_interop::make_stream(engine, eigen_tp));
    return tp_stream;
  } else {
    stream* tp_stream = new stream(engine);
    return tp_stream;
  }
#else
  stream* tp_stream = new stream(engine);
  return tp_stream;
#endif  
}
class MklDnnShape {
 private:
  struct MklShapeData {
    bool is_mkl_tensor_ = false;
    size_t dimension_ = 0;
    dnnl_dims_t sizes_;  
    MklTensorFormat tf_data_format_ = MklTensorFormat::FORMAT_BLOCKED;
    memory::data_type T_ = memory::data_type::undef;
    MEMORY_DESC mkl_md_;
    dnnl_dims_t map_;
  };
  MklShapeData data_;
  typedef std::remove_extent<dnnl_dims_t>::type dnnl_dim_t;
#define INVALID_DIM_SIZE -1
 public:
  MklDnnShape() : data_{} {
    for (size_t i = 0; i < sizeof(data_.sizes_) / sizeof(data_.sizes_[0]);
         ++i) {
      data_.sizes_[i] = -1;
    }
    for (size_t i = 0; i < sizeof(data_.map_) / sizeof(data_.map_[0]); ++i) {
      data_.map_[i] = -1;
    }
  }
  ~MklDnnShape() {}
  MklDnnShape(const MklDnnShape&) = delete;
  void operator=(const MklDnnShape&) = delete;  
  inline bool operator==(const MklDnnShape& input_shape) const {
    if (this->IsMklTensor() != input_shape.IsMklTensor()) {
      return false;
    }
    if (this->IsMklTensor()) {
      auto const& cur_md = this->GET_MEMORY_DESC_USING_MKLDNN_SHAPE_PTR;
      auto const& input_shape_md =
          input_shape.GET_MEMORY_DESC_USING_MKLDNN_SHAPE_PTR;
      return (this->GetTfShape() == input_shape.GetTfShape()) &&
             ARE_MEMORY_DESCS_EQUAL(cur_md, input_shape_md);
    }
    return true;
  }
  inline bool operator==(const TensorShape& input_shape) const {
    if (!this->IsMklTensor()) {
      return false;
    }
    return this->GetTfShape() == input_shape;
  }
  inline const bool IsMklTensor() const { return data_.is_mkl_tensor_; }
  inline void SetMklTensor(bool is_mkl_tensor) {
    data_.is_mkl_tensor_ = is_mkl_tensor;
  }
  inline void SetDimensions(const size_t dimension) {
    data_.dimension_ = dimension;
  }
  inline size_t GetDimension(char dimension) const {
    int index = GetMklDnnTensorDimIndex(dimension);
    CHECK(index >= 0 && index < this->GetDimension())
        << "Invalid index from the dimension: " << index << ", " << dimension;
    return this->DimSize(index);
  }
  inline size_t GetDimension3D(char dimension) const {
    int index = GetMklDnnTensor3DDimIndex(dimension);
    CHECK(index >= 0 && index < this->GetDimension())
        << "Invalid index from the dimension: " << index << ", " << dimension;
    return this->DimSize(index);
  }
  inline int32 GetMklDnnTensorDimIndex(char dimension) const {
    switch (dimension) {
      case 'N':
        return MklDnnDims::Dim_N;
      case 'C':
        return MklDnnDims::Dim_C;
      case 'H':
        return MklDnnDims::Dim_H;
      case 'W':
        return MklDnnDims::Dim_W;
      default:
        LOG(FATAL) << "Invalid dimension: " << dimension;
        return -1;  
    }
  }
  inline int32 GetMklDnnTensor3DDimIndex(char dimension) const {
    switch (dimension) {
      case 'N':
        return MklDnnDims3D::Dim3d_N;
      case 'C':
        return MklDnnDims3D::Dim3d_C;
      case 'D':
        return MklDnnDims3D::Dim3d_D;
      case 'H':
        return MklDnnDims3D::Dim3d_H;
      case 'W':
        return MklDnnDims3D::Dim3d_W;
      default:
        LOG(FATAL) << "Invalid dimension: " << dimension;
        return -1;  
    }
  }
  inline size_t GetDimension() const { return data_.dimension_; }
  inline const int* GetSizes() const {
    return reinterpret_cast<const int*>(&data_.sizes_[0]);
  }
  inline memory::dims GetSizesAsMklDnnDims() const {
    memory::dims retVal;
    if (data_.is_mkl_tensor_) {
      size_t dimensions = sizeof(data_.sizes_) / sizeof(data_.sizes_[0]);
      for (size_t i = 0; i < dimensions; i++) {
        if (data_.sizes_[i] != INVALID_DIM_SIZE)
          retVal.push_back(data_.sizes_[i]);
      }
    } else {
      CHECK_EQ(data_.is_mkl_tensor_, true);
    }
    return retVal;
  }
  inline int64 DimSize(int index) const {
    CHECK_LT(index, sizeof(data_.sizes_) / sizeof(data_.sizes_[0]));
    return data_.sizes_[index];
  }
  inline TensorShape GetTfShape() const {
    CHECK_EQ(data_.is_mkl_tensor_, true);
    std::vector<int32> shape(data_.dimension_, -1);
    if (data_.tf_data_format_ != MklTensorFormat::FORMAT_BLOCKED) {
      for (size_t idx = 0; idx < data_.dimension_; ++idx) {
        shape[idx] = data_.sizes_[TfDimIdx(idx)];
      }
    } else {
      for (size_t idx = 0; idx < data_.dimension_; ++idx) {
        shape[idx] = data_.sizes_[idx];
      }
    }
    TensorShape ts;
    bool ret = TensorShapeUtils::MakeShape(shape, &ts).ok();
    CHECK_EQ(ret, true);
    return ts;
  }
  inline void SetElemType(memory::data_type dt) { data_.T_ = dt; }
  inline const memory::data_type GetElemType() { return data_.T_; }
#ifndef ENABLE_ONEDNN_V3
  inline void SetMklLayout(memory::desc* md) {
    CHECK_NOTNULL(md);
    data_.mkl_md_ = md->data;
  }
#else
  inline void SetMklLayout(const memory::desc& md) { data_.mkl_md_ = md; }
#endif  
  inline const memory::desc GetMklLayout() const {
    return memory::desc(data_.mkl_md_);
  }
  inline MklTensorFormat GetTfDataFormat() const {
    return data_.tf_data_format_;
  }
  inline void SetTfLayout(size_t dims, const memory::dims& sizes,
                          MklTensorFormat format) {
    DCHECK_EQ(dims, sizes.size())
        << "SetTfLayout: Number of dimensions does not"
           "match with dimension array";
    data_.dimension_ = dims;
    for (size_t ii = 0; ii < dims; ++ii) {
      data_.sizes_[ii] = sizes[ii];
    }
    data_.tf_data_format_ = format;
    if (format != MklTensorFormat::FORMAT_BLOCKED) {
      if (dims == 2) {
        data_.map_[0] = MklDnnDims::Dim_N;
        data_.map_[1] = MklDnnDims::Dim_C;
      } else {
        SetTfDimOrder(dims, format);
      }
    }
  }
  inline const memory::desc GetTfLayout() const {
    memory::dims dims;
    for (size_t ii = 0; ii < data_.dimension_; ++ii) {
      dims.push_back(data_.sizes_[ii]);
    }
    if (data_.tf_data_format_ == MklTensorFormat::FORMAT_BLOCKED) {
      auto strides = CalculateTFStrides(dims);
      dnnl_memory_desc_t blocked_md;
      TF_CHECK_OK(
          CreateBlockedMemDescHelper(dims, strides, data_.T_, &blocked_md));
      return memory::desc(blocked_md);
    } else {
      auto format_tag =
          MklTensorFormatToMklDnnDataFormat(data_.tf_data_format_);
      return memory::desc(dims, data_.T_, format_tag);
    }
  }
  inline const memory::desc GetCurLayout() const {
    return IsMklTensor() ? GetMklLayout() : GetTfLayout();
  }
  inline void SetTfDimOrder(const size_t dimension, const dnnl_dims_t map) {
    CHECK(dimension == data_.dimension_);
    for (size_t ii = 0; ii < dimension; ii++) {
      data_.map_[ii] = map[ii];
    }
  }
  inline void SetTfDimOrder(const size_t dimension, TensorFormat data_format) {
    if (dimension == 5) {
      CHECK(dimension == data_.dimension_);
      data_.map_[GetTensorDimIndex<3>(data_format, '0')] =
          MklDnnDims3D::Dim3d_D;
      data_.map_[GetTensorDimIndex<3>(data_format, '1')] =
          MklDnnDims3D::Dim3d_H;
      data_.map_[GetTensorDimIndex<3>(data_format, '2')] =
          MklDnnDims3D::Dim3d_W;
      data_.map_[GetTensorDimIndex<3>(data_format, 'C')] =
          MklDnnDims3D::Dim3d_C;
      data_.map_[GetTensorDimIndex<3>(data_format, 'N')] =
          MklDnnDims3D::Dim3d_N;
    } else {
      CHECK_EQ(dimension, 4);
      CHECK(dimension == data_.dimension_);
      data_.map_[GetTensorDimIndex<2>(data_format, 'W')] = MklDnnDims::Dim_W;
      data_.map_[GetTensorDimIndex<2>(data_format, 'H')] = MklDnnDims::Dim_H;
      data_.map_[GetTensorDimIndex<2>(data_format, 'C')] = MklDnnDims::Dim_C;
      data_.map_[GetTensorDimIndex<2>(data_format, 'N')] = MklDnnDims::Dim_N;
    }
  }
  inline void SetTfDimOrder(const size_t dimension, MklTensorFormat format) {
    TensorFormat data_format = MklDnnDataFormatToTFDataFormat(format);
    SetTfDimOrder(dimension, data_format);
  }
  inline const dnnl_dim_t* GetTfToMklDimMap() const { return &data_.map_[0]; }
  inline size_t TfDimIdx(int index) const { return data_.map_[index]; }
  inline int64 TfDimSize(int index) const {
    return data_.sizes_[TfDimIdx(index)];
  }
  inline bool IsMklChannelDim(int d) const {
    return TfDimIdx(d) == MklDnnDims::Dim_C;
  }
  inline bool IsMklBatchDim(int d) const {
    return TfDimIdx(d) == MklDnnDims::Dim_N;
  }
  inline bool IsMklWidthDim(int d) const {
    return TfDimIdx(d) == MklDnnDims::Dim_W;
  }
  inline bool IsMklHeightDim(int d) const {
    return TfDimIdx(d) == MklDnnDims::Dim_H;
  }
  inline bool IsTensorInNCHWFormat() const {
    TensorFormat data_format = FORMAT_NCHW;
    return (IsMklBatchDim(GetTensorDimIndex<2>(data_format, 'N')) &&
            IsMklChannelDim(GetTensorDimIndex<2>(data_format, 'C')) &&
            IsMklHeightDim(GetTensorDimIndex<2>(data_format, 'H')) &&
            IsMklWidthDim(GetTensorDimIndex<2>(data_format, 'W')));
  }
  inline bool IsTensorInNHWCFormat() const {
    TensorFormat data_format = FORMAT_NHWC;
    return (IsMklBatchDim(GetTensorDimIndex<2>(data_format, 'N')) &&
            IsMklChannelDim(GetTensorDimIndex<2>(data_format, 'C')) &&
            IsMklHeightDim(GetTensorDimIndex<2>(data_format, 'H')) &&
            IsMklWidthDim(GetTensorDimIndex<2>(data_format, 'W')));
  }
  inline size_t GetSerializeBufferSize() const { return sizeof(MklShapeData); }
  void SerializeMklDnnShape(unsigned char* buf, size_t buf_size) const {
    CHECK(buf_size >= GetSerializeBufferSize())
        << "Buffer size is too small to SerializeMklDnnShape";
    *reinterpret_cast<MklShapeData*>(buf) = data_;
  }
  void DeSerializeMklDnnShape(const unsigned char* buf, size_t buf_size) {
    CHECK(buf_size >= sizeof(data_.is_mkl_tensor_))
        << "Buffer size is too small in DeSerializeMklDnnShape";
    const bool is_mkl_tensor = *reinterpret_cast<const bool*>(buf);
    if (is_mkl_tensor) {  
      CHECK(buf_size >= GetSerializeBufferSize())
          << "Buffer size is too small in DeSerializeMklDnnShape";
      data_ = *reinterpret_cast<const MklShapeData*>(buf);
    }
  }
};
inline Eigen::ThreadPoolInterface* EigenThreadPoolFromTfContext(
    OpKernelContext* context) {
  return context->device()
      ->tensorflow_cpu_worker_threads()
      ->workers->AsEigenThreadPool();
}
typedef std::vector<MklDnnShape> MklDnnShapeList;
template <typename T>
class MklDnnData;
inline void ExecutePrimitive(const std::vector<primitive>& net,
                             const std::vector<MemoryArgsMap>* net_args,
                             const engine& cpu_engine,
                             OpKernelContext* context = nullptr) {
  DCHECK(net_args);
  DCHECK_EQ(net.size(), net_args->size());
  std::unique_ptr<stream> cpu_stream;
  tsl::OneDnnThreadPool eigen_tp;
  if (context != nullptr) {
    Eigen::ThreadPoolInterface* eigen_interface =
        EigenThreadPoolFromTfContext(context);
    eigen_tp =
        tsl::OneDnnThreadPool(eigen_interface, ThreadPoolUseCallerThread());
    cpu_stream.reset(CreateStream(&eigen_tp, cpu_engine));
  } else {
    cpu_stream.reset(CreateStream(nullptr, cpu_engine));
  }
  for (size_t i = 0; i < net.size(); ++i) {
    net.at(i).execute(*cpu_stream, net_args->at(i));
  }
  cpu_stream->wait();
}
template <typename T>
inline Status ConvertMklToTF(OpKernelContext* context,
                             const Tensor& input_mkl_tensor,
                             const MklDnnShape& input_mkl_shape,
                             Tensor* output_tf_tensor) {
  try {
    if (!input_mkl_shape.IsMklTensor()) {
      *output_tf_tensor = input_mkl_tensor;
      return OkStatus();
    }
    TensorShape output_tf_shape = input_mkl_shape.GetTfShape();
    TF_CHECK_OK(context->allocate_temp(DataTypeToEnum<T>::v(), output_tf_shape,
                                       output_tf_tensor));
    engine cpu_engine(engine::kind::cpu, 0);
    MklDnnData<T> input(&cpu_engine);
    auto input_mkl_md = input_mkl_shape.GetMklLayout();
    auto output_tf_md = input_mkl_shape.GetTfLayout();
    input.SetUsrMem(input_mkl_md, &input_mkl_tensor);
    if (input.IsReorderNeeded(output_tf_md)) {
      std::vector<primitive> net;
      std::vector<MemoryArgsMap> net_args;
      bool status = input.CheckReorderToOpMem(output_tf_md, output_tf_tensor,
                                              net, net_args, cpu_engine);
      if (!status) {
        return absl::InternalError(
            "ConvertMklToTF(): Failed to create reorder for input");
      }
      ExecutePrimitive(net, &net_args, cpu_engine, context);
    } else {
      bool status =
          output_tf_tensor->CopyFrom(input_mkl_tensor, output_tf_shape);
      if (!status) {
        return absl::InternalError(
            "ConvertMklToTF(): Failed to forward input tensor to output");
      }
    }
    return OkStatus();
  } catch (dnnl::error& e) {
    string error_msg = "Status: " + std::to_string(e.status) +
                       ", message: " + string(e.message) + ", in file " +
                       string(__FILE__) + ":" + std::to_string(__LINE__);
    LOG(FATAL) << "Operation received an exception: " << error_msg;
  }
}
inline void GetMklShape(OpKernelContext* ctext, int n, MklDnnShape* mklshape,
                        bool eager_mode) {
  if (!eager_mode) {
    mklshape->DeSerializeMklDnnShape(
        ctext->input(GetTensorMetaDataIndex(n, ctext->num_inputs()))
            .flat<uint8>()
            .data(),
        ctext->input(GetTensorMetaDataIndex(n, ctext->num_inputs()))
                .flat<uint8>()
                .size() *
            sizeof(uint8));
  } else {
    mklshape->SetMklTensor(false);
  }
}
inline void GetMklShape(OpKernelContext* ctext, int n, MklDnnShape* mklshape) {
  GetMklShape(ctext, n, mklshape, false);
}
inline const Tensor& MklGetInput(OpKernelContext* ctext, int n) {
  return ctext->input(GetTensorDataIndex(n, ctext->num_inputs()));
}
inline void GetMklInputList(OpKernelContext* ctext, StringPiece name,
                            OpInputList* input_tensors) {
  CHECK_NOTNULL(input_tensors);
  TF_CHECK_OK(ctext->input_list(name, input_tensors));
}
inline void GetMklShapeList(OpKernelContext* ctext, StringPiece name,
                            MklDnnShapeList* mkl_shapes,
                            bool native_format = false) {
  if (!native_format) {
    OpInputList input_mkl_tensors;
    GetMklInputList(ctext, strings::StrCat("mkl_", name), &input_mkl_tensors);
    for (int i = 0; i < input_mkl_tensors.size(); i++) {
      (*mkl_shapes)[i].DeSerializeMklDnnShape(
          input_mkl_tensors[i].flat<uint8>().data(),
          input_mkl_tensors[i].flat<uint8>().size() * sizeof(uint8));
    }
  } else {
    for (int i = 0; i < mkl_shapes->size(); ++i) {
      (*mkl_shapes)[i].SetMklTensor(false);
    }
  }
}
inline TensorShape GetTfShape(OpKernelContext* context, size_t input_idx,
                              bool eager_mode = false) {
  CHECK_NOTNULL(context);
  CHECK_LT(input_idx, context->num_inputs());
  MklDnnShape input_mkl_shape;
  GetMklShape(context, input_idx, &input_mkl_shape, eager_mode);
  if (input_mkl_shape.IsMklTensor() && !eager_mode) {
    return input_mkl_shape.GetTfShape();
  } else {
    const Tensor& t = MklGetInput(context, input_idx);
    return t.shape();
  }
}
inline void AllocateOutputSetMklShape(OpKernelContext* ctext, int n,
                                      const MklDnnShape& mkl_shape) {
  Tensor* second_tensor = nullptr;
  TensorShape second_shape;
  second_shape.AddDim(mkl_shape.GetSerializeBufferSize());
  OP_REQUIRES_OK(ctext, ctext->allocate_output(
                            GetTensorMetaDataIndex(n, ctext->num_outputs()),
                            second_shape, &second_tensor));
  mkl_shape.SerializeMklDnnShape(
      second_tensor->flat<uint8>().data(),
      second_tensor->flat<uint8>().size() * sizeof(uint8));
}
inline void AllocateOutputSetMklShape(OpKernelContext* ctext, int n,
                                      Tensor** output,
                                      const TensorShape& tf_shape,
                                      const MklDnnShape& mkl_shape,
                                      bool eager_mode = false) {
  OP_REQUIRES_OK(
      ctext, ctext->allocate_output(GetTensorDataIndex(n, ctext->num_outputs()),
                                    tf_shape, output));
  if (!eager_mode) {
    Tensor* second_tensor = nullptr;
    TensorShape second_shape;
    second_shape.AddDim(mkl_shape.GetSerializeBufferSize());
    OP_REQUIRES_OK(ctext, ctext->allocate_output(
                              GetTensorMetaDataIndex(n, ctext->num_outputs()),
                              second_shape, &second_tensor));
    mkl_shape.SerializeMklDnnShape(
        second_tensor->flat<uint8>().data(),
        second_tensor->flat<uint8>().size() * sizeof(uint8));
  }
}
template <typename T>
inline void AllocTmpBuffer(OpKernelContext* context, Tensor* tensor_out,
                           const memory::desc& pd, void** buf_out) {
  TensorShape tf_shape;
  tf_shape.AddDim(pd.get_size() / sizeof(T) + 1);
  OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::v(),
                                                 tf_shape, tensor_out));
  *buf_out = static_cast<void*>(tensor_out->flat<T>().data());
}
template <typename T>
inline void AllocTmpBuffer(OpKernelContext* context, Tensor* tensor_out,
                           TensorShape tf_shape) {
  OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::v(),
                                                 tf_shape, tensor_out));
}
template <typename T>
struct UserScratchPad {
  template <typename MklPrim>
  inline void AllocateSPTensor(MklPrim* mkl_prim, OpKernelContext* context) {
    allocated_ = false;
    auto spad_md = mkl_prim->GetScratchPadDesc();
    size_t spad_size = spad_md.get_size();
    if (spad_size == 0) return;
    size_t allocate_size = (spad_size + sizeof(T) - 1) / sizeof(T);
    TensorShape tf_shape;
    tf_shape.AddDim(allocate_size);
    AllocTmpBuffer<T>(context, &scratch_pad_, tf_shape);
    allocated_ = true;
  }
  inline void* Get() {
    if (allocated_) {
      return static_cast<void*>(scratch_pad_.flat<T>().data());
    } else {
      return nullptr;
    }
  }
 private:
  Tensor scratch_pad_;
  bool allocated_ = false;
};
inline void GetStridesFromSizes(MklTensorFormat data_format, size_t* strides,
                                const size_t* sizes) {
  DCHECK_NE(data_format, MklTensorFormat::FORMAT_INVALID);
  if (data_format == MklTensorFormat::FORMAT_NHWC) {
    strides[0] = sizes[2];
    strides[1] = sizes[0] * sizes[2];
    strides[2] = 1;
    strides[3] = sizes[0] * sizes[1] * sizes[2];
  } else {
    strides[0] = 1;
    strides[1] = sizes[0];
    strides[2] = sizes[0] * sizes[1];
    strides[3] = sizes[0] * sizes[1] * sizes[2];
  }
}
inline void CopyMklTensorInToOut(OpKernelContext* context, int idx_in,
                                 int idx_out) {
  int num_inputs = context->num_inputs();
  int num_outputs = context->num_outputs();
  int idx_data_in = GetTensorDataIndex(idx_in, num_inputs);
  int idx_meta_in = GetTensorMetaDataIndex(idx_in, num_inputs);
  int idx_data_out = GetTensorDataIndex(idx_out, num_outputs);
  int idx_meta_out = GetTensorMetaDataIndex(idx_out, num_outputs);
  const Tensor& data = context->input(idx_data_in);
  const Tensor& meta = context->input(idx_meta_in);
  Tensor output(data.dtype());
  Tensor meta_output(meta.dtype());
  CHECK(output.CopyFrom(data, data.shape()));
  CHECK(meta_output.CopyFrom(meta, meta.shape()));
  context->set_output(idx_data_out, output);
  context->set_output(idx_meta_out, meta_output);
}
inline void CopyTfTensorInToOutWithShape(OpKernelContext* context, int idx_in,
                                         int idx_out,
                                         const TensorShape& shape) {
  int num_inputs = context->num_inputs();
  int num_outputs = context->num_outputs();
  int idx_data_in = GetTensorDataIndex(idx_in, num_inputs);
  int idx_data_out = GetTensorDataIndex(idx_out, num_outputs);
  const Tensor& data = context->input(idx_data_in);
  MklDnnShape mkl_shape_output;
  mkl_shape_output.SetMklTensor(false);
  AllocateOutputSetMklShape(context, idx_out, mkl_shape_output);
  Tensor output(data.dtype());
  CHECK(output.CopyFrom(data, shape));
  context->set_output(idx_data_out, output);
}
inline void ForwardTfTensorInToOut(OpKernelContext* context, int idx_in,
                                   int idx_out) {
  int num_inputs = context->num_inputs();
  int num_outputs = context->num_outputs();
  int idx_data_in = GetTensorDataIndex(idx_in, num_inputs);
  int idx_data_out = GetTensorDataIndex(idx_out, num_outputs);
  MklDnnShape dnn_shape_output;
  dnn_shape_output.SetMklTensor(false);
  AllocateOutputSetMklShape(context, idx_out, dnn_shape_output);
  if (IsRefType(context->input_dtype(idx_data_in))) {
    context->forward_ref_input_to_ref_output(idx_data_in, idx_data_out);
  } else {
    context->set_output(idx_data_out, context->input(idx_data_in));
  }
}
inline void ForwardMklTensorInToOut(OpKernelContext* context, int idx_in,
                                    int idx_out) {
  int num_inputs = context->num_inputs();
  int num_outputs = context->num_outputs();
  int idx_data_in = GetTensorDataIndex(idx_in, num_inputs);
  int idx_meta_in = GetTensorMetaDataIndex(idx_in, num_inputs);
  int idx_data_out = GetTensorDataIndex(idx_out, num_outputs);
  int idx_meta_out = GetTensorMetaDataIndex(idx_out, num_outputs);
  if (IsRefType(context->input_dtype(idx_data_in))) {
    context->forward_ref_input_to_ref_output(idx_data_in, idx_data_out);
    context->forward_ref_input_to_ref_output(idx_meta_in, idx_meta_out);
  } else {
    context->set_output(idx_data_out, context->input(idx_data_in));
    context->set_output(idx_meta_out, context->input(idx_meta_in));
  }
}
inline void SetDummyMklDnnShapeOutput(OpKernelContext* context,
                                      uint32 idx_data_out) {
  MklDnnShape mkl_shape_output;
  mkl_shape_output.SetMklTensor(false);
  AllocateOutputSetMklShape(context, idx_data_out, mkl_shape_output);
}
inline bool ForwardMklTensorInToOutWithMklShape(OpKernelContext* context,
                                                int idx_in, int idx_out,
                                                Tensor** output,
                                                const MklDnnShape& mkl_shape,
                                                bool always_forward = true) {
  int num_inputs = context->num_inputs();
  int num_outputs = context->num_outputs();
  int idx_data_in = GetTensorDataIndex(idx_in, num_inputs);
  int idx_data_out = GetTensorDataIndex(idx_out, num_outputs);
  bool is_forwarded = false;
  const Tensor& input_tensor = context->input(idx_data_in);
  const auto output_shape = input_tensor.shape();
  if (always_forward) {
    if (IsRefType(context->input_dtype(idx_data_in))) {
      context->forward_ref_input_to_ref_output(idx_data_in, idx_data_out);
    } else {
      context->set_output(idx_data_out, input_tensor);
    }
  } else {
    is_forwarded = context->forward_input_to_output_with_shape(
        idx_data_in, idx_data_out, output_shape, output);
  }
  if (is_forwarded || always_forward) {
    AllocateOutputSetMklShape(context, idx_out, mkl_shape);
    return true;
  }
  return false;
}
inline void ForwardMklMetaDataInToOut(OpKernelContext* context,
                                      uint32 idx_data_in,
                                      uint32_t idx_data_out) {
  uint32 idx_meta_in =
      GetTensorMetaDataIndex(idx_data_in, context->num_inputs());
  uint32 idx_meta_out =
      GetTensorMetaDataIndex(idx_data_out, context->num_outputs());
  if (IsRefType(context->input_dtype(idx_data_in))) {
    context->forward_ref_input_to_ref_output(idx_meta_in, idx_meta_out);
  } else {
    context->set_output(idx_meta_out, context->input(idx_meta_in));
  }
}
inline Tensor GetMklMetaTensor() {
  MklDnnShape non_mkl_shape;
  non_mkl_shape.SetMklTensor(false);
  auto size = static_cast<int64_t>(non_mkl_shape.GetSerializeBufferSize());
  Tensor tensor(DT_UINT8, {size});
  non_mkl_shape.SerializeMklDnnShape(tensor.flat<uint8>().data(),
                                     size * sizeof(uint8));
  return tensor;
}
template <typename T>
static memory::data_type MklDnnType();
template <>
memory::data_type MklDnnType<float>() {
  return memory::data_type::f32;
}
template <>
memory::data_type MklDnnType<quint8>() {
  return memory::data_type::u8;
}
template <>
memory::data_type MklDnnType<uint8>() {
  return memory::data_type::u8;
}
template <>
memory::data_type MklDnnType<qint8>() {
  return memory::data_type::s8;
}
template <>
memory::data_type MklDnnType<qint32>() {
  return memory::data_type::s32;
}
template <>
memory::data_type MklDnnType<bfloat16>() {
  return memory::data_type::bf16;
}
template <>
memory::data_type MklDnnType<Eigen::half>() {
  return memory::data_type::f16;
}
inline memory::format_tag MklTensorFormatToMklDnnDataFormat(
    MklTensorFormat format) {
  if (format == MklTensorFormat::FORMAT_NHWC) return memory::format_tag::nhwc;
  if (format == MklTensorFormat::FORMAT_NCHW) return memory::format_tag::nchw;
  if (format == MklTensorFormat::FORMAT_NDHWC) return memory::format_tag::ndhwc;
  if (format == MklTensorFormat::FORMAT_NCDHW) return memory::format_tag::ncdhw;
  if (format == MklTensorFormat::FORMAT_X) return memory::format_tag::x;
  if (format == MklTensorFormat::FORMAT_NC) return memory::format_tag::nc;
  if (format == MklTensorFormat::FORMAT_TNC) return memory::format_tag::tnc;
  return memory::format_tag::undef;
}
inline MklTensorFormat TFDataFormatToMklDnn3DDataFormat(TensorFormat format) {
  if (format == FORMAT_NHWC) return MklTensorFormat::FORMAT_NDHWC;
  if (format == FORMAT_NCHW) return MklTensorFormat::FORMAT_NCDHW;
  TF_CHECK_OK(absl::InvalidArgumentError("Unsupported data format"));
  return MklTensorFormat::FORMAT_INVALID;
}
inline MklTensorFormat TFDataFormatToMklDnnDataFormat(TensorFormat format) {
  if (format == FORMAT_NHWC) return MklTensorFormat::FORMAT_NHWC;
  if (format == FORMAT_NCHW) return MklTensorFormat::FORMAT_NCHW;
  TF_CHECK_OK(absl::InvalidArgumentError("Unsupported data format"));
  return MklTensorFormat::FORMAT_INVALID;
}
inline TensorFormat MklDnnDataFormatToTFDataFormat(MklTensorFormat format) {
  if (format == MklTensorFormat::FORMAT_NHWC ||
      format == MklTensorFormat::FORMAT_NDHWC)
    return FORMAT_NHWC;
  if (format == MklTensorFormat::FORMAT_NCHW ||
      format == MklTensorFormat::FORMAT_NCDHW)
    return FORMAT_NCHW;
  TF_CHECK_OK(absl::InvalidArgumentError("Unsupported data format"));
  return FORMAT_NHWC;
}
inline memory::dims TFShapeToMklDnnDims(const TensorShape& shape) {
  memory::dims dims(shape.dims());
  for (int d = 0; d < shape.dims(); ++d) {
    dims[d] = shape.dim_size(d);
  }
  return dims;
}
inline memory::dims TFShapeToMklDnnDimsInNCHW(const TensorShape& shape,
                                              TensorFormat format) {
  DCHECK_NE(TFDataFormatToMklDnnDataFormat(format),
            MklTensorFormat::FORMAT_INVALID);
  int n = shape.dim_size(GetTensorDimIndex(format, 'N'));
  int c = shape.dim_size(GetTensorDimIndex(format, 'C'));
  int h = shape.dim_size(GetTensorDimIndex(format, 'H'));
  int w = shape.dim_size(GetTensorDimIndex(format, 'W'));
  return memory::dims({n, c, h, w});
}
inline memory::dims TFShapeToMklDnnDimsInNCDHW(const TensorShape& shape,
                                               TensorFormat format) {
  DCHECK_NE(TFDataFormatToMklDnn3DDataFormat(format),
            MklTensorFormat::FORMAT_INVALID);
  int n = shape.dim_size(GetTensorDimIndex<3>(format, 'N'));
  int c = shape.dim_size(GetTensorDimIndex<3>(format, 'C'));
  int d = shape.dim_size(GetTensorDimIndex<3>(format, '0'));
  int h = shape.dim_size(GetTensorDimIndex<3>(format, '1'));
  int w = shape.dim_size(GetTensorDimIndex<3>(format, '2'));
  return memory::dims({n, c, d, h, w});
}
inline memory::dims MklDnnDimsInNCHW(const memory::dims& in_dims,
                                     TensorFormat format) {
  DCHECK_NE(TFDataFormatToMklDnnDataFormat(format),
            MklTensorFormat::FORMAT_INVALID);
  int n = in_dims[GetTensorDimIndex(format, 'N')];
  int c = in_dims[GetTensorDimIndex(format, 'C')];
  int h = in_dims[GetTensorDimIndex(format, 'H')];
  int w = in_dims[GetTensorDimIndex(format, 'W')];
  return memory::dims({n, c, h, w});
}
inline memory::dims MklDnnDimsInNCDHW(const memory::dims& in_dims,
                                      TensorFormat format) {
  DCHECK_NE(TFDataFormatToMklDnnDataFormat(format),
            MklTensorFormat::FORMAT_INVALID);
  int n = in_dims[GetTensorDimIndex<3>(format, 'N')];
  int c = in_dims[GetTensorDimIndex<3>(format, 'C')];
  int d = in_dims[GetTensorDimIndex<3>(format, '0')];
  int h = in_dims[GetTensorDimIndex<3>(format, '1')];
  int w = in_dims[GetTensorDimIndex<3>(format, '2')];
  return memory::dims({n, c, d, h, w});
}
inline TensorShape MklDnnDimsToTFShape(const memory::dims& dims) {
  std::vector<int32> shape(dims.size(), -1);
  for (int d = 0; d < dims.size(); d++) {
    shape[d] = dims[d];
  }
  TensorShape ret;
  CHECK_EQ(TensorShapeUtils::MakeShape(shape, &ret).ok(), true);
  return ret;
}
inline memory::dims CalculateTFStrides(const memory::dims& dims_tf_order) {
  CHECK_GT(dims_tf_order.size(), 0);
  memory::dims strides(dims_tf_order.size());
  int last_dim_idx = dims_tf_order.size() - 1;
  strides[last_dim_idx] = 1;
  for (int d = last_dim_idx - 1; d >= 0; d--) {
    strides[d] = strides[d + 1] * dims_tf_order[d + 1];
  }
  return strides;
}
inline Status CreateBlockedMemDescHelper(const memory::dims& dim,
                                         const memory::dims& strides,
                                         memory::data_type dtype,
                                         dnnl_memory_desc_t* blocked_md) {
  DCHECK_EQ(dim.size(), strides.size());
  const int kNumDims = dim.size();
  dnnl_dim_t* input_dims = new dnnl_dim_t[kNumDims];
  dnnl_dim_t* input_strides = new dnnl_dim_t[kNumDims];
  for (int i = 0; i < kNumDims; ++i) {
    input_dims[i] = dim[i];
    input_strides[i] = strides[i];
  }
  try {
    CREATE_MEMORY_DESC_USING_STRIDES(blocked_md, kNumDims, input_dims,
                                     memory::convert_to_c(dtype),
                                     input_strides);
    delete[] input_dims;
    delete[] input_strides;
  } catch (dnnl::error& e) {
    delete[] input_dims;
    delete[] input_strides;
    return absl::InternalError(
        absl::StrCat("Failed to create blocked memory descriptor.",
                     "Status: ", e.status, ", message: ", e.message));
  }
  return OkStatus();
}
inline void CreateAndExecuteReorder(const ReorderPd& reorder_desc,
                                    const memory& src_mem,
                                    const memory& dst_mem, const engine& engine,
                                    OpKernelContext* ctx = nullptr,
                                    memory* scale_mem = nullptr) {
  std::vector<primitive> net;
  net.push_back(dnnl::reorder(reorder_desc));
  std::vector<MemoryArgsMap> net_args;
#ifndef ENABLE_ONEDNN_V3
  net_args.push_back({{DNNL_ARG_FROM, src_mem}, {DNNL_ARG_TO, dst_mem}});
#else
  if (scale_mem != nullptr) {
    net_args.push_back({{DNNL_ARG_FROM, src_mem},
                        {DNNL_ARG_TO, dst_mem},
                        {DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST, *scale_mem}});
  } else {
    net_args.push_back({{DNNL_ARG_FROM, src_mem}, {DNNL_ARG_TO, dst_mem}});
  }
#endif  
  ExecutePrimitive(net, &net_args, engine, ctx);
}
class MklReorderPrimitive;
template <typename T>
inline MklReorderPrimitive* FindOrCreateReorder(const memory* from,
                                                const memory* to);
template <typename T>
class MklDnnData {
 private:
  memory* user_memory_;
  memory* reorder_memory_;
  memory::desc* op_md_;
  bool bIs3D;
  void* allocated_buffer_;
  const engine* cpu_engine_;
 public:
  explicit MklDnnData(const engine* e)
      : user_memory_(nullptr),
        reorder_memory_(nullptr),
        op_md_(nullptr),
        bIs3D(false),
        allocated_buffer_(nullptr),
        cpu_engine_(e) {}
  MklDnnData(const MklDnnData&) = default;
  MklDnnData& operator=(const MklDnnData&) = delete;
  ~MklDnnData() {
    if (allocated_buffer_ != nullptr) {
      cpu_allocator()->DeallocateRaw(allocated_buffer_);
    }
    cpu_engine_ = nullptr;  
    delete (user_memory_);
    delete (reorder_memory_);
    delete (op_md_);
  }
  inline void* GetTensorBuffer(const Tensor* tensor) const {
    CHECK_NOTNULL(tensor);
    return const_cast<void*>(
        static_cast<const void*>(tensor->flat<T>().data()));
  }
  void SetIs3DData(bool bIs3D_) { bIs3D = bIs3D_; }
  bool GetIs3D() { return bIs3D; }
  inline void SetUsrMem(const memory::dims& dim, memory::format_tag fm,
                        void* data_buffer = nullptr) {
    auto md = memory::desc(dim, MklDnnType<T>(), fm);
    SetUsrMem(md, data_buffer);
  }
  inline void SetUsrMem(const memory::dims& dim, memory::format_tag fm,
                        const Tensor* tensor) {
    DCHECK(tensor);
    SetUsrMem(dim, fm, GetTensorBuffer(tensor));
  }
  static inline memory::desc CreateBlockedMemDesc(const memory::dims& dim,
                                                  const memory::dims& strides) {
    dnnl_memory_desc_t blocked_md;
    TF_CHECK_OK(
        CreateBlockedMemDescHelper(dim, strides, MklDnnType<T>(), &blocked_md));
    return memory::desc(blocked_md);
  }
  inline void SetUsrMem(const memory::dims& dim, const memory::dims& strides,
                        void* data_buffer = nullptr) {
    CHECK_EQ(dim.size(), strides.size());
    auto blocked_md = MklDnnData<T>::CreateBlockedMemDesc(dim, strides);
    SetUsrMem(blocked_md, data_buffer);
  }
  inline void SetUsrMem(const memory::dims& dim, const memory::dims& strides,
                        const Tensor* tensor) {
    CHECK_NOTNULL(tensor);
    SetUsrMem(dim, strides, GetTensorBuffer(tensor));
  }
  inline void SetUsrMem(const memory::desc& md, const Tensor* tensor) {
    CHECK_NOTNULL(tensor);
    SetUsrMem(md, GetTensorBuffer(tensor));
  }
  inline void SetUsrMem(const memory::desc& pd, void* data_buffer = nullptr) {
    DCHECK(cpu_engine_);
    if (user_memory_) delete user_memory_;
    if (data_buffer) {
      user_memory_ = new memory(pd, *cpu_engine_, data_buffer);
    } else {
      user_memory_ = new memory(pd, *cpu_engine_);
    }
  }
  inline const memory* GetUsrMem() const { return user_memory_; }
  inline memory::desc GetUsrMemDesc() const {
    DCHECK(user_memory_);
    return user_memory_->get_desc();
  }
  inline void* GetUsrMemDataHandle() const {
    CHECK_NOTNULL(user_memory_);
    return user_memory_->get_data_handle();
  }
  inline void SetUsrMemDataHandle(void* data_buffer,
                                  std::shared_ptr<stream> t_stream = nullptr) {
    CHECK_NOTNULL(user_memory_);
    CHECK_NOTNULL(data_buffer);
#if !defined(ENABLE_ONEDNN_OPENMP) && !defined(ENABLE_ONEDNN_V3)
    user_memory_->set_data_handle(data_buffer, *t_stream);
#else
    user_memory_->set_data_handle(data_buffer);
#endif  
  }
  inline void SetUsrMemDataHandle(const Tensor* tensor,
                                  std::shared_ptr<stream> t_stream = nullptr) {
    SetUsrMemDataHandle(GetTensorBuffer(tensor), t_stream);
  }
  inline void AllocateBuffer(size_t size) {
    const int64 kMemoryAlignment = 64;  
    allocated_buffer_ = cpu_allocator()->AllocateRaw(kMemoryAlignment, size);
  }
  inline void* GetAllocatedBuffer() { return allocated_buffer_; }
  inline const memory& GetOpMem() const {
    return reorder_memory_ ? *reorder_memory_ : *user_memory_;
  }
  inline void SetOpMemDesc(const memory::dims& dim, memory::format_tag fm) {
    op_md_ = new memory::desc(dim, MklDnnType<T>(), fm);
  }
  inline const memory::desc& GetOpMemDesc() const { return *op_md_; }
  inline bool IsReorderNeeded(const memory::desc& op_pd) const {
    DCHECK(user_memory_);
    return op_pd != user_memory_->get_desc();
  }
  inline primitive CreateReorder(const memory* from, const memory* to) const {
    CHECK_NOTNULL(from);
    CHECK_NOTNULL(to);
    return reorder(*from, *to);
  }
  inline bool CheckReorderToOpMem(const memory::desc& op_md,
                                  std::vector<primitive>& net,
                                  std::vector<MemoryArgsMap>& net_args,
                                  const engine& engine) {
    DCHECK(user_memory_);
    DCHECK_EQ(net.size(), net_args.size());
    if (IsReorderNeeded(op_md)) {
      reorder_memory_ = new memory(op_md, engine);
      net.push_back(CreateReorder(user_memory_, reorder_memory_));
      net_args.push_back(MemoryArgsMap{{DNNL_ARG_FROM, *user_memory_},
                                       {DNNL_ARG_TO, *reorder_memory_}});
      return true;
    }
    return false;
  }
  inline bool CheckReorderToOpMem(const memory::desc& op_md,
                                  const engine& engine,
                                  OpKernelContext* context = nullptr) {
    DCHECK(user_memory_);
    if (IsReorderNeeded(op_md)) {
      reorder_memory_ = new memory(op_md, engine);
      auto* prim = FindOrCreateReorder<T>(user_memory_, reorder_memory_);
      std::shared_ptr<stream> cpu_stream;
      tsl::OneDnnThreadPool eigen_tp;
      if (context != nullptr) {
        Eigen::ThreadPoolInterface* eigen_interface =
            EigenThreadPoolFromTfContext(context);
        eigen_tp =
            tsl::OneDnnThreadPool(eigen_interface, ThreadPoolUseCallerThread());
        cpu_stream.reset(CreateStream(&eigen_tp, prim->GetEngine()));
      } else {
        cpu_stream.reset(CreateStream(nullptr, prim->GetEngine()));
      }
      std::vector<primitive> net;
      net.push_back(*(prim->GetPrimitive()));
      std::vector<MemoryArgsMap> net_args;
      net_args.push_back(
          {{DNNL_ARG_FROM, *user_memory_}, {DNNL_ARG_TO, *reorder_memory_}});
      execute_primitives(net, cpu_stream, net_args);
      return true;
    }
    return false;
  }
  inline bool CheckReorderToOpMem(const memory::desc& op_md,
                                  void* reorder_data_handle,
                                  std::vector<primitive>& net,
                                  std::vector<MemoryArgsMap>& net_args,
                                  const engine& engine) {
    DCHECK(reorder_data_handle);
    DCHECK(user_memory_);
    if (IsReorderNeeded(op_md)) {
      reorder_memory_ = new memory(op_md, engine, reorder_data_handle);
      net.push_back(CreateReorder(user_memory_, reorder_memory_));
      net_args.push_back(MemoryArgsMap{{DNNL_ARG_FROM, *user_memory_},
                                       {DNNL_ARG_TO, *reorder_memory_}});
      return true;
    }
    return false;
  }
  inline bool CheckReorderToOpMem(const memory::desc& op_md,
                                  void* reorder_data_handle,
                                  const engine& engine,
                                  OpKernelContext* context = nullptr) {
    DCHECK(reorder_data_handle);
    DCHECK(user_memory_);
    if (IsReorderNeeded(op_md)) {
      reorder_memory_ = new memory(op_md, engine, reorder_data_handle);
      auto* prim = FindOrCreateReorder<T>(user_memory_, reorder_memory_);
      std::shared_ptr<stream> cpu_stream;
      tsl::OneDnnThreadPool eigen_tp;
      if (context != nullptr) {
        Eigen::ThreadPoolInterface* eigen_interface =
            EigenThreadPoolFromTfContext(context);
        eigen_tp =
            tsl::OneDnnThreadPool(eigen_interface, ThreadPoolUseCallerThread());
        cpu_stream.reset(CreateStream(&eigen_tp, prim->GetEngine()));
      } else {
        cpu_stream.reset(CreateStream(nullptr, prim->GetEngine()));
      }
      std::vector<primitive> net;
      net.push_back(*(prim->GetPrimitive()));
      std::vector<MemoryArgsMap> net_args;
      net_args.push_back(
          {{DNNL_ARG_FROM, *user_memory_}, {DNNL_ARG_TO, *reorder_memory_}});
      execute_primitives(net, cpu_stream, net_args);
      return true;
    }
    return false;
  }
  inline bool CheckReorderToOpMem(const memory::desc& op_md,
                                  Tensor* reorder_tensor,
                                  std::vector<primitive>& net,
                                  std::vector<MemoryArgsMap>& net_args,
                                  const engine& engine) {
    DCHECK(reorder_tensor);
    return CheckReorderToOpMem(op_md, GetTensorBuffer(reorder_tensor), net,
                               net_args, engine);
  }
  inline bool CheckReorderToOpMem(const memory::desc& op_pd,
                                  Tensor* reorder_tensor,
                                  OpKernelContext* ctx = nullptr) {
    DCHECK(reorder_tensor);
    return CheckReorderToOpMem(op_pd, GetTensorBuffer(reorder_tensor),
                               *cpu_engine_, ctx);
  }
  inline bool PrepareReorderToUserMemIfReq(const memory::desc& op_pd) {
    DCHECK(user_memory_);
    if (IsReorderNeeded(op_pd)) {
      reorder_memory_ = new memory(op_pd, *cpu_engine_);
      return true;
    }
    return false;
  }
  inline void InsertReorderToUserMem(std::vector<primitive>& net,
                                     std::vector<MemoryArgsMap>& net_args) {
    DCHECK(user_memory_);
    DCHECK(reorder_memory_);
    net.push_back(CreateReorder(reorder_memory_, user_memory_));
    net_args.push_back(MemoryArgsMap{{DNNL_ARG_FROM, *reorder_memory_},
                                     {DNNL_ARG_TO, *user_memory_}});
  }
  inline void InsertReorderToUserMem(OpKernelContext* ctx = nullptr) {
    DCHECK(user_memory_);
    DCHECK(reorder_memory_);
    DCHECK(cpu_engine_);
    std::vector<primitive> net;
    auto* prim = FindOrCreateReorder<T>(reorder_memory_, user_memory_);
    net.push_back(*(prim->GetPrimitive()));
    std::vector<MemoryArgsMap> net_args;
    net_args.push_back(
        {{DNNL_ARG_FROM, *reorder_memory_}, {DNNL_ARG_TO, *user_memory_}});
    std::shared_ptr<stream> cpu_stream;
    tsl::OneDnnThreadPool eigen_tp;
    if (ctx != nullptr) {
      Eigen::ThreadPoolInterface* eigen_interface =
          EigenThreadPoolFromTfContext(ctx);
      eigen_tp =
          tsl::OneDnnThreadPool(eigen_interface, ThreadPoolUseCallerThread());
      cpu_stream.reset(CreateStream(&eigen_tp, prim->GetEngine()));
    } else {
      cpu_stream.reset(CreateStream(nullptr, prim->GetEngine()));
    }
    execute_primitives(net, cpu_stream, net_args);
  }
};
class MklPrimitive {
 public:
  virtual ~MklPrimitive() {}
  MklPrimitive() {}
  MklPrimitive(const engine& cpu_engine) { cpu_engine_ = cpu_engine; }
  unsigned char* DummyData = nullptr;
  engine cpu_engine_ = engine(engine::kind::cpu, 0);
  const engine& GetEngine() { return cpu_engine_; }
};
const dnnl::memory::dims NONE_DIMS = {};
template <typename T>
class LRUCache {
 public:
  explicit LRUCache(size_t capacity) {
    capacity_ = capacity;
    Clear();
  }
  T* GetOp(const string& key) {
#if defined(DNNL_AARCH64_USE_ACL) && defined(ENABLE_ONEDNN_OPENMP)
    mutex_lock lock(lru_mu_);
#endif
    auto it = cache_.find(key);
    if (it == cache_.end()) {
      return nullptr;
    }
    lru_list_.erase(it->second.lru_iterator);
    lru_list_.push_front(it->first);
    it->second.lru_iterator = lru_list_.begin();
    return it->second.op;
  }
  void SetOp(const string& key, T* op) {
#if defined(DNNL_AARCH64_USE_ACL) && defined(ENABLE_ONEDNN_OPENMP)
    mutex_lock lock(lru_mu_);
#endif
    if (lru_list_.size() >= capacity_) {
      Delete();
    }
    lru_list_.push_front(key);
    Entry entry(op, lru_list_.begin());
    cache_.emplace(std::make_pair(key, std::move(entry)));
#if defined(DNNL_AARCH64_USE_ACL) && defined(ENABLE_ONEDNN_OPENMP)
    FinishedAllocation(key);
#endif
  }
  void Clear() {
    if (lru_list_.empty()) return;
    cache_.clear();
    lru_list_.clear();
  }
#if defined(DNNL_AARCH64_USE_ACL) && defined(ENABLE_ONEDNN_OPENMP)
  bool IsAllocating(const string& key) {
    mutex_lock lock(in_flight_mu_);
    return in_flight_.find(key) != in_flight_.end();
  }
  void Allocate(const string& key) {
    mutex_lock lock(in_flight_mu_);
    in_flight_.insert(key);
  }
  void FinishedAllocation(const string& key) {
    mutex_lock lock(in_flight_mu_);
    in_flight_.erase(key);
  }
#endif
 private:
  struct Entry {
    T* op;
    std::list<string>::iterator lru_iterator;
    Entry(T* op, std::list<string>::iterator it) {
      this->op = op;
      this->lru_iterator = it;
    }
    Entry(Entry&& source) noexcept
        : lru_iterator(std::move(source.lru_iterator)) {
      op = std::move(source.op);
      source.op = std::forward<T*>(nullptr);
    }
    ~Entry() {
      if (op != nullptr) delete op;
    }
  };
  bool Delete() {
    if (lru_list_.empty()) return false;
    string key = lru_list_.back();
    lru_list_.pop_back();
    cache_.erase(key);
    return true;
  }
  size_t capacity_;
  std::unordered_map<string, Entry> cache_;
  std::list<string> lru_list_;
#if defined(DNNL_AARCH64_USE_ACL) && defined(ENABLE_ONEDNN_OPENMP)
  mutex lru_mu_;
  std::set<string> in_flight_;
  TF_GUARDED_BY(in_flight_mu_)
  mutex in_flight_mu_;
#endif
};
template <typename T>
class MklPrimitiveFactory {
 public:
  MklPrimitiveFactory() {}
  ~MklPrimitiveFactory() {}
  MklPrimitive* GetOp(const string& key) {
#if !defined(DNNL_AARCH64_USE_ACL) || !defined(ENABLE_ONEDNN_OPENMP)
    auto& lru_cache = MklPrimitiveFactory<T>::GetLRUCache();
    return lru_cache.GetOp(key);
#else
    while (true) {
      mutex_lock lock(primitive_creation_mu_);
      auto& lru_cache = MklPrimitiveFactory<T>::GetLRUCache();
      MklPrimitive* primitive = lru_cache.GetOp(key);
      if (primitive != nullptr) {
        return primitive;
      }
      if (!lru_cache.IsAllocating(key)) {
        lru_cache.Allocate(key);
        return nullptr;
      }
      primitive_creation_cv_.wait(lock);
    }
#endif
  }
  void SetOp(const string& key, MklPrimitive* op) {
#if !defined(DNNL_AARCH64_USE_ACL) || !defined(ENABLE_ONEDNN_OPENMP)
    auto& lru_cache = MklPrimitiveFactory<T>::GetLRUCache();
    lru_cache.SetOp(key, op);
#else
    {
      mutex_lock lock(primitive_creation_mu_);
      auto& lru_cache = MklPrimitiveFactory<T>::GetLRUCache();
      lru_cache.SetOp(key, op);
    }
    primitive_creation_cv_.notify_all();
#endif
  }
  static inline bool IsLegacyPlatform() {
#ifdef DNNL_AARCH64_USE_ACL
    return false;
#else
    static const bool is_legacy_platform =
        (!port::TestCPUFeature(port::CPUFeature::AVX512F) &&
         !port::TestCPUFeature(port::CPUFeature::AVX2));
    return is_legacy_platform;
#endif  
  }
  static inline bool IsPrimitiveMemOptEnabled() {
    static const bool is_primitive_mem_opt_enabled = [] {
      bool value = true;
      TF_CHECK_OK(
          ReadBoolFromEnvVar("TF_MKL_OPTIMIZE_PRIMITIVE_MEMUSE", true, &value));
      return value;
    }();
    return is_primitive_mem_opt_enabled;
  }
#ifdef DNNL_AARCH64_USE_ACL
  static int IncrementCounter() {
    static std::atomic_int counter{1};
    return counter.fetch_add(1);
  }
#endif
 private:
  static inline LRUCache<MklPrimitive>& GetLRUCache() {
    static const int kCapacity = 1024;  
#if !defined(DNNL_AARCH64_USE_ACL) || !defined(ENABLE_ONEDNN_OPENMP)
    static thread_local LRUCache<MklPrimitive> lru_cache_(kCapacity);
#else
    static LRUCache<MklPrimitive> lru_cache_(kCapacity);
#endif
    return lru_cache_;
  }
#if defined(DNNL_AARCH64_USE_ACL) && defined(ENABLE_ONEDNN_OPENMP)
  mutex primitive_creation_mu_;
  condition_variable primitive_creation_cv_;
#endif
};
class FactoryKeyCreator {
 public:
  FactoryKeyCreator() { key_.reserve(kMaxKeyLength); }
  ~FactoryKeyCreator() {}
  void AddAsKey(const string& str) { Append(str); }
  void AddAsKey(const dnnl::memory::dims& dims) {
    for (unsigned int i = 0; i < dims.size(); i++) {
      AddAsKey<int>(dims[i]);
    }
  }
  template <typename T>
  void AddAsKey(const T data) {
    auto buffer = reinterpret_cast<const char*>(&data);
    Append(StringPiece(buffer, sizeof(T)));
  }
  void AddAsKey(const void* data) {
    auto buffer = reinterpret_cast<const char*>(&data);
    Append(StringPiece(buffer, sizeof(data)));
  }
  string GetKey() { return key_; }
 private:
  string key_;
  const char delimiter = 'x';
  const int kMaxKeyLength = 256;
  void Append(StringPiece s) {
    key_.append(string(s));
    key_.append(1, delimiter);
  }
};
class MklReorderPrimitive : public MklPrimitive {
 public:
  explicit MklReorderPrimitive(const memory* from, const memory* to)
      : MklPrimitive(engine(engine::kind::cpu, 0)) {
    Setup(from, to);
  }
  ~MklReorderPrimitive() {}
  std::shared_ptr<primitive> GetPrimitive() { return context_.reorder_prim; }
  void SetMemory(const memory* from, const memory* to) {
    context_.src_mem->set_data_handle(from->get_data_handle());
    context_.dst_mem->set_data_handle(to->get_data_handle());
  }
  std::shared_ptr<dnnl::stream> GetStream() { return stream_; }
 private:
  struct ReorderContext {
    std::shared_ptr<dnnl::memory> src_mem;
    std::shared_ptr<dnnl::memory> dst_mem;
    std::shared_ptr<primitive> reorder_prim;
    ReorderContext()
        : src_mem(nullptr), dst_mem(nullptr), reorder_prim(nullptr) {}
  } context_;
  std::shared_ptr<dnnl::stream> stream_;
  void Setup(const memory* from, const memory* to) {
    context_.src_mem.reset(
        new memory(from->get_desc(), cpu_engine_, DummyData));
    context_.dst_mem.reset(new memory(to->get_desc(), cpu_engine_, DummyData));
    context_.reorder_prim = std::make_shared<dnnl::reorder>(
        reorder(*context_.src_mem, *context_.dst_mem));
    stream_.reset(new stream(cpu_engine_));
  }
};
template <typename T>
class MklReorderPrimitiveFactory : public MklPrimitiveFactory<T> {
 public:
  static MklReorderPrimitive* Get(const memory* from, const memory* to) {
    auto reorderPrim = static_cast<MklReorderPrimitive*>(
        MklReorderPrimitiveFactory<T>::GetInstance().GetReorder(from, to));
    if (reorderPrim == nullptr) {
      reorderPrim = new MklReorderPrimitive(from, to);
      MklReorderPrimitiveFactory<T>::GetInstance().SetReorder(from, to,
                                                              reorderPrim);
    }
    reorderPrim->SetMemory(from, to);
    return reorderPrim;
  }
  static MklReorderPrimitiveFactory& GetInstance() {
    static MklReorderPrimitiveFactory instance_;
    return instance_;
  }
  static string CreateKey(const memory* from, const memory* to) {
    string prefix = "reorder";
    FactoryKeyCreator key_creator;
    auto const& from_desc = from->GET_MEMORY_DESC;
    auto const& to_desc = to->GET_MEMORY_DESC;
    memory::dims INIT_DIMS_FROM_DESC(from_dims, from_desc);
    memory::dims INIT_DIMS_FROM_DESC(to_dims, to_desc);
    auto from_strides = from_desc.GET_STRIDES;
    auto from_inner_nblks = from_desc.GET_INNER_NBLKS;
    auto from_inner_blks = from_desc.GET_INNER_BLKS;
    auto from_inner_idxs = from_desc.GET_INNER_IDXS;
    auto to_inner_nblks = to_desc.GET_INNER_NBLKS;
    auto to_inner_blks = to_desc.GET_INNER_BLKS;
    auto to_inner_idxs = to_desc.GET_INNER_IDXS;
    auto to_strides = to_desc.GET_STRIDES;
#ifndef ENABLE_ONEDNN_V3
    memory::dims from_inner_blks_1(from_inner_blks,
                                   &from_inner_blks[from_inner_nblks]);
    memory::dims from_inner_idxs_1(from_inner_idxs,
                                   &from_inner_idxs[from_inner_nblks]);
    memory::dims to_inner_blks_1(to_inner_blks, &to_inner_blks[to_inner_nblks]);
    memory::dims to_inner_idxs_1(to_inner_idxs, &to_inner_idxs[to_inner_nblks]);
    memory::dims from_strides_outer_blocks(from_strides,
                                           &from_strides[from_desc.ndims]);
    memory::dims to_strides_outer_blocks(to_strides,
                                         &to_strides[to_desc.ndims]);
#endif  
    key_creator.AddAsKey(prefix);
#ifdef DNNL_AARCH64_USE_ACL
    key_creator.AddAsKey(std::this_thread::get_id());
#endif
    key_creator.AddAsKey(static_cast<int>(from_desc.GET_MEMORY_DESC_FLAGS));
    key_creator.AddAsKey(static_cast<int>(from_inner_nblks));
    key_creator.AddAsKey(GET_INNER_DIMS(from_inner_blks, from_inner_blks_1));
    key_creator.AddAsKey(GET_INNER_DIMS(from_inner_idxs, from_inner_idxs_1));
    key_creator.AddAsKey(static_cast<int>(from_desc.GET_DATA_TYPE));
    key_creator.AddAsKey(from_dims);
    key_creator.AddAsKey(
        GET_STRIDES_DIMS(from_strides, from_strides_outer_blocks));
    key_creator.AddAsKey(static_cast<int>(to_desc.GET_MEMORY_DESC_FLAGS));
    key_creator.AddAsKey(static_cast<int>(to_inner_nblks));
    key_creator.AddAsKey(GET_INNER_DIMS(to_inner_blks, to_inner_blks_1));
    key_creator.AddAsKey(GET_INNER_DIMS(to_inner_idxs, to_inner_idxs_1));
    key_creator.AddAsKey(static_cast<int>(to_desc.GET_DATA_TYPE));
    key_creator.AddAsKey(to_dims);
    key_creator.AddAsKey(GET_STRIDES_DIMS(to_strides, to_strides_outer_blocks));
    return key_creator.GetKey();
  }
 private:
  MklReorderPrimitiveFactory() {}
  ~MklReorderPrimitiveFactory() {}
  MklPrimitive* GetReorder(const memory* from, const memory* to) {
    string key = CreateKey(from, to);
    return this->GetOp(key);
  }
  void SetReorder(const memory* from, const memory* to, MklPrimitive* op) {
    string key = CreateKey(from, to);
    this->SetOp(key, op);
  }
};
template <typename T>
inline MklReorderPrimitive* FindOrCreateReorder(const memory* from,
                                                const memory* to) {
  CHECK_NOTNULL(from);
  CHECK_NOTNULL(to);
  MklReorderPrimitive* reorder_prim =
      MklReorderPrimitiveFactory<T>::Get(from, to);
  return reorder_prim;
}
inline bool IsConv1x1StrideNot1(memory::dims filter_dims,
                                memory::dims strides) {
  if (filter_dims.size() != 4 || strides.size() != 2) return false;
  return ((filter_dims[2] == 1) && (filter_dims[3] == 1) &&
          ((strides[0] != 1) || (strides[1] != 1)));
}
#undef ARE_MEMORY_DESCS_EQUAL
#undef CREATE_MEMORY_DESC_USING_STRIDES
#undef GET_DATA_TYPE
#undef GET_DIMS
#undef GET_INNER_BLKS
#undef GET_INNER_DIMS
#undef GET_INNER_IDXS
#undef GET_INNER_NBLKS
#undef GET_MEMORY_DESC
#undef GET_MEMORY_DESC_FLAGS
#undef GET_MEMORY_DESC_USING_MKLDNN_SHAPE_PTR
#undef GET_NDIMS
#undef GET_STRIDES
#undef GET_STRIDES_DIMS
#undef INIT_DIMS_FROM_DESC
#undef MEMORY_DESC
}  
#define REGISTER_TEST_FLOAT32(TEST) REGISTER_TEST(TEST, DT_FLOAT, Float32Input);
#define REGISTER_TEST_BFLOAT16(TEST) \
  REGISTER_TEST(TEST, DT_BFLOAT16, BFloat16Input);
#define REGISTER_TEST_ALL_TYPES(TEST) \
  REGISTER_TEST_FLOAT32(TEST);        \
  REGISTER_TEST_BFLOAT16(TEST);
#else
#define REGISTER_TEST_ALL_TYPES(TEST) REGISTER_TEST_FLOAT32(TEST);
#endif  
#endif  