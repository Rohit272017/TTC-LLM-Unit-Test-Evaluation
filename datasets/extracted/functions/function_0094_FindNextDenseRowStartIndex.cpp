#include "tensorflow/core/kernels/sparse_utils.h"
#include <cstddef>
#include <cstdint>
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/status.h"
namespace tensorflow {
namespace sparse_utils {
template <typename Tindices>
Tindices FindNextDenseRowStartIndex(
    const Tindices sparse_index_begin,
    const typename TTypes<Tindices>::ConstMatrix& indices_mat) {
  Tindices begin = sparse_index_begin;
  Tindices end = indices_mat.dimension(0);
  const Tindices orig_sparse_index_end = end;
  const Tindices orig_dense_index_begin = indices_mat(begin, 0);
  if (orig_dense_index_begin == static_cast<int64_t>(indices_mat(end - 1, 0))) {
    return orig_sparse_index_end;
  }
  Tindices increment = 1;
  while (begin + increment < end &&
         indices_mat(begin + increment, 0) == orig_dense_index_begin) {
    increment *= 2;
  }
  if (begin + increment < end) {
    end = begin + increment;
  }
  begin += increment / 2;
  const Tindices dense_row_index_to_find = orig_dense_index_begin;
  while (begin < end) {
    const Tindices m = begin + (end - begin) / 2;
    const Tindices m_dense_row_index = static_cast<Tindices>(indices_mat(m, 0));
    if (m_dense_row_index == dense_row_index_to_find &&
        (m + 1 == orig_sparse_index_end ||
         static_cast<Tindices>(indices_mat(m + 1, 0)) !=
             dense_row_index_to_find)) {
      return m + 1;
    } else if (m_dense_row_index <= dense_row_index_to_find) {
      begin = m + 1;
    } else {
      end = m;
    }
  }
  return orig_sparse_index_end;
}
template <typename Tindices>
std::vector<Tindices> GetStartIndicesOfEachDenseRow(
    const typename TTypes<Tindices>::ConstMatrix& indices_mat,
    bool* contains_empty_rows) {
  int64_t start_sparse_index_of_cur_dense_row = 0;
  std::vector<Tindices> segment_indices;
  const Tindices num_entries_in_sparse_tensor = indices_mat.dimension(0);
  const Tindices num_dense_rows_in_sparse_tensor =
      1 + indices_mat(num_entries_in_sparse_tensor - 1, 0);
  segment_indices.reserve(1 + num_dense_rows_in_sparse_tensor);
  segment_indices.push_back(0);
  for (Tindices i = 0; i < indices_mat(0, 0); ++i) {
    segment_indices.push_back(0);
  }
  *contains_empty_rows = indices_mat(0, 0) > 0;
  while (true) {
    const Tindices start_sparse_index_of_next_dense_row =
        FindNextDenseRowStartIndex<Tindices>(
            start_sparse_index_of_cur_dense_row, indices_mat);
    if (start_sparse_index_of_next_dense_row == num_entries_in_sparse_tensor) {
      segment_indices.push_back(start_sparse_index_of_next_dense_row);
      break;
    }
    for (Tindices i = 0;
         i < indices_mat(start_sparse_index_of_next_dense_row, 0) -
                 indices_mat(start_sparse_index_of_cur_dense_row, 0);
         ++i) {
      segment_indices.push_back(start_sparse_index_of_next_dense_row);
    }
    *contains_empty_rows |=
        indices_mat(start_sparse_index_of_next_dense_row, 0) -
            indices_mat(start_sparse_index_of_cur_dense_row, 0) >
        1;
    start_sparse_index_of_cur_dense_row = start_sparse_index_of_next_dense_row;
  }
  return segment_indices;
}
template <typename Tindices>
std::vector<Tindices> ParseRowStartIndices(
    const tensorflow::Tensor& tensor,
    const Tindices num_nonzero_entries_in_sparse_mat) {
  std::vector<Tindices> out;
  auto vec = tensor.vec<Tindices>();
  out.reserve(vec.size() + 1);
  for (size_t i = 0; i < vec.dimension(0); ++i) {
    out.push_back(vec(i));
  }
  out.push_back(num_nonzero_entries_in_sparse_mat);
  return out;
}
template <typename Tindices>
bool ContainsEmptyRows(const std::vector<Tindices>& row_start_indices) {
  for (size_t i = 1; i < row_start_indices.size() - 1; ++i) {
    if (row_start_indices.at(i) - row_start_indices.at(i - 1) == 0) {
      return true;
    }
  }
  return false;
}
namespace {
Status ValidateSparseTensorShape(const Tensor& indices, const Tensor& values,
                                 const Tensor& shape) {
  if (!TensorShapeUtils::IsMatrix(indices.shape())) {
    return errors::InvalidArgument("Sparse indices must be rank 2 but is rank ",
                                   indices.shape().dim_sizes().size());
  }
  if (!TensorShapeUtils::IsVector(values.shape())) {
    return errors::InvalidArgument("Sparse values must be rank 1 but is rank ",
                                   values.shape().dims());
  }
  if (!TensorShapeUtils::IsVector(shape.shape())) {
    return errors::InvalidArgument("Sparse shape must be rank 1 but is rank ",
                                   shape.shape().dims());
  }
  int64_t nnz = indices.dim_size(0);
  int64_t ndims = indices.dim_size(1);
  if (values.dim_size(0) != nnz) {
    return errors::InvalidArgument("Number of elements in indices (", nnz,
                                   ") and values (", values.dim_size(0),
                                   ") do not match");
  }
  if (shape.NumElements() != ndims) {
    return errors::InvalidArgument("Index rank (", ndims, ") and shape rank (",
                                   shape.NumElements(), ") do not match");
  }
  return absl::OkStatus();
}
template <typename IndexTensor>
string CreateIndexString(const IndexTensor& indices, int64_t row) {
  const int64_t ndims = indices.dimension(1);
  string index_str = strings::StrCat("indices[", row, ", :] = [");
  for (int64_t dim = 0; dim < ndims; ++dim) {
    strings::StrAppend(&index_str, indices(row, dim),
                       dim < ndims - 1 ? ", " : "]");
  }
  if (ndims == 0) {
    strings::StrAppend(&index_str, "]");
  }
  return index_str;
}
template <typename Tindices>
Status ValidateSparseTensorIndicesUnordered(const Tensor& indices,
                                            const Tensor& shape) {
  const auto indices_mat = indices.flat_inner_dims<Tindices>();
  const auto shape_vec = shape.flat<Tindices>();
  int64_t nnz = indices.dim_size(0);
  int64_t ndims = indices.dim_size(1);
  for (int64_t i = 0; i < nnz; ++i) {
    for (int64_t dim = 0; dim < ndims; ++dim) {
      const Tindices idx = indices_mat(i, dim);
      if (TF_PREDICT_FALSE(idx < 0 || idx >= shape_vec(dim))) {
        string index_str = CreateIndexString(indices_mat, i);
        return errors::InvalidArgument("Sparse index tuple ", index_str,
                                       " is out of bounds");
      }
    }
  }
  return absl::OkStatus();
}
template <typename Tindices>
Status ValidateSparseTensorIndicesOrdered(const Tensor& indices,
                                          const Tensor& shape) {
  const auto indices_mat = indices.flat_inner_dims<Tindices>();
  const auto shape_vec = shape.flat<Tindices>();
  int64_t nnz = indices.dim_size(0);
  int64_t ndims = indices.dim_size(1);
  if (nnz == 0) {
    return absl::OkStatus();
  }
  for (int64_t dim = 0; dim < ndims; ++dim) {
    const Tindices idx = indices_mat(0, dim);
    if (TF_PREDICT_FALSE(idx < 0 || idx >= shape_vec(dim))) {
      string index_str = CreateIndexString(indices_mat, 0);
      return errors::InvalidArgument("Sparse index tuple ", index_str,
                                     " is out of bounds");
    }
  }
  for (int64_t i = 1; i < nnz; ++i) {
    bool different = false;
    for (int64_t dim = 0; dim < ndims; ++dim) {
      const Tindices idx = indices_mat(i, dim);
      const Tindices prev_idx = indices_mat(i - 1, dim);
      if (TF_PREDICT_TRUE(different)) {
        if (TF_PREDICT_FALSE(idx < 0 || idx >= shape_vec(dim))) {
          string index_str = CreateIndexString(indices_mat, i);
          return errors::InvalidArgument("Sparse index tuple ", index_str,
                                         " is out of bounds");
        }
      } else {
        if (TF_PREDICT_FALSE(idx < prev_idx || idx >= shape_vec(dim))) {
          string index_str = CreateIndexString(indices_mat, i);
          if (TF_PREDICT_FALSE(idx < 0 || idx >= shape_vec(dim))) {
            return errors::InvalidArgument("Sparse index tuple ", index_str,
                                           " is out of bounds");
          } else {
            return errors::InvalidArgument("Sparse index tuple ", index_str,
                                           " is out of order");
          }
        } else if (TF_PREDICT_TRUE(idx > prev_idx)) {
          different = true;
        }
      }  
    }    
    if (TF_PREDICT_FALSE(!different)) {
      string index_str = CreateIndexString(indices_mat, i);
      return errors::InvalidArgument("Sparse index tuple ", index_str,
                                     " is repeated");
    }
  }  
  return absl::OkStatus();
}
}  
template <typename Tindices>
Status ValidateSparseTensor(const Tensor& indices, const Tensor& values,
                            const Tensor& shape,
                            IndexValidation index_validation) {
  TF_RETURN_IF_ERROR(ValidateSparseTensorShape(indices, values, shape));
  switch (index_validation) {
    case IndexValidation::kOrdered:
      return ValidateSparseTensorIndicesOrdered<Tindices>(indices, shape);
    case IndexValidation::kUnordered:
      return ValidateSparseTensorIndicesUnordered<Tindices>(indices, shape);
    case IndexValidation::kNone: {
    }
  }
  return absl::OkStatus();
}
#define REGISTER_SPARSE_UTIL_FUNCTIONS(TypeIndex)                           \
  template TypeIndex FindNextDenseRowStartIndex<TypeIndex>(                 \
      const TypeIndex sparse_index_begin,                                   \
      const TTypes<TypeIndex>::ConstMatrix& indices_mat);                   \
  template std::vector<TypeIndex> GetStartIndicesOfEachDenseRow<TypeIndex>( \
      const TTypes<TypeIndex>::ConstMatrix& indices_mat,                    \
      bool* contains_empty_rows);                                           \
  template bool ContainsEmptyRows<TypeIndex>(                               \
      const std::vector<TypeIndex>& row_start_indices);                     \
  template std::vector<TypeIndex> ParseRowStartIndices<TypeIndex>(          \
      const tensorflow::Tensor& tensor,                                     \
      const TypeIndex num_nonzero_entries_in_sparse_mat);                   \
  template Status ValidateSparseTensor<TypeIndex>(                          \
      const Tensor& indices, const Tensor& values, const Tensor& shape,     \
      IndexValidation index_validation)
REGISTER_SPARSE_UTIL_FUNCTIONS(int32);
REGISTER_SPARSE_UTIL_FUNCTIONS(int64);
REGISTER_SPARSE_UTIL_FUNCTIONS(uint8);
REGISTER_SPARSE_UTIL_FUNCTIONS(uint16);
REGISTER_SPARSE_UTIL_FUNCTIONS(uint32);
REGISTER_SPARSE_UTIL_FUNCTIONS(uint64);
}  
}  