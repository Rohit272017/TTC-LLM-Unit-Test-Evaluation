#ifndef TENSORSTORE_INDEX_SPACE_OUTPUT_INDEX_MAP_H_
#define TENSORSTORE_INDEX_SPACE_OUTPUT_INDEX_MAP_H_
#include <cassert>
#include "tensorstore/array.h"
#include "tensorstore/index_space/internal/transform_rep.h"
#include "tensorstore/index_space/output_index_method.h"
#include "tensorstore/strided_layout.h"
#include "tensorstore/util/element_pointer.h"
namespace tensorstore {
template <DimensionIndex InputRank = dynamic_rank>
class OutputIndexMapRef {
 public:
  class IndexArrayView {
   public:
    SharedArrayView<const Index, InputRank, offset_origin> shared_array_ref()
        const {
      return {element_pointer(), layout()};
    }
    ArrayView<const Index, InputRank, offset_origin> array_ref() const {
      return {element_pointer(), layout()};
    }
    const SharedElementPointer<const Index>& element_pointer() const {
      return index_array_data_->element_pointer;
    }
    IndexInterval index_range() const { return index_array_data_->index_range; }
    StaticOrDynamicRank<InputRank> rank() const {
      return StaticRankCast<InputRank, unchecked>(
          static_cast<DimensionIndex>(rep_->input_rank));
    }
    StridedLayoutView<InputRank, offset_origin> layout() const {
      return StridedLayoutView<InputRank, offset_origin>(
          rank(), rep_->input_origin().data(), rep_->input_shape().data(),
          index_array_data_->byte_strides);
    }
    span<const Index, InputRank> byte_strides() const {
      return {index_array_data_->byte_strides, rank()};
    }
   private:
    template <DimensionIndex>
    friend class OutputIndexMapRef;
    explicit IndexArrayView(
        internal_index_space::IndexArrayData* index_array_data,
        internal_index_space::TransformRep* rep)
        : index_array_data_(index_array_data), rep_(rep) {}
    internal_index_space::IndexArrayData* index_array_data_;
    internal_index_space::TransformRep* rep_;
  };
  OutputIndexMapRef() = default;
  OutputIndexMapRef& operator=(const OutputIndexMapRef&) = default;
  StaticOrDynamicRank<InputRank> input_rank() const {
    return StaticRankCast<InputRank, unchecked>(
        static_cast<DimensionIndex>(rep_->input_rank));
  }
  OutputIndexMethod method() const { return map_->method(); }
  Index offset() const { return map_->offset(); }
  Index stride() const { return map_->stride(); }
  DimensionIndex input_dimension() const { return map_->input_dimension(); }
  IndexArrayView index_array() const {
    return IndexArrayView(&map_->index_array_data(), rep_);
  }
 private:
  template <DimensionIndex, DimensionIndex, ContainerKind>
  friend class OutputIndexMapRange;
  template <DimensionIndex>
  friend class OutputIndexMapIterator;
  explicit OutputIndexMapRef(internal_index_space::OutputIndexMap* map,
                             internal_index_space::TransformRep* rep)
      : map_(map), rep_(rep) {}
  internal_index_space::OutputIndexMap* map_ = nullptr;
  internal_index_space::TransformRep* rep_ = nullptr;
};
template <DimensionIndex InputRank = dynamic_rank>
class OutputIndexMapIterator {
 public:
  using value_type = OutputIndexMapRef<InputRank>;
  using reference = OutputIndexMapRef<InputRank>;
  using difference_type = DimensionIndex;
  using pointer = value_type*;
  using iterator_category = std::random_access_iterator_tag;
  OutputIndexMapIterator() = default;
  OutputIndexMapRef<InputRank> operator*() const { return ref_; }
  const OutputIndexMapRef<InputRank>* operator->() const { return &ref_; }
  OutputIndexMapRef<InputRank> operator[](DimensionIndex n) const {
    auto new_ref = ref_;
    new_ref.map_ += n;
    return new_ref;
  }
  OutputIndexMapIterator& operator+=(DimensionIndex n) {
    ref_.map_ += n;
    return *this;
  }
  OutputIndexMapIterator& operator-=(DimensionIndex n) { return *this += (-n); }
  OutputIndexMapIterator& operator++() {
    ++ref_.map_;
    return *this;
  }
  OutputIndexMapIterator& operator--() {
    --ref_.map_;
    return *this;
  }
  OutputIndexMapIterator operator++(int) {
    auto temp = *this;
    ++ref_.map_;
    return temp;
  }
  OutputIndexMapIterator operator--(int) {
    auto temp = *this;
    --ref_.map_;
    return temp;
  }
  friend DimensionIndex operator-(OutputIndexMapIterator a,
                                  OutputIndexMapIterator b) {
    return a.map() - b.map();
  }
  friend OutputIndexMapIterator operator+(OutputIndexMapIterator it,
                                          DimensionIndex n) {
    it += n;
    return it;
  }
  friend OutputIndexMapIterator operator+(DimensionIndex n,
                                          OutputIndexMapIterator it) {
    it += n;
    return it;
  }
  friend OutputIndexMapIterator operator-(OutputIndexMapIterator it,
                                          DimensionIndex n) {
    it -= n;
    return it;
  }
  friend bool operator==(OutputIndexMapIterator a, OutputIndexMapIterator b) {
    return a.map() == b.map();
  }
  friend bool operator!=(OutputIndexMapIterator a, OutputIndexMapIterator b) {
    return a.map() != b.map();
  }
  friend bool operator<(OutputIndexMapIterator a, OutputIndexMapIterator b) {
    return a.map() < b.map();
  }
  friend bool operator<=(OutputIndexMapIterator a, OutputIndexMapIterator b) {
    return a.map() <= b.map();
  }
  friend bool operator>(OutputIndexMapIterator a, OutputIndexMapIterator b) {
    return a.map() > b.map();
  }
  friend bool operator>=(OutputIndexMapIterator a, OutputIndexMapIterator b) {
    return a.map() >= b.map();
  }
 private:
  internal_index_space::OutputIndexMap* map() const { return ref_.map_; }
  template <DimensionIndex, DimensionIndex, ContainerKind>
  friend class OutputIndexMapRange;
  OutputIndexMapRef<InputRank> ref_;
  explicit OutputIndexMapIterator(internal_index_space::OutputIndexMap* map,
                                  internal_index_space::TransformRep* rep)
      : ref_(map, rep) {}
};
template <DimensionIndex InputRank = dynamic_rank,
          DimensionIndex OutputRank = dynamic_rank, ContainerKind CKind = view>
class OutputIndexMapRange {
 public:
  using value_type = OutputIndexMapRef<InputRank>;
  using reference = value_type;
  using iterator = OutputIndexMapIterator<InputRank>;
  using difference_type = DimensionIndex;
  constexpr static DimensionIndex extent = OutputRank;
  OutputIndexMapRange() = default;
  explicit OutputIndexMapRange(
      IndexTransform<InputRank, OutputRank, CKind> transform)
      : transform_(std::move(transform)) {}
  template <DimensionIndex OtherInputRank, DimensionIndex OtherOutputRank,
            ContainerKind OtherCKind,
            typename = std::enable_if_t<
                (RankConstraint::Implies(OtherInputRank, InputRank) &&
                 RankConstraint::Implies(OtherOutputRank, OutputRank))>>
  OutputIndexMapRange(
      OutputIndexMapRange<OtherInputRank, OtherOutputRank, OtherCKind> other)
      : transform_(std::move(other.transform_)) {}
  StaticOrDynamicRank<OutputRank> size() const {
    return transform_.output_rank();
  }
  bool empty() const { return size() == 0; }
  iterator begin() const {
    return iterator(rep()->output_index_maps().data(), rep());
  }
  iterator end() const {
    return iterator(rep()->output_index_maps().data() + size(), rep());
  }
  OutputIndexMapRef<InputRank> operator[](DimensionIndex output_dim) const {
    assert(output_dim >= 0 && output_dim < size());
    return OutputIndexMapRef<InputRank>(
        rep()->output_index_maps().data() + output_dim, rep());
  }
  StaticOrDynamicRank<InputRank> input_rank() const {
    return transform_.input_rank();
  }
 private:
  template <DimensionIndex, DimensionIndex, ContainerKind>
  friend class OutputIndexMapRange;
  internal_index_space::TransformRep* rep() const {
    return internal_index_space::TransformAccess::rep(transform_);
  }
  IndexTransform<InputRank, OutputRank, CKind> transform_;
};
}  
#endif  