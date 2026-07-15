#include "kernel_operator.h"

namespace {

constexpr uint32_t kMaxDimSize = 4096;
constexpr uint32_t kDim1VectorAlignment = 256;
constexpr uint32_t kBlockCount = 40;
constexpr uint32_t kClearElementsPerCore = 128;
constexpr uint32_t kIndicesPerCore = 8;
constexpr uint32_t kBufferNum = 2;
constexpr uint32_t kMaxDim0FunctionalSmallIndexCount = 256;
constexpr uint32_t kGeneralPath = 0;
constexpr uint32_t kDim0InplaceSmallDeduplicatePath = 2;
constexpr uint32_t kDim0FunctionalSmallDirectPath = 3;
constexpr uint32_t kDim0FunctionalSmallMembershipPath = 4;

__aicore__ inline uint32_t AlignUp(uint32_t value, uint32_t alignment) {
  return (value + alignment - 1) / alignment * alignment;
}

template <typename T>
struct IndexFillValue {
  using Type = T;

  __aicore__ static inline Type Convert(float value, uint32_t) {
    return static_cast<T>(value);
  }
};

template <>
struct IndexFillValue<bfloat16_t> {
  using Type = half;

  __aicore__ static inline Type Convert(float, uint32_t value_bits) {
    union {
      uint16_t bits;
      half fp16_bits;
    } scalar_bits;
    scalar_bits.bits = static_cast<uint16_t>(value_bits);
    return scalar_bits.fp16_bits;
  }
};

template <typename T>
struct IndexFillSelect {
  __aicore__ static inline void Run(const AscendC::LocalTensor<T> &output,
                                    const AscendC::LocalTensor<uint8_t> &mask,
                                    const AscendC::LocalTensor<T> &input,
                                    T value,
                                    uint32_t count) {
    AscendC::Select(output, mask, input, value, AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, count);
  }
};

template <>
struct IndexFillSelect<bfloat16_t> {
  __aicore__ static inline void Run(const AscendC::LocalTensor<bfloat16_t> &output,
                                    const AscendC::LocalTensor<uint8_t> &mask,
                                    const AscendC::LocalTensor<bfloat16_t> &input,
                                    half value,
                                    uint32_t count) {
    AscendC::Select(output.ReinterpretCast<half>(),
                    mask,
                    input.ReinterpretCast<half>(),
                    value,
                    AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE,
                    count);
  }
};

template <typename T>
struct IndexFillDuplicate {
  __aicore__ static inline void Run(const AscendC::LocalTensor<T> &output, T value, uint32_t count) {
    AscendC::Duplicate(output, value, count);
  }
};

template <>
struct IndexFillDuplicate<bfloat16_t> {
  __aicore__ static inline void Run(const AscendC::LocalTensor<bfloat16_t> &output,
                                    half value,
                                    uint32_t count) {
    AscendC::Duplicate(output.ReinterpretCast<half>(), value, count);
  }
};

template <typename T>
struct IndexFillTailStore {
  __aicore__ static inline void Run(const AscendC::GlobalTensor<T> &global,
                                    const AscendC::LocalTensor<T> &local,
                                    uint32_t count) {
    AscendC::DataCopyExtParams copy_params {1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
    AscendC::DataCopyPad(global, local, copy_params);
  }
};

template <>
struct IndexFillTailStore<bfloat16_t> {
  __aicore__ static inline void Run(const AscendC::GlobalTensor<bfloat16_t> &global,
                                    const AscendC::LocalTensor<bfloat16_t> &local,
                                    uint32_t count) {
    AscendC::GlobalTensor<half> global_half;
    global_half.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(global.GetPhyAddr(0)), count);
    AscendC::DataCopyExtParams copy_params {1, static_cast<uint32_t>(count * sizeof(half)), 0, 0, 0};
    AscendC::DataCopyPad(global_half, local.ReinterpretCast<half>(), copy_params);
  }
};

template <typename T>
class IndexFillDim0InplaceSmallKernel {
 public:
  __aicore__ inline void Init(GM_ADDR index,
                              GM_ADDR output,
                              float value,
                              uint32_t value_bits,
                              uint32_t rows,
                              uint32_t cols,
                              uint32_t index_count,
                              uint32_t block_count,
                              bool deduplicate) {
    rows_ = rows;
    cols_ = cols;
    index_count_ = index_count;
    block_count_ = block_count;
    deduplicate_ = deduplicate;
    index_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(index), index_count_);
    output_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(output), rows_ * cols_);
    value_ = IndexFillValue<T>::Convert(value, value_bits);

    pipe_.InitBuffer(index_buf_, kIndicesPerCore * sizeof(int64_t));
    pipe_.InitBuffer(output_queue_, kBufferNum, kMaxDimSize * sizeof(T));
  }

  __aicore__ inline void Process() {
    const uint32_t core = AscendC::GetBlockIdx();
    auto index_local = index_buf_.Get<int64_t>();
    for (uint32_t offset = core * kIndicesPerCore; offset < index_count_;
         offset += block_count_ * kIndicesPerCore) {
      AscendC::DataCopy(index_local, index_gm_[offset], kIndicesPerCore);
      auto mte2_to_scalar = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_S));
      AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(mte2_to_scalar);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(mte2_to_scalar);
      for (uint32_t i = 0; i < kIndicesPerCore; ++i) {
        int64_t row = index_local.GetValue(i);
        row = row < 0 ? row + static_cast<int64_t>(rows_) : row;
        bool duplicate = false;
        if (deduplicate_) {
          for (uint32_t j = 0; j < i; ++j) {
            int64_t previous_row = index_local.GetValue(j);
            previous_row = previous_row < 0 ? previous_row + static_cast<int64_t>(rows_) : previous_row;
            duplicate = duplicate || previous_row == row;
          }
        }
        if (!duplicate) {
          FillRow(static_cast<uint32_t>(row));
        }
      }
    }
  }

 private:
  __aicore__ inline void CopyLocalToGlobal(const AscendC::GlobalTensor<T> &global,
                                           const AscendC::LocalTensor<T> &local,
                                           uint32_t count) {
    if ((count * sizeof(T)) % 32 == 0) {
      AscendC::DataCopy(global, local, count);
      return;
    }
    IndexFillTailStore<T>::Run(global, local, count);
  }

  __aicore__ inline void FillRow(uint32_t row) {
    for (uint32_t col = 0; col < cols_; col += kMaxDimSize) {
      const uint32_t count = cols_ - col < kMaxDimSize ? cols_ - col : kMaxDimSize;
      auto output_local = output_queue_.AllocTensor<T>();
      IndexFillDuplicate<T>::Run(output_local, value_, count);
      output_queue_.EnQue(output_local);
      output_local = output_queue_.DeQue<T>();
      CopyLocalToGlobal(output_gm_[row * cols_ + col], output_local, count);
      output_queue_.FreeTensor(output_local);
    }
  }

  AscendC::TPipe pipe_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> index_buf_;
  AscendC::TQue<AscendC::TPosition::VECOUT, kBufferNum> output_queue_;
  AscendC::GlobalTensor<int64_t> index_gm_;
  AscendC::GlobalTensor<T> output_gm_;
  typename IndexFillValue<T>::Type value_;
  uint32_t rows_;
  uint32_t cols_;
  uint32_t index_count_;
  uint32_t block_count_;
  bool deduplicate_;
};

template <typename T>
class IndexFillDim0FunctionalSmallKernel {
 public:
  __aicore__ inline void Init(GM_ADDR input,
                              GM_ADDR index,
                              GM_ADDR output,
                              float value,
                              uint32_t value_bits,
                              uint32_t rows,
                              uint32_t cols,
                              uint32_t index_count,
                              uint32_t block_count,
                              bool direct_match) {
    rows_ = rows;
    cols_ = cols;
    index_count_ = index_count;
    block_count_ = block_count;
    direct_match_ = direct_match;
    input_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(input), rows_ * cols_);
    index_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(index), index_count_);
    output_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(output), rows_ * cols_);
    value_ = IndexFillValue<T>::Convert(value, value_bits);

    pipe_.InitBuffer(index_buf_, kMaxDim0FunctionalSmallIndexCount * sizeof(int64_t));
    pipe_.InitBuffer(membership_buf_, kMaxDimSize * sizeof(half));
    pipe_.InitBuffer(input_queue_, kBufferNum, kMaxDimSize * sizeof(T));
    pipe_.InitBuffer(output_queue_, kBufferNum, kMaxDimSize * sizeof(T));
  }

  __aicore__ inline void Process() {
    LoadIndex();
    if (!direct_match_) {
      BuildLocalMembership();
    }
    const uint32_t core = AscendC::GetBlockIdx();
    for (uint32_t row = core; row < rows_; row += block_count_) {
      const bool selected =
          direct_match_ ? IsSelected(row) : membership_local_.ReinterpretCast<uint16_t>().GetValue(row) != 0;
      if (selected) {
        FillRow(row);
      } else {
        CopyRow(row);
      }
    }
  }

 private:
  __aicore__ inline void LoadIndex() {
    index_local_ = index_buf_.Get<int64_t>();
    AscendC::DataCopy(index_local_, index_gm_, index_count_);
    auto mte2_to_scalar = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_S));
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(mte2_to_scalar);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(mte2_to_scalar);
  }

  __aicore__ inline bool IsSelected(uint32_t row) {
    bool selected = false;
    for (uint32_t i = 0; i < index_count_; ++i) {
      int64_t index_row = index_local_.GetValue(i);
      index_row = index_row < 0 ? index_row + static_cast<int64_t>(rows_) : index_row;
      selected = selected || index_row == static_cast<int64_t>(row);
    }
    return selected;
  }

  __aicore__ inline void BuildLocalMembership() {
    membership_local_ = membership_buf_.Get<half>();
    AscendC::Duplicate(membership_local_, static_cast<half>(0), AlignUp(rows_, 16));
    auto vector_to_scalar = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_S));
    AscendC::SetFlag<AscendC::HardEvent::V_S>(vector_to_scalar);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(vector_to_scalar);
    for (uint32_t i = 0; i < index_count_; ++i) {
      int64_t index_row = index_local_.GetValue(i);
      index_row = index_row < 0 ? index_row + static_cast<int64_t>(rows_) : index_row;
      membership_local_.SetValue(static_cast<uint32_t>(index_row), static_cast<half>(1));
    }
  }

  __aicore__ inline void CopyGlobalToLocal(const AscendC::LocalTensor<T> &local,
                                           const AscendC::GlobalTensor<T> &global,
                                           uint32_t count) {
    if ((count * sizeof(T)) % 32 == 0) {
      AscendC::DataCopy(local, global, count);
      return;
    }
    AscendC::DataCopyExtParams copy_params {1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
    AscendC::DataCopyPadExtParams<T> pad_params;
    AscendC::DataCopyPad(local, global, copy_params, pad_params);
  }

  __aicore__ inline void CopyLocalToGlobal(const AscendC::GlobalTensor<T> &global,
                                           const AscendC::LocalTensor<T> &local,
                                           uint32_t count) {
    if ((count * sizeof(T)) % 32 == 0) {
      AscendC::DataCopy(global, local, count);
      return;
    }
    IndexFillTailStore<T>::Run(global, local, count);
  }

  __aicore__ inline void CopyRow(uint32_t row) {
    for (uint32_t col = 0; col < cols_; col += kMaxDimSize) {
      const uint32_t count = cols_ - col < kMaxDimSize ? cols_ - col : kMaxDimSize;
      auto input_local = input_queue_.AllocTensor<T>();
      CopyGlobalToLocal(input_local, input_gm_[row * cols_ + col], count);
      input_queue_.EnQue(input_local);
      input_local = input_queue_.DeQue<T>();
      auto mte2_to_mte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_MTE3));
      AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(mte2_to_mte3);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(mte2_to_mte3);
      CopyLocalToGlobal(output_gm_[row * cols_ + col], input_local, count);
      auto mte3_to_mte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE2));
      AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(mte3_to_mte2);
      AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mte3_to_mte2);
      input_queue_.FreeTensor(input_local);
    }
  }

  __aicore__ inline void FillRow(uint32_t row) {
    for (uint32_t col = 0; col < cols_; col += kMaxDimSize) {
      const uint32_t count = cols_ - col < kMaxDimSize ? cols_ - col : kMaxDimSize;
      auto output_local = output_queue_.AllocTensor<T>();
      IndexFillDuplicate<T>::Run(output_local, value_, count);
      output_queue_.EnQue(output_local);
      output_local = output_queue_.DeQue<T>();
      CopyLocalToGlobal(output_gm_[row * cols_ + col], output_local, count);
      output_queue_.FreeTensor(output_local);
    }
  }

  AscendC::TPipe pipe_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> index_buf_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> membership_buf_;
  AscendC::TQue<AscendC::TPosition::VECIN, kBufferNum> input_queue_;
  AscendC::TQue<AscendC::TPosition::VECOUT, kBufferNum> output_queue_;
  AscendC::GlobalTensor<T> input_gm_;
  AscendC::GlobalTensor<int64_t> index_gm_;
  AscendC::GlobalTensor<T> output_gm_;
  AscendC::LocalTensor<int64_t> index_local_;
  AscendC::LocalTensor<half> membership_local_;
  typename IndexFillValue<T>::Type value_;
  uint32_t rows_;
  uint32_t cols_;
  uint32_t index_count_;
  uint32_t block_count_;
  bool direct_match_;
};

template <typename T>
class IndexFillFusedKernel {
 public:
  __aicore__ inline void Init(GM_ADDR input,
                              GM_ADDR index,
                              GM_ADDR output,
                              GM_ADDR membership,
                              float value,
                              uint32_t value_bits,
                              uint32_t rows,
                              uint32_t cols,
                              uint32_t index_count,
                              uint32_t dim,
                              uint32_t inplace) {
    rows_ = rows;
    cols_ = cols;
    index_count_ = index_count;
    dim_ = dim;
    inplace_ = inplace != 0;
    dim_size_ = dim_ == 0 ? rows_ : cols_;
    membership_elements_ = AlignUp(dim_size_, dim_ == 1 ? kDim1VectorAlignment : 16);
    active_builder_cores_ = (index_count_ + kIndicesPerCore - 1) / kIndicesPerCore;
    active_builder_cores_ = active_builder_cores_ < kBlockCount ? active_builder_cores_ : kBlockCount;
    input_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(input), rows_ * cols_);
    index_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(index), index_count_);
    output_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(output), rows_ * cols_);
    membership_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(membership), membership_elements_);
    value_ = IndexFillValue<T>::Convert(value, value_bits);

    pipe_.InitBuffer(clear_buf_, kClearElementsPerCore * sizeof(half));
    pipe_.InitBuffer(index_buf_, kIndicesPerCore * sizeof(int64_t));
    pipe_.InitBuffer(membership_buf_, kMaxDimSize * sizeof(half));
    pipe_.InitBuffer(mask_buf_, kMaxDimSize / 8);
    pipe_.InitBuffer(input_queue_, kBufferNum, kMaxDimSize * sizeof(T));
    pipe_.InitBuffer(output_queue_, kBufferNum, kMaxDimSize * sizeof(T));
  }

  __aicore__ inline void Process() {
    ClearMembership();
    AscendC::SyncAll();
    MarkMembership();
    AscendC::SyncAll();

    if (dim_ == 0) {
      ProcessDim0();
    } else {
      ProcessDim1();
    }
  }

 private:
  __aicore__ inline void ClearMembership() {
    const uint32_t core = AscendC::GetBlockIdx();
    const uint32_t offset = core * kClearElementsPerCore;
    if (offset >= membership_elements_) {
      return;
    }

    auto clear_local = clear_buf_.Get<half>();
    AscendC::Duplicate(clear_local, static_cast<half>(0), kClearElementsPerCore);
    auto vector_to_mte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vector_to_mte3);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vector_to_mte3);
    const uint32_t count = membership_elements_ - offset < kClearElementsPerCore
                               ? membership_elements_ - offset
                               : kClearElementsPerCore;
    AscendC::DataCopy(membership_gm_[offset], clear_local, count);
    auto mte3_to_scalar = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_S));
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(mte3_to_scalar);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(mte3_to_scalar);
  }

  __aicore__ inline void MarkMembership() {
    const uint32_t core = AscendC::GetBlockIdx();
    if (core >= active_builder_cores_) {
      return;
    }

    auto local_membership = membership_buf_.Get<half>();
    AscendC::Duplicate(local_membership, static_cast<half>(0), membership_elements_);
    auto vector_to_scalar = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_S));
    AscendC::SetFlag<AscendC::HardEvent::V_S>(vector_to_scalar);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(vector_to_scalar);

    auto index_local = index_buf_.Get<int64_t>();
    for (uint32_t offset = core * kIndicesPerCore; offset < index_count_;
         offset += active_builder_cores_ * kIndicesPerCore) {
      AscendC::DataCopy(index_local, index_gm_[offset], kIndicesPerCore);
      auto mte2_to_scalar = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_S));
      AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(mte2_to_scalar);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(mte2_to_scalar);
      for (uint32_t i = 0; i < kIndicesPerCore; ++i) {
        int64_t index_value = index_local.GetValue(i);
        index_value = index_value < 0 ? index_value + static_cast<int64_t>(dim_size_) : index_value;
        if (index_value >= 0 && index_value < static_cast<int64_t>(dim_size_)) {
          local_membership.SetValue(index_value, static_cast<half>(1));
        }
      }
    }
    auto scalar_to_mte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::S_MTE3));
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(scalar_to_mte3);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(scalar_to_mte3);

    AscendC::DataCopyExtParams copy_params {1,
                                            static_cast<uint32_t>(membership_elements_ * sizeof(half)),
                                            0,
                                            0,
                                            0};
    AscendC::SetAtomicAdd<half>();
    AscendC::DataCopyPad(membership_gm_, local_membership, copy_params);
    AscendC::SetAtomicNone();

    auto mte3_to_scalar = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_S));
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(mte3_to_scalar);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(mte3_to_scalar);
  }

  __aicore__ inline void LoadMembership() {
    membership_local_ = membership_buf_.Get<half>();
    AscendC::DataCopy(membership_local_, membership_gm_, membership_elements_);
  }

  __aicore__ inline void ProcessDim0() {
    LoadMembership();
    auto mte2_to_scalar = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_S));
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(mte2_to_scalar);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(mte2_to_scalar);

    const uint32_t core = AscendC::GetBlockIdx();
    for (uint32_t row = core; row < rows_; row += kBlockCount) {
      const bool selected = membership_local_.ReinterpretCast<uint16_t>().GetValue(row) != 0;
      if (selected) {
        FillRow(row);
      } else if (!inplace_) {
        CopyRow(row);
      }
    }
  }

  __aicore__ inline void ProcessDim1() {
    LoadMembership();
    mask_local_ = mask_buf_.Get<uint8_t>();
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::CompareScalar(mask_local_,
                           membership_local_,
                           static_cast<half>(0),
                           AscendC::CMPMODE::EQ,
                           membership_elements_);
    AscendC::PipeBarrier<PIPE_ALL>();

    const uint32_t core = AscendC::GetBlockIdx();
    for (uint32_t row = core; row < rows_; row += kBlockCount) {
      CopyInDim1(row);
      ComputeDim1();
      CopyOutDim1(row);
    }
  }

  __aicore__ inline void CopyGlobalToLocal(const AscendC::LocalTensor<T> &local,
                                           const AscendC::GlobalTensor<T> &global,
                                           uint32_t count) {
    if ((count * sizeof(T)) % 32 == 0) {
      AscendC::DataCopy(local, global, count);
      return;
    }
    AscendC::DataCopyExtParams copy_params {1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
    AscendC::DataCopyPadExtParams<T> pad_params;
    AscendC::DataCopyPad(local, global, copy_params, pad_params);
  }

  __aicore__ inline void CopyLocalToGlobal(const AscendC::GlobalTensor<T> &global,
                                           const AscendC::LocalTensor<T> &local,
                                           uint32_t count) {
    if ((count * sizeof(T)) % 32 == 0) {
      AscendC::DataCopy(global, local, count);
      return;
    }
    IndexFillTailStore<T>::Run(global, local, count);
  }

  __aicore__ inline void CopyInDim1(uint32_t row) {
    auto input_local = input_queue_.AllocTensor<T>();
    CopyGlobalToLocal(input_local, input_gm_[row * cols_], cols_);
    input_queue_.EnQue(input_local);
  }

  __aicore__ inline void ComputeDim1() {
    auto input_local = input_queue_.DeQue<T>();
    auto output_local = output_queue_.AllocTensor<T>();
    IndexFillSelect<T>::Run(output_local, mask_local_, input_local, value_, membership_elements_);
    output_queue_.EnQue(output_local);
    input_queue_.FreeTensor(input_local);
  }

  __aicore__ inline void CopyOutDim1(uint32_t row) {
    auto output_local = output_queue_.DeQue<T>();
    CopyLocalToGlobal(output_gm_[row * cols_], output_local, cols_);
    output_queue_.FreeTensor(output_local);
  }

  __aicore__ inline void CopyRow(uint32_t row) {
    for (uint32_t col = 0; col < cols_; col += kMaxDimSize) {
      const uint32_t count = cols_ - col < kMaxDimSize ? cols_ - col : kMaxDimSize;
      auto input_local = input_queue_.AllocTensor<T>();
      CopyGlobalToLocal(input_local, input_gm_[row * cols_ + col], count);
      input_queue_.EnQue(input_local);
      input_local = input_queue_.DeQue<T>();
      auto mte2_to_mte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_MTE3));
      AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(mte2_to_mte3);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(mte2_to_mte3);
      CopyLocalToGlobal(output_gm_[row * cols_ + col], input_local, count);
      auto mte3_to_mte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE2));
      AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(mte3_to_mte2);
      AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mte3_to_mte2);
      input_queue_.FreeTensor(input_local);
    }
  }

  __aicore__ inline void FillRow(uint32_t row) {
    for (uint32_t col = 0; col < cols_; col += kMaxDimSize) {
      const uint32_t count = cols_ - col < kMaxDimSize ? cols_ - col : kMaxDimSize;
      auto output_local = output_queue_.AllocTensor<T>();
      IndexFillDuplicate<T>::Run(output_local, value_, count);
      output_queue_.EnQue(output_local);
      output_local = output_queue_.DeQue<T>();
      CopyLocalToGlobal(output_gm_[row * cols_ + col], output_local, count);
      output_queue_.FreeTensor(output_local);
    }
  }

  AscendC::TPipe pipe_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> clear_buf_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> index_buf_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> membership_buf_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> mask_buf_;
  AscendC::TQue<AscendC::TPosition::VECIN, kBufferNum> input_queue_;
  AscendC::TQue<AscendC::TPosition::VECOUT, kBufferNum> output_queue_;
  AscendC::GlobalTensor<T> input_gm_;
  AscendC::GlobalTensor<int64_t> index_gm_;
  AscendC::GlobalTensor<T> output_gm_;
  AscendC::GlobalTensor<half> membership_gm_;
  AscendC::LocalTensor<half> membership_local_;
  AscendC::LocalTensor<uint8_t> mask_local_;
  typename IndexFillValue<T>::Type value_;
  uint32_t rows_;
  uint32_t cols_;
  uint32_t index_count_;
  uint32_t dim_;
  uint32_t dim_size_;
  uint32_t membership_elements_;
  uint32_t active_builder_cores_;
  bool inplace_;
};

template <typename T>
__aicore__ inline void RunIndexFill(GM_ADDR input,
                                    GM_ADDR index,
                                    GM_ADDR output,
                                    GM_ADDR membership,
                                    float value,
                                    uint32_t value_bits,
                                    uint32_t rows,
                                    uint32_t cols,
                                    uint32_t index_count,
                                    uint32_t dim,
                                    uint32_t inplace,
                                    uint32_t path_code,
                                    uint32_t block_count) {
  if (path_code == kDim0FunctionalSmallDirectPath ||
      path_code == kDim0FunctionalSmallMembershipPath) {
    IndexFillDim0FunctionalSmallKernel<T> kernel;
    kernel.Init(input,
                index,
                output,
                value,
                value_bits,
                rows,
                cols,
                index_count,
                block_count,
                path_code == kDim0FunctionalSmallDirectPath);
    kernel.Process();
    return;
  }
  if (path_code != kGeneralPath) {
    IndexFillDim0InplaceSmallKernel<T> kernel;
    kernel.Init(index,
                output,
                value,
                value_bits,
                rows,
                cols,
                index_count,
                block_count,
                path_code == kDim0InplaceSmallDeduplicatePath);
    kernel.Process();
    return;
  }
  IndexFillFusedKernel<T> kernel;
  kernel.Init(input,
              index,
              output,
              membership,
              value,
              value_bits,
              rows,
              cols,
              index_count,
              dim,
              inplace);
  kernel.Process();
}

}  // namespace

extern "C" __global__ __aicore__ void flag_gems_index_fill_fused_2d(GM_ADDR input,
                                                                    GM_ADDR index,
                                                                    GM_ADDR output,
                                                                    GM_ADDR membership,
                                                                    float value,
                                                                    uint32_t value_bits,
                                                                    uint32_t dtype_code,
                                                                    uint32_t rows,
                                                                    uint32_t cols,
                                                                    uint32_t index_count,
                                                                    uint32_t dim,
                                                                    uint32_t inplace,
                                                                    uint32_t path_code,
                                                                    uint32_t block_count) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
  if (dtype_code == 0) {
    RunIndexFill<half>(input, index, output, membership, value, value_bits, rows,
                       cols, index_count, dim, inplace, path_code, block_count);
  } else if (dtype_code == 1) {
    RunIndexFill<bfloat16_t>(input, index, output, membership, value, value_bits,
                             rows, cols, index_count, dim, inplace, path_code, block_count);
  } else if (dtype_code == 2) {
    RunIndexFill<float>(input, index, output, membership, value, value_bits, rows,
                        cols, index_count, dim, inplace, path_code, block_count);
  }
}
