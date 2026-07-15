#include "kernel_operator.h"

namespace {

constexpr uint32_t kMaxCols = 4096;
constexpr uint32_t kBlockCount = 40;
constexpr uint32_t kClearElementsPerCore = 128;
constexpr uint32_t kIndicesPerCore = 8;
constexpr uint32_t kBufferNum = 2;

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
                              uint32_t index_count) {
    rows_ = rows;
    cols_ = cols;
    index_count_ = index_count;
    active_builder_cores_ = (index_count_ + kIndicesPerCore - 1) / kIndicesPerCore;
    active_builder_cores_ = active_builder_cores_ < kBlockCount ? active_builder_cores_ : kBlockCount;
    input_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(input), rows_ * cols_);
    index_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(index), index_count_);
    output_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(output), rows_ * cols_);
    membership_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(membership), cols_);
    value_ = IndexFillValue<T>::Convert(value, value_bits);

    pipe_.InitBuffer(clear_buf_, kClearElementsPerCore * sizeof(half));
    pipe_.InitBuffer(index_buf_, kIndicesPerCore * sizeof(int64_t));
    pipe_.InitBuffer(membership_buf_, kMaxCols * sizeof(half));
    pipe_.InitBuffer(mask_buf_, kMaxCols / 8);
    pipe_.InitBuffer(input_queue_, kBufferNum, kMaxCols * sizeof(T));
    pipe_.InitBuffer(output_queue_, kBufferNum, kMaxCols * sizeof(T));
  }

  __aicore__ inline void Process() {
    ClearMembership();
    AscendC::SyncAll();
    MarkMembership();
    AscendC::SyncAll();
    BuildMask();

    const uint32_t core = AscendC::GetBlockIdx();
    for (uint32_t row = core; row < rows_; row += kBlockCount) {
      CopyIn(row);
      Compute();
      CopyOut(row);
    }
  }

 private:
  __aicore__ inline void ClearMembership() {
    const uint32_t core = AscendC::GetBlockIdx();
    const uint32_t offset = core * kClearElementsPerCore;
    if (offset >= cols_) {
      return;
    }

    auto clear_local = clear_buf_.Get<half>();
    AscendC::Duplicate(clear_local, static_cast<half>(0), kClearElementsPerCore);
    auto vector_to_mte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vector_to_mte3);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vector_to_mte3);
    const uint32_t count = cols_ - offset < kClearElementsPerCore ? cols_ - offset : kClearElementsPerCore;
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
    AscendC::Duplicate(local_membership, static_cast<half>(0), cols_);
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
        index_value = index_value < 0 ? index_value + static_cast<int64_t>(cols_) : index_value;
        if (index_value >= 0 && index_value < static_cast<int64_t>(cols_)) {
          local_membership.SetValue(index_value, static_cast<half>(1));
        }
      }
    }
    auto scalar_to_mte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::S_MTE3));
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(scalar_to_mte3);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(scalar_to_mte3);

    AscendC::DataCopyExtParams copy_params {1, static_cast<uint32_t>(cols_ * sizeof(half)), 0, 0, 0};
    AscendC::SetAtomicAdd<half>();
    AscendC::DataCopyPad(membership_gm_, local_membership, copy_params);
    AscendC::SetAtomicNone();

    auto mte3_to_scalar = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_S));
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(mte3_to_scalar);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(mte3_to_scalar);
  }

  __aicore__ inline void BuildMask() {
    membership_local_ = membership_buf_.Get<half>();
    mask_local_ = mask_buf_.Get<uint8_t>();
    AscendC::DataCopy(membership_local_, membership_gm_, cols_);
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::CompareScalar(mask_local_, membership_local_, static_cast<half>(0), AscendC::CMPMODE::EQ, cols_);
    AscendC::PipeBarrier<PIPE_ALL>();
  }

  __aicore__ inline void CopyIn(uint32_t row) {
    auto input_local = input_queue_.AllocTensor<T>();
    AscendC::DataCopy(input_local, input_gm_[row * cols_], cols_);
    input_queue_.EnQue(input_local);
  }

  __aicore__ inline void Compute() {
    auto input_local = input_queue_.DeQue<T>();
    auto output_local = output_queue_.AllocTensor<T>();
    IndexFillSelect<T>::Run(output_local, mask_local_, input_local, value_, cols_);
    output_queue_.EnQue(output_local);
    input_queue_.FreeTensor(input_local);
  }

  __aicore__ inline void CopyOut(uint32_t row) {
    auto output_local = output_queue_.DeQue<T>();
    AscendC::DataCopy(output_gm_[row * cols_], output_local, cols_);
    output_queue_.FreeTensor(output_local);
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
  uint32_t active_builder_cores_;
};

}  // namespace

extern "C" __global__ __aicore__ void flag_gems_index_fill_fused_2d_dim1(GM_ADDR input,
                                                                         GM_ADDR index,
                                                                         GM_ADDR output,
                                                                         GM_ADDR membership,
                                                                         float value,
                                                                         uint32_t value_bits,
                                                                         uint32_t dtype_code,
                                                                         uint32_t rows,
                                                                         uint32_t cols,
                                                                         uint32_t index_count) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
  if (dtype_code == 0) {
    IndexFillFusedKernel<half> kernel;
    kernel.Init(input, index, output, membership, value, value_bits, rows, cols, index_count);
    kernel.Process();
  } else if (dtype_code == 1) {
    IndexFillFusedKernel<bfloat16_t> kernel;
    kernel.Init(input, index, output, membership, value, value_bits, rows, cols, index_count);
    kernel.Process();
  } else {
    IndexFillFusedKernel<float> kernel;
    kernel.Init(input, index, output, membership, value, value_bits, rows, cols, index_count);
    kernel.Process();
  }
}
