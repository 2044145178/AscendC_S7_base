#include "kernel_operator.h"
using namespace AscendC;
constexpr uint32_t BufferNum = 1;
// Find the closest power of two, except 0.
__aicore__ inline uint32_t FindClosestPowerOfTwo(uint32_t n) {
    ASCENDC_ASSERT(n != 0, { KERNEL_LOG(KERNEL_ERROR, "input n must be non-zero!"); });
    constexpr uint32_t totalShiftBits = 63;
    return totalShiftBits - ScalarCountLeadingZero(n);
}
// 会修改srcTensor的内容
// 支持：half/int16_t/int32_t/float
template <typename T>
__aicore__ inline T BinaryReduceAdd(const LocalTensor<T> &srcTensor, uint32_t count) {
    uint32_t k = FindClosestPowerOfTwo(count);
    uint32_t splitK = 1 << k;
    uint32_t remain = count - splitK;
    if (remain != 0) {
        Add(srcTensor, srcTensor, srcTensor[splitK], remain);
    }
    while (splitK * sizeof(T) > 32) {
        splitK >>= 1;
        Add(srcTensor, srcTensor, srcTensor[splitK], splitK);
    }
    if constexpr (std::is_same<T, half>::value) {
        float sum_fp32 = 0.0f;
        for (uint32_t i = 0; i < splitK; ++i) {
            sum_fp32 += static_cast<float>(srcTensor(i));
        }
        return static_cast<half>(sum_fp32);
    } else {
        T sum = 0;
        for (uint32_t i = 0; i < splitK; ++i) {
            sum += srcTensor(i);
        }
        return sum;
    }
}
template <typename T>
class KernelTrace {
public:
    __aicore__ inline KernelTrace() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR out, uint32_t N, uint32_t M, TPipe *pipeIn) {
        if constexpr (std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value) {
            // int8_t/uint8_t 需要用到 float 临时变量
            this->tmp_sum_fp16 = static_cast<half>(+0.0f);
        } else {
            this->tmp_sum = static_cast<T>(+0.0f);
        }
        this->pipe = pipeIn;
        this->size = min(N, M);
        this->M = M;
        this->N = N;

        inputGm.SetGlobalBuffer((__gm__ T *)x, N * M);
        outGm.SetGlobalBuffer((__gm__ T *)out, 32);
        constexpr uint32_t maxUBSizeKB = 220;// 220K
        if constexpr (std::is_same<T, float>::value || std::is_same<T, int32_t>::value || std::is_same<T, int16_t>::value) {
            // 1 = 1(in)
            uint32_t spaceSize = maxUBSizeKB / 1 / BufferNum * 1024;
            spaceSize = min((uint32_t)(size * 32), spaceSize);

            this->n_elements_per_iter = spaceSize / 32;
            this->n_elements_per_iter = min(4095U, this->n_elements_per_iter);
            this->smallLoopTimes = (size + n_elements_per_iter - 1) / n_elements_per_iter;

            pipe->InitBuffer(inputBuf, BufferNum, spaceSize);
            pipe->InitBuffer(outBuf, BufferNum, 256);
        } else if constexpr (std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value) {
            // 3 = 1(input int8_t) + 2(half tmp)
            uint32_t spaceSize = maxUBSizeKB / 3 / BufferNum * 1024;
            spaceSize = min((uint32_t)(size * 32), spaceSize);

            this->n_elements_per_iter = spaceSize / 32;
            this->n_elements_per_iter = min(4095U, this->n_elements_per_iter);
            this->smallLoopTimes = (size + n_elements_per_iter - 1) / n_elements_per_iter;
            pipe->InitBuffer(inputBuf, BufferNum, spaceSize);
            pipe->InitBuffer(outBuf, BufferNum, 256);
            pipe->InitBuffer(tmpBuf, BufferNum, spaceSize * 2);
        }
    }
    __aicore__ inline void Process_iter(uint32_t offset, uint32_t iterSize) {
        LocalTensor<T> inputLocal = inputBuf.AllocTensor<T>();

        // 从全局内存读取数据到本地内存
        DataCopyExtParams copyParams{static_cast<uint16_t>(iterSize), sizeof(T), static_cast<uint32_t>((M) * sizeof(T)), 0, 0}; // iterSize∈[1, 4095]
        DataCopyPadExtParams<T> padParams{true, 0, (32 / sizeof(T) - 1), 0};
        DataCopyPad(inputLocal, inputGm[offset], copyParams, padParams);

        inputBuf.EnQue<T>(inputLocal);
        inputLocal = inputBuf.DeQue<T>();

        LocalTensor<T> outLocal = outBuf.AllocTensor<T>();
        uint32_t realSize = 32 / sizeof(T) * iterSize;

        // 计算
        if constexpr (std::is_same<T, float>::value || std::is_same<T, int32_t>::value || std::is_same<T, int16_t>::value) {
            tmp_sum += BinaryReduceAdd(inputLocal, realSize);
        } else if constexpr (std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value) {
            LocalTensor<half> inputLocal_fp16 = tmpBuf.AllocTensor<half>();
            Cast(inputLocal_fp16, inputLocal, RoundMode::CAST_NONE, iterSize);
            float tmp_sum_fp32 = static_cast<float>(tmp_sum_fp16) + static_cast<float>(BinaryReduceAdd(inputLocal_fp16, realSize));
            tmp_sum_fp16 = static_cast<half>(tmp_sum_fp32);
            tmpBuf.FreeTensor<half>(inputLocal_fp16);
        }
        inputBuf.FreeTensor<T>(inputLocal);
        outBuf.EnQue<T>(outLocal);
        outLocal = outBuf.DeQue<T>();
        outBuf.FreeTensor<T>(outLocal);
    }
    __aicore__ inline void Process() {
        uint32_t blockBeginIndex = 0;
        for (uint32_t i = 0; i < smallLoopTimes - 1; ++i) {
            Process_iter(blockBeginIndex, n_elements_per_iter);
            blockBeginIndex += n_elements_per_iter * (M + 1);
        }
        uint32_t iterSize = min(n_elements_per_iter, size - blockBeginIndex / (M + 1));
        Process_iter(blockBeginIndex, iterSize);

        if constexpr (std::is_same<T, float>::value || std::is_same<T, int32_t>::value || std::is_same<T, int16_t>::value) {
            // printf("trace result: %f\n", static_cast<float>(tmp_sum));
            outGm.SetValue(0, tmp_sum);
            DataCacheCleanAndInvalid<T, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(outGm);
            // printf("outGm:%f\n",static_cast<float>(outGm.GetValue(0)));
        } else if constexpr (std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value) {
            outGm.SetValue(0, static_cast<T>(static_cast<int8_t>(tmp_sum_fp16)));
            DataCacheCleanAndInvalid<T, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(outGm);
        }
    }

private:
    TPipe *pipe;
    GlobalTensor<T> inputGm;
    GlobalTensor<T> targetGm;
    GlobalTensor<T> outGm;
    TQue<QuePosition::VECIN, 1> inputBuf;
    TQue<QuePosition::VECOUT, 1> outBuf;
    TQue<QuePosition::VECCALC, 1> tmpBuf;
    uint32_t size;
    uint32_t smallLoopTimes;
    uint32_t n_elements_per_iter;
    uint32_t N;
    uint32_t M;

    half tmp_sum_fp16;
    T tmp_sum;
};
extern "C" __global__ __aicore__ void trace(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
    TPipe pipe;
    if constexpr (!std::is_same<DTYPE_X, int64_t>::value) {
        KernelTrace<DTYPE_X> kernel;
        kernel.Init(x, y, tiling_data.N, tiling_data.M, &pipe);
        kernel.Process();
    } else {
        return;
    }
}