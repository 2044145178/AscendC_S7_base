#include "kernel_operator.h"
using namespace AscendC;
constexpr uint32_t BufferNum = 2;
template <typename T>
class KernelSoftplus {
public:
    using ComputeType = typename std::conditional<std::is_same<T, half>::value, half, float>::type;
    __aicore__ inline KernelSoftplus() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR out, uint32_t smallSize, uint32_t incSize, uint32_t formerNum, float beta, float threshold, TPipe *pipeIn) {
        this->pipe = pipeIn;
        this->beta = beta;
        this->threshold = threshold;
        this->beta_val = static_cast<ComputeType>(beta);
        this->inverse_beta_val = static_cast<ComputeType>(1.0f / beta);
        this->threshold_val = static_cast<ComputeType>(threshold);
        uint32_t srcBeginIndex = 0;
        if (GetBlockIdx() < formerNum) {
            this->size = smallSize + incSize;
            srcBeginIndex = this->size * GetBlockIdx();
        } else {
            this->size = smallSize;
            srcBeginIndex = this->size * GetBlockIdx() + formerNum * incSize;
        }
        inputGm.SetGlobalBuffer((__gm__ T *)x + srcBeginIndex, size);
        outGm.SetGlobalBuffer((__gm__ T *)out + srcBeginIndex, size);
        constexpr uint32_t maxUBSizeKB = 220;// 220K
        if constexpr (std::is_same<T, float>::value || std::is_same<T, half>::value) {
            // 2 ~= 1(in) + 1(out) + 1/16(tmp for half)
            uint32_t spaceSize = maxUBSizeKB / 2 / BufferNum * 1024;
            spaceSize = min((uint32_t)(size * sizeof(T)), spaceSize);

            this->n_elements_per_iter = spaceSize / sizeof(T);
            this->smallLoopTimes = (size + n_elements_per_iter - 1) / n_elements_per_iter;

            pipe->InitBuffer(inputBuf, BufferNum, spaceSize);
            pipe->InitBuffer(outBuf, BufferNum, spaceSize);
            pipe->InitBuffer(tmpBuf, BufferNum, spaceSize / (sizeof(T) * 8) + 256); // tmpBuf 用于存放 uint8 mask
        } else if (std::is_same<T, bfloat16_t>::value) {
            // 6 = 1(input bf16) + 1(out bf16) + 4(2*float tmp)
            uint32_t spaceSize = maxUBSizeKB / 6 / BufferNum * 1024;
            spaceSize = min((uint32_t)(size * sizeof(T)), spaceSize);

            this->n_elements_per_iter = spaceSize / sizeof(T);
            this->smallLoopTimes = (size + n_elements_per_iter - 1) / n_elements_per_iter;

            pipe->InitBuffer(inputBuf, BufferNum, spaceSize);
            pipe->InitBuffer(outBuf, BufferNum, spaceSize);
            // tmpBuf 用于存放 float 中间结果，大小需为 bf16 的 4 倍 (即 2 倍 float 元素数量)
            pipe->InitBuffer(tmpBuf, BufferNum, spaceSize * 4);
        }
    }
    template <bool IsBetaOne>
    __aicore__ inline void Process_iter(uint32_t offset, uint32_t iterSize) {
        LocalTensor<T> inputLocal = inputBuf.AllocTensor<T>();

        // 从全局内存读取数据到本地内存
        DataCopy(inputLocal, inputGm[offset], iterSize);
        inputBuf.EnQue<T>(inputLocal);
        inputLocal = inputBuf.DeQue<T>();

        LocalTensor<T> outLocal = outBuf.AllocTensor<T>();
        // 计算
        LocalTensor<uint8_t> tmpLocal = tmpBuf.AllocTensor<uint8_t>();

        if constexpr (std::is_same<T, float>::value || std::is_same<T, half>::value) {
            Compute_iter<IsBetaOne>(outLocal, inputLocal, tmpLocal, iterSize);

        } else if constexpr (std::is_same<T, bfloat16_t>::value) {
            LocalTensor<float> inputLocal_fp32 = tmpLocal.template ReinterpretCast<float>();
            LocalTensor<uint8_t> maskLocal = outLocal.template ReinterpretCast<uint8_t>();
            LocalTensor<float> outLocal_fp32 = inputLocal_fp32[n_elements_per_iter];

            Cast(inputLocal_fp32, inputLocal, RoundMode::CAST_NONE, iterSize);
            Compute_iter<IsBetaOne>(outLocal_fp32, inputLocal_fp32, maskLocal, iterSize);
            Cast(outLocal, outLocal_fp32, RoundMode::CAST_RINT, iterSize);
        }
        tmpBuf.FreeTensor<uint8_t>(tmpLocal);

        inputBuf.FreeTensor<T>(inputLocal);
        outBuf.EnQue<T>(outLocal);
        outLocal = outBuf.DeQue<T>();
        DataCopy(outGm[offset], outLocal, iterSize);
        outBuf.FreeTensor<T>(outLocal);
    }
    template <bool IsBetaOne>
    __aicore__ inline void Compute_iter(const LocalTensor<ComputeType> &outLocal, const LocalTensor<ComputeType> &inputLocal, const LocalTensor<uint8_t> &maskLocal, uint32_t iterSize) {
        // out = log(1 + exp(beta * x)) / beta

        if constexpr (!IsBetaOne) {
            Muls(inputLocal, inputLocal, beta_val, iterSize);
        }

        // inputLocal不动
        CompareScalar(maskLocal, inputLocal, threshold_val, CMPMODE::GT, iterSize); // maskLocal = (inputLocal > threshold_T)
        // exp_part = exp(beta * x)
        Exp(outLocal, inputLocal, iterSize);

        Adds(outLocal, outLocal, static_cast<ComputeType>(1), iterSize);

        Ln(outLocal, outLocal, iterSize);

        // Select first, then multiply by 1/beta
        // If > threshold, select inputLocal (beta * x). After mul 1/beta, it becomes x.
        // If <= threshold, select outLocal (ln(1+exp)). After mul 1/beta, it becomes result.
        Select(outLocal, maskLocal, inputLocal, outLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE, iterSize);

        if constexpr (!IsBetaOne) {
            Muls(outLocal, outLocal, inverse_beta_val, iterSize);
        }
    }
    template <bool IsBetaOne>
    __aicore__ inline void ProcessImpl() {
        uint32_t blockBeginIndex = 0;
        for (uint32_t i = 0; i < smallLoopTimes - 1; ++i) {
            Process_iter<IsBetaOne>(blockBeginIndex, n_elements_per_iter);
            blockBeginIndex += n_elements_per_iter;
        }
        uint32_t iterSize = min(n_elements_per_iter, size - blockBeginIndex);
        Process_iter<IsBetaOne>(blockBeginIndex, iterSize);
    }
    __aicore__ inline void Process() {
        if (beta == 1.0f) {
            ProcessImpl<true>();
        } else {
            ProcessImpl<false>();
        }
    }

private:
    TPipe *pipe;
    GlobalTensor<T> inputGm;
    GlobalTensor<T> outGm;
    TQue<QuePosition::VECIN, 1> inputBuf;
    TQue<QuePosition::VECOUT, 1> outBuf;
    TQue<QuePosition::VECCALC, 1> tmpBuf;
    uint32_t size;
    uint32_t smallLoopTimes;
    uint32_t n_elements_per_iter;

    uint32_t totalSize;
    uint32_t inputSize;
    uint32_t otherSize;

    float beta;
    float threshold;
    ComputeType beta_val;
    ComputeType inverse_beta_val;
    ComputeType threshold_val;
};
extern "C" __global__ __aicore__ void softplus(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
    TPipe pipe;
    KernelSoftplus<DTYPE_X> kernel;
    kernel.Init(x, y, tiling_data.smallSize, tiling_data.incSize, tiling_data.formerNum, tiling_data.beta, tiling_data.threshold, &pipe);
    kernel.Process();
}