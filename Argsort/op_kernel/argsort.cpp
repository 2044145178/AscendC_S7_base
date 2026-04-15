#include "kernel_operator.h"
#include <type_traits>
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 1;
/**
 * 找到最接近且不小于 y 的 c * 4^k，其中 c ∈ {1, 2, 3}
 * @param y 输入的正整数
 * @param c_ptr 返回的 c 值
 * @param k_ptr 返回的 k 值
 * @return 满足条件的值 c * 4^k
 */
__aicore__ inline int find_nearest_4_power(int y, int* c_ptr, int* k_ptr) {
    if (y <= 0) {
        *c_ptr = 1;
        *k_ptr = 0;
        return 1;
    }

    if (y <= 3) {
        *c_ptr = y;
        *k_ptr = 0;
        return y;
    }

    int best_val = 1 << 30;  // 初始化为一个大数
    int best_c = 1;
    int best_k = 0;

    // 计算最大可能的 k 值
    // 对于 y <= 15000，4^7 = 16384 > 15000，所以 k 最大为 7
    int max_k = 0;
    int pow4 = 1;
    while (pow4 <= y * 3) {  // 乘以3是因为 c 最大为 3
        max_k++;
        if (max_k > 10) break;  // 安全限制
        pow4 *= 4;
    }
    // 遍历所有可能的 k 值
    for (int k = 0; k <= max_k; k++) {
        // 计算 4^k
        int base = 1;
        for (int i = 0; i < k; i++) {
            base *= 4;
        }

        // 检查 c = 1, 2, 3
        for (int c = 1; c <= 3; c++) {
            int val = c * base;

            if (val >= y && val < best_val) {
                best_val = val;
                best_c = c;
                best_k = k;
            }

            // 如果 val 已经大于 y，且当前的 c 更大，那么值只会更大，可以跳过
            if (val > y && c == 3) {
                break;
            }
        }

        // 如果最小的 c*4^k (c=1) 已经大于 best_val，说明后面不会有更优解
        if (base > best_val && best_val < (1 << 30)) {
            break;
        }
    }
    *c_ptr = best_c;
    *k_ptr = best_k;
    return best_val;
}
// 支持fp32和int32
template <typename T>
__aicore__ inline void my_CreateVecIndex(const LocalTensor<T>& src, T initVal, uint32_t count) {
    // 边界检查
    if (count == 0) return;

    // 第一阶段：直接设置前8个元素 [0,1,2,3,4,5,6,7]
    int32_t initCount = min((int32_t)count, 8);
    for (int32_t i = 0; i < initCount; i++) {
        src.SetValue(i, (T)(i + initVal));
    }

        // 如果元素数量不超过8，直接返回
    if (count <= 8) {
        return;
    }

    int32_t currentIndex = 8;

    // 第二阶段：使用向量操作批量生成 [8,15], [16,23], ..., [56,63]
    // 每次处理8个元素，最多处理7次（8到63，共56个元素）
    const int32_t vectorSize = 8;
    const int32_t maxVectorOps = 7; // 限制向量操作次数，避免超出范围

    for (int32_t batch = 0; batch < maxVectorOps && currentIndex < (int32_t)count; batch++) {
        int32_t elementsToProcess = min(vectorSize, (int32_t)count - currentIndex);
        Adds(src[currentIndex], src, (T)currentIndex, elementsToProcess);
        currentIndex += vectorSize;
    }

    // 第三阶段：处理剩余元素，使用更大的批次（64个元素）
    while (currentIndex < (int32_t)count) {
        int32_t elementsToProcess = min(64, (int32_t)count - currentIndex);
        Adds(src[currentIndex], src, (T)currentIndex, elementsToProcess);
        currentIndex += 64;
    }
}
// 返回值：true  -> 结果落在 tmp1
//        false -> 结果落在 tmp2
template <typename T>
__aicore__ inline bool MergeSortPipeline(const LocalTensor<T>& tmp1, const LocalTensor<T>& tmp2, int32_t best_y, int32_t best_k, int32_t best_c, int32_t firstIndex = 0) {
    // 1. 初始 32 路排序
    uint32_t total_elements = best_y * 32;
    LocalTensor<int32_t> tmp_index = tmp1[total_elements].template ReinterpretCast<int32_t>();// 前total_elements是元素，后total_elements是索引
    my_CreateVecIndex(tmp_index, firstIndex, total_elements);

    Sort32<T>(tmp2, tmp1, tmp_index.ReinterpretCast<uint32_t>(), best_y);

    MrgSort4Info params;
    uint32_t elementLength = 32;
    uint16_t curr_repeatTimes = best_y / 4;
    bool dstIs1 = false;          // false: 当前结果落在 tmp2

    // 2. k 轮 4 路归并
    for (uint32_t i = 0; i < best_k; ++i) {
        // 填 elementLength
        for (int j = 0; j < 4; ++j) params.elementLengths[j] = elementLength;

        MrgSortSrcList<T> srcList;
        const LocalTensor<T> tmpSrc = dstIs1 ? tmp1 : tmp2;
        const LocalTensor<T> tmpDst = dstIs1 ? tmp2 : tmp1;
        dstIs1 = !dstIs1;

        srcList.src1 = tmpSrc[0];
        srcList.src2 = tmpSrc[elementLength * 8 / sizeof(T) * 1];
        srcList.src3 = tmpSrc[elementLength * 8 / sizeof(T) * 2];
        srcList.src4 = tmpSrc[elementLength * 8 / sizeof(T) * 3];

        params.ifExhaustedSuspension = false;
        params.validBit = 15;
        params.repeatTimes = curr_repeatTimes;
        MrgSort(tmpDst, srcList, params);

        elementLength *= 4;
        curr_repeatTimes /= 4;
    }

    // 3. 尾部处理
    if (best_c == 2) {
        for (int j = 0; j < 4; ++j) params.elementLengths[j] = elementLength;

        MrgSortSrcList<T> srcList;
        const LocalTensor<T> tmpSrc = dstIs1 ? tmp1 : tmp2;
        const LocalTensor<T> tmpDst = dstIs1 ? tmp2 : tmp1;
        dstIs1 = !dstIs1;

        srcList.src1 = tmpSrc[0];
        srcList.src2 = tmpSrc[elementLength * 8 / sizeof(T) * 1];
        // src3/src4 不用

        params.ifExhaustedSuspension = false;
        params.validBit = 3;
        params.repeatTimes = 1;
        MrgSort(tmpDst, srcList, params);
    } else if (best_c == 3) {
        for (int j = 0; j < 4; ++j) params.elementLengths[j] = elementLength;

        MrgSortSrcList<T> srcList;
        LocalTensor<T> tmpSrc = dstIs1 ? tmp1 : tmp2;
        LocalTensor<T> tmpDst = dstIs1 ? tmp2 : tmp1;
        dstIs1 = !dstIs1;

        srcList.src1 = tmpSrc[0];
        srcList.src2 = tmpSrc[elementLength * 8 / sizeof(T) * 1];
        srcList.src3 = tmpSrc[elementLength * 8 / sizeof(T) * 2];
        // src4 不用

        params.ifExhaustedSuspension = false;
        params.validBit = 7;
        params.repeatTimes = 1;
        MrgSort(tmpDst, srcList, params);
    }

    return dstIs1;   // true: 结果在 tmp1；false: 结果在 tmp2
}

/**
 * 对“主段+尾部”做归并排序，并保证最终有序结果落在与主段相同的 buffer 中
 * @param tmp1          主段 buffer1（已含主段有序结果）
 * @param tmp2          主段 buffer2
 * @param best_y        主段 32 元组数
 * @param best_yy       尾部 32 元组数
 * @param best_kk       尾部归并参数
 * @param best_cc       尾部归并参数
 * @param dstIs1        主段当前结果是否在 tmp1（输入输出参数，函数返回时更新）
 * @return              void，dstIs1 引用更新，结果始终与主段同 buffer
 */
template <typename T>
__aicore__ inline bool MergeTailAndMain(const LocalTensor<T>& tmp1, const LocalTensor<T>& tmp2, uint32_t best_y, uint32_t best_yy, uint32_t best_kk, uint32_t best_cc, bool dstIs1) {
    // 1. 尾部排序
    auto tmp1_tail = tmp1[6144 * 8 / sizeof(T)];
    auto tmp2_tail = tmp2[6144 * 8 / sizeof(T)];
    bool dstIs1_tail = MergeSortPipeline<T>(tmp1_tail, tmp2_tail, best_yy, best_kk, best_cc, 6144);

    // 2. 若 tail 结果不在主段同一 buffer，则搬过去
    if (dstIs1 != dstIs1_tail) {
        auto tmp_src = dstIs1_tail ? tmp1_tail : tmp2_tail;
        auto tmp_dst = dstIs1 ? tmp1_tail : tmp2_tail;
        Adds(tmp_dst, tmp_src, (T)0, best_yy * 32);
        dstIs1_tail = dstIs1;
    }

    // 3. 前后两段 2 路归并
    MrgSort4Info params;
    params.elementLengths[0] = best_y * 32 / 2;
    params.elementLengths[1] = best_y * 32 / 2;
    params.elementLengths[2] = best_yy * 32 / 2;
    params.elementLengths[3] = best_yy * 32 / 2;

    MrgSortSrcList<T> srcList;
    const LocalTensor<T> tmpSrc = dstIs1 ? tmp1 : tmp2;
    const LocalTensor<T> tmpDst = dstIs1 ? tmp2 : tmp1;

    srcList.src1 = tmpSrc[0];
    srcList.src2 = tmpSrc[(best_y * 32 / 2) * 8 / sizeof(T)];
    srcList.src3 = tmpSrc[(best_y * 32) * 8 / sizeof(T)];  // 紧接主段之后
    srcList.src4 = tmpSrc[(best_y * 32 + best_yy * 32 / 2) * 8 / sizeof(T)];

    params.ifExhaustedSuspension = false;
    params.validBit = 15;
    params.repeatTimes = 1;
    MrgSort(tmpDst, srcList, params);
    return !dstIs1;   // 更新结果位置
}

template <typename T>
class KernelSort {
public:
    __aicore__ inline KernelSort() {}
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR outIndex, GM_ADDR workspace, uint32_t big_repeat, uint32_t single_sort_size, bool descending, uint32_t interval, TPipe* pipeIn) {
        this->interval = interval;
        this->big_repeat = big_repeat;
        this->single_sort_size = single_sort_size;
        this->single_sort_size_32_aligned = (this->single_sort_size + 31) / 32 * 32;
        // 1~10000元素个数界限：32, 64, 96, 128, 256, 384, 512, 1024, 1536, 2048, 4096, 6144, 8192, 16384

        if (this->single_sort_size_32_aligned <= 6144) {
            this->best_y = find_nearest_4_power(this->single_sort_size_32_aligned / 32, &this->best_c, &this->best_k);
            this->best_yy = 0;
            this->tail_elements = 0;
            this->total_elements = this->best_y * 32;
        } else {// 多了就分成两份处理
            this->best_y = find_nearest_4_power(6144 / 32, &this->best_c, &this->best_k);
            this->best_yy = find_nearest_4_power((4096) / 32, &this->best_cc, &this->best_kk);// MrgSort好像有bug，测试6144+2048会报错
            this->tail_elements = this->best_yy * 32;
            this->total_elements = (this->best_y + this->best_yy) * 32;
        }
        this->descending = descending;

        inputGm.SetGlobalBuffer((__gm__ T*)input);
        outIndexGm.SetGlobalBuffer((__gm__ int64_t*)outIndex);
        outIndex_32Gm.SetGlobalBuffer((__gm__ int32_t*)outIndex);

        tmp_inputGm.SetGlobalBuffer((__gm__ T*)workspace);
        tmp_outIndexGm.SetGlobalBuffer((__gm__ int64_t*)(workspace + this->single_sort_size * 32));
        tmp_outIndex_32Gm.SetGlobalBuffer((__gm__ int32_t*)(workspace + this->single_sort_size * 32));

        pipeIn->InitBuffer(inQueueInput, BUFFER_NUM, 10 * 1024 * 8 + 1024);
        pipeIn->InitBuffer(outQueueIndex, BUFFER_NUM, 10 * 1024 * 8 + 1024);
    }
    __aicore__ inline void Process() {
        {
            for (int32_t i = 0; i < this->big_repeat; i++) {
                for (int32_t offset = 0; offset < this->interval; offset++) {
                    if (interval == 1) {
                        CopyIn(i, offset);
                    } else {
                        if (offset * sizeof(T) % 32 == 0) {
                            CopyInTmpGM(i, offset);
                        }
                        CopyInGroup(i, offset);
                    }

                    Compute();

                    if (interval == 1) {
                        CopyOut(i, offset);
                    } else {
                        CopyOutGroup(i, offset);
                    }
                }
            }
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress, int32_t offset = 0) {
        LocalTensor<T> inputLocal = inQueueInput.AllocTensor<T>();
        if (interval == 1) {
            DataCopy(inputLocal, inputGm[progress * this->single_sort_size], this->single_sort_size_32_aligned);
        } else {
            for (uint32_t i = 0; i < this->single_sort_size; i++) {
                inputLocal.SetValue(i, inputGm.GetValue(progress * this->single_sort_size * interval + i * interval + offset));
            }
        }
        inQueueInput.EnQue(inputLocal);
    }
    __aicore__ inline void CopyInTmpGM(int32_t progress, int32_t offset = 0) {// 单次最多2K元素32Bytes对齐搬运
        if constexpr (std::is_same_v<T, int64_t>) {
        } else {
            uint16_t n_index_per_copy = min(1024u, this->single_sort_size);// 32KB
            LocalTensor<T> srcLocal = outQueueIndex.AllocTensor<T>();
            bool isFirstMTE2 = true;
            uint32_t blockLen = 32;
            uint32_t srcStride = interval * sizeof(T) - 32;
            if (interval * sizeof(T) <= 32) {
                srcStride = 0;
                blockLen = interval * sizeof(T);
            }
            for (uint32_t currentIndex = 0; currentIndex < this->single_sort_size; currentIndex += n_index_per_copy) {
                uint16_t copy_size = (currentIndex + n_index_per_copy <= this->single_sort_size) ? n_index_per_copy : (this->single_sort_size - currentIndex);
                if (isFirstMTE2) {
                    isFirstMTE2 = false;
                } else {
                    TQueSync<PIPE_MTE3, PIPE_MTE2> sync;
                    sync.SetFlag(0);
                    sync.WaitFlag(0);
                }
                DataCopyExtParams copyInParams{copy_size, blockLen, srcStride, 0, 0};
                DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
                DataCopyPad(srcLocal, inputGm[progress * this->single_sort_size * interval + currentIndex * interval + offset], copyInParams, padParams);
                TQueSync<PIPE_MTE2, PIPE_MTE3> sync1;
                sync1.SetFlag(1);
                sync1.WaitFlag(1);
                DataCopyExtParams copyOutParams{copy_size, 32, 0, 0, 0};
                DataCopyPad(tmp_inputGm[currentIndex * 32 / sizeof(T)], srcLocal, copyOutParams);
            }
            outQueueIndex.FreeTensor(srcLocal);
        }
    }

    __aicore__ inline void CopyInGroup(int32_t progress, int32_t offset = 0) {// 单次最多2K个32Bytes对齐搬运，但是当前设置为1K
        LocalTensor<T> inputLocal = inQueueInput.AllocTensor<T>();
        if constexpr (std::is_same_v<T, int64_t>) {
        } else {
            uint16_t n_index_per_copy = min(1024u, this->single_sort_size);
            LocalTensor<int32_t> outBuf = outQueueIndex.AllocTensor<int32_t>();
            LocalTensor<int32_t> indexLocal = outBuf;// 4KB
            LocalTensor<T> srcLocal = outBuf[1024].template ReinterpretCast<T>();// 32KB
            int32_t offset_in_32Bytes = (offset * sizeof(T)) % 32;
            my_CreateVecIndex(indexLocal, 0, n_index_per_copy);
            ShiftLeft(indexLocal, indexLocal, 5, n_index_per_copy);// 相当于乘以32,32Bytes
            Adds(indexLocal, indexLocal, offset_in_32Bytes, n_index_per_copy);// 偏移量加入到index中

            for (uint32_t currentIndex = 0; currentIndex < this->single_sort_size; currentIndex += n_index_per_copy) {
                uint16_t copy_size = (currentIndex + n_index_per_copy <= this->single_sort_size) ? n_index_per_copy : (this->single_sort_size - currentIndex);
                TQueSync<PIPE_V, PIPE_MTE2> sync;
                sync.SetFlag(0);
                sync.WaitFlag(0);
                // srcStride:相邻连续数据块的间隔（前面一个数据块的尾与后面数据块的头的间隔）。
                DataCopyExtParams copyParams{copy_size, 32, 0, 0, 0};
                DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
                DataCopyPad(srcLocal, tmp_inputGm[currentIndex * 32 / sizeof(T)], copyParams, padParams);
                TQueSync<PIPE_MTE2, PIPE_V> sync1;
                sync1.SetFlag(1);
                sync1.WaitFlag(1);
                if constexpr (std::is_same_v<T, bfloat16_t>) {
                    Gather(inputLocal[currentIndex].template ReinterpretCast<half>(), srcLocal.template ReinterpretCast<half>(), indexLocal.template ReinterpretCast<uint32_t>(), 0, copy_size);
                } else {
                    Gather(inputLocal[currentIndex], srcLocal, indexLocal.template ReinterpretCast<uint32_t>(), 0, copy_size);
                }
            }
            outQueueIndex.FreeTensor(outBuf);
        }
        inQueueInput.EnQue(inputLocal);
    }
    __aicore__ inline void Compute() {
        LocalTensor<T> inputLocal = inQueueInput.DeQue<T>();
        LocalTensor<int64_t> outIndexLocal = outQueueIndex.AllocTensor<int64_t>();// 需要this->best_y * 32元素 * 8byte * 2
        bool dstIs1, dstIs1_tail;
        LocalTensor<float> tmp1_fp32 = outIndexLocal.template ReinterpretCast<float>();
        LocalTensor<float> tmp2_fp32 = inputLocal.template ReinterpretCast<float>();// 利用inputLocal作为第二个buffer，节省空间
        if constexpr (std::is_same_v<T, half> || std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>) {
            auto tmp1 = tmp1_fp32.template ReinterpretCast<half>();
            auto tmp2 = tmp2_fp32.template ReinterpretCast<half>();// 一个元素占8字节
            if (this->tail_elements == 0) {
                Duplicate(tmp1, (half)(-65504), total_elements);
            } else {// 需要处理尾部元素
                Duplicate(tmp1, (half)(-65504), 6144);
                Duplicate(tmp1[6144 * 8 / sizeof(half)], (half)(-65504), total_elements - 6144);
            }
            if constexpr (std::is_same_v<T, half>) {
                if (this->tail_elements == 0) {
                    Adds(tmp1, inputLocal, (half)0, this->single_sort_size);
                } else {// 需要处理尾部元素
                    Adds(tmp1, inputLocal, (half)0, 6144);
                    Adds(tmp1[6144 * 8 / sizeof(half)], inputLocal[6144], (half)0, this->single_sort_size - 6144);
                }
            } else {
                if (this->tail_elements == 0) {
                    Cast(tmp1, inputLocal, RoundMode::CAST_NONE, this->single_sort_size);
                } else {// 需要处理尾部元素
                    Cast(tmp1, inputLocal, RoundMode::CAST_NONE, 6144);
                    Cast(tmp1[6144 * 8 / sizeof(half)], inputLocal[6144], RoundMode::CAST_NONE, this->single_sort_size - 6144);
                }
            }
            if (descending == false) {
                if (this->tail_elements == 0) {
                    Muls(tmp1, tmp1, (half)(-1), this->single_sort_size);// 全部元素取反
                } else {// 需要处理尾部元素
                    Muls(tmp1, tmp1, (half)(-1), 6144);// 全部元素取反
                    Muls(tmp1[6144 * 8 / sizeof(half)], tmp1[6144 * 8 / sizeof(half)], (half)(-1), this->single_sort_size - 6144);// 全部元素取反
                }
            }
            dstIs1 = MergeSortPipeline<half>(tmp1, tmp2, best_y, best_k, best_c);
            if (this->tail_elements > 0) {// 需要处理尾部元素
                dstIs1 = MergeTailAndMain(tmp1, tmp2, best_y, best_yy, best_kk, best_cc, dstIs1);
            }
        } else {
            auto tmp1 = tmp1_fp32;
            auto tmp2 = tmp2_fp32;
            // 这里的锅，后面的部分没有设置到
            if (this->tail_elements == 0) {
                Duplicate(tmp1, (float)(-3.4028235e38), total_elements);
            } else {// 需要处理尾部元素
                Duplicate(tmp1, (float)(-3.4028235e38), 6144);
                Duplicate(tmp1[6144 * 8 / sizeof(float)], (float)(-3.4028235e38), total_elements - 6144);
            }
            if constexpr (std::is_same_v<T, float>) {
                if (this->tail_elements == 0) {
                    Adds(tmp1, inputLocal, (float)0, this->single_sort_size);
                } else {// 需要处理尾部元素
                    Adds(tmp1, inputLocal, (float)0, 6144);
                    Adds(tmp1[6144 * 8 / sizeof(float)], inputLocal[6144], (float)0, this->single_sort_size - 6144);
                }
            } else if constexpr (std::is_same_v<T, int64_t>) {
                if (this->tail_elements == 0) {
                    Cast(tmp1, inputLocal, RoundMode::CAST_RINT, this->single_sort_size);
                } else {// 需要处理尾部元素
                    Cast(tmp1, inputLocal, RoundMode::CAST_RINT, 6144);
                    Cast(tmp1[6144 * 8 / sizeof(float)], inputLocal[6144], RoundMode::CAST_RINT, this->single_sort_size - 6144);
                }
            } else {// bf16 int32 int16
                if (this->tail_elements == 0) {
                    Cast(tmp1, inputLocal, RoundMode::CAST_NONE, this->single_sort_size);
                } else {// 需要处理尾部元素
                    Cast(tmp1, inputLocal, RoundMode::CAST_NONE, 6144);
                    Cast(tmp1[6144 * 8 / sizeof(float)], inputLocal[6144], RoundMode::CAST_NONE, this->single_sort_size - 6144);
                }
            }
            if (descending == false) {
                if (this->tail_elements == 0) {
                    Muls(tmp1, tmp1, (float)(-1), this->single_sort_size);// 全部元素取反
                } else {// 需要处理尾部元素
                    Muls(tmp1, tmp1, (float)(-1), 6144);// 全部元素取反
                    Muls(tmp1[6144 * 8 / sizeof(float)], tmp1[6144 * 8 / sizeof(float)], (float)(-1), this->single_sort_size - 6144);// 全部元素取反
                }
            }
            dstIs1 = MergeSortPipeline<float>(tmp1, tmp2, best_y, best_k, best_c);
            if (this->tail_elements > 0) {// 需要处理尾部元素
                dstIs1 = MergeTailAndMain(tmp1, tmp2, best_y, best_yy, best_kk, best_cc, dstIs1);
            }
        }
        LocalTensor<int32_t> tmpOut = (dstIs1 ? tmp1_fp32 : tmp2_fp32).template ReinterpretCast<int32_t>();
        LocalTensor<int32_t> tmpOut_32 = (dstIs1 ? tmp2_fp32 : tmp1_fp32).template ReinterpretCast<int32_t>();
        LocalTensor<int64_t> tmpOut_64 = tmpOut.ReinterpretCast<int64_t>();
        uint64_t rsvdCnt = 0;
        GatherMask(tmpOut_32, tmpOut, 2, false, 0, {1, static_cast<uint16_t>(best_y + best_yy), 8, 0}, rsvdCnt);// (best_y+best_yy) * 32元素 * 8bit / 256bit
        if (this->interval > 1) {// 需要间隔搬运,保存低32位
            Adds(outIndexLocal.template ReinterpretCast<int32_t>(), tmpOut_32, 0, this->single_sort_size_32_aligned);
        } else {
            Cast(tmpOut_64, tmpOut_32, RoundMode::CAST_NONE, this->single_sort_size_32_aligned);
            Adds(outIndexLocal.template ReinterpretCast<int32_t>(), tmpOut_64.template ReinterpretCast<int32_t>(), 0, this->single_sort_size_32_aligned * 2);
        }

        inQueueInput.FreeTensor(inputLocal);
        outQueueIndex.EnQue(outIndexLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress, int32_t offset = 0) {
        LocalTensor<int64_t> outIndexLocal = outQueueIndex.DeQue<int64_t>();
        if (interval == 1) {
            DataCopy(outIndexGm[progress * this->single_sort_size], outIndexLocal, this->single_sort_size_32_aligned);
        } else {
            for (uint32_t i = 0; i < this->single_sort_size; i++) {
                outIndexGm.SetValue(progress * this->single_sort_size * interval + i * interval + offset, outIndexLocal.GetValue(i));
            }
        }
        outQueueIndex.FreeTensor(outIndexLocal);
    }

    __aicore__ inline void CopyOutGroup(int32_t progress, int32_t offset = 0) {// 单次最多搬运2048元素
        LocalTensor<int32_t> outIndexLocal = outQueueIndex.DeQue<int32_t>();// DataCopyPad不支持int64，所以用int32_t接收
        if constexpr (std::is_same_v<T, int64_t>) {
        } else {
            uint16_t n_index_per_copy = min(32u, this->single_sort_size);
            // 前this->total_elements是数据
            LocalTensor<int32_t> outBuf = outIndexLocal[this->total_elements].template ReinterpretCast<int32_t>();// 40KB
            LocalTensor<int32_t> indexLocal = outBuf;// 32*4B
            // DataCopyPad不支持int64，所以用int32_t接收
            LocalTensor<int32_t> dstLocal = outBuf[32u].template ReinterpretCast<int32_t>();

            Duplicate(dstLocal, 0, 32 / sizeof(int32_t) * n_index_per_copy);// 初始化为0
            my_CreateVecIndex(indexLocal, 0, n_index_per_copy);
            ShiftLeft(indexLocal, indexLocal, 5, n_index_per_copy);// 相当于乘以32,32Bytes

            bool isFirstScatter = true;

            for (uint32_t currentIndex = 0; currentIndex < this->single_sort_size; currentIndex += n_index_per_copy) {
                uint16_t copy_size = (currentIndex + n_index_per_copy <= this->single_sort_size) ? n_index_per_copy : (this->single_sort_size - currentIndex);
                if (!isFirstScatter) {
                    TQueSync<PIPE_MTE3, PIPE_V> sync;
                    sync.SetFlag(2);
                    sync.WaitFlag(2);
                } else {
                    isFirstScatter = false;
                }
                Scatter(dstLocal, outIndexLocal[currentIndex], indexLocal.template ReinterpretCast<uint32_t>(), 0, copy_size);

                TQueSync<PIPE_V, PIPE_MTE3> sync1;
                sync1.SetFlag(3);
                sync1.WaitFlag(3);
                // srcStride:相邻连续数据块的间隔（前面一个数据块的尾与后面数据块的头的间隔）。
                DataCopyExtParams copyParams{copy_size, sizeof(int64_t), 0, static_cast<uint32_t>((interval - 1) * sizeof(int64_t)), 0};
                DataCopyPad(outIndex_32Gm[(progress * this->single_sort_size * interval + currentIndex * interval + offset) * 2], dstLocal, copyParams);
            }
        }
        outQueueIndex.FreeTensor(outIndexLocal);
    }
    __aicore__ inline void CopyOutTmpGM(int32_t progress, int32_t offset = 0) {// 单次最多搬运4096-32 = 4064元素
        if constexpr (std::is_same_v<T, int64_t>) {
        } else {
            uint16_t n_index_per_copy = min(1024u, this->single_sort_size);
            LocalTensor<int32_t> srcLocal = outQueueIndex.AllocTensor<int32_t>();
            bool isFirstMTE2 = true;
            for (uint32_t currentIndex = 0; currentIndex < this->single_sort_size; currentIndex += n_index_per_copy) {
                uint16_t copy_size = (currentIndex + n_index_per_copy <= this->single_sort_size) ? n_index_per_copy : (this->single_sort_size - currentIndex);
                if (isFirstMTE2) {
                    isFirstMTE2 = false;
                } else {
                    TQueSync<PIPE_MTE3, PIPE_MTE2> sync;
                    sync.SetFlag(4);
                    sync.WaitFlag(4);
                }
                    // srcStride:相邻连续数据块的间隔（前面一个数据块的尾与后面数据块的头的间隔）。
                DataCopyExtParams copyInParams{copy_size, 32, 0, 0, 0};
                DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
                DataCopyPad(srcLocal, tmp_outIndex_32Gm[currentIndex * 32 / sizeof(int32_t)], copyInParams, padParams);
                TQueSync<PIPE_MTE2, PIPE_MTE3> sync1;
                sync1.SetFlag(5);
                sync1.WaitFlag(5);
                    // srcStride:相邻连续数据块的间隔（前面一个数据块的尾与后面数据块的头的间隔）。
                DataCopyExtParams copyOutParams{copy_size, 32, 0, static_cast<uint32_t>(interval * sizeof(int64_t) - 32), 0};
                DataCopyPad(outIndex_32Gm[(progress * this->single_sort_size * interval + currentIndex * interval + offset) * 2], srcLocal, copyOutParams);
            }
            outQueueIndex.FreeTensor(srcLocal);
        }
    }

private:
    TQue<QuePosition::VECIN, 1> inQueueInput;
    TQue<QuePosition::VECOUT, 1> outQueueIndex;
    GlobalTensor<T> inputGm;
    GlobalTensor<T> tmp_inputGm;
    GlobalTensor<int64_t> outIndexGm;
    GlobalTensor<int32_t> outIndex_32Gm;
    GlobalTensor<int64_t> tmp_outIndexGm;
    GlobalTensor<int32_t> tmp_outIndex_32Gm;

    uint32_t big_repeat;
    uint32_t single_sort_size;
    uint32_t single_sort_size_32_aligned;
    bool descending;
    // 32元素组数 = best_y = best_c*4^best_k
    int best_c;
    int best_k;
    int best_y;
    int best_cc;
    int best_kk;
    int best_yy;
    uint32_t total_elements;
    uint32_t tail_elements;
    uint32_t interval;
};
extern "C" __global__ __aicore__ void argsort(GM_ADDR x, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
    TPipe pipe;
    uint32_t total_elements = tiling_data.big_repeat * tiling_data.single_sort_size * tiling_data.interval;
    if constexpr (std::is_same_v<DTYPE_X, bool>) {
        KernelSort<uint8_t> op;
        op.Init(x, out, workspace, tiling_data.big_repeat, tiling_data.single_sort_size, tiling_data.descending, tiling_data.interval, &pipe);
        op.Process();
    } else {
        KernelSort<DTYPE_X> op;
        op.Init(x, out, workspace, tiling_data.big_repeat, tiling_data.single_sort_size, tiling_data.descending, tiling_data.interval, &pipe);
        op.Process();
    }
}