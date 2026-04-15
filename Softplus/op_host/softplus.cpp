
#include "softplus_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    SoftplusTilingData tiling;
    const auto runtime_attrs = context->GetAttrs();
    tiling.set_beta(*runtime_attrs->GetFloat(0));
    tiling.set_threshold(*runtime_attrs->GetFloat(1));
    const uint32_t totalSize = context->GetInputTensor(0)->GetShapeSize();
    uint32_t size = totalSize;
    uint32_t aivNum = 1; // Ascend310B
    auto dt = context->GetInputTensor(0)->GetDataType(); //  ge::DT_FLOAT
    int DataTypeSize = 0;
    if (dt == ge::DT_FLOAT || dt == ge::DT_INT32) {
        DataTypeSize = 4;
    } else if (dt == ge::DT_BF16 || dt == ge::DT_FLOAT16 || dt == ge::DT_INT16) {
        DataTypeSize = 2;
    } else if (dt == ge::DT_BOOL || dt == ge::DT_INT8 || dt == ge::DT_UINT8) {
        DataTypeSize = 1;
    } else if (dt == ge::DT_INT64) {
        DataTypeSize = 8;
    }
    // blockSize以byte为单位
    uint32_t blockSize = 1024;
    uint32_t blockNum = (size * DataTypeSize + blockSize - 1) / blockSize;
    aivNum = std::min(aivNum, blockNum);

    uint32_t smallSize = blockNum / aivNum * blockSize / DataTypeSize;
    uint32_t incSize = blockSize / DataTypeSize;
    uint16_t formerNum = blockNum % aivNum;

    tiling.set_totalSize(totalSize);
    tiling.set_smallSize(smallSize);
    tiling.set_incSize(incSize);
    tiling.set_formerNum(formerNum);
    context->SetBlockDim(aivNum);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context) {
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext* context) {
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class Softplus : public OpDef {
public:
    explicit Softplus(const char* name) : OpDef(name) {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("beta").Float();
        this->Attr("threshold").Float();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b");
    }
};

OP_ADD(Softplus);
} // namespace ops
