
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(SoftplusTilingData)
TILING_DATA_FIELD_DEF(uint32_t, smallSize);
TILING_DATA_FIELD_DEF(uint32_t, incSize);
TILING_DATA_FIELD_DEF(uint32_t, totalSize);
TILING_DATA_FIELD_DEF(uint32_t, formerNum);
TILING_DATA_FIELD_DEF(float, beta);
TILING_DATA_FIELD_DEF(float, threshold);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Softplus, SoftplusTilingData)
} // namespace optiling
