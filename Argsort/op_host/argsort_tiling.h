
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ArgsortTilingData)
TILING_DATA_FIELD_DEF(uint32_t, big_repeat);
TILING_DATA_FIELD_DEF(uint32_t, single_sort_size);
TILING_DATA_FIELD_DEF(uint32_t, interval);
TILING_DATA_FIELD_DEF(bool, descending);

END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Argsort, ArgsortTilingData)
} // namespace optiling
