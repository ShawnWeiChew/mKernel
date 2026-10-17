#pragma once

namespace ag_gemm_warp_specialized {

enum class TileTraversal { DEVICE_M_LOCAL_N, M_GLOBAL_N };

struct TileCoordinate {
    int row;     // Cluster row within a logical device.
    int column;
    int device;  // Absolute logical device, shared by loads and stores.
};

// This is the only traversal-dependent code in the benchmark kernel. Both
// modes enumerate the same rectangular tile set, including any shard padding.
template <int WIDTH, TileTraversal TRAVERSAL>
#ifdef __CUDACC__
__host__ __device__
#endif
inline TileCoordinate gemm_tile_coordinate(
    int rows_per_device, int columns, int tile, int devices, int first_device) {
    static_assert(WIDTH > 0);
    int rows, device;
    if constexpr (TRAVERSAL == TileTraversal::DEVICE_M_LOCAL_N) {
        const int tiles_per_device = rows_per_device * columns;
        device = (tile / tiles_per_device + first_device) % devices;
        tile %= tiles_per_device;
        rows = rows_per_device;
    } else {
        rows = rows_per_device * devices;
        device = 0;
    }
    const int group = tile / (rows * WIDTH);
    const int full_columns = columns / WIDTH * WIDTH;
    int row, column;
    if (tile < rows * full_columns) {
        row = (tile % (rows * WIDTH)) / WIDTH;
        column = group * WIDTH + tile % WIDTH;
    } else {
        const int remainder = tile - rows * full_columns;
        row = remainder / (columns - full_columns);
        column = full_columns + remainder % (columns - full_columns);
    }
    if (group & 1) row = rows - row - 1;
    if constexpr (TRAVERSAL == TileTraversal::M_GLOBAL_N) {
        device = row / rows_per_device;
        row %= rows_per_device;
    }
    return {row, column, device};
}

}  // namespace ag_gemm_warp_specialized
