// Test the actual host/device coordinate function used by both kernel variants.
#include "operators/ag_gemm/gemm_tile_traversal.h"
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace ag_gemm_warp_specialized;

template <int Width, TileTraversal Traversal>
std::vector<int> check(int m, int n, int devices, int rank) {
    const int local_rows = m / devices / 128;
    const int cluster_rows = (local_rows + 3) / 4;
    const int columns = n / 256;
    const int count = cluster_rows * devices * columns;
    std::vector<int> visits(count, 0), output(m / 128 * columns, 0), order(count);
    int padding = 0;
    // Persistent CTA scheduling is identical for both modes.
    for (int cluster = 0; cluster < 74; ++cluster) {
        for (int id = cluster; id < count; id += 74) {
            const auto tile = gemm_tile_coordinate<Width, Traversal>(
                cluster_rows, columns, id, devices, rank);
            if (tile.row < 0 || tile.row >= cluster_rows || tile.column < 0 ||
                tile.column >= columns || tile.device < 0 || tile.device >= devices)
                throw std::runtime_error("out-of-bounds coordinate");
            const int global_row = tile.device * cluster_rows + tile.row;
            const int index = global_row * columns + tile.column;
            if (++visits[index] != 1) throw std::runtime_error("duplicate cluster tile");
            order[id] = index;
            // Check the specified nesting independently of the snake formula.
            if constexpr (Traversal == TileTraversal::DEVICE_M_LOCAL_N) {
                const int expected_device = (id / (cluster_rows * columns) + rank) % devices;
                if (tile.device != expected_device) throw std::runtime_error("wrong device order");
            } else {
                const int group = tile.column / Width;
                const int group_width = std::min(Width, columns - group * Width);
                const int offset = id - group * Width * cluster_rows * devices;
                const int expected_row = (group % 2 == 0)
                    ? offset / group_width : cluster_rows * devices - 1 - offset / group_width;
                if (global_row != expected_row || tile.column != group * Width + offset % group_width)
                    throw std::runtime_error("wrong global snake order");
            }
            for (int consumer = 0; consumer < 2; ++consumer) {
                for (int cta = 0; cta < 2; ++cta) {
                    const int row = tile.row * 4 + consumer * 2 + cta;
                    if (row >= local_rows) { ++padding; continue; }
                    const int out = (tile.device * local_rows + row) * columns + tile.column;
                    if (++output[out] != 1) throw std::runtime_error("duplicate output tile");
                }
            }
        }
    }
    for (int v : visits) if (v != 1) throw std::runtime_error("missing cluster tile");
    for (int v : output) if (v != 1) throw std::runtime_error("missing output tile");
    if (padding != count * 4 - static_cast<int>(output.size()))
        throw std::runtime_error("wrong padding count");
    return order;
}

template <int Width>
void run() {
    for (int devices : {1, 2, 4, 8, 16}) {
        for (int m : {2048, 4096, 8192, 16384, 32768}) {
            // Include rectangular shapes and incomplete supergroups.
            for (int n : {256, 1280, 2048, 4096, 8192, 16384, 32768}) {
                const auto global = check<Width, TileTraversal::M_GLOBAL_N>(m, n, devices, 0);
                for (int rank = 0; rank < devices; ++rank) {
                    const auto local = check<Width, TileTraversal::DEVICE_M_LOCAL_N>(m, n, devices, rank);
                    if (devices == 1 && local != global)
                        throw std::runtime_error("one-device traversals should be identical");
                    if (devices > 1 && n / 256 > Width && local == global)
                        throw std::runtime_error("multi-device traversal order did not change");
                }
            }
        }
    }
}

int main() {
    run<4>();
    run<8>();
    std::cout << "PASS: actual traversal mapper, complete tile coverage, equal padding, "
                 "distinct orders, rank rotations, and partial supergroups\n";
}
