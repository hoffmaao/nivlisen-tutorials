// Minimal Fill-Spill-Merge CLI for the Nivlisen tutorials (meltwater routing).
//
// Routes water over a DEM with Fill-Spill-Merge (Barnes, Callaghan & Wickert,
// 2020, "Computing water flow through complex landscapes, Part 3: Fill-Spill-
// Merge", Earth Surface Dynamics 8, 431-445), using the header-only RichDEM /
// dephier / FSM libraries from
//   https://github.com/r-barnes/Barnes2020-FillSpillMerge  (MIT).
//
// I/O is raw little-endian float64 so it drops straight into numpy: it reads a
// DEM and a per-cell water depth as (ny, nx) C-order arrays, runs FSM, and
// writes the routed standing-water depth back as (ny, nx) float64. Cells whose
// elevation is NaN or < -9990 are treated as OCEAN (open drains); the domain
// must contain at least one such cell for the depression hierarchy to exist.
//
// Usage: fsm_wrapper <ny> <nx> <dem.bin> <water_in.bin> <water_out.bin>
//
// The Docker image compiles this against the upstream headers (see the
// Fill-Spill-Merge build stage in the Dockerfile); no runtime libraries are
// linked because all I/O is raw binary:
//   g++ -O2 -std=c++17 -I<fsm>/include \
//       -I<fsm>/submodules/dephier/include \
//       -I<fsm>/submodules/dephier/submodules/richdem/include \
//       -o fsm_wrapper fsm_wrapper.cpp

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <richdem/richdem.hpp>
#include <fsm/fill_spill_merge.hpp>

int main(int argc, char *argv[]) {
    if (argc != 6) {
        fprintf(stderr, "Usage: %s <ny> <nx> <dem.bin> <water_in.bin> <water_out.bin>\n", argv[0]);
        return 1;
    }

    int ny = atoi(argv[1]);
    int nx = atoi(argv[2]);
    int ncells = ny * nx;

    // Read DEM
    std::vector<double> dem_data(ncells);
    FILE *f = fopen(argv[3], "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", argv[3]); return 1; }
    if (fread(dem_data.data(), sizeof(double), ncells, f) != (size_t)ncells) {
        fprintf(stderr, "Short read on %s\n", argv[3]); return 1;
    }
    fclose(f);

    // Read water input
    std::vector<double> water_in(ncells);
    f = fopen(argv[4], "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", argv[4]); return 1; }
    if (fread(water_in.data(), sizeof(double), ncells, f) != (size_t)ncells) {
        fprintf(stderr, "Short read on %s\n", argv[4]); return 1;
    }
    fclose(f);

    // Build RichDEM arrays
    richdem::Array2D<double> topo(nx, ny);
    richdem::Array2D<double> wtd(nx, ny);  // water table depth = water input

    for (int i = 0; i < ny; i++) {
        for (int j = 0; j < nx; j++) {
            double elev = dem_data[i * nx + j];
            double win  = water_in[i * nx + j];
            if (elev != elev || elev < -9990) {  // NaN or nodata
                topo(j, i) = topo.noData();
                wtd(j, i)  = wtd.noData();
            } else {
                topo(j, i) = elev;
                wtd(j, i)  = win;  // water input goes in wtd
            }
        }
    }

    // Prepare label and flowdir arrays for FSM
    richdem::Array2D<richdem::dephier::dh_label_t> label(nx, ny, richdem::dephier::NO_DEP);
    richdem::Array2D<int8_t> flowdirs(nx, ny, richdem::NO_FLOW);

    // Mark ocean cells
    for (int i = 0; i < ny; i++)
        for (int j = 0; j < nx; j++)
            if (topo(j, i) == topo.noData())
                label(j, i) = richdem::dephier::OCEAN;

    // Build depression hierarchy
    auto deps = richdem::dephier::GetDepressionHierarchy<double, richdem::Topology::D8>(topo, label, flowdirs);
    // Run Fill-Spill-Merge to redistribute water
    richdem::dephier::FillSpillMerge(topo, label, flowdirs, deps, wtd);

    // Write water depth output
    std::vector<double> water_out(ncells, 0.0);
    for (int i = 0; i < ny; i++) {
        for (int j = 0; j < nx; j++) {
            if (topo(j, i) != topo.noData()) {
                water_out[i * nx + j] = wtd(j, i);
            }
        }
    }

    f = fopen(argv[5], "wb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", argv[5]); return 1; }
    fwrite(water_out.data(), sizeof(double), ncells, f);
    fclose(f);

    return 0;
}
