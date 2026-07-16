// Batch Fill-Spill-Merge server for the Nivlisen tutorials (meltwater routing).
//
// Same Fill-Spill-Merge routing as fsm_wrapper.cpp (Barnes, Callaghan & Wickert,
// 2020, "Computing water flow through complex landscapes, Part 3: Fill-Spill-
// Merge", Earth Surface Dynamics 8, 431-445), but built for routing *many*
// water inputs over *one* fixed DEM. Building the depression hierarchy
// (GetDepressionHierarchy) is the expensive step and depends only on the
// terrain, so we do it once; FillSpillMerge then resets and reuses that
// hierarchy for each new water field (it calls ResetDH internally, and takes
// topo/label/flowdirs as const), which is cheap. This turns the per-node melt
// routing (~one route per mesh node) from N hierarchy builds into one.
//
// Protocol (raw little-endian float64, so it drops straight into numpy):
//   argv: fsm_batch <ny> <nx> <dem.bin>
//   - reads the DEM once from <dem.bin> as an (ny, nx) C-order array; cells
//     whose elevation is NaN or < -9990 are OCEAN (open drains).
//   - then loops: read one (ny, nx) water-input frame from stdin, route it, and
//     write the (ny, nx) routed standing-water depth to stdout (flushed). A
//     clean EOF on stdin (no bytes at a frame boundary) ends the process.
//
// Build: identical include paths to fsm_wrapper (see the Dockerfile FSM stage);
// header-only, nothing to link.

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <richdem/richdem.hpp>
#include <fsm/fill_spill_merge.hpp>

// Read exactly n doubles from stdin. Returns 1 on success, 0 on a clean EOF at
// a frame boundary (nothing read), -1 on a short/interrupted frame (error).
static int read_frame(double *buf, size_t n) {
    size_t got = 0;
    while (got < n) {
        size_t r = fread(buf + got, sizeof(double), n - got, stdin);
        if (r == 0) return (got == 0) ? 0 : -1;
        got += r;
    }
    return 1;
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <ny> <nx> <dem.bin>\n", argv[0]);
        return 1;
    }
    int ny = atoi(argv[1]);
    int nx = atoi(argv[2]);
    size_t ncells = (size_t)ny * nx;

    // --- Read the DEM once ---------------------------------------------------
    std::vector<double> dem_data(ncells);
    FILE *f = fopen(argv[3], "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", argv[3]); return 1; }
    if (fread(dem_data.data(), sizeof(double), ncells, f) != ncells) {
        fprintf(stderr, "Short read on %s\n", argv[3]); return 1;
    }
    fclose(f);

    // --- Build the depression hierarchy once (the expensive step) ------------
    richdem::Array2D<double> topo(nx, ny);
    for (int i = 0; i < ny; i++)
        for (int j = 0; j < nx; j++) {
            double elev = dem_data[(size_t)i * nx + j];
            topo(j, i) = (elev != elev || elev < -9990) ? topo.noData() : elev;
        }

    richdem::Array2D<richdem::dephier::dh_label_t> label(nx, ny, richdem::dephier::NO_DEP);
    richdem::Array2D<int8_t> flowdirs(nx, ny, richdem::NO_FLOW);
    for (int i = 0; i < ny; i++)
        for (int j = 0; j < nx; j++)
            if (topo(j, i) == topo.noData())
                label(j, i) = richdem::dephier::OCEAN;

    auto deps = richdem::dephier::GetDepressionHierarchy<double, richdem::Topology::D8>(
        topo, label, flowdirs);

    // --- Route each water frame streamed on stdin ----------------------------
    // topo/label/flowdirs are const to FillSpillMerge and `deps` is reset inside
    // it, so the one hierarchy above serves every frame.
    richdem::Array2D<double> wtd(nx, ny);
    std::vector<double> water_in(ncells), water_out(ncells);
    for (;;) {
        int st = read_frame(water_in.data(), ncells);
        if (st == 0) break;                       // clean EOF: done
        if (st < 0) { fprintf(stderr, "fsm_batch: short water frame\n"); return 1; }

        for (int i = 0; i < ny; i++)
            for (int j = 0; j < nx; j++)
                wtd(j, i) = (topo(j, i) == topo.noData())
                                ? wtd.noData()
                                : water_in[(size_t)i * nx + j];

        richdem::dephier::FillSpillMerge(topo, label, flowdirs, deps, wtd);

        for (int i = 0; i < ny; i++)
            for (int j = 0; j < nx; j++)
                water_out[(size_t)i * nx + j] =
                    (topo(j, i) == topo.noData()) ? 0.0 : wtd(j, i);

        if (fwrite(water_out.data(), sizeof(double), ncells, stdout) != ncells) {
            fprintf(stderr, "fsm_batch: short write\n"); return 1;
        }
        fflush(stdout);
    }
    return 0;
}
