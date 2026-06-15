#!/usr/bin/env python3
"""Prepare the small, low-resolution dataset shipped with these tutorials.

This is a *one-time, author-side* script: it clips the full BedMachine
Antarctica and MEaSUREs velocity mosaics (which live in the production
``~/projects/nivlisen`` tree) to the Nivlisen domain, subsamples to a coarse
grid, and writes a single small NetCDF plus the domain outline. The committed
outputs are what the notebooks actually load, so end users never need the
multi-gigabyte source mosaics or NASA Earthdata credentials.

Run (only needed to regenerate the committed data):
    python data/prepare_data.py

Outputs (committed to the repo):
    data/nivlisen_data.nc   x, y, bed, thickness, surface, mask, vx, vy, errx, erry
    data/nivlisen_domain.gpkg   domain / basin / neighbour polygons (EPSG:3031)

Provenance:
    BedMachine Antarctica v4 (NSIDC-0756) — bed, thickness, mask
    MEaSUREs Antarctic Ice Velocity 450 m v2 (NSIDC-0484) — VX, VY, ERRX, ERRY
    MEaSUREs Antarctic Boundaries v2 (NSIDC-0709) — Nivl drainage basin
"""

import os
import glob
import shutil
import numpy as np
import xarray as xr
import geopandas as gpd

# Source production project (only needed when regenerating).
SRC = os.path.expanduser("~/projects/nivlisen")
HERE = os.path.dirname(os.path.abspath(__file__))

SUBSAMPLE = 4          # 500 m BedMachine * 4 -> 2 km tutorial grid
PAD_M = 8.0e3          # padding around the domain bounds

# Physical constants for the (hydrostatic) surface, matching the inversion.
RHO_I, RHO_W = 917.0, 1024.0


def _find(d, pattern):
    m = glob.glob(os.path.join(d, pattern))
    if not m:
        raise FileNotFoundError(f"No {pattern} in {d}")
    return m[0]


def main():
    # Domain outline (from the production meshing step) sets the clip window.
    dom_fn = os.path.join(SRC, "mesh", "nivlisen_domain.gpkg")
    gdf = gpd.read_file(dom_fn)
    domain = gdf[gdf["name"] == "domain"].geometry.values[0]
    minx, miny, maxx, maxy = domain.bounds
    print(f"Domain bounds (km): "
          f"[{minx/1e3:.0f},{maxx/1e3:.0f}] x [{miny/1e3:.0f},{maxy/1e3:.0f}]")

    def clip(ds, keys):
        x, y = ds["x"].values, ds["y"].values
        ix = np.where((x >= minx - PAD_M) & (x <= maxx + PAD_M))[0][::SUBSAMPLE]
        iy = np.where((y >= miny - PAD_M) & (y <= maxy + PAD_M))[0][::SUBSAMPLE]
        out = {k: ds[k].values[np.ix_(iy, ix)].astype("float32") for k in keys}
        return out, x[ix].astype("float64"), y[iy].astype("float64")

    # BedMachine: bed, thickness, mask
    bm = xr.open_dataset(_find(os.path.join(SRC, "data", "bedmachine"), "*.nc"))
    bmv, x, y = clip(bm, ["bed", "thickness", "mask"])
    bm.close()

    # Velocity: VX, VY, ERRX, ERRY  (interpolate onto the BedMachine grid)
    vel = xr.open_dataset(_find(os.path.join(SRC, "data", "velocity"), "*.nc"))
    vel_i = vel.interp(x=xr.DataArray(x, dims="x"), y=xr.DataArray(y, dims="y"),
                       method="nearest")
    velv = {k.lower(): vel_i[k].values.astype("float32")
            for k in ["VX", "VY", "ERRX", "ERRY"]}
    vel.close()

    # Hydrostatic surface (same definition the inversion uses).
    h = np.maximum(bmv["thickness"], 10.0)
    surface = np.maximum(bmv["bed"] + h,
                         (1.0 - RHO_I / RHO_W) * h).astype("float32")

    out = xr.Dataset(
        {
            "bed": (("y", "x"), bmv["bed"]),
            "thickness": (("y", "x"), bmv["thickness"]),
            "surface": (("y", "x"), surface),
            "mask": (("y", "x"), bmv["mask"]),
            "vx": (("y", "x"), velv["vx"]),
            "vy": (("y", "x"), velv["vy"]),
            "errx": (("y", "x"), velv["errx"]),
            "erry": (("y", "x"), velv["erry"]),
        },
        coords={"x": ("x", x), "y": ("y", y)},
        attrs={
            "title": "Nivlisen low-resolution tutorial dataset (EPSG:3031)",
            "crs": "EPSG:3031",
            "grid_spacing_m": 500.0 * SUBSAMPLE,
            "source_bed_thickness": "BedMachine Antarctica v4 (NSIDC-0756)",
            "source_velocity": "MEaSUREs Antarctic Ice Velocity 450 m v2 (NSIDC-0484)",
        },
    )
    out_fn = os.path.join(HERE, "nivlisen_data.nc")
    enc = {v: {"zlib": True, "complevel": 4} for v in out.data_vars}
    out.to_netcdf(out_fn, encoding=enc)
    print(f"wrote {out_fn}  ({os.path.getsize(out_fn)/1e6:.2f} MB, grid {out.bed.shape})")

    # Domain / basin polygons for the region map.
    dst = os.path.join(HERE, "nivlisen_domain.gpkg")
    shutil.copy2(dom_fn, dst)
    print(f"wrote {dst}  ({os.path.getsize(dst)/1e3:.0f} kB)")


if __name__ == "__main__":
    main()
