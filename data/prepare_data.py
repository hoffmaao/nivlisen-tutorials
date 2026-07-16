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
    data/nivlisen_surface_rema_200m.tif   REMA surface elevation, 200 m (routing)

The REMA surface is fetched separately (it only needs the committed domain
outline and public internet, not the source mosaics):
    python data/prepare_data.py rema

Provenance:
    BedMachine Antarctica v4 (NSIDC-0756) — bed, thickness, mask
    MEaSUREs Antarctic Ice Velocity 450 m v2 (NSIDC-0484) — VX, VY, ERRX, ERRY
    MEaSUREs Antarctic Boundaries v2 (NSIDC-0709) — Nivl drainage basin
    REMA Mosaic v2.0 (PGC) — surface elevation for supraglacial meltwater routing
"""

import os
import sys
import glob
import shutil
import numpy as np
import xarray as xr
import geopandas as gpd

# Source production project (only needed when regenerating the BedMachine /
# velocity clip; the REMA step below does not need it).
SRC = os.path.expanduser("~/projects/nivlisen")
HERE = os.path.dirname(os.path.abspath(__file__))

SUBSAMPLE = 4          # 500 m BedMachine * 4 -> 2 km tutorial grid
PAD_M = 8.0e3          # padding around the domain bounds

# Physical constants for the (hydrostatic) surface, matching the inversion.
RHO_I, RHO_W = 917.0, 1024.0

# --- REMA surface for meltwater routing --------------------------------------
# The melt-sensitivity notebook routes supraglacial water down the *observed*
# ice surface, which needs real topography at a finer grid than the 2 km
# (hydrostatic) model surface. We pull the REMA Mosaic v2.0 (Howat et al.,
# public on the PGC AWS open-data bucket, no login), clip it to the Nivlisen
# domain, and resample to 200 m. The mosaic is served as a virtual raster of
# Cloud-Optimised GeoTIFFs in EPSG:3031 (same projection as everything else),
# so a single windowed, decimated read fetches only the tiles we need.
REMA_VRT = ("/vsicurl/https://pgc-opendata-dems.s3.us-west-2.amazonaws.com/"
            "rema/mosaics/v2.0/32m_dem_tiles.vrt")
RES_REMA = 200.0       # routing-grid spacing for the REMA surface


def fetch_rema_surface():
    """Fetch + clip + resample the REMA surface to 200 m over the domain.

    Reads the committed domain outline for the clip window (so this runs without
    the source mosaics), windowed-reads the REMA v2.0 mosaic at 200 m with area
    averaging, and writes a small compressed GeoTIFF. This is the elevation the
    notebook routes meltwater over."""
    import math
    import rasterio
    from rasterio.windows import from_bounds
    from rasterio.enums import Resampling
    from rasterio.transform import from_origin

    # Anonymous, public S3; only fetch the window we ask for.
    os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".vrt,.tif")
    os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")
    os.environ.setdefault("AWS_DEFAULT_REGION", "us-west-2")
    os.environ.setdefault("VSI_CACHE", "TRUE")

    dom_fn = os.path.join(HERE, "nivlisen_domain.gpkg")
    gdf = gpd.read_file(dom_fn)
    domain = gdf[gdf["name"] == "domain"].geometry.values[0]
    minx, miny, maxx, maxy = domain.bounds
    # Pad and snap to the 200 m grid (a superset of the model mesh).
    x0 = math.floor((minx - PAD_M) / RES_REMA) * RES_REMA
    x1 = math.ceil((maxx + PAD_M) / RES_REMA) * RES_REMA
    y0 = math.floor((miny - PAD_M) / RES_REMA) * RES_REMA
    y1 = math.ceil((maxy + PAD_M) / RES_REMA) * RES_REMA
    nx = int(round((x1 - x0) / RES_REMA))
    ny = int(round((y1 - y0) / RES_REMA))
    print(f"REMA target grid {ny}x{nx} @ {RES_REMA:.0f} m  "
          f"x[{x0:.0f},{x1:.0f}] y[{y0:.0f},{y1:.0f}]")

    with rasterio.open(REMA_VRT) as src:
        win = from_bounds(x0, y0, x1, y1, src.transform)
        dem = src.read(1, window=win, out_shape=(ny, nx),
                       resampling=Resampling.average, boundless=True,
                       fill_value=src.nodata).astype("float32")
        nodata = float(src.nodata)
    valid = (dem != nodata) & np.isfinite(dem)
    print(f"REMA read: {valid.mean()*100:.1f}% valid, "
          f"elev {dem[valid].min():.0f}-{dem[valid].max():.0f} m")

    transform = from_origin(x0, y1, RES_REMA, RES_REMA)   # north-up
    out_fn = os.path.join(HERE, "nivlisen_surface_rema_200m.tif")
    with rasterio.open(
        out_fn, "w", driver="GTiff", height=ny, width=nx, count=1,
        dtype="float32", crs="EPSG:3031", transform=transform, nodata=nodata,
        tiled=True, compress="deflate", predictor=2,
    ) as dst:
        dst.write(dem, 1)
        dst.update_tags(
            source="REMA Mosaic v2.0 (PGC, s3://pgc-opendata-dems), 32 m",
            processing="clipped to Nivlisen domain, area-averaged to 200 m",
            purpose="surface elevation for supraglacial meltwater routing",
        )
    print(f"wrote {out_fn}  ({os.path.getsize(out_fn)/1e6:.2f} MB, grid {dem.shape})")


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
    # `rema` fetches just the REMA surface (needs only the committed domain
    # outline + internet); the default clips BedMachine / velocity and needs the
    # source mosaics in ~/projects/nivlisen.
    if len(sys.argv) > 1 and sys.argv[1] == "rema":
        fetch_rema_surface()
    else:
        main()
