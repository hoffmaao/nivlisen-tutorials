# Nivlisen ice-shelf inversion tutorials

A short, self-contained series of Jupyter notebooks that infer the basal
**friction** and ice **fluidity** of the Nivlisen ice shelf and its grounded
catchment (Dronning Maud Land, East Antarctica) from observed surface
velocities, and then quantify the **uncertainty** of that estimate.

They are a teaching translation of a production study
([`~/projects/nivlisen`](../nivlisen)) into the style of the
[icepack tutorials](https://icepack.github.io/notebooks/tutorials/) — narrated,
self-contained, and runnable at low resolution in a few minutes on a laptop.

| Notebook | What it does |
|---|---|
| [`00-domain.ipynb`](notebooks/00-domain.ipynb) | Introduces the region, loads the gridded data (bed, thickness, velocity), carves the ice-only domain with a delineated **calving front**, and builds + plots a coarse mesh. |
| [`01-inversion.ipynb`](notebooks/01-inversion.ipynb) | Sets up the icepack shallow-stream (SSA) model with a water-pressure flotation friction law, then inverts for log-friction θ and log-fluidity φ with icepack's `StatisticsProblem` (Recinos-2023-style σ-weighted misfit + Whittle–Matérn prior). |
| [`02-uncertainty.ipynb`](notebooks/02-uncertainty.ipynb) | Computes the posterior uncertainty by a matrix-free eigendecomposition of the prior-preconditioned Gauss–Newton Hessian — Hessian–vector products via `firedrake.adjoint` (fenics_ice / Recinos UQ framework). |

The science (cost function, friction law, prior, UQ) is explained in the
notebooks; reusable plumbing lives in [`src/nivlisen_tutorial.py`](src/nivlisen_tutorial.py).

```
nivlisen-tutorials/
├── Dockerfile              custom image: Firedrake + icepack + tools
├── requirements.txt        Python deps layered on top of Firedrake
├── README.md               this file
├── data/
│   ├── nivlisen_data.nc     small clipped low-res grid (bed/thickness/vel/…)
│   ├── nivlisen_domain.gpkg domain & basin outlines
│   └── prepare_data.py      (author-side) how the small data were made
├── src/nivlisen_tutorial.py shared helpers (data, mesh, prior, model, plots)
├── mesh/                    meshes written by notebook 00
└── notebooks/               00-domain, 01-inversion, 02-uncertainty
```

The committed dataset (~0.35 MB) is already clipped and subsampled, so you do
**not** need the multi-gigabyte source mosaics or NASA Earthdata credentials to
run the tutorials.

---

## Running with Docker (recommended)

The notebooks need Firedrake, PETSc and icepack — a stack that is fiddly to
install by hand. The included `Dockerfile` builds an image with everything, and
you mount this repository into the container so your edits and outputs stay on
the host.

**1. Build the image** (from this directory):

```bash
docker build -t nivlisen-tutorials .
```

**2. Run it, mounting this directory** into the container's work folder
(`/home/firedrake/work`). The image activates the Firedrake environment and
starts JupyterLab automatically:

```bash
docker run --rm -it -p 8888:8888 \
    -v "$PWD":/home/firedrake/work \
    nivlisen-tutorials
```

- `-v "$PWD":/home/firedrake/work` mounts a host directory into the container.
  `$PWD` is a shell variable holding your **current working directory**, so —
  run from the repository root — `"$PWD"` expands to the path of this repo and
  shares it into the container at `/home/firedrake/work`. The notebooks, `src/`,
  `data/` and any meshes or outputs you generate are then the *same files* on the
  host and in the container (nothing is lost when the container exits). The
  quotes keep it working if the path has spaces. On Windows use `%cd%`
  (`cmd.exe`) or `${PWD}` (PowerShell) instead of `"$PWD"`.
- `-p 8888:8888` forwards JupyterLab to your browser.

**3. Open JupyterLab** at <http://localhost:8888> and run the notebooks in
order, starting with `notebooks/00-domain.ipynb`. (Token auth is disabled in
the image for convenience on a local machine — add one for any shared host.)

The image runs as the **`firedrake`** user with Firedrake in a virtual
environment at `~/firedrake`. The container's start-up command activates it for
you (`source ~/firedrake/bin/activate`) before launching JupyterLab.

To get a **shell** in the container instead (e.g. to run things by hand),
**activate the Firedrake environment**, then start JupyterLab yourself:

```bash
docker run --rm -it -p 8888:8888 \
    -v "$PWD":/home/firedrake/work nivlisen-tutorials bash
# then, inside the container:
source ~/firedrake/bin/activate
OMP_NUM_THREADS=1 jupyter lab --ip=0.0.0.0 --no-browser \
    --ServerApp.token= --ServerApp.root_dir=/home/firedrake/work
```

### Notes / troubleshooting

- **Base image:** the `Dockerfile` pins
  `firedrakeproject/firedrake-vanilla:2025-01` — the image icepack currently
  recommends, which ships a Firedrake venv at `~/firedrake`. Override it with
  `--build-arg FIREDRAKE_TAG=<tag>`, but be aware that newer Firedrake images
  changed their layout (root, no venv), so the `source ~/firedrake/bin/activate`
  steps here assume the 2025-01 venv layout. See
  [Docker Hub](https://hub.docker.com/u/firedrakeproject).
- If a `rasterio`/`geopandas` wheel fails to build for missing GDAL headers, add
  this next to the `apt-get` line in the `Dockerfile`, then rebuild:
  ```dockerfile
  RUN sudo apt-get update && sudo apt-get install -y --no-install-recommends \
          libgdal-dev gdal-bin && sudo rm -rf /var/lib/apt/lists/*
  ```
- **Apple Silicon / ARM hosts:** icepack's `gmsh` dependency only publishes
  **linux x86-64** wheels (no aarch64 wheel), so the build must run as
  `linux/amd64`. The `Dockerfile` already pins this (its
  `FROM --platform=linux/amd64` line), so `docker build` produces an x86-64
  image and runs under emulation with no extra flags — just expect the build to
  be slower. If `docker run` warns about a platform mismatch, add
  `--platform linux/amd64` to the run command as well. (Without the pin you'd
  see `ERROR: No matching distribution found for gmsh`.)

---

## Running in an existing Firedrake environment (no Docker)

If you already have a Firedrake virtual environment with icepack, you can skip
Docker entirely and run the notebooks there.

**1. Activate the Firedrake virtual environment** — point this at *your* venv
(this is the `source .../bin/activate` step):

```bash
source /path/to/firedrake/bin/activate
```

**2. Install the data / notebook dependencies** on top of it (icepack itself is
assumed to be in the venv already; this adds xarray, rasterio, geopandas,
JupyterLab, …):

```bash
pip install -r requirements.txt
```

**3. Launch JupyterLab** from the repository root (`OMP_NUM_THREADS=1` avoids
thread oversubscription on serial runs):

```bash
OMP_NUM_THREADS=1 jupyter lab
```

**4. Open JupyterLab** at <http://localhost:8888> (it usually opens your browser
for you) and run the notebooks in order, starting with
`notebooks/00-domain.ipynb`.

---

## Regenerating the data (optional)

`data/nivlisen_data.nc` is committed, so this is rarely needed. To rebuild it
from the full mosaics you need the production `~/projects/nivlisen` tree
populated; then `python data/prepare_data.py`.
