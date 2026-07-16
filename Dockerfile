# Custom image for the Nivlisen inversion tutorials.
#
# Built on the Firedrake image icepack currently recommends —
# firedrake-vanilla:2025-01 (Firedrake + PETSc inside a venv at ~/firedrake,
# running as the `firedrake` user) — then adds icepack and the geospatial /
# notebook tools. icepack's inversion tools (StatisticsProblem /
# firedrake.adjoint) are all we need — no icepack2 or tlm_adjoint.
#
# Build (from this directory):
#     docker build -t nivlisen-tutorials .
#
# Run, mounting THIS directory into the container so notebooks/data persist and
# JupyterLab serves them (see README.md):
#     docker run --rm -it -p 8888:8888 \
#         -v "$PWD":/home/firedrake/work nivlisen-tutorials
#
# Apple Silicon / ARM: the FROM line pins linux/amd64 (the Firedrake image and
# the gmsh wheel are x86-64 only), so the build and run work under emulation.
# Expect a harmless "requested image's platform ... does not match host" warning.
ARG FIREDRAKE_TAG=2025-01
FROM --platform=linux/amd64 firedrakeproject/firedrake-vanilla:${FIREDRAKE_TAG}

# `source` needs bash. The Firedrake venv lives at ~/firedrake and the default
# user (firedrake) has passwordless sudo.
SHELL ["/bin/bash", "-c"]

# gmsh (the mesh generator the meshing helper drives) + patchelf (fixes binary
# wheel RPATHs), installed via apt as in the icepack install guide. g++ and git
# additionally build the Fill-Spill-Merge routing engine (next stage).
RUN sudo apt-get update \
 && sudo apt-get install -y --no-install-recommends patchelf gmsh g++ git \
 && sudo rm -rf /var/lib/apt/lists/*

# --- Fill-Spill-Merge meltwater routing (Barnes et al., 2020) -----------------
# The melt-sensitivity notebook routes surface meltwater downslope with
# Fill-Spill-Merge. The upstream RichDEM / dephier / FSM libraries are
# header-only (MIT); we clone them at a pinned commit and compile our small
# raw-binary CLI wrapper (fsm/fsm_wrapper.cpp) against them, leaving the
# `fsm_wrapper` binary on PATH. I/O is raw float64, so there is nothing to link.
COPY fsm/fsm_wrapper.cpp /tmp/fsm/fsm_wrapper.cpp
ARG FSM_COMMIT=1c499ea475c09b9f4c5da74ee5cc995de169db63
RUN git clone https://github.com/r-barnes/Barnes2020-FillSpillMerge.git /tmp/fsm-src \
 && git -C /tmp/fsm-src checkout ${FSM_COMMIT} \
 && git -C /tmp/fsm-src submodule update --init --recursive \
 && g++ -O2 -std=c++17 \
        -I/tmp/fsm-src/include \
        -I/tmp/fsm-src/submodules/dephier/include \
        -I/tmp/fsm-src/submodules/dephier/submodules/richdem/include \
        -o /tmp/fsm_wrapper /tmp/fsm/fsm_wrapper.cpp \
 && sudo mv /tmp/fsm_wrapper /usr/local/bin/fsm_wrapper \
 && sudo rm -rf /tmp/fsm-src /tmp/fsm

# --- Ice-flow / adjoint stack -------------------------------------------------
# icepack builds on the pyadjoint already in the image and pulls in the gmsh
# Python bindings; that is the whole solver stack.
RUN source ~/firedrake/bin/activate \
 && pip install --no-cache-dir git+https://github.com/icepack/icepack.git

# --- Geospatial data + notebook tools ----------------------------------------
RUN source ~/firedrake/bin/activate \
 && pip install --no-cache-dir \
        xarray netCDF4 rasterio geopandas shapely scipy \
        colorcet matplotlib jupyterlab nbconvert ipympl

# Mount point for this repository (see `docker run -v ...`).
RUN mkdir -p /home/firedrake/work
WORKDIR /home/firedrake/work

EXPOSE 8888

# Activate the Firedrake venv, then serve JupyterLab from the mounted repo.
# OMP_NUM_THREADS=1 avoids thread oversubscription (Firedrake's recommendation).
ENV OMP_NUM_THREADS=1
CMD ["/bin/bash", "-c", "source /home/firedrake/firedrake/bin/activate && exec jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --ServerApp.token= --ServerApp.root_dir=/home/firedrake/work"]
