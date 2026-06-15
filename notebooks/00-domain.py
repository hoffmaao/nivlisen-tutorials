# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 1. The Nivlisen ice shelf: domain, data, and mesh
#
# This is the first of three notebooks that infer the basal **friction** and ice
# **fluidity** of the Nivlisen ice shelf and its grounded catchment, and then
# estimate the **uncertainty** of that inference. They are a teaching version of
# a production study, written in the style of the
# [icepack tutorials](https://icepack.github.io/notebooks/tutorials/).
#
# **Nivlisen** is an ice shelf in Dronning Maud Land, East Antarctica
# (≈ 70°S, 11°E). Ice from the grounded *Nivl* drainage basin flows north,
# crosses the grounding line, and spreads out as the floating Nivlisen shelf
# before calving into the ocean. We model the whole system — grounded ice *and*
# shelf — with a single shallow-stream (SSA) model from
# [icepack](https://icepack.github.io/).
#
# In this notebook we:
#
# 1. load the gridded observations (bed, thickness, surface, velocity),
# 2. look at the region,
# 3. carve out the **ice-only domain**, drawing a proper **calving front**, and
# 4. build a coarse triangular **mesh** with its boundary split into the parts
#    where ice flows *in* and the part that faces the *ocean*.
#
# Everything runs at low resolution so it finishes in a minute or two.

# %% [markdown]
# ## Setup
#
# We add the repository's `src/` folder to the path (it holds small helper
# functions, kept out of the notebooks so the science stays front and centre)
# and use paths *relative to this notebook*. Because the repository is mounted
# into the container, anything we write under `../mesh` or `../output` persists
# on the host and is picked up by the later notebooks.

# %%
import sys, os
sys.path.insert(0, os.path.abspath("../src"))

import numpy as np
import matplotlib.pyplot as plt
import firedrake as fd

import nivlisen_tutorial as nt

DATA = "../data/nivlisen_data.nc"
DOMAIN = "../data/nivlisen_domain.gpkg"
MESH_OUT = "../mesh/nivlisen_tutorial.msh"
os.makedirs("../mesh", exist_ok=True)

# %% [markdown]
# ## The data
#
# A single small NetCDF holds everything we need on a coarse (2 km) grid in the
# Antarctic Polar Stereographic projection (EPSG:3031):
#
# - **bed**, **thickness**, **surface** — from
#   [BedMachine Antarctica v3](https://nsidc.org/data/nsidc-0756),
# - **vx, vy** and their errors **errx, erry** — from the
#   [MEaSUREs 450 m velocity mosaic](https://nsidc.org/data/nsidc-0484),
# - **mask** — BedMachine's ice/ocean/grounded classification.
#
# The full mosaics are many gigabytes; the committed file is the small piece
# clipped to Nivlisen (see `data/prepare_data.py` for exactly how it was made),
# so you never need the originals or any data credentials.

# %%
ds = nt.load_data(DATA)
domain, basin, neighbours = nt.load_domain(DOMAIN)

speed = np.hypot(ds["vx"], ds["vy"])
print("grid:", dict(ds.sizes), " spacing:", ds.attrs["grid_spacing_m"], "m")
print(f"observed speed: {float(speed.min()):.0f} – {float(speed.max()):.0f} m/yr")

# %% [markdown]
# ## A look at the region
#
# The catchment is a long, narrow grounded basin (south) that widens into the
# floating shelf (north). The black outline is the **buffered domain** from the
# production study — the basin plus the shelf, extended a few km into the ocean.

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 8), sharey=True)

xkm, ykm = ds["x"].values / 1e3, ds["y"].values / 1e3
gx, gy = np.array(domain.exterior.xy[0]) / 1e3, np.array(domain.exterior.xy[1]) / 1e3

im0 = axes[0].pcolormesh(xkm, ykm, ds["surface"], cmap="terrain", shading="auto")
axes[0].set_title("surface elevation (m)")
fig.colorbar(im0, ax=axes[0], shrink=0.7)

im1 = axes[1].pcolormesh(xkm, ykm, speed, vmax=float(np.nanpercentile(speed, 99)),
                         cmap="magma", shading="auto")
axes[1].set_title("observed speed (m/yr)")
fig.colorbar(im1, ax=axes[1], shrink=0.7)

for ax in axes:
    ax.plot(gx, gy, "k-", lw=1.5)
    ax.set_aspect("equal")
    ax.set_xlabel("x (km)")
axes[0].set_ylabel("y (km)")
fig.suptitle("Nivlisen ice shelf and the Nivl catchment", y=0.93)
fig.tight_layout()
plt.show()

# %% [markdown]
# The shelf and the fast outlet glaciers feeding it show up clearly in the speed
# map (right); the slow interior of the catchment is nearly stagnant. This is the
# velocity field we will try to reproduce by inverting for the friction and
# fluidity.

# %% [markdown]
# ## Carving the ice-only domain and the calving front
#
# The production study used icepack2, which can cope with the ice thickness
# going to zero, so its mesh was *buffered* a few km into the open ocean. Here we
# use the **primal** SSA model, which needs a positive thickness everywhere — so
# we must trim the domain back to where there actually *is* ice.
#
# `nt.ice_extent` does this by **subtracting the open ocean** (BedMachine
# `mask == 0`) from the buffered domain. The new boundary appears only along the
# seaward edge — that is the **calving front**. Every other boundary (the ice
# divides we share with neighbouring catchments, the interior cut, and any
# inland rock) is inherited unchanged from the smooth basin-shapefile outline.
# (Subtracting ocean, rather than intersecting with the ice mask, matters: it
# keeps the inland boundary smooth instead of chasing every ragged gap in the
# ice raster.)

# %%
ice = nt.ice_extent(ds, domain)
print(f"ice domain area: {ice.area/1e6:,.0f} km²  "
      f"(buffered was {domain.area/1e6:,.0f} km²)")

fig, ax = plt.subplots(figsize=(6, 8))
ax.plot(*np.array(domain.exterior.xy)/1e3, "0.6", lw=1.2, ls="--",
        label="buffered domain (into ocean)")
ax.plot(*np.array(ice.exterior.xy)/1e3, "C3", lw=1.8, label="ice domain (calving front)")
ax.set_aspect("equal"); ax.legend(loc="lower left")
ax.set_xlabel("x (km)"); ax.set_ylabel("y (km)")
ax.set_title("Trimming the ocean buffer back to the ice front")
plt.show()

# %% [markdown]
# The red outline sits on top of the grey dashed one everywhere except along the
# **northern shelf front**, where it pulls back out of the ocean. That seaward
# arc is the calving front; the rest is inflow boundary.

# %% [markdown]
# ## Building the mesh
#
# `nt.build_mesh` triangulates the ice domain and — importantly — splits its
# boundary into two physical groups, returning their tags:
#
# - **inflow** (tag 1): the inland boundary, where ice flows in from the
#   neighbouring catchments. The inverse model will *clamp* the velocity here to
#   the observations (a Dirichlet condition).
# - **calving front** (tag 2): the seaward arc, where the model applies the
#   ice/ocean back-pressure (the *terminus* condition).
#
# A boundary segment is "inflow" if it lies on the buffered outline and
# "calving" otherwise. The production project uses an *adaptive* mesh (fine near
# the grounding line and front); a uniform coarse mesh keeps this tutorial fast
# and is plenty to illustrate the method. Increase `RESOLUTION_M` for an even
# faster run, decrease it for more detail.

# %%
RESOLUTION_M = 5000.0   # 5 km elements — coarse and fast

ids = nt.build_mesh(ice, RESOLUTION_M, MESH_OUT, domain)
print("boundary tags:", ids)

mesh = fd.Mesh(MESH_OUT)
print(f"mesh: {mesh.num_vertices()} vertices, {mesh.num_cells()} cells")

# length of each boundary group, as a sanity check
from firedrake import Constant, assemble, ds as ds_meas
for name, tag in [("inflow", ids["inflow"][0]), ("calving", ids["calving"][0])]:
    L = float(assemble(Constant(1.0) * ds_meas(tag, domain=mesh)))
    print(f"  {name:8s} (tag {tag}) = {L/1e3:.0f} km")

# %%
fig, ax = plt.subplots(figsize=(6, 8))
nt.plot_mesh(mesh, ax)
ax.set_title(f"Tutorial mesh ({mesh.num_cells()} cells, ~{RESOLUTION_M/1e3:.0f} km)")
plt.show()

# %% [markdown]
# ## Where is the ice grounded?
#
# One last thing the inversion will need: a map of where the ice is **grounded**
# versus **floating**. We get it from the *flotation criterion* — comparing the
# basal water pressure with the ice overburden pressure:
#
# $$ p_W = \rho_W\,g\max(0,\,h-s),\quad p_I = \rho_I g\,h,\quad
#    \phi = 1 - \frac{p_W}{p_I}. $$
#
# The **grounded fraction** $\phi$ is 1 where the bed carries the ice and falls
# smoothly to 0 at flotation. Basal friction only acts where $\phi>0$, so this
# map tells us where friction is even a player — the grounded catchment — versus
# the freely-floating shelf, where the ice deforms but does not slide on a bed.

# %%
Q = fd.FunctionSpace(mesh, "CG", 1)
from firedrake import Function, max_value
h = Function(Q).interpolate(max_value(nt.interpolate_field(ds, "thickness", Q),
                                      Constant(10.0, domain=mesh)))
s = nt.interpolate_field(ds, "surface", Q)
phi = Function(Q, name="grounded_fraction").interpolate(nt.flotation_factor(h, s))
print(f"{100*float((phi.dat.data_ro < 0.5).mean()):.0f}% of the domain is floating")

fig, ax = plt.subplots(figsize=(6, 8))
c = nt.plot_field(phi, ax, cmap="Blues", vmin=0, vmax=1)
fig.colorbar(c, ax=ax, shrink=0.7, label="grounded fraction ϕ")
ax.set_title("Grounded (1) vs floating (0)")
plt.show()

# %% [markdown]
# The grounded basin (south, $\phi\approx1$) and the floating Nivlisen shelf
# (north, $\phi\approx0$) are clearly separated, with the **grounding line**
# running across the middle — exactly where we would expect it.
#
# The mesh and the boundary tags are all the next notebook needs. We saved the
# mesh to `../mesh/nivlisen_tutorial.msh`; because that path lives in the mounted
# repository, **notebook 2 (the inversion)** loads exactly this mesh, with the
# convention **tag 1 = inflow, tag 2 = calving front**.
#
# ➡️ Continue with [`01-inversion.ipynb`](01-inversion.ipynb).
