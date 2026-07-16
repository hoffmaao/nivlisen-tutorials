# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 1. The Nivlisen ice shelf: domain, data, and mesh
#
# This is the first of a series of notebooks that infer the basal **friction** and ice **fluidity** of the Nivlisen ice shelf and its grounded catchment, then estimate the **uncertainty** of that inference, and finally use the surface elevation of the catchment and the model adjoint to understand how water routed over the ice sheet surface affects grounded ice discharge.
#
# **Nivlisen** is an ice shelf in Dronning Maud Land, East Antarctica (70°S, 11°E). Ice from the grounded *Nivlisen* catchment flows north, crosses the grounding line, and spreads out over the floating Nivlisen shelf before calving into the ocean. We model the system using the shallow-stream (SSA) model implemented in [icepack](https://icepack.github.io/).
#
# In this notebook we:
#
# 1. load the gridded observations (bed, thickness, surface, velocity),
# 2. visualize the **domain**, and
# 3. build an **adaptive mesh**, refined where the effective strain rates are highest.
#
# Everything runs at modest resolution so it should finish in a couple hours.

# %% [markdown]
# ## Setup
#
# We add the repository's `src/` folder to the path (it holds small helper functions). Because the repository is mounted
# into the container, anything we write under `../mesh` or `../output` will stay in the container and can be picked up by the later notebooks. We save output to interpret out side of the container in the output folder.

# %%
import sys, os
sys.path.insert(0, os.path.abspath("../src"))

import numpy as np
import matplotlib.pyplot as plt
import firedrake as fd

import nivlisen_tutorial as nt

DATA = "../data/nivlisen_data.nc"
DOMAIN = "../data/nivlisen_domain.gpkg"
ICE_DOMAIN = "../data/nivlisen_ice_domain.gpkg"
MESH_OUT = "../mesh/nivlisen_tutorial.msh"
os.makedirs("../mesh", exist_ok=True)

# %% [markdown]
# ## The data
#
# We start by loading in our data. These include include:
#
# - **bed**, **thickness**, **surface** — from
#   [BedMachine Antarctica v3](https://nsidc.org/data/nsidc-0756),
# - **vx, vy** and their errors **errx, erry** — from the
#   [MEaSUREs 450 m velocity mosaic](https://nsidc.org/data/nsidc-0484),
# - **mask** — BedMachine's ice/ocean/grounded classification.
#
# The full continent wide products are many gigabytes; the committed file is the small piece clipped to Nivlisen (see `data/prepare_data.py` for exactly how it was made).
#
# We also load the **model domain** — the outline of where there there is ice,
# `nivlisen_ice_domain.gpkg`.

# %%
ds = nt.load_data(DATA)
domain, basin, neighbours = nt.load_domain(DOMAIN)
ice = nt.load_ice_domain(ICE_DOMAIN)        # pre-carved ice-only domain

speed = np.hypot(ds["vx"], ds["vy"])
print("grid:", dict(ds.sizes), " spacing:", ds.attrs["grid_spacing_m"], "m")
print(f"observed speed: {float(speed.min()):.0f} – {float(speed.max()):.0f} m/yr")
print(f"ice domain area: {ice.area/1e6:,.0f} km²")

# %% [markdown]
# ## A look at the region
#
# The catchment is a long, narrow grounded basin (south) that widens into the floating shelf (north). The black outline is the **ice domain** we model. Its
# northern edge is the calving front.

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 8), sharey=True)

xkm, ykm = ds["x"].values / 1e3, ds["y"].values / 1e3
gx, gy = np.array(ice.exterior.xy[0]) / 1e3, np.array(ice.exterior.xy[1]) / 1e3

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
fig.tight_layout()
plt.show()

# %% [markdown]
# The shelf and the fast outlet glaciers feeding it show up clearly in the speed map (right); the slow interior of the catchment moves comparitvely slowly. This is the
# velocity field we will try to reproduce by inverting for the friction and the prefactor in the glen flow law.

# %% [markdown]
# ## Building an adaptive mesh
#
# `nt.build_mesh` triangulates the ice domain and — importantly — splits its boundary into two "physical groups", returning their tags:
#
# - **inflow** (tag 1): the inland boundary, where ice flows in from the  neighbouring catchments. The inverse model will *clamp* the velocity here to the observations (a Dirichlet condition).
# - **calving front** (tag 2): the seaward arc, where the model applies the ice/ocean back-pressure (the *terminus* condition).
#
# A boundary segment is "inflow" if it lies on the buffered outline and "calving" otherwise.
#
# Rather than a uniform mesh, we **spend resolution where the ice is working hardest**. From the observed velocity we form the effective strain rate 
# $\dot\varepsilon_e=\sqrt{\dot\varepsilon_{xx}^2+\dot\varepsilon_{yy}^2+
# \dot\varepsilon_{xx}\dot\varepsilon_{yy}+\dot\varepsilon_{xy}^2}$ and, through
# Glen's flow law, the effective stress $\tau_e=(\dot\varepsilon_e/A)^{1/n}$.
# Where the effective stress is large (e.g. the shear margins, the grounding line, the fast trunk), we shrink the target element size toward `FINE`, coarsening to `COARSE` in
# the near-stagnant interior.

# %%
from scipy.interpolate import RegularGridInterpolator
import icepack

x, y = ds["x"].values, ds["y"].values
vx, vy = np.nan_to_num(ds["vx"].values), np.nan_to_num(ds["vy"].values)

# strain-rate tensor from the observed velocity (np.gradient uses the coordinates)
dvx_dy, dvx_dx = np.gradient(vx, y, x)
dvy_dy, dvy_dx = np.gradient(vy, y, x)
exx, eyy, exy = dvx_dx, dvy_dy, 0.5 * (dvx_dy + dvy_dx)
eps_e = np.sqrt(exx**2 + eyy**2 + exx * eyy + exy**2)      # effective strain rate (1/yr)
A0, n = float(icepack.rate_factor(fd.Constant(260.0))), 3.0
tau_e = (np.maximum(eps_e, 1e-12) / A0) ** (1.0 / n)       # effective stress (Glen's law)

# combined refinement metric — normalised on the moving ice so both stress and
# strain rate contribute — then a target element size, fine where deformation is
# highest and coarse where the ice is near-stagnant.
speed = np.hypot(vx, vy)
moving = speed > 1.0
refine = eps_e / np.percentile(eps_e[moving], 60) + tau_e / np.percentile(tau_e[moving], 60)
r_hi = np.percentile(refine[moving], 90)

FINE, COARSE = 2500.0, 7000.0                             # target element sizes (m)
size_grid = np.clip(FINE * r_hi / np.maximum(refine, 1e-6), FINE, COARSE)
size_grid[~moving] = COARSE

# a callable size field for the mesher (RegularGridInterpolator wants ascending y)
yy, gg = (y[::-1], size_grid[::-1]) if y[0] > y[-1] else (y, size_grid)
_size = RegularGridInterpolator((yy, x), gg, bounds_error=False, fill_value=COARSE)
def size_field(px, py):
    return float(_size([[py, px]])[0])

# %% [markdown]
# The deformation metric (left; brightest at the shear margins and fast trunk)
# and the resulting target element size (right) — the mesher follows the latter.

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 8), sharey=True)
im0 = axes[0].pcolormesh(x/1e3, y/1e3, np.where(moving, refine, np.nan), cmap="magma",
                         vmax=float(np.percentile(refine[moving], 98)), shading="auto")
axes[0].set_title("deformation metric (stress + strain rate)")
fig.colorbar(im0, ax=axes[0], shrink=0.7)
im1 = axes[1].pcolormesh(x/1e3, y/1e3, size_grid/1e3, cmap="viridis_r", shading="auto")
axes[1].set_title("target element size (km)")
fig.colorbar(im1, ax=axes[1], shrink=0.7)
for ax in axes:
    ax.plot(*np.array(ice.exterior.xy)/1e3, "w-", lw=1.0)
    ax.set_aspect("equal"); ax.set_xlabel("x (km)")
axes[0].set_ylabel("y (km)")
fig.tight_layout(); plt.show()

# %%
ids = nt.build_mesh(ice, FINE, MESH_OUT, domain, size_field=size_field)
print("boundary tags:", ids)

mesh = fd.Mesh(MESH_OUT)
print(f"adaptive mesh: {mesh.num_vertices()} vertices, {mesh.num_cells()} cells "
      f"(target {FINE/1e3:.1f}–{COARSE/1e3:.0f} km)")

# length of each boundary group, as a sanity check
from firedrake import Constant, assemble, ds as ds_meas
for name, tag in [("inflow", ids["inflow"][0]), ("calving", ids["calving"][0])]:
    L = float(assemble(Constant(1.0) * ds_meas(tag, domain=mesh)))
    print(f"  {name:8s} (tag {tag}) = {L/1e3:.0f} km")

# %%
fig, ax = plt.subplots(figsize=(6, 8))
nt.plot_mesh(mesh, ax)
ax.set_title(f"Adaptive mesh ({mesh.num_cells()} cells, {FINE/1e3:.1f}–{COARSE/1e3:.0f} km)")
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
# The grounded basin (south, $\phi\approx1$) and the floating Nivlisen shelf (north, $\phi\approx0$) are clearly separated, with the **grounding line** running across the middle — exactly where we would expect it.
#
# The mesh and the boundary tags are all the next notebook needs. We saved the mesh to `../mesh/nivlisen_tutorial.msh`; because that path lives in the mounted repository, **notebook 2 (the inversion)** loads exactly this mesh, with the convention **tag 1 = inflow, tag 2 = calving front**.
#
# ➡️ Continue with [`01-inversion.ipynb`](01-inversion.ipynb).
