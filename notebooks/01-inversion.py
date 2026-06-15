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
# # 2. Inverting for friction and fluidity
#
# We can measure how fast the ice flows (from satellites), but two of the things
# that *control* that flow are hidden underneath it:
#
# - the **basal friction** — how strongly the bed resists sliding, and
# - the ice **fluidity** — how easily the ice deforms (it depends on temperature,
#   damage, fabric…).
#
# In this notebook we **invert** the observed surface velocity for both fields at
# once. We treat them as spatially-varying log-adjustments to reference values,
#
# $$ C = C_0\,e^{\theta}, \qquad A = A_0\,e^{\varphi}, $$
#
# (the exponential keeps them positive) and find the $\theta$ (log-friction) and
# $\varphi$ (log-fluidity) that make the modelled velocity match the
# observations, regularised so the answer stays smooth and physically
# reasonable. We use icepack's
# [`StatisticsProblem`](https://icepack.github.io/), which wraps the adjoint
# gradient and the optimiser for us.

# %% [markdown]
# ## Setup: mesh, data, and fields on the mesh
#
# We load the mesh that notebook 1 wrote — with its **tag 1 = inflow**, **tag 2 =
# calving front** boundary convention — then interpolate each gridded field onto
# it. `Q` is the scalar space we invert in (one value of $\theta$ and $\varphi$
# per vertex); `V` is the vector space for the velocity.

# %%
import sys, os
sys.path.insert(0, os.path.abspath("../src"))

import numpy as np
import matplotlib.pyplot as plt
import firedrake as fd
from firedrake import Constant, Function, max_value, exp, sqrt, assemble, dx, inner
import icepack
from icepack.statistics import StatisticsProblem, MaximumProbabilityEstimator

import nivlisen_tutorial as nt

mesh = fd.Mesh("../mesh/nivlisen_tutorial.msh")
ds = nt.load_data("../data/nivlisen_data.nc")
Q = fd.FunctionSpace(mesh, "CG", 1)
V = fd.VectorFunctionSpace(mesh, "CG", 1)
INFLOW, CALVING = [1], [2]      # boundary tags written by notebook 1
print(f"mesh: {mesh.num_vertices()} vertices, {mesh.num_cells()} cells; "
      f"{Q.dim()} control DOFs per field")

# geometry and observed velocity, interpolated onto the mesh (thickness floored
# at 10 m so the floating shelf front stays positive for the primal model)
h = Function(Q, name="thickness").interpolate(
    max_value(nt.interpolate_field(ds, "thickness", Q), Constant(10.0, domain=mesh)))
s = Function(Q, name="surface").interpolate(nt.interpolate_field(ds, "surface", Q))
u_obs = Function(V, name="u_obs")
u_obs.sub(0).assign(nt.interpolate_field(ds, "vx", Q))
u_obs.sub(1).assign(nt.interpolate_field(ds, "vy", Q))

# %% [markdown]
# ### Observational uncertainty
#
# We weight the data misfit by the per-pixel velocity error $\sigma$ (so
# well-measured fast ice counts more than noisy slow ice). We take it from the
# MEaSUREs error fields, floored at 1 m/yr.

# %%
sigma_x = Function(Q, name="sigma_x").interpolate(
    max_value(nt.interpolate_field(ds, "errx", Q), Constant(1.0, domain=mesh)))
sigma_y = Function(Q, name="sigma_y").interpolate(
    max_value(nt.interpolate_field(ds, "erry", Q), Constant(1.0, domain=mesh)))

# %% [markdown]
# ## The ice-flow model
#
# We use icepack's **shallow-stream (SSA)** model, `IceStream`, solving for the
# depth-averaged velocity $u$. Two ingredients carry our controls — they live in
# `src/nivlisen_tutorial.py`:
#
# - **viscosity** uses the fluidity $A=A_0 e^{\varphi}$ (softer ice flows faster),
# - **friction** is a Weertman law with coefficient $C=C_0 e^{\theta}$, switched
#   off where the ice floats.
#
# ### Friction and the flotation criterion
#
# Basal friction only acts where the ice is **grounded**. Approaching the
# grounding line the bed drops below sea level, sea water pressurises the base,
# and the effective pressure falls to zero at flotation:
#
# $$ p_W = \rho_W g\max(0,\,h-s),\quad p_I=\rho_I g\,h,\quad
#    \phi = 1-\frac{p_W}{p_I}. $$
#
# We multiply the basal drag by the **grounded fraction** $\phi$ (1 grounded,
# $\to 0$ at flotation), so friction vanishes under the shelf — the icepack
# tutorials' water-pressure friction. We also multiply the *control* $\theta$ by
# a hard grounded mask, so the optimiser only adjusts friction where it can
# actually matter.

# %%
grounded = nt.grounded_mask(h, s, Q)
print(f"grounded: {int((grounded.dat.data_ro > 0.5).sum())}/{Q.dim()} vertices")

# %% [markdown]
# ### Reference scales, the solver, and a first forward solve
#
# We set a reference fluidity `A0` (from a reference ice temperature of 260 K)
# and friction `C0`, then build the flow solver. The **inflow** boundary is
# clamped to the observed velocity (Dirichlet); the **calving front** gets the
# ice/ocean terminus condition automatically. With the controls at zero
# ($\theta=\varphi=0$) we run a first forward solve — the modelled velocity for
# our prior guess.

# %%
A0 = Constant(float(icepack.rate_factor(Constant(260.0))), domain=mesh)
C0 = Constant(0.01, domain=mesh)
solver = nt.make_solver(dirichlet_ids=INFLOW, ice_front_ids=CALVING)

def simulation(controls):
    """Forward SSA solve for given (log_friction, log_fluidity)."""
    theta, phi = controls
    A = Function(Q).interpolate(A0 * exp(phi))
    C = Function(Q).interpolate(C0 * exp(theta * grounded))
    return solver.diagnostic_solve(velocity=u_obs, thickness=h, surface=s,
                                   fluidity=A, friction=C)

u_prior = simulation([Function(Q), Function(Q)])
print(f"prior modelled u_max = "
      f"{float(Function(Q).interpolate(sqrt(inner(u_prior, u_prior))).dat.data_ro.max()):.0f} m/yr")

# %% [markdown]
# ## The inverse problem (Recinos et al., 2023)
#
# We minimise a cost made of a **data-misfit** term and a **prior** (regulariser):
#
# $$
# J = \underbrace{\tfrac12\!\int \Big[\big(\tfrac{u-u_\text{obs}}{\sigma_u}\big)^2
#       +\big(\tfrac{v-v_\text{obs}}{\sigma_v}\big)^2\Big]\,dx}_{\text{σ-weighted misfit}}
#   \; + \; \lambda\!\!\sum_{c\in\{\theta,\varphi\}}\!
#       \underbrace{\tfrac12\!\int\big(\delta\,c^2+\gamma\,|\nabla c|^2\big)\,dx}_{\text{Whittle–Matérn prior}},
#   \qquad \gamma=\delta\,\ell^2 .
# $$
#
# The prior keeps the controls smooth (correlation length $\ell$) and pulls them
# toward zero (amplitude penalty $\delta$); $\lambda$ trades data-fit against
# smoothness. In a production run you would scan $\lambda$ (an *L-curve*) to pick
# the trade-off; here we fix one sensible value to keep things short.
#
# icepack wants these as three plain functions: `simulation` (above),
# `loss_functional` (the misfit integrand), and `regularization`.

# %%
DELTA = 1.0       # prior amplitude penalty δ
ELL = 7.5e3       # prior correlation length ℓ (m)
LAMBDA = 10.0     # overall prior strength (the L-curve knob)

def loss_functional(u):
    return 0.5 * (((u[0] - u_obs[0]) / sigma_x)**2
                  + ((u[1] - u_obs[1]) / sigma_y)**2) * dx

def regularization(controls):
    theta, phi = controls
    return LAMBDA * (nt.regularization(theta, DELTA, ELL)
                     + nt.regularization(phi, DELTA, ELL))

# %% [markdown]
# ### Run the optimisation
#
# `StatisticsProblem` ties the three pieces to the two controls;
# `MaximumProbabilityEstimator` then drives the cost downhill from
# $\theta=\varphi=0$ using a Newton/BFGS optimiser, getting the gradient by the
# adjoint method (one extra solve per iteration, *regardless* of how many DOFs).
# On this coarse mesh a few dozen iterations is plenty.

# %%
theta = Function(Q, name="log_friction")
phi = Function(Q, name="log_fluidity")

problem = StatisticsProblem(
    simulation=simulation,
    loss_functional=loss_functional,
    regularization=regularization,
    controls=[theta, phi],
)
estimator = MaximumProbabilityEstimator(
    problem, gradient_tolerance=1e-4, step_tolerance=1e-2, max_iterations=40,
)
theta, phi = estimator.solve()

m0 = float(assemble(loss_functional(u_prior)))
u_opt = simulation([theta, phi])
m1 = float(assemble(loss_functional(u_opt)))
print(f"misfit {m0:.3e} → {m1:.3e}  ({100*(1-m1/m0):.0f}% reduction)")
print(f"θ ∈ [{theta.dat.data_ro.min():.2f}, {theta.dat.data_ro.max():.2f}], "
      f"φ ∈ [{phi.dat.data_ro.min():.2f}, {phi.dat.data_ro.max():.2f}]")

# %% [markdown]
# ## Results
#
# The modelled velocity at the optimum, next to the observations.

# %%
u_opt = Function(V, name="velocity").assign(u_opt)
speed_obs = Function(Q).interpolate(sqrt(u_obs[0]**2 + u_obs[1]**2))
speed_opt = Function(Q).interpolate(sqrt(u_opt[0]**2 + u_opt[1]**2))
vmax = max(50.0, float(np.percentile(speed_obs.dat.data_ro, 99)))

fig, axes = plt.subplots(1, 3, figsize=(16, 6))
c0 = nt.plot_field(speed_obs, axes[0], vmin=0, vmax=vmax, cmap="magma")
axes[0].set_title("observed speed")
c1 = nt.plot_field(speed_opt, axes[1], vmin=0, vmax=vmax, cmap="magma")
axes[1].set_title("modelled speed (inverted)")
diff = Function(Q).interpolate(speed_opt - speed_obs)
c2 = nt.plot_field(diff, axes[2], vmin=-vmax/2, vmax=vmax/2, cmap="RdBu_r")
axes[2].set_title("modelled − observed")
for c, ax in zip([c0, c1, c2], axes):
    fig.colorbar(c, ax=ax, shrink=0.6, label="m/yr")
fig.tight_layout(); plt.show()

# %% [markdown]
# The inferred controls themselves: $\theta$ (higher ⇒ *more* friction, a
# stickier bed) and $\varphi$ (higher ⇒ softer ice). Note that **friction** is
# the masked control — it is adjusted only on grounded ice — whereas **fluidity
# is a free control over the whole domain**, grounded *and* floating. We plot
# both on a robust (98th-percentile) symmetric colour scale so the broad
# variation is visible rather than swamped by a few extreme pixels.

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 6))
tm = float(np.percentile(np.abs(theta.dat.data_ro), 98)) or 1.0
ct = nt.plot_field(theta, axes[0], cmap="RdBu_r", vmin=-tm, vmax=tm)
axes[0].set_title("θ  (log-friction adjustment)")
pm = float(np.percentile(np.abs(phi.dat.data_ro), 98)) or 1.0
cp = nt.plot_field(phi, axes[1], cmap="PuOr_r", vmin=-pm, vmax=pm)
axes[1].set_title("φ  (log-fluidity adjustment)")
fig.colorbar(ct, ax=axes[0], shrink=0.6); fig.colorbar(cp, ax=axes[1], shrink=0.6)
fig.tight_layout(); plt.show()

# %% [markdown]
# Friction is high in the slow grounded interior (the bed holds the ice back)
# and is left untouched under the floating shelf. The fluidity adjustment varies
# across the *whole* domain — it is strongest on the shelf and fast shear
# margins, where the ice deforms and the velocity therefore pins it down well.
# In the slow deep interior the ice barely deforms, so the surface velocity says
# little about its fluidity and the estimate there leans on the prior — the
# uncertainty notebook makes exactly this distinction quantitative.

# %% [markdown]
# ## Save the result
#
# We write the MAP estimate to `../output/inversion.h5` with Firedrake's
# checkpointing — the mesh together with $\theta$, $\varphi$, the velocity, the
# geometry, the grounded mask, and the uncertainties. Because `../output` is
# inside the mounted repository, the **next notebook (uncertainty)** loads
# exactly these fields from the same path.

# %%
os.makedirs("../output", exist_ok=True)
grounded.rename("grounded")
with fd.CheckpointFile("../output/inversion.h5", "w") as chk:
    chk.save_mesh(mesh)
    for f in (theta, phi, u_opt, u_obs, h, s, grounded, sigma_x, sigma_y):
        chk.save_function(f)
print("saved ../output/inversion.h5")

# %% [markdown]
# We have a friction and fluidity map that reproduces the observed flow — but how
# *well-constrained* is it? Where did the data actually pin the controls down, and
# where are we just believing the prior? That is the job of the last notebook.
#
# ➡️ Continue with [`02-uncertainty.ipynb`](02-uncertainty.ipynb).
