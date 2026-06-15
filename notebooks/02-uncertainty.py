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
# # 3. How certain is the inversion?
#
# Notebook 2 gave us *a* friction and fluidity field that fits the data — but the
# data do not pin down every detail. Where the ice flows fast and is
# well-observed, the velocity strongly constrains the controls; in the slow
# interior the velocity barely depends on them, so we are mostly believing the
# **prior**. This notebook quantifies that.
#
# We use the Bayesian / **fenics_ice–Recinos** framework. Near the optimum the
# posterior is approximately Gaussian (a *Laplace approximation*) with covariance
#
# $$ \Gamma_\text{post} = \big(H_\text{mis} + A\big)^{-1}, $$
#
# where $H_\text{mis}$ is the Hessian of the data-misfit (how sharply the misfit
# curves — i.e. how much the data constrain each direction) and $A=\delta M+\gamma K$
# is the **prior precision** (the Hessian of the regulariser). The key object is
# the **prior-preconditioned misfit Hessian** $A^{-1}H_\text{mis}$: its
# eigenvalues say, direction by direction, whether the data ($\lambda\gg1$) or the
# prior ($\lambda\ll1$) wins.
#
# We never form $H_\text{mis}$ as a matrix. We only need its *action* on a vector
# (a Hessian–vector product, by the adjoint method) and feed that to an iterative
# eigensolver — exactly what a production study does, where the controls have
# millions of DOFs.

# %% [markdown]
# ## Load the inversion and rebuild the model at the optimum

# %%
import sys, os
sys.path.insert(0, os.path.abspath("../src"))

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, eigsh, splu
import matplotlib.pyplot as plt
import firedrake as fd
from firedrake import (
    Constant, Function, max_value, exp, assemble, dx, inner, grad,
    TrialFunction, TestFunction,
)
import firedrake.adjoint as fda
import icepack

import nivlisen_tutorial as nt

# everything notebook 2 saved (same path, persisted via the mounted repo)
with fd.CheckpointFile("../output/inversion.h5", "r") as chk:
    mesh = chk.load_mesh()
    theta_map = chk.load_function(mesh, "log_friction")
    phi_map = chk.load_function(mesh, "log_fluidity")
    u_MAP = chk.load_function(mesh, "velocity")
    h = chk.load_function(mesh, "thickness")
    s = chk.load_function(mesh, "surface")
    grounded = chk.load_function(mesh, "grounded")
    sigma_x = chk.load_function(mesh, "sigma_x")
    sigma_y = chk.load_function(mesh, "sigma_y")
    u_obs = chk.load_function(mesh, "u_obs")

Q = fd.FunctionSpace(mesh, "CG", 1)
ndof = Q.dim()
print(f"{mesh.num_cells()} cells, {ndof} control DOFs per field → {2*ndof} total")

A0 = Constant(float(icepack.rate_factor(Constant(260.0))), domain=mesh)
C0 = Constant(0.01, domain=mesh)
solver = nt.make_solver(dirichlet_ids=[1], ice_front_ids=[2])

# Prior hyperparameters — MUST match the inversion (notebook 2).
DELTA, ELL, LAMBDA = 1.0, 7.5e3, 10.0
delta_eff = LAMBDA * DELTA            # prior precision A = delta_eff·M + gamma_eff·K
ell = ELL

# %% [markdown]
# ## The Gauss–Newton misfit Hessian by the adjoint method
#
# We need the Hessian of the data-misfit with respect to the controls. We use the
# **Gauss–Newton** approximation, obtained with a small trick: evaluate the misfit
# against the *modelled* MAP velocity instead of the observations. Then the
# residual is zero at the optimum, the awkward second-derivative-of-the-model term
# drops out, and what remains is guaranteed positive semi-definite — exactly the
# curvature we want.
#
# We tape one forward solve with `firedrake.adjoint`. The resulting
# `ReducedFunctional` then gives Hessian–vector products $v\mapsto H_\text{mis}v$
# by the second-order adjoint, each costing about two linearised solves —
# *independent* of the number of DOFs. (Calling `.derivative()` once first primes
# the adjoint tape so the Hessian evaluations are well-defined.)
#
# One subtlety matters for getting the eigenvalues right: we ask for the Hessian
# action in the **cotangent (dual) space** — `riesz_representation=None` — so it
# lives in the same space as the assembled prior precision $A$ below. (The
# default would apply an inverse-mass *Riesz map*, which silently rescales the
# spectrum by the mass matrix and gives meaningless eigenvalues.) This is the
# same convention fenics_ice and the production code use.

# %%
theta = Function(Q, name="theta").assign(theta_map)
phi = Function(Q, name="phi").assign(phi_map)

fda.continue_annotation()
A = Function(Q).interpolate(A0 * exp(phi))
C = Function(Q).interpolate(C0 * exp(theta * grounded))
u = solver.diagnostic_solve(velocity=u_obs, thickness=h, surface=s,
                            fluidity=A, friction=C)
# Gauss–Newton misfit: against the modelled MAP velocity (residual ≡ 0 here)
J_gn = assemble(0.5 * (((u[0] - u_MAP[0]) / sigma_x)**2
                       + ((u[1] - u_MAP[1]) / sigma_y)**2) * dx)
rf = fda.ReducedFunctional(J_gn, [fda.Control(theta), fda.Control(phi)])
fda.pause_annotation()

print(f"J_GN at MAP = {float(rf([theta, phi])):.2e}  (≈0 by construction)")
_ = rf.derivative()                   # prime the adjoint before Hessian products

_e0, _e1 = Function(Q), Function(Q)

def Hmis(x):
    """Action of the Gauss–Newton misfit Hessian on a stacked [θ; φ] vector,
    returned in the cotangent space to match the prior precision A."""
    _e0.dat.data[:] = x[:ndof]
    _e1.dat.data[:] = x[ndof:]
    Hd = rf.hessian([_e0, _e1], options={"riesz_representation": None})
    return np.concatenate([Hd[0].dat.data_ro, Hd[1].dat.data_ro])

H_op = LinearOperator((2 * ndof, 2 * ndof), matvec=Hmis, dtype=float)

# %% [markdown]
# ## The prior precision operator
#
# The prior precision $A=\delta M+\gamma K$ (mass + stiffness, with
# $\gamma=\delta\ell^2$) we *can* assemble cheaply — it is sparse. We build it as
# a scipy matrix, block-diagonal across $(\theta,\varphi)$, and pre-factorise it
# so the eigensolver can apply $A^{-1}$ fast.

# %%
trial, test = TrialFunction(Q), TestFunction(Q)
A_petsc = assemble(nt.prior_bilinear(trial, test, delta_eff, ell)).M.handle
ai, aj, av = A_petsc.getValuesCSR()
A_block1 = sp.csr_matrix((av, aj, ai), shape=A_petsc.getSize())
A_prior = sp.block_diag([A_block1, A_block1], format="csc")   # θ and φ share the prior
A_lu = splu(A_prior)
Minv = LinearOperator(A_prior.shape, matvec=A_lu.solve, dtype=float)
print(f"prior precision A: {A_prior.shape}, nnz={A_prior.nnz}")

# %% [markdown]
# ## The prior-preconditioned Hessian spectrum
#
# We solve the generalised eigenproblem $H_\text{mis}\,v=\lambda\,A\,v$ for the
# leading `K` modes with ARPACK (matrix-free in $H_\text{mis}$, using $A^{-1}$ as
# the preconditioner). The eigenvalues $\lambda$ are the *data-to-prior
# information ratio* in each direction:
#
# - $\lambda \gg 1$ — the data dominate; that pattern of friction/fluidity is well
#   constrained,
# - $\lambda \ll 1$ — the prior dominates; the data say little about that pattern.
#
# The eigenvectors come out $A$-orthonormal ($v_j^\top A\,v_k=\delta_{jk}$), which
# is exactly what we need for the posterior covariance below.

# %%
K = min(150, 2 * ndof - 2)            # number of leading modes to extract
v0 = np.ones(2 * ndof)                 # deterministic start (reproducible)
evals, evecs = eigsh(H_op, k=K, M=A_prior, Minv=Minv, which="LM", v0=v0, tol=1e-5)
order = np.argsort(evals)[::-1]
evals, evecs = evals[order], evecs[:, order]

n_informed = int((evals > 1).sum())
n_eff = float(np.sum(evals[evals > 0] / (1.0 + evals[evals > 0])))
print(f"{n_informed} of the leading {K} modes are data-dominated (λ > 1)")
print(f"effective # of constrained parameters  n_eff = Σ λ/(1+λ) = {n_eff:.1f}")

fig, ax = plt.subplots(figsize=(7, 4))
ax.semilogy(np.maximum(evals, 1e-12), ".-")
ax.axhline(1.0, color="r", ls="--", label="λ = 1 (data = prior)")
ax.set_xlabel("mode index"); ax.set_ylabel("eigenvalue λ")
ax.set_title("Prior-preconditioned Hessian spectrum"); ax.legend()
plt.show()

# %% [markdown]
# The spectrum decays from data-dominated modes ($\lambda>1$, left) down through
# the crossover at $\lambda=1$ into prior-dominated modes. Only the first handful
# of patterns are genuinely informed by the velocity — that count is `n_eff`, the
# *effective number of parameters* the data constrain.

# %% [markdown]
# ### The best-constrained patterns
#
# The leading eigenvectors are the friction/fluidity patterns the data constrain
# most — unsurprisingly they light up the fast-flowing ice. Here are the θ
# (friction) components of the first few.

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for k, ax in enumerate(axes):
    mode = Function(Q, name=f"mode{k}")
    mode.dat.data[:] = evecs[:ndof, k]
    m = float(np.abs(mode.dat.data_ro).max()) or 1.0
    c = nt.plot_field(mode, ax, cmap="RdBu_r", vmin=-m, vmax=m)
    ax.set_title(f"mode {k+1}  (λ = {evals[k]:.1f})")
fig.suptitle("Leading constrained patterns of log-friction θ", y=1.02)
fig.tight_layout(); plt.show()

# %% [markdown]
# ## Posterior uncertainty reduction
#
# The Laplace posterior covariance has the low-rank form
#
# $$ \Gamma_\text{post} = A^{-1} - \sum_{k} \frac{\lambda_k}{1+\lambda_k}\,
#    v_k v_k^\top, $$
#
# i.e. the prior covariance $A^{-1}$, minus a correction along each
# data-constrained direction (modes with $\lambda_k\gg1$ contribute a full
# $v_kv_k^\top$; modes with $\lambda_k\ll1$ contribute almost nothing). The
# **uncertainty reduction** is the relative drop in pointwise variance,
#
# $$ 1-\frac{\operatorname{diag}\Gamma_\text{post}}{\operatorname{diag}A^{-1}}
#    = \frac{\sum_k \frac{\lambda_k}{1+\lambda_k}\,(v_k)_i^2}
#           {\operatorname{diag}(A^{-1})_i}, $$
#
# near 1 where the data taught us a lot, near 0 where we still rely on the prior.
# (We form $\operatorname{diag}A^{-1}$ once from the factorised prior.)

# %%
# pointwise prior variance = diag(A^{-1}); θ and φ share the prior, so we only
# factor the single block and solve against the identity once.
lu1 = splu(A_block1.tocsc())
diag_Ainv = lu1.solve(np.eye(ndof)).diagonal()
prior_var = np.concatenate([diag_Ainv, diag_Ainv])
# low-rank variance removed by the data, Σ_k λ/(1+λ) (v_k)²
weight = evals / (1.0 + evals)
removed = (evecs**2) @ weight
reduction = np.clip(removed / prior_var, 0.0, 1.0)

def to_Q(vec):
    f = Function(Q); f.dat.data[:] = vec; return f

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
for ax, blk, name in zip(axes, [slice(0, ndof), slice(ndof, None)],
                         ["friction θ", "fluidity φ"]):
    c = nt.plot_field(to_Q(reduction[blk]), ax, vmin=0, vmax=1, cmap="viridis")
    fig.colorbar(c, ax=ax, shrink=0.6)
    ax.set_title(f"uncertainty reduction — {name}")
fig.suptitle("Where the velocity data constrain the controls (1 = fully)", y=0.98)
fig.tight_layout(); plt.show()

# %% [markdown]
# The bright regions — the fast outlet glaciers and shear margins — are where the
# surface velocity actually informs the friction and fluidity; the dark interior
# is essentially unconstrained by the data, so its inferred values are really just
# the prior. This map is the honest companion to the inversion: it tells you which
# parts of the friction/fluidity estimate to trust.

# %%
# Save the uncertainty-reduction fields alongside the inversion for reuse.
red_theta = Function(Q, name="reduction_theta"); red_theta.dat.data[:] = reduction[:ndof]
red_phi = Function(Q, name="reduction_phi"); red_phi.dat.data[:] = reduction[ndof:]
os.makedirs("../output", exist_ok=True)
with fd.CheckpointFile("../output/uncertainty.h5", "w") as chk:
    chk.save_mesh(mesh)
    chk.save_function(red_theta)
    chk.save_function(red_phi)
print("saved ../output/uncertainty.h5")

# %% [markdown]
# ## Recap
#
# Across the three notebooks we introduced the Nivlisen domain and data, carved
# the ice-only mesh with a delineated calving front, inferred the basal friction
# and ice fluidity with a σ-weighted misfit and a Whittle–Matérn prior (Recinos
# et al., 2023), and quantified the posterior uncertainty by an eigenanalysis of
# the prior-preconditioned Gauss–Newton Hessian (the fenics_ice framework). The
# same workflow — on an adaptive high-resolution mesh, with thousands of Hessian
# modes — is what the production study runs.
