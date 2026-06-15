r"""Shared infrastructure for the Nivlisen inversion tutorials (icepack primal).

This module collects the *plumbing* the three notebooks reuse — loading the
gridded data, carving the ice-only domain out of the buffered region, building a
low-resolution mesh with a properly delineated calving front, interpolating
raster fields onto the mesh, the icepack ice-flow model (shallow-stream / SSA
with a grounded-mask friction), the Whittle–Matérn prior, and plotting helpers.

The notebooks explain the *science*; this module keeps them uncluttered. It is
written to be readable — each function is short and documented.

Everything runs **serially**, which is all the low-resolution tutorials need.
"""

import numpy as np
import firedrake as fd
from firedrake import (
    Constant, Function, FunctionSpace, VectorFunctionSpace,
    max_value, min_value, exp, sqrt, inner, grad, dx, conditional,
)
import icepack
from icepack.constants import (
    ice_density as rho_I, water_density as rho_W, gravity as g,
)


# ─────────────────────────────────────────────────────────────────────────
# Data + domain + mesh
# ─────────────────────────────────────────────────────────────────────────

def load_data(path="../data/nivlisen_data.nc"):
    """Open the small gridded tutorial dataset (xarray Dataset, EPSG:3031)."""
    import xarray as xr
    return xr.open_dataset(path)


def load_domain(path="../data/nivlisen_domain.gpkg"):
    """Return (domain, basin, neighbours) shapely polygons. ``domain`` is the
    *buffered* region from the production meshing; we trim it to ice below."""
    import geopandas as gpd
    gdf = gpd.read_file(path)
    def layer(name):
        rows = gdf[gdf["name"] == name]
        return rows.geometry.values[0] if len(rows) else None
    return layer("domain"), layer("basin"), layer("neighbours")


def ice_extent(ds, buffered_domain, simplify_m=2000.0):
    r"""Carve the **ice-only** domain out of the buffered production region.

    The production mesh was buffered a few km into the ocean (icepack2 handled
    ``h=0`` natively). The *primal* SSA model needs ``h>0``, so we trim the
    seaward edge back to the **calving front** — the ice/ocean boundary.

    We do this by subtracting only the **open ocean** (BedMachine ``mask == 0``)
    from the buffered domain. This deviates from the smooth production outline
    *only* along the true ice front: everywhere else — the shared ice divides,
    the interior cut, and any inland rock or data gaps — the boundary stays
    exactly the smooth buffered (basin-shapefile-derived) outline. We smooth the
    raster ocean polygon first so the front is not a pixel staircase, and drop
    any interior holes so the domain stays simply-connected. (Subtracting ocean
    rather than *intersecting* with the ice mask is deliberate: intersecting
    would also cut rough interior boundaries wherever the ice mask happened to
    fall short of the basin outline.) Returns one shapely Polygon.
    """
    from shapely.geometry import shape as _shape, Polygon
    from shapely.ops import unary_union
    from rasterio.features import shapes as rio_shapes
    from rasterio.transform import Affine

    x, y = ds["x"].values, ds["y"].values
    ocean = (ds["mask"].values == 0).astype("uint8")
    dx_, dy_ = x[1] - x[0], y[1] - y[0]
    transform = Affine.translation(x[0] - dx_ / 2, y[0] - dy_ / 2) * Affine.scale(dx_, dy_)
    polys = [_shape(g) for g, v in rio_shapes(ocean, mask=ocean.astype(bool),
                                              transform=transform) if v == 1]
    # Smooth the raster staircase BEFORE subtracting, so the calving front is a
    # clean curve while the rest of the boundary stays the smooth basin outline.
    ocean_region = unary_union(polys).buffer(0).simplify(simplify_m)
    extent = buffered_domain.difference(ocean_region).buffer(0)
    if extent.geom_type == "MultiPolygon":
        extent = max(extent.geoms, key=lambda p: p.area)
    return Polygon(extent.exterior)        # drop interior holes; keep smooth ring


def build_mesh(ice_domain, resolution_m, out_path, buffered_domain,
               inflow_tol=3000.0):
    r"""Mesh ``ice_domain`` at ``resolution_m`` with a delineated calving front.

    Boundary segments are classified into two physical groups:

    - **inflow** (tag 1): where the ice domain meets the edge of the original
      buffered region — i.e. the shared ice divides and the inland cut, where
      ice flows in from neighbouring basins. These get a Dirichlet (clamped)
      velocity condition in the model.
    - **calving** (tag 2): the seaward ice front, *interior* to the buffered
      region. This is the ocean boundary; the model applies the ice/ocean
      back-pressure (terminus) condition there.

    A segment is "inflow" if its midpoint lies within ``inflow_tol`` of the
    buffered-domain boundary, otherwise "calving". Returns
    ``{"inflow": [1], "calving": [2]}`` for use as ``dirichlet_ids`` etc.
    """
    import gmsh
    from shapely.geometry import Point

    # Resample the boundary ring uniformly at ~resolution_m. This keeps the
    # smooth shape (notably the production inland boundary) but decimates its
    # dense vertices, so the mesh is ~resolution_m everywhere instead of being
    # pinned fine wherever the input outline happened to be densely sampled.
    ring = np.asarray(ice_domain.exterior.coords)          # closed (last == first)
    seglen = np.hypot(*np.diff(ring, axis=0).T)
    cum = np.concatenate([[0.0], np.cumsum(seglen)])
    total = float(cum[-1])
    n_nodes = max(8, int(round(total / resolution_m)))
    targets = np.linspace(0.0, total, n_nodes, endpoint=False)
    dense = np.column_stack([np.interp(targets, cum, ring[:, 0]),
                             np.interp(targets, cum, ring[:, 1])])

    bdy = buffered_domain.boundary
    mids = 0.5 * (dense + np.roll(dense, -1, axis=0))
    is_inflow = np.array([bdy.distance(Point(mx, my)) < inflow_tol
                          for mx, my in mids])

    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 1)
    gmsh.model.add("nivlisen")
    pts = [gmsh.model.geo.addPoint(x, y, 0, resolution_m) for x, y in dense]
    lines, inflow_lines, calving_lines = [], [], []
    for i in range(len(pts)):
        ln = gmsh.model.geo.addLine(pts[i], pts[(i + 1) % len(pts)])
        lines.append(ln)
        (inflow_lines if is_inflow[i] else calving_lines).append(ln)
    loop = gmsh.model.geo.addCurveLoop(lines)
    surf = gmsh.model.geo.addPlaneSurface([loop])
    # physical groups: inflow → tag 1, calving → tag 2, ice surface → tag 3
    gmsh.model.geo.addPhysicalGroup(1, inflow_lines or lines, tag=1, name="inflow")
    gmsh.model.geo.addPhysicalGroup(1, calving_lines, tag=2, name="calving")
    gmsh.model.geo.addPhysicalGroup(2, [surf], tag=3, name="ice")
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(2)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.write(out_path)
    gmsh.finalize()
    return {"inflow": [1], "calving": [2] if calving_lines else []}


def interpolate_field(ds, name, Q, fill=0.0):
    """Interpolate gridded field ``ds[name]`` onto the CG1 space ``Q`` (bilinear)."""
    from scipy.interpolate import RegularGridInterpolator
    x, y = ds["x"].values, ds["y"].values
    arr = np.asarray(ds[name].values, dtype=float)
    if y[0] > y[-1]:
        y, arr = y[::-1], arr[::-1, :]
    arr = np.nan_to_num(arr, nan=fill)
    interp = RegularGridInterpolator((y, x), arr, bounds_error=False, fill_value=fill)
    mesh = Q.mesh()
    xy = Function(VectorFunctionSpace(mesh, "CG", 1)).interpolate(
        mesh.coordinates).dat.data_ro
    f = Function(Q, name=name)
    f.dat.data[:] = interp(np.column_stack([xy[:, 1], xy[:, 0]]))
    return f


# ─────────────────────────────────────────────────────────────────────────
# Ice-flow model (icepack primal: IceStream with flotation-masked friction)
# ─────────────────────────────────────────────────────────────────────────

def flotation_factor(h, s):
    r"""Smooth grounded fraction from the water-pressure flotation criterion.

    Following the icepack tutorials, the basal water pressure and the ice
    overburden pressure are

    .. math:: p_W = \rho_W\,g\,\max(0,\,h-s), \qquad p_I = \rho_I\,g\,h,

    and :math:`\phi = 1 - p_W/p_I` is the **grounded fraction**: it is 1 on
    grounded ice (the base sits above sea level, so :math:`p_W=0`) and falls
    smoothly to 0 at flotation (where :math:`p_W\to p_I`). The ramp is over the
    natural ice-thickness scale — no artificial length is imposed. Multiplying
    the basal drag by :math:`\phi` switches friction off under the shelf.
    """
    p_W = rho_W * g * max_value(0.0, h - s)
    p_I = rho_I * g * h
    return max_value(0.0, 1.0 - p_W / p_I)


def grounded_mask(h, s, Q):
    r"""Hard 0/1 grounded indicator (:math:`\phi>0.01`) as a ``Q`` Function.

    Used to restrict the friction *control* to grounded ice: building
    :math:`C=C_0\,e^{\theta\,\mathrm{grounded}}` leaves :math:`C=C_0` under the
    shelf, so the optimiser never adjusts friction where the ice floats (and
    where :math:`\phi` would zero it out anyway)."""
    return Function(Q, name="grounded").interpolate(
        conditional(flotation_factor(h, s) > 0.01, Constant(1.0), Constant(0.0)))


def friction(**kwargs):
    r"""Weertman bed friction switched off on floating ice.

    The sliding coefficient ``C`` (passed in already built as
    :math:`C_0 e^{\theta\cdot\mathrm{grounded}}`) is multiplied by the smooth
    grounded fraction :math:`\phi` and handed to icepack's built-in
    :func:`icepack.models.friction.bed_friction`."""
    from operator import itemgetter
    u, h, s, C = itemgetter("velocity", "thickness", "surface", "friction")(kwargs)
    return icepack.models.friction.bed_friction(
        velocity=u, friction=C * flotation_factor(h, s))


def viscosity(**kwargs):
    r"""Depth-averaged viscosity with fluidity ``A`` (passed in already built as
    :math:`A_0 e^{\varphi}`)."""
    from operator import itemgetter
    A, u, h = itemgetter("fluidity", "velocity", "thickness")(kwargs)
    return icepack.models.viscosity.viscosity_depth_averaged(
        velocity=u, thickness=h, fluidity=A)


def make_solver(dirichlet_ids, ice_front_ids):
    """An icepack ``IceStream`` flow solver with our friction/viscosity.

    ``dirichlet_ids`` clamp the **inflow** boundary (ice entering from the
    neighbouring catchment); ``ice_front_ids`` mark the **calving front**, where
    the model applies the ice/ocean back-pressure (terminus) condition. We use
    icepack's default diagnostic solver, which globalises the SSA nonlinearity
    with Picard iterations before switching to Newton — robust even when the
    inversion proposes extreme friction/fluidity."""
    model = icepack.models.IceStream(friction=friction, viscosity=viscosity)
    return icepack.solvers.FlowSolver(
        model, dirichlet_ids=dirichlet_ids, ice_front_ids=ice_front_ids)


# ─────────────────────────────────────────────────────────────────────────
# Whittle–Matérn prior (Recinos et al. 2023 / fenics_ice)
# ─────────────────────────────────────────────────────────────────────────

def prior_gamma(delta, ell):
    r"""Smoothness weight :math:`\gamma=\delta\,\ell^2`."""
    return float(delta) * float(ell) ** 2


def regularization(c, delta, ell):
    r"""Prior energy :math:`R(c)=\tfrac12\int(\delta c^2+\gamma|\nabla c|^2)\,dx`."""
    gamma = prior_gamma(delta, ell)
    return 0.5 * (float(delta) * inner(c, c)
                  + gamma * inner(grad(c), grad(c))) * dx


def prior_bilinear(trial, test, delta, ell):
    r"""Prior precision :math:`A=\delta M+\gamma K` as a bilinear form (for UQ)."""
    gamma = prior_gamma(delta, ell)
    return (float(delta) * inner(trial, test)
            + gamma * inner(grad(trial), grad(test))) * dx


# ─────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────

def plot_field(f, ax, **kw):
    """tripcolor a field with a thin boundary outline; returns the collection."""
    coll = fd.tripcolor(f, axes=ax, **kw)
    fd.triplot(f.function_space().mesh(), axes=ax, interior_kw={"linewidth": 0},
               boundary_kw={"linewidth": 0.8, "color": "k"})
    _km_axes(ax)
    return coll


def plot_mesh(mesh, ax, bnd_ids=None):
    """Plot the triangulation; if ``bnd_ids`` is given, colour the calving front."""
    fd.triplot(mesh, axes=ax,
               interior_kw={"linewidth": 0.2, "color": "k", "alpha": 0.5},
               boundary_kw={"linewidth": 1.0, "color": "0.5"})
    _km_axes(ax)


def _km_axes(ax):
    from matplotlib.ticker import FuncFormatter
    ax.set_aspect("equal")
    km = FuncFormatter(lambda v, _: f"{v/1e3:.0f}")
    ax.xaxis.set_major_formatter(km)
    ax.yaxis.set_major_formatter(km)
    ax.set_xlabel("x (km, EPSG:3031)")
    ax.set_ylabel("y (km, EPSG:3031)")
