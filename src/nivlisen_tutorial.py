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


def load_ice_domain(path="../data/nivlisen_ice_domain.gpkg"):
    """Load the pre-carved ice-only domain polygon (EPSG:3031).

    This is the :func:`ice_extent` result saved to disk, so the notebooks load a
    clean ice boundary (with the calving front already delineated) directly,
    instead of re-deriving it from the raster each time."""
    import geopandas as gpd
    return gpd.read_file(path).geometry.values[0]


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
               inflow_tol=3000.0, size_field=None):
    r"""Mesh ``ice_domain`` with a delineated calving front.

    ``resolution_m`` sets a uniform target element size. Pass ``size_field`` — a
    callable ``(x, y) -> target size (m)`` — for an **adaptive** mesh: the
    boundary is resampled at the local target spacing and gmsh's size callback
    drives the *interior* to the same field, so elements shrink where the field
    is small (e.g. high stress / strain rate) and grow where it is large.

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

    sz = size_field if size_field is not None else (lambda x, y: resolution_m)

    # Walk the boundary ring, placing a node every local ``sz`` metres — uniform
    # when ``sz`` is constant, finer where the size field is small. This decimates
    # the densely-sampled input outline while honouring the target spacing.
    ring = np.asarray(ice_domain.exterior.coords)          # closed (last == first)
    seglen = np.hypot(*np.diff(ring, axis=0).T)
    cum = np.concatenate([[0.0], np.cumsum(seglen)])
    total = float(cum[-1])
    xs, ys, s = [], [], 0.0
    while s < total:
        x = float(np.interp(s, cum, ring[:, 0]))
        y = float(np.interp(s, cum, ring[:, 1]))
        xs.append(x); ys.append(y)
        s += max(sz(x, y), 1.0)
    dense = np.column_stack([xs, ys])
    if len(dense) > 8 and np.hypot(*(dense[-1] - dense[0])) < 0.5 * sz(*dense[0]):
        dense = dense[:-1]                                  # drop node collapsed onto the start

    bdy = buffered_domain.boundary
    mids = 0.5 * (dense + np.roll(dense, -1, axis=0))
    is_inflow = np.array([bdy.distance(Point(mx, my)) < inflow_tol
                          for mx, my in mids])

    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 1)
    gmsh.model.add("nivlisen")
    pts = [gmsh.model.geo.addPoint(x, y, 0, sz(x, y)) for x, y in dense]
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
    if size_field is not None:
        # drive interior element size from the field, not just the boundary nodes
        gmsh.model.mesh.setSizeCallback(lambda dim, tag, x, y, z, lc: sz(x, y))
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
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

def plot_field(f, ax, show_mesh=False, **kw):
    """tripcolor a field with a thin boundary outline; returns the collection.

    With ``show_mesh=True`` the mesh is overlaid as faint transparent triangles."""
    coll = fd.tripcolor(f, axes=ax, **kw)
    interior_kw = ({"linewidth": 0.15, "color": "k", "alpha": 0.25} if show_mesh
                   else {"linewidth": 0})
    fd.triplot(f.function_space().mesh(), axes=ax, interior_kw=interior_kw,
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


# ─────────────────────────────────────────────────────────────────────────
# Grounding-zone flux + surface-melt routing (notebook 03)
# ─────────────────────────────────────────────────────────────────────────

# m³ of ice per year → Gt per year (ρ_ice = 917 kg/m³).
ICE_TO_GT = 917.0 / 1e12


def grounding_line_gate(h, s, Q, length=4000.0):
    r"""Smooth 0/1 grounded indicator whose gradient marks the grounding line.

    The grounding-zone flux is evaluated with the divergence trick
    :math:`Q_{GL}=\int h\,\mathbf u\cdot\nabla\chi\,dx`: for a grounded indicator
    :math:`\chi` (1 grounded, 0 floating), :math:`\nabla\chi` is a narrow ridge
    on the flotation contour, so the integral collects :math:`h\,\mathbf u\cdot
    \hat n` across the grounding line (no explicit contour tracing needed). We
    take the smooth flotation fraction (:func:`flotation_factor`), Helmholtz-
    smooth it over ``length`` metres so the ridge is a clean band rather than a
    mesh staircase, and threshold at 0.5. Returns a ``Q`` Function."""
    phi_g = Function(Q).interpolate(flotation_factor(h, s))
    L = Constant(float(length))
    chi, w, t = Function(Q), fd.TestFunction(Q), fd.TrialFunction(Q)
    fd.solve((t * w + L * L * inner(grad(t), grad(w))) * dx == phi_g * w * dx, chi)
    return Function(Q, name="gl_gate").interpolate(conditional(chi > 0.5, 1.0, 0.0))


def grounding_zone_flux(u, h, gate):
    r"""Signed grounding-zone flux :math:`\int h\,\mathbf u\cdot\nabla\chi\,dx`
    (m³/yr) for velocity ``u``, thickness ``h`` and a gate ``chi`` from
    :func:`grounding_line_gate`. Left as an ``assemble`` (not floored to a
    Python float) so it tapes under ``firedrake.adjoint`` for the sensitivity;
    scale by :data:`ICE_TO_GT` for Gt/yr."""
    return fd.assemble(h * inner(u, grad(gate)) * dx)


def routing_grid(fe_coords, field, ice_polygon, res=2000.0):
    r"""Rasterise a nodal ``field`` onto a regular ``res``-metre grid for
    routing. Cells outside ``ice_polygon`` become **OCEAN** drains (-9999),
    so surface water leaving the ice exits at the true margin (not the mesh's
    convex hull). Returns ``(grid, xs, ys, inside)`` — the field on ice and
    -9999 off it, the axes, and the boolean ice mask, all shaped ``(ny, nx)``."""
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
    from matplotlib.path import Path
    x0, x1 = fe_coords[:, 0].min(), fe_coords[:, 0].max()
    y0, y1 = fe_coords[:, 1].min(), fe_coords[:, 1].max()
    xs, ys = np.arange(x0, x1 + res, res), np.arange(y0, y1 + res, res)
    gx, gy = np.meshgrid(xs, ys)
    inside = Path(np.asarray(ice_polygon.exterior.coords)).contains_points(
        np.column_stack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
    g = LinearNDInterpolator(fe_coords, field)(gx, gy)
    g = np.where(np.isnan(g), NearestNDInterpolator(fe_coords, field)(gx, gy), g)
    return np.where(inside, g, -9999.0), xs, ys, inside


def routing_grid_rema(rema_path, fe_coords, ice_polygon):
    r"""Rasterised routing grid taken straight from the committed REMA surface.

    Where :func:`routing_grid` rasterises the smooth 2 km model surface, this
    reads the observed **REMA 200 m** elevation (``data/nivlisen_surface_rema_
    200m.tif``) — real supraglacial topography, fine enough that every mesh node
    owns routing cells. We crop the raster to the mesh extent, flip it to
    ascending-``y`` (the convention the rest of the routing code uses), and mark
    everything off the ice — outside ``ice_polygon`` or a REMA void — as
    **OCEAN** (-9999), so surface water leaving the ice drains at the true
    margin. Returns ``(grid, xs, ys, inside)`` like :func:`routing_grid`."""
    import rasterio
    from rasterio.windows import from_bounds
    from matplotlib.path import Path
    x0, x1 = float(fe_coords[:, 0].min()), float(fe_coords[:, 0].max())
    y0, y1 = float(fe_coords[:, 1].min()), float(fe_coords[:, 1].max())
    with rasterio.open(rema_path) as src:
        win = from_bounds(x0, y0, x1, y1, src.transform).round_lengths().round_offsets()
        nodata = float(src.nodata)
        dem = src.read(1, window=win, boundless=True, fill_value=nodata).astype("float64")
        t = src.window_transform(win)
    ny, nx = dem.shape
    dem = dem[::-1, :]                                    # north-up → ascending y
    xs = t.c + t.a * (np.arange(nx) + 0.5)               # cell centres, ascending
    ys = (t.f + t.e * (np.arange(ny) + 0.5))[::-1]       # t.e < 0, so reverse
    gx, gy = np.meshgrid(xs, ys)
    inside = Path(np.asarray(ice_polygon.exterior.coords)).contains_points(
        np.column_stack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
    inside &= (dem != nodata) & np.isfinite(dem)
    return np.where(inside, dem, -9999.0), xs, ys, inside


def route_fsm(dem, water_in, fsm_bin="fsm_wrapper"):
    r"""Route ``water_in`` (m of water per cell) downslope over ``dem`` (m;
    NaN or < -9990 is an OCEAN drain) with the compiled Fill-Spill-Merge binary
    (Barnes et al., 2020). Returns the routed standing-water depth ``(ny, nx)``.
    Arrays are exchanged as raw ``(ny, nx)`` C-order float64 — the wrapper's
    contract; the binary is on ``PATH`` in the Docker image."""
    import subprocess, tempfile, shutil
    binpath = shutil.which(fsm_bin) or fsm_bin
    ny, nx = dem.shape
    with tempfile.TemporaryDirectory() as td:
        d, wi, wo = f"{td}/dem.bin", f"{td}/win.bin", f"{td}/wout.bin"
        np.ascontiguousarray(dem, dtype="<f8").tofile(d)
        np.ascontiguousarray(water_in, dtype="<f8").tofile(wi)
        subprocess.run([binpath, str(ny), str(nx), d, wi, wo],
                       check=True, capture_output=True)
        return np.fromfile(wo, dtype="<f8").reshape(ny, nx)


class FSMRouter:
    r"""Route many water fields over one fixed DEM with Fill-Spill-Merge.

    Building the depression hierarchy is the expensive part of Fill-Spill-Merge
    and depends only on the terrain, so the per-node melt routing — one route
    per mesh node — should not rebuild it every time. The ``fsm_batch`` binary
    builds it once and then routes each water field streamed to it. This class
    holds that long-running process: ``route(water_in)`` sends one ``(ny, nx)``
    field and returns the routed standing-water depth. Raw ``(ny, nx)`` C-order
    float64 in and out, matching :func:`route_fsm`. Use as a context manager (or
    call :meth:`close`) so the process and its DEM file are cleaned up::

        with nt.FSMRouter(dem) as router:
            for k in range(N):
                wout = router.route(win_k)
    """

    def __init__(self, dem, fsm_bin="fsm_batch"):
        import subprocess, tempfile, shutil
        self.ny, self.nx = dem.shape
        self._nbytes = self.ny * self.nx * 8
        binpath = shutil.which(fsm_bin) or fsm_bin
        self._td = tempfile.mkdtemp()
        demf = f"{self._td}/dem.bin"
        np.ascontiguousarray(dem, dtype="<f8").tofile(demf)
        # The process reads the DEM and builds the hierarchy at startup, then
        # blocks on stdin waiting for water frames. stderr is dropped: the FSM
        # engine prints a progress bar per route, which would otherwise flood
        # the notebook with thousands of lines.
        self.proc = subprocess.Popen(
            [binpath, str(self.ny), str(self.nx), demf],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def route(self, water_in):
        self.proc.stdin.write(np.ascontiguousarray(water_in, dtype="<f8").tobytes())
        self.proc.stdin.flush()
        chunks, got = [], 0
        while got < self._nbytes:                 # pipes may return short reads
            c = self.proc.stdout.read(self._nbytes - got)
            if not c:
                raise RuntimeError("fsm_batch exited before returning a frame")
            chunks.append(c); got += len(c)
        return np.frombuffer(b"".join(chunks), dtype="<f8").reshape(self.ny, self.nx)

    def close(self):
        import shutil, subprocess
        try:
            if self.proc.poll() is None:
                self.proc.stdin.close()            # EOF ends the process loop
                try:
                    self.proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    self.proc.kill()               # unblock a wedged child
                    self.proc.wait()
        finally:
            shutil.rmtree(self._td, ignore_errors=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def grid_node_map(xs, ys, fe_coords):
    r"""Nearest mesh node for each grid cell (a Voronoi assignment). Compute
    once and reuse across many routings. Returns an ``(ny, nx)`` int array."""
    from scipy.spatial import cKDTree
    gx, gy = np.meshgrid(xs, ys)
    return cKDTree(fe_coords).query(
        np.column_stack([gx.ravel(), gy.ravel()]))[1].reshape(gx.shape)


def grid_to_nodes(grid_field, nn, inside, n_nodes):
    r"""Mass-conservingly average a per-cell grid field (m ice-equivalent) onto
    the mesh nodes: each ice cell is credited to its nearest node (``nn`` from
    :func:`grid_node_map`), and the node value is the area-weighted mean over
    the cells it owns. Returns an ``(n_nodes,)`` array."""
    idx = nn[inside].ravel()
    ncells = np.bincount(idx, minlength=n_nodes)
    total = np.bincount(idx, weights=grid_field[inside].ravel(), minlength=n_nodes)
    return np.where(ncells > 0, total / np.maximum(ncells, 1), 0.0)
