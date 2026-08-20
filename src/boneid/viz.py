"""boneid.viz — meshcat visualization of a serial-chain skeleton.

Style, matching `boneid.core`: plain functions over the dataclasses. There are
no member functions and no visualization "objects" of our own; the meshcat
`Visualizer` handle is passed around as an ordinary argument, exactly like a
file handle. State lives in the meshcat scene tree (a set of named paths), not
in Python.

Conventions
-----------
* Data is **z-up**, SI, lab frame — same as `boneid.core`. meshcat's default
  scene is also z-up (its grid helper is rotated into the x-y plane and the
  camera's "up" is +z), so world matrices go straight to `set_transform` with
  no axis remap.
* three.js primitives have their own local axes: `Cylinder` is centered at the
  origin with its axis along local **+Y**, `Box` is centered and axis-aligned.
  Every helper here therefore builds a 4x4 that maps local +Y onto the desired
  world direction. See `_frame_from_direction`.
* Scene layout (all under one root so a single `delete` clears everything):

      <root>/skeleton/<segment_name>     one mesh per segment
      <root>/pelvis/band                 wide cylinder spanning the two hips
      <root>/pelvis/sacrum               hip-midpoint -> L5S1 connector
      <root>/torso/trunk                 mass-sized cylinder L5S1 -> shoulders
      <root>/markers/<marker_name>       one sphere per marker
      <root>/grf/shaft                   ground-reaction-force arrow shaft
      <root>/grf/head                    ...and its cone head
      <root>/ground                      thin box standing in for the floor

  `<root>` defaults to the module constant `ROOT` ("boneid"), and every
  drawing function takes a `root=` argument so two independent things can be
  drawn in one scene without colliding: the two-leg report passes
  `root="boneid/r"` and `root="boneid/l"`, which keeps two skeletons and two
  GRF arrows alive at once and still lets `clear_scene` (which deletes
  `ROOT`) wipe the lot.

* Meshes are created **once** (`build_scene`) and afterwards only posed with
  `set_transform`. That is what makes `animate` cheap and what lets meshcat's
  animation clips (which only carry position/quaternion/scale tracks) work.

Making the body read as ONE body
--------------------------------
Two legs drawn as bare cylinder chains look like two disconnected sticks with a
thin pelvis floating above a gap, and the ~40 kg living above L5S1 is simply
absent. Two additive pieces fix that without inventing anatomy:

* **Pelvis band** (`draw_pelvis_band`) — a wide horizontal cylinder whose axis
  runs hip_r -> hip_l, so both thighs visibly plug into one pelvis, plus an
  optional short "sacrum" cylinder from the hip midpoint up to L5S1 that closes
  the gap between the hips and the base of the trunk.
* **Torso** (`draw_torso`) — one cylinder from L5S1 to a caller-supplied top
  point (for a standard markerset: mid-acromion from RAC/LAC). Its radius is
  *not* a style choice: it is derived from the trunk mass the caller passes,
  `r = sqrt(m / (rho * pi * h))` at `rho = 1000 kg/m^3` with `h` the mean
  L5S1->top distance, so the drawn volume displaces the mass that is really
  there. 42 kg over 0.5 m gives r ~ 0.16 m.

Both are optional and off by default: `build_scene`, `draw_skeleton` and
`animate` all take `pelvis_span=` and `torso=` keyword arguments (see
`draw_skeleton` for the tuple contract), so existing callers are unaffected.

Looping
-------
Animations must loop (an animation that plays once and parks on its last frame
reads as broken), so `animate`'s `repetitions` defaults to the module constant
`LOOP_REPETITIONS`.

Force arrow scaling
-------------------
`FORCE_SCALE_M_PER_N = 1e-3`, i.e. **1 kN of force is drawn 1 m long**. Pass
`force_scale=` to override. The arrow is drawn from its application point along
+F, so an upward ground reaction points up out of the floor.

Quirks found in meshcat 0.3.x (documented so nobody re-discovers them)
----------------------------------------------------------------------
* `Visualizer.close()` raises `AttributeError` — `ViewerWindow` has no
  `close()`. Use `stop_viewer(vis)` below instead.
* There is no `Cone` geometry. A cone is `Cylinder(h, radiusTop=0,
  radiusBottom=r)`.
* `Visualizer.set_transform` decomposes the matrix into position/quaternion/
  **scale**, so a non-uniform scale in the 4x4 is fine for static drawing. The
  *animation* path is different: `AnimationFrameVisualizer.set_transform` only
  emits position + quaternion (via `quaternion_from_matrix`, which is garbage
  for a scaled matrix). Animated stretching must therefore go through a
  separate `set_property("scale", "vector3", ...)` track — which is what
  `animate` does for the force arrow.
* Constructing a `Visualizer` prints the viewer URL to stdout and spawns a
  `meshcat-server` subprocess. It works headless (no browser needed);
  `static_html()` round-trips through that subprocess.
"""

from __future__ import annotations

import contextlib
import io
import os
import time
import xml.etree.ElementTree as ET
from typing import Any

import numpy as np

import meshcat
import meshcat.geometry as g
import meshcat.animation as anim

from .core import GroundWrench, InverseDynamicsResult, Skeleton, SegmentKinematics
from .core import cop_from_wrench

# --------------------------------------------------------------------------
# Tunables (module-level constants, overridable per call)
# --------------------------------------------------------------------------

ROOT = "boneid"                 #: root path of everything this module draws

FORCE_SCALE_M_PER_N = 1e-3      #: 1 kN of force -> 1 m of arrow
GIRTH_FRACTION = 0.09           #: limb cylinder radius = 9% of segment length
TOP_GIRTH_FRACTION = 0.15       #: the top/torso segment is visibly bulkier
MARKER_RADIUS = 0.012           #: raw-marker sphere radius, m
JOINT_RADIUS_FRACTION = 0.9     #: joint ball radius, as a fraction of girth

LOOP_REPETITIONS = 10000        #: `animate` default: effectively "loop forever"

#: pelvis band radius as a fraction of the hip-to-hip HALF distance
PELVIS_BAND_FRACTION = 0.55
#: sacrum connector radius, as a fraction of the pelvis band radius
SACRUM_RADIUS_FRACTION = 0.8
#: body density used to turn a trunk mass into a drawn radius, kg/m^3
BODY_DENSITY = 1000.0

SEGMENT_COLOR = 0xB0C4DE        #: light steel blue limbs
TOP_COLOR = 0x8FA8C8            #: slightly darker torso
PELVIS_COLOR = 0x9BB0CC         #: pelvis band / sacrum
TORSO_COLOR = 0x7C93B5          #: trunk volume — distinct from the limbs
JOINT_COLOR = 0x404A55
MARKER_COLOR = 0xFF3B30
GRF_COLOR = 0x22CC55
GROUND_COLOR = 0xE8E8E8


# --------------------------------------------------------------------------
# Small geometric helpers
# --------------------------------------------------------------------------

def _unit(v: np.ndarray) -> np.ndarray:
    """Normalize a 3-vector; return +Z for a (near-)zero vector."""
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n < 1e-12:
        return np.array([0.0, 0.0, 1.0])
    return v / n


def _frame_from_direction(direction: np.ndarray) -> np.ndarray:
    """3x3 rotation whose **second** column (local +Y) is `direction`.

    three.js `CylinderGeometry` runs along its local +Y, so this is the
    rotation that aligns a cylinder (or cone) with an arbitrary world
    direction. The other two columns are an arbitrary but right-handed
    completion — limbs are rotationally symmetric so it does not matter.
    """
    y = _unit(direction)
    # pick a seed that is not parallel to y
    seed = np.array([0.0, 0.0, 1.0]) if abs(y[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    x = _unit(np.cross(seed, y))
    z = np.cross(x, y)
    return np.column_stack([x, y, z])


def segment_transform(prox: np.ndarray, dist: np.ndarray,
                      nominal_length: float | None = None) -> np.ndarray:
    """4x4 posing a unit-height, +Y-aligned primitive as the segment prox->dist.

    The primitive is assumed to be *centered at its origin with height 1*, so
    the returned matrix both stretches it (scale along local Y) and places it
    at the segment midpoint. `nominal_length` (e.g. `skeleton.length[s]`) is
    used when the two endpoints coincide, which happens in degenerate frames.
    """
    prox = np.asarray(prox, dtype=float)
    dist = np.asarray(dist, dtype=float)
    delta = dist - prox
    length = float(np.linalg.norm(delta))
    if length < 1e-9:
        length = float(nominal_length) if nominal_length else 1e-6
    m = np.eye(4)
    m[:3, :3] = _frame_from_direction(delta) @ np.diag([1.0, length, 1.0])
    m[:3, 3] = 0.5 * (prox + dist)
    return m


def translation(p: np.ndarray) -> np.ndarray:
    """4x4 pure translation (thin wrapper, kept so callers avoid tf imports)."""
    m = np.eye(4)
    m[:3, 3] = np.asarray(p, dtype=float)
    return m


def _rigid(rotation: np.ndarray, position: np.ndarray) -> np.ndarray:
    """4x4 from a 3x3 rotation and a 3-vector (no scale — animation safe)."""
    m = np.eye(4)
    m[:3, :3] = rotation
    m[:3, 3] = np.asarray(position, dtype=float)
    return m


def _at(data: np.ndarray, frame_index: int = 0) -> np.ndarray:
    """A single `[3]` point from either a `[3]` constant or a `[T,3]` track."""
    arr = np.asarray(data, dtype=float)
    return arr if arr.ndim == 1 else arr[int(frame_index)]


def _mean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Mean |a - b| over time, accepting `[3]` or `[T,3]` for either side."""
    a = np.atleast_2d(np.asarray(a, dtype=float))
    b = np.atleast_2d(np.asarray(b, dtype=float))
    d = np.linalg.norm(a - b, axis=-1)
    d = d[np.isfinite(d)]
    return float(np.mean(d)) if d.size else 0.0


def pelvis_band_radius(hip_l: np.ndarray, hip_r: np.ndarray,
                       band_fraction: float = PELVIS_BAND_FRACTION) -> float:
    """Radius of the pelvis band, m.

    `band_fraction` (default 55%) of the hip-to-hip **half** distance, averaged
    over time when `[T,3]` trajectories are passed. A ~0.18 m hip separation
    therefore gives ~0.05 m, comfortably fatter than a thigh cylinder
    (`GIRTH_FRACTION` of ~0.4 m is ~0.036 m), so the thighs plug into it.
    """
    return float(max(0.5 * _mean_distance(hip_l, hip_r) * band_fraction, 1e-3))


def torso_radius(mass: float, height: float,
                 density: float = BODY_DENSITY) -> float:
    """Radius, m, of a cylinder of `height` m that weighs `mass` kg.

    Solid-cylinder inversion of `m = rho * pi * r^2 * h`:

        r = sqrt(m / (rho * pi * h))

    with `rho = BODY_DENSITY = 1000 kg/m^3` (near enough to whole-body density
    for a drawing). This is the honesty rule for the trunk: the unmodeled mass
    above L5S1 is drawn at the size that mass actually occupies, rather than at
    a size that looks tidy. A 42 kg trunk over a 0.5 m L5S1->acromion distance
    gives r ~= 0.163 m.
    """
    h = max(float(height), 1e-3)
    return float(np.sqrt(max(float(mass), 0.0) / (float(density) * np.pi * h)))


def _frame_stretch(frame, path: str, prox: np.ndarray, dist: np.ndarray,
                   nominal_length: float | None = None) -> None:
    """Pose a unit +Y cylinder between two points *inside an animation frame*.

    `segment_transform` bakes the stretch into the 4x4, which an animation
    track cannot carry (its quaternion extraction is garbage for a scaled
    matrix — see the module docstring). So this splits the same pose into a
    rigid transform plus an explicit `scale` track along local Y.
    """
    prox = np.asarray(prox, dtype=float)
    dist = np.asarray(dist, dtype=float)
    delta = dist - prox
    length = float(np.linalg.norm(delta))
    if length < 1e-9:
        length = float(nominal_length) if nominal_length else 1e-6
    frame[path].set_transform(
        _rigid(_frame_from_direction(delta), 0.5 * (prox + dist)))
    frame[path].set_property("scale", "vector3", [1.0, length, 1.0])


def segment_girth(skeleton: Skeleton, s: int,
                  girth_fraction: float = GIRTH_FRACTION,
                  top_girth_fraction: float = TOP_GIRTH_FRACTION) -> float:
    """Radius to draw segment `s` with, in metres.

    Limb radius is `girth_fraction` of the segment's length; the top of the
    chain (index S-1, the torso in a distal->proximal chain) gets
    `top_girth_fraction` so it reads as a trunk rather than a fourth stick.
    """
    n_seg = len(skeleton.segment_names)
    frac = top_girth_fraction if s == n_seg - 1 else girth_fraction
    return float(max(skeleton.length[s], 1e-3) * frac)


# --------------------------------------------------------------------------
# Viewer lifecycle
# --------------------------------------------------------------------------

def start_viewer(open_browser: bool = False, zmq_url: str | None = None,
                 attempts: int = 4, quiet: bool = True) -> meshcat.Visualizer:
    """Start a meshcat viewer (and its zmq/web server) and return the handle.

    Headless-safe: nothing here needs a display. `open_browser=True` only calls
    `webbrowser.open`, and failures there are swallowed so a headless CI box
    does not take the whole run down.

    meshcat picks its own free port when `zmq_url is None`, but the pick is
    racy — two processes starting at once can collide, and the subprocess can
    be slow enough that the first ZMQ request times out. We therefore retry
    `attempts` times with a short backoff. `quiet=True` suppresses meshcat's
    unconditional "You can open the visualizer by ..." banner on stdout.

    Raises the last underlying exception if every attempt fails.
    """
    last_error: BaseException | None = None
    for i in range(max(1, attempts)):
        try:
            sink: Any = io.StringIO() if quiet else None
            if sink is not None:
                with contextlib.redirect_stdout(sink):
                    vis = meshcat.Visualizer(zmq_url=zmq_url)
            else:
                vis = meshcat.Visualizer(zmq_url=zmq_url)
            if open_browser:
                try:
                    vis.open()
                except Exception:       # no browser / no DISPLAY: keep going
                    pass
            return vis
        except Exception as exc:        # port collision, slow subprocess, ...
            last_error = exc
            time.sleep(0.25 * (i + 1))
    raise RuntimeError(
        "could not start a meshcat viewer after "
        f"{attempts} attempts: {last_error!r}") from last_error


def stop_viewer(vis: meshcat.Visualizer) -> None:
    """Shut down the viewer's server subprocess.

    `Visualizer.close()` is broken in meshcat 0.3.x (`ViewerWindow` has no
    `close`), so we reach for the subprocess handle ourselves. Safe to call
    more than once, and safe when the viewer was attached to an external
    server (`server_proc is None`).
    """
    window = getattr(vis, "window", None)
    proc = getattr(window, "server_proc", None)
    if proc is None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except Exception:
        with contextlib.suppress(Exception):
            proc.kill()


def set_camera(vis: meshcat.Visualizer,
               target: np.ndarray = (0.0, 0.0, 0.8),
               offset: np.ndarray = (1.3, -1.3, 0.25)) -> None:
    """Point the default camera at `target` from `target + offset`.

    meshcat's camera lives under `/Cameras/default`: translating that node
    moves the orbit *target*, and the camera object's `position` property is
    its offset from the target, expressed in the camera node's frame (x right,
    y up, z toward the viewer once meshcat has applied its z-up rotation).
    """
    target = np.asarray(target, dtype=float)
    offset = np.asarray(offset, dtype=float)
    vis["/Cameras/default"].set_transform(translation(target))
    vis["/Cameras/default/rotated/<object>"].set_property(
        "position", [float(offset[0]), float(offset[2]), float(-offset[1])])
    vis["/Cameras/default/rotated/<object>"].set_property("zoom", 1.0)


# --------------------------------------------------------------------------
# Scene construction (called once; everything after is set_transform)
# --------------------------------------------------------------------------

def draw_ground(vis: meshcat.Visualizer, size: float = 1.8,
                path: str | None = None, root: str = ROOT) -> None:
    """Draw a floor at z=0: a thin box plus meshcat's built-in grid.

    The built-in `/Grid` helper is already in the x-y plane (meshcat rotates it
    for its z-up convention), so we simply make sure it is visible; the box
    gives the shadowless viewer something to read depth against.
    """
    path = path or f"{root}/ground"
    vis[path].set_object(
        g.Box([size, size, 0.004]),
        g.MeshLambertMaterial(color=GROUND_COLOR, opacity=0.3,
                              transparent=True))
    vis[path].set_transform(translation([0.0, 0.0, -0.002]))
    vis["/Grid"].set_property("visible", True)


def _pelvis_span_parts(pelvis_span):
    """Unpack a `pelvis_span` argument into `(hip_l, hip_r, l5s1_or_None)`.

    Accepts `(hip_l, hip_r)` or `(hip_l, hip_r, l5s1)`; each entry is a `[3]`
    point or a `[T,3]` trajectory.
    """
    parts = tuple(pelvis_span)
    if len(parts) == 2:
        return parts[0], parts[1], None
    if len(parts) == 3:
        return parts
    raise ValueError(
        "pelvis_span must be (hip_l, hip_r) or (hip_l, hip_r, l5s1), "
        f"got {len(parts)} entries")


def _torso_parts(torso):
    """Unpack a `torso` argument into `(l5s1, top, mass_kg)`."""
    parts = tuple(torso)
    if len(parts) != 3:
        raise ValueError("torso must be (l5s1, top, mass_kg), "
                         f"got {len(parts)} entries")
    return parts[0], parts[1], float(parts[2])


def build_pelvis_band(vis: meshcat.Visualizer, radius: float,
                      sacrum: bool = True, root: str = ROOT) -> None:
    """Create the pelvis-band meshes (unit height, +Y axis) under `<root>`.

    `band` spans the two hips, `cap_l`/`cap_r` round its ends off so it reads
    as a pelvis rather than a cut pipe, and `sacrum` (optional) is the short
    connector from the hip midpoint up to L5S1.
    """
    material = g.MeshLambertMaterial(color=PELVIS_COLOR)
    vis[f"{root}/pelvis/band"].set_object(
        g.Cylinder(height=1.0, radius=radius), material)
    for side in ("l", "r"):
        vis[f"{root}/pelvis/cap_{side}"].set_object(g.Sphere(radius), material)
    if sacrum:
        vis[f"{root}/pelvis/sacrum"].set_object(
            g.Cylinder(height=1.0, radius=radius * SACRUM_RADIUS_FRACTION),
            material)


def build_torso(vis: meshcat.Visualizer, radius: float,
                root: str = ROOT) -> None:
    """Create the trunk mesh (unit height, +Y axis) under `<root>/torso`."""
    vis[f"{root}/torso/trunk"].set_object(
        g.Cylinder(height=1.0, radius=radius),
        g.MeshLambertMaterial(color=TORSO_COLOR))


def draw_pelvis_band(vis: meshcat.Visualizer, hip_l: np.ndarray,
                     hip_r: np.ndarray, l5s1: np.ndarray | None = None,
                     frame_index: int = 0,
                     band_fraction: float = PELVIS_BAND_FRACTION,
                     radius: float | None = None, build: bool = True,
                     root: str = ROOT) -> float:
    """Draw one pelvis spanning the two hip centers; return the radius used.

    `hip_l`, `hip_r` and `l5s1` are each a `[3]` point or a `[T,3]` trajectory
    (indexed at `frame_index`). The band is a cylinder whose axis runs
    hip_r -> hip_l with radius `pelvis_band_radius(hip_l, hip_r,
    band_fraction)` — i.e. 55% of the hip-to-hip half distance — so both thigh
    cylinders end *inside* one solid pelvis instead of in mid-air. When `l5s1`
    is given, a narrower "sacrum" cylinder runs from the hip midpoint up to
    L5S1, closing the gap between the hips and the base of the trunk.

    Pass an explicit `radius` (and `build=False`) to reuse meshes that
    `build_scene` / `build_pelvis_band` already created — the meshes are unit
    cylinders, so re-posing them is pure `set_transform`.
    """
    if radius is None:
        radius = pelvis_band_radius(hip_l, hip_r, band_fraction)
    if build:
        build_pelvis_band(vis, radius, sacrum=l5s1 is not None, root=root)
    p_l = _at(hip_l, frame_index)
    p_r = _at(hip_r, frame_index)
    vis[f"{root}/pelvis/band"].set_transform(segment_transform(p_r, p_l))
    vis[f"{root}/pelvis/cap_l"].set_transform(translation(p_l))
    vis[f"{root}/pelvis/cap_r"].set_transform(translation(p_r))
    if l5s1 is not None:
        vis[f"{root}/pelvis/sacrum"].set_transform(
            segment_transform(0.5 * (p_l + p_r), _at(l5s1, frame_index)))
    return float(radius)


def draw_torso(vis: meshcat.Visualizer, l5s1: np.ndarray, top: np.ndarray,
               mass: float, frame_index: int = 0,
               density: float = BODY_DENSITY, radius: float | None = None,
               build: bool = True, root: str = ROOT) -> float:
    """Draw the unmodeled upper body as one mass-sized cylinder; return radius.

    `l5s1` and `top` are `[3]` points or `[T,3]` trajectories (indexed at
    `frame_index`); `top` is whatever the markerset actually measures as the
    top of the trunk — for a standard set, mid-acromion from RAC/LAC. `mass` is
    the trunk mass in kg the caller believes in (for the two-leg model, the
    subject mass minus the modeled leg + pelvis segments).

    **Sizing rule.** The radius is not cosmetic. The cylinder is made to
    displace `mass` at `density` (default `BODY_DENSITY = 1000 kg/m^3`) over
    its own measured height `h` = mean |top - l5s1|:

        r = sqrt(mass / (density * pi * h))

    so a 42 kg trunk over a 0.5 m L5S1->acromion distance is drawn at
    r ~= 0.16 m. That is deliberately bulky: it is what 42 kg looks like, and
    it is the whole point — the drawing must not hide the mass the model does
    not account for. Height is re-measured every frame (the cylinder is
    stretched between the two points); only the radius is fixed, from the mean
    height, so the trunk does not pulse.
    """
    if radius is None:
        radius = torso_radius(mass, _mean_distance(top, l5s1), density)
    if build:
        build_torso(vis, radius, root=root)
    vis[f"{root}/torso/trunk"].set_transform(
        segment_transform(_at(l5s1, frame_index), _at(top, frame_index)))
    return float(radius)


def build_scene(vis: meshcat.Visualizer, skeleton: Skeleton,
                girth_fraction: float = GIRTH_FRACTION,
                top_girth_fraction: float = TOP_GIRTH_FRACTION,
                ground: bool = True, force_arrow: bool = True,
                root: str = ROOT,
                pelvis_span=None, torso=None) -> None:
    """Create every mesh once, under `<root>/...`, in its canonical pose.

    Segments are unit-height cylinders along local +Y (so `segment_transform`
    can stretch them to the per-frame joint distance), each capped with a
    darker sphere at the proximal joint. The force arrow is a unit-height
    shaft plus a fixed-size cone head. Idempotent: calling it again just
    replaces the objects.

    `pelvis_span` and `torso` (see `draw_skeleton` for the tuple contract) add
    the pelvis band and the mass-sized trunk. They are sized here — from the
    whole trajectory, so the radii are constant over the trial — and only posed
    afterwards, which is what lets `animate` drive them with scale tracks.
    """
    n_seg = len(skeleton.segment_names)
    for s, name in enumerate(skeleton.segment_names):
        radius = segment_girth(skeleton, s, girth_fraction, top_girth_fraction)
        color = TOP_COLOR if s == n_seg - 1 else SEGMENT_COLOR
        vis[f"{root}/skeleton/{name}"].set_object(
            g.Cylinder(height=1.0, radius=radius),
            g.MeshLambertMaterial(color=color))
        vis[f"{root}/skeleton/{name}_joint"].set_object(
            g.Sphere(radius * JOINT_RADIUS_FRACTION),
            g.MeshLambertMaterial(color=JOINT_COLOR))
    if force_arrow:
        # unit shaft: centered at origin, height 1 along +Y; the transform
        # applied per frame both scales and places it.
        vis[f"{root}/grf/shaft"].set_object(
            g.Cylinder(height=1.0, radius=0.012),
            g.MeshLambertMaterial(color=GRF_COLOR))
        vis[f"{root}/grf/head"].set_object(
            g.Cylinder(height=0.06, radiusTop=0.0, radiusBottom=0.03),
            g.MeshLambertMaterial(color=GRF_COLOR))
    if pelvis_span is not None:
        hip_l, hip_r, l5s1 = _pelvis_span_parts(pelvis_span)
        build_pelvis_band(vis, pelvis_band_radius(hip_l, hip_r),
                          sacrum=l5s1 is not None, root=root)
    if torso is not None:
        t_l5s1, t_top, t_mass = _torso_parts(torso)
        build_torso(vis, torso_radius(t_mass, _mean_distance(t_top, t_l5s1)),
                    root=root)
    if ground:
        draw_ground(vis, root=root)


def clear_scene(vis: meshcat.Visualizer) -> None:
    """Delete everything this module drew (leaves grid/camera alone)."""
    vis[ROOT].delete()


# --------------------------------------------------------------------------
# Per-frame drawing
# --------------------------------------------------------------------------

def draw_skeleton(vis: meshcat.Visualizer, skeleton: Skeleton,
                  kin: SegmentKinematics, frame_index: int,
                  girth_fraction: float = GIRTH_FRACTION,
                  top_girth_fraction: float = TOP_GIRTH_FRACTION,
                  build: bool = True, root: str = ROOT,
                  pelvis_span=None, torso=None) -> None:
    """Pose every segment at `frame_index`.

    Each segment mesh is stretched and placed between `kin.prox_pos[t, s]` and
    `kin.dist_pos[t, s]`; the joint ball sits on the proximal end. Segment
    orientation about its own long axis comes from `kin.r_world[t, s]` — we
    use it for the joint ball's frame so that a future non-symmetric mesh
    (a foot, say) drops straight in. `build=True` creates the meshes on the
    first call; pass `build=False` in a tight loop when you know
    `build_scene` already ran.

    The meshes themselves are created once; this function only issues
    `set_transform`.

    Two optional arguments make the drawing read as one body rather than a
    heap of sticks (both default to `None`, i.e. off):

    * `pelvis_span=(hip_l, hip_r)` or `(hip_l, hip_r, l5s1)` — draws the pelvis
      band that the two thighs plug into (see `draw_pelvis_band`).
    * `torso=(l5s1, top, mass_kg)` — draws the unmodeled upper body at the size
      its mass really occupies (see `draw_torso` for the sizing rule).

    Every entry is a `[3]` point or a `[T,3]` trajectory; trajectories are
    indexed at `frame_index`, so the same tuple works here and in `animate`.
    """
    if build:
        build_scene(vis, skeleton, girth_fraction, top_girth_fraction,
                    root=root, pelvis_span=pelvis_span, torso=torso)
    t = int(frame_index)
    for s, name in enumerate(skeleton.segment_names):
        prox = kin.prox_pos[t, s]
        dist = kin.dist_pos[t, s]
        vis[f"{root}/skeleton/{name}"].set_transform(
            segment_transform(prox, dist, skeleton.length[s]))
        vis[f"{root}/skeleton/{name}_joint"].set_transform(
            _rigid(kin.r_world[t, s], prox))
    if pelvis_span is not None:
        hip_l, hip_r, l5s1 = _pelvis_span_parts(pelvis_span)
        draw_pelvis_band(vis, hip_l, hip_r, l5s1, frame_index=t,
                         build=False, root=root)
    if torso is not None:
        t_l5s1, t_top, t_mass = _torso_parts(torso)
        draw_torso(vis, t_l5s1, t_top, t_mass, frame_index=t,
                   build=False, root=root)


def force_application_point(ground: GroundWrench, frame_index: int,
                            plane_height: float = 0.0,
                            min_fz: float = 10.0) -> np.ndarray:
    """Where to draw the GRF arrow's tail at `frame_index`.

    `GroundWrench` is deliberately COP-free (the wrench is about a fixed lab
    point), but an arrow drawn from that fixed point is useless visually. So
    we convert to the equivalent center of pressure with `core.cop_from_wrench`
    and fall back to `ground.point` whenever the vertical force is too small
    for the COP division to mean anything (that is exactly the NaN case
    `cop_from_wrench` flags).
    """
    t = int(frame_index)
    cop, _ = cop_from_wrench(ground.force[t:t + 1], ground.moment[t:t + 1],
                             ground.point, plane_height, min_fz)
    p = cop[0]
    if not np.all(np.isfinite(p)):
        return np.asarray(ground.point, dtype=float)
    return p


def arrow_transforms(origin: np.ndarray, vector: np.ndarray,
                     scale: float = FORCE_SCALE_M_PER_N,
                     head_length: float = 0.06):
    """Return `(shaft_4x4, head_4x4, length)` for an arrow at `origin`.

    `vector` is a physical quantity (newtons); `scale` converts it to metres
    (default 1e-3 = **1 kN per metre**). The shaft transform includes a local
    +Y stretch, so feed it to `set_transform` (fine — meshcat decomposes the
    scale out) but *not* to an animation track; `animate` splits it into a
    rigid transform plus a scale track for that reason.
    """
    origin = np.asarray(origin, dtype=float)
    vector = np.asarray(vector, dtype=float)
    length = float(np.linalg.norm(vector)) * float(scale)
    rot = _frame_from_direction(vector)
    body = max(length - head_length, 1e-4)
    shaft = np.eye(4)
    shaft[:3, :3] = rot @ np.diag([1.0, body, 1.0])
    shaft[:3, 3] = origin + rot[:, 1] * (0.5 * body)
    head = _rigid(rot, origin + rot[:, 1] * (body + 0.5 * head_length))
    return shaft, head, length


def draw_ground_force(vis: meshcat.Visualizer, ground: GroundWrench,
                      frame_index: int,
                      force_scale: float = FORCE_SCALE_M_PER_N,
                      head_length: float = 0.06,
                      plane_height: float = 0.0, root: str = ROOT) -> None:
    """Draw the ground reaction as an arrow (shaft cylinder + cone head).

    Length is |F| * `force_scale` (1 kN = 1 m by default); the tail sits at the
    COP-equivalent point (see `force_application_point`). Below the COP's
    force threshold the arrow degenerates to ~zero length at the wrench point,
    which reads correctly as "no contact".
    """
    t = int(frame_index)
    origin = force_application_point(ground, t, plane_height)
    shaft, head, _ = arrow_transforms(origin, ground.force[t], force_scale,
                                      head_length)
    vis[f"{root}/grf/shaft"].set_transform(shaft)
    vis[f"{root}/grf/head"].set_transform(head)


def draw_markers(vis: meshcat.Visualizer, markers: dict[str, np.ndarray],
                 frame_index: int, radius: float = MARKER_RADIUS,
                 color: int = MARKER_COLOR, build: bool = True,
                 root: str = ROOT) -> None:
    """Draw raw marker data as small spheres — "just visualize the data".

    `markers` maps a marker name to either a `[T,3]` trajectory (indexed at
    `frame_index`) or a single `[3]` position (drawn as-is, `frame_index`
    ignored). Non-finite positions (gaps/dropouts) hide the sphere instead of
    throwing it to the origin.
    """
    t = int(frame_index)
    for name, data in markers.items():
        arr = np.asarray(data, dtype=float)
        p = arr if arr.ndim == 1 else arr[t]
        path = f"{root}/markers/{name}"
        if build:
            vis[path].set_object(g.Sphere(radius),
                                 g.MeshLambertMaterial(color=color))
        finite = bool(np.all(np.isfinite(p)))
        vis[path].set_property("visible", finite)
        if finite:
            vis[path].set_transform(translation(p))


# --------------------------------------------------------------------------
# URDF skeleton (an alternative to the cylinder look)
# --------------------------------------------------------------------------
#
# Some viewers want an anatomical body rather than a robot made of pipes. We
# vendor one under `assets/urdf/` — human-gazebo's `humanSubject01`, LGPL-2.1,
# see `assets/urdf/LICENSE-note.md` — chosen because its geometry is
# **primitives only** (12 box, 11 cylinder, 2 sphere, zero <mesh> elements), so
# meshcat renders it with no mesh loader and no external asset files at all.
#
# We deliberately do NOT implement URDF *kinematics*: joints, limits and the
# link tree are ignored. Our own `SegmentKinematics` already knows where every
# segment is, so the caller maps URDF link names onto our segment indices and
# each link's visuals are posed directly from that segment's endpoints. This
# keeps the whole feature to an XML walk plus `set_transform`.

def _urdf_floats(text: str | None, default) -> np.ndarray:
    """Parse a whitespace-separated float attribute ('0 0 0.1')."""
    if not text:
        return np.asarray(default, dtype=float)
    return np.asarray([float(v) for v in text.split()], dtype=float)


def _rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    """URDF fixed-axis roll-pitch-yaw (X then Y then Z) as a 3x3."""
    r, p, y = (float(v) for v in rpy)
    cr, sr, cp, sp, cy, sy = (np.cos(r), np.sin(r), np.cos(p), np.sin(p),
                              np.cos(y), np.sin(y))
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=float)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=float)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=float)
    return rz @ ry @ rx


def _frame_from_z_direction(direction: np.ndarray) -> np.ndarray:
    """3x3 rotation whose **third** column (local +Z) is `direction`.

    URDF links in this model grow along their own +Z from the joint, which is
    the opposite convention to the +Y that three.js cylinders use (see
    `_frame_from_direction`).
    """
    z = _unit(direction)
    seed = np.array([0.0, 0.0, 1.0]) if abs(z[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    x = _unit(np.cross(seed, z))
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


#: URDF cylinders run along +Z, three.js cylinders along +Y — this is the fix.
_Y_TO_Z = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])


def parse_urdf(urdf_path: str | os.PathLike) -> dict[str, list[dict]]:
    """Read a URDF's **visual** geometry: `{link_name: [visual, ...]}`.

    Only `<link><visual>` is read — `<joint>`, `<inertial>` and `<collision>`
    are ignored on purpose (see the section comment above). Links with no
    `<visual>` never appear in the result, which conveniently drops the 44
    massless `*_f1`/`*_f2` helper links of the vendored human model.

    Each visual is a plain dict:

    * `kind`      — `"box"`, `"cylinder"` or `"sphere"`.
    * `params`    — `{"size": [x,y,z]}` / `{"length":, "radius":}` /
                    `{"radius":}`, in metres, exactly as the file states them.
    * `origin`    — 4x4 link-frame transform of the primitive, **already
                    including** the +Z-to-+Y correction for cylinders, so it can
                    go straight to `set_transform`.
    * `extent_z`  — the primitive's length along the link's +Z (box `size[2]`,
                    cylinder `length`, sphere `2*radius`). `draw_urdf_skeleton`
                    uses it to derive a scale factor from `Skeleton.length`.
    * `color`     — `(0xRRGGBB, opacity)` if the visual names a material with an
                    `<rgba>`, else `None`.

    Anything with a `<mesh>` (or any geometry type meshcat has no primitive
    for) is skipped rather than faked; a URDF made entirely of meshes therefore
    parses to an empty mapping, which `draw_urdf_skeleton` reports as an error.

    Unsupported-by-design: `package://` resolution, xacro, and `<joint>`
    kinematics.
    """
    root_el = ET.parse(os.fspath(urdf_path)).getroot()

    def _rgba_of(mat) -> tuple[int, float] | None:
        """(0xRRGGBB, opacity) from a <material>'s own <color>, else None."""
        color_el = None if mat is None else mat.find("color")
        if color_el is None:
            return None
        rgba = _urdf_floats(color_el.get("rgba"), [0.5, 0.5, 0.5, 1.0])
        packed = ((int(round(255 * rgba[0])) << 16)
                  | (int(round(255 * rgba[1])) << 8)
                  | int(round(255 * rgba[2])))
        return packed, float(rgba[3]) if len(rgba) > 3 else 1.0

    # robot-level <material name=...> definitions that visuals refer to by name
    materials = {mat.get("name"): rgba
                 for mat in root_el.findall("material")
                 if mat.get("name") and (rgba := _rgba_of(mat)) is not None}

    def _material_of(visual) -> tuple[int, float] | None:
        mat = visual.find("material")
        if mat is None:
            return None
        return _rgba_of(mat) or materials.get(mat.get("name", ""))

    out: dict[str, list[dict]] = {}
    for link in root_el.findall("link"):
        name = link.get("name")
        visuals: list[dict] = []
        for visual in link.findall("visual"):
            geom = visual.find("geometry")
            if geom is None:
                continue
            origin_el = visual.find("origin")
            xyz = _urdf_floats(origin_el.get("xyz") if origin_el is not None
                               else None, [0.0, 0.0, 0.0])
            rpy = _urdf_floats(origin_el.get("rpy") if origin_el is not None
                               else None, [0.0, 0.0, 0.0])
            rot = _rpy_matrix(rpy)
            box, cyl, sph = (geom.find("box"), geom.find("cylinder"),
                             geom.find("sphere"))
            if box is not None:
                size = _urdf_floats(box.get("size"), [0.1, 0.1, 0.1])
                kind, params, extent = "box", {"size": size}, float(size[2])
            elif cyl is not None:
                length = float(cyl.get("length", 0.1))
                radius = float(cyl.get("radius", 0.05))
                kind = "cylinder"
                params = {"length": length, "radius": radius}
                extent = length
                rot = rot @ _Y_TO_Z          # +Z (URDF) -> +Y (three.js)
            elif sph is not None:
                radius = float(sph.get("radius", 0.05))
                kind, params, extent = "sphere", {"radius": radius}, 2.0 * radius
            else:
                continue                     # <mesh> etc: skipped, never faked
            visuals.append({"kind": kind, "params": params,
                            "origin": _rigid(rot, xyz),
                            "extent_z": extent,
                            "color": _material_of(visual)})
        if visuals:
            out[name] = visuals
    return out


def _urdf_geometry(visual: dict):
    """meshcat geometry object for one parsed visual."""
    if visual["kind"] == "box":
        return g.Box([float(v) for v in visual["params"]["size"]])
    if visual["kind"] == "cylinder":
        return g.Cylinder(height=visual["params"]["length"],
                          radius=visual["params"]["radius"])
    return g.Sphere(visual["params"]["radius"])


def draw_urdf_skeleton(vis: meshcat.Visualizer,
                       urdf_path: str | os.PathLike,
                       segment_map: dict[str, int],
                       kin: SegmentKinematics, frame_index: int,
                       scale_map: dict[str, float] | None = None,
                       skeleton: Skeleton | None = None,
                       color: int | None = None,
                       build: bool = True, root: str = ROOT) -> dict[str, float]:
    """Draw a URDF humanoid's links posed from our own segment kinematics.

    An alternative to the cylinder look: instead of one pipe per segment, each
    named URDF link's primitives are drawn where that segment actually is.

    * `urdf_path` — a URDF whose visuals are primitives. The vendored
      `assets/urdf/humanSubject01_66dof.urdf` (LGPL-2.1, see
      `assets/urdf/LICENSE-note.md`) is the intended one.
    * `segment_map` — `{urdf_link_name: segment_index}` into `kin`'s segment
      axis, supplied by the caller because only the caller knows which chain it
      built, e.g. `{"RightFoot": 0, "RightLowerLeg": 1, "RightUpperLeg": 2,
      "Pelvis": 3}`. Naming a link the URDF does not have is an error rather
      than a silent no-op.
    * `scale_map` — optional `{link_name: float}` uniform scale. Where a link
      has no entry and `skeleton` is given, the scale is derived per segment as
      `skeleton.length[s] / <link's own +Z extent>`, so a 1.75 m template model
      is resized to this subject's actual segment lengths. Failing both, 1.0.

      That derivation is right for links that *grow* along +Z — which is every
      limb cylinder, the case that matters — and wrong for links whose long
      axis is not +Z, notably the **feet** (a box whose length runs along X, so
      its +Z extent is the sole thickness and the auto-scale comes out far too
      large). Give those links an explicit `scale_map` entry. This is left as a
      caller decision rather than guessed at per link name.
    * `color` — override every link's material; by default each visual keeps
      the color its URDF material declares (the vendored model declares one
      near-black material, so passing `color=SEGMENT_COLOR` is often nicer).

    **Posing.** URDF link frames in this model sit at the proximal joint with
    the link growing along local +Z, so each link node is placed at
    `kin.prox_pos[t, s]` and rotated so that its +Z points at
    `kin.dist_pos[t, s]`. Our `kin.r_world` is deliberately *not* used: its
    axis convention is whatever the marker set produced, whereas the endpoint
    pair is unambiguous. Uniform scale rides in the node's 4x4 (three.js
    propagates it to the child primitives).

    Scene layout is `<root>/urdf/<link_name>/v<i>`, one node per link so a
    later re-pose is a single `set_transform` per link.

    Returns `{link_name: scale_used}`.

    Raises `ValueError` if the URDF has no primitive visuals at all, or if
    `segment_map` names a link that has none.
    """
    links = parse_urdf(urdf_path)
    if not links:
        raise ValueError(
            f"{os.fspath(urdf_path)} has no primitive <visual> geometry that "
            "meshcat can draw (mesh-only URDFs are not supported)")
    missing = sorted(set(segment_map) - set(links))
    if missing:
        raise ValueError(
            f"URDF links {missing} have no visual geometry; available: "
            f"{sorted(links)}")

    scale_map = dict(scale_map or {})
    t = int(frame_index)
    used: dict[str, float] = {}
    for link_name, seg in segment_map.items():
        visuals = links[link_name]
        if link_name in scale_map:
            scale = float(scale_map[link_name])
        elif skeleton is not None:
            span = max((v["extent_z"] for v in visuals), default=0.0)
            scale = (float(skeleton.length[seg]) / span) if span > 1e-9 else 1.0
        else:
            scale = 1.0
        used[link_name] = scale

        node = f"{root}/urdf/{link_name}"
        if build:
            for i, visual in enumerate(visuals):
                rgb, opacity = (visual["color"] or (SEGMENT_COLOR, 1.0))
                if color is not None:
                    rgb = color
                material = g.MeshLambertMaterial(
                    color=rgb, opacity=opacity, transparent=opacity < 1.0)
                vis[f"{node}/v{i}"].set_object(_urdf_geometry(visual), material)
                vis[f"{node}/v{i}"].set_transform(visual["origin"])
        prox = kin.prox_pos[t, seg]
        dist = kin.dist_pos[t, seg]
        pose = np.eye(4)
        pose[:3, :3] = _frame_from_z_direction(dist - prox) * scale
        pose[:3, 3] = prox
        vis[node].set_transform(pose)
    return used


# --------------------------------------------------------------------------
# Animation
# --------------------------------------------------------------------------

def animate(vis: meshcat.Visualizer, skeleton: Skeleton,
            kin: SegmentKinematics, ground: GroundWrench | None = None,
            idres: InverseDynamicsResult | None = None,
            realtime_scale: float = 1.0, decimate: int = 1,
            force_scale: float = FORCE_SCALE_M_PER_N,
            head_length: float = 0.06, play: bool = True,
            repetitions: int = LOOP_REPETITIONS,
            plane_height: float = 0.0,
            root: str = ROOT,
            animation: anim.Animation | None = None,
            camera: bool = True,
            girth_fraction: float = GIRTH_FRACTION,
            top_girth_fraction: float = TOP_GIRTH_FRACTION,
            pelvis_span=None, torso=None,
            ) -> anim.Animation:
    """Build and send a meshcat `Animation` over the whole trial.

    Frames `0, decimate, 2*decimate, ...` become animation keyframes; playback
    fps is `kin.rate / decimate * realtime_scale`, so `realtime_scale=0.25`
    plays quarter speed and `decimate=4` on 200 Hz data still plays real time.

    If `ground` is given the force arrow is animated too. Animation tracks only
    carry position/quaternion/scale, so the arrow is animated as a *rigid*
    shaft transform plus an explicit `scale` track along local Y — feeding a
    scaled matrix to an animation frame would corrupt its quaternion (see the
    module docstring).

    `plane_height` is the z of the plane the force arrow's tail is drawn on
    (the same knob `force_application_point` takes): the wrench is COP-free, so
    the arrow is drawn where its line of action crosses this plane. Set it to
    the floor height when the data's floor is not at z=0.

    `idres` is accepted for symmetry with the rest of the API and is currently
    only used to color the trial: nothing about the geometry depends on it, so
    passing it is optional and never changes the pose.

    `pelvis_span` and `torso` add the pelvis band and the mass-sized trunk and
    animate them per frame; they take exactly the tuples `draw_skeleton`
    documents, except that here the entries should be `[T,3]` trajectories
    sharing `kin`'s time base (a constant `[3]` still works and simply does not
    move). For the two-leg report `pelvis_span=(hip_l, hip_r, l5s1)` comes from
    the two thigh proximal ends and the pelvis proximal end, and
    `torso=(l5s1, mid_acromion, m_trunk)` from the RAC/LAC markers plus the
    trunk mass the model does not carry.

    `repetitions` defaults to `LOOP_REPETITIONS` so the clip loops: an
    animation that plays once and parks on its final frame reads as broken.

    `root` namespaces every path this call touches, and `animation` lets a
    second call append its tracks to an existing clip set instead of starting
    a fresh one — together they are how the two-leg report animates two
    skeletons and two GRF arrows in one scene (`root="boneid/r"`, then
    `root="boneid/l", animation=<the first one>`). `camera=False` skips the
    automatic camera aim, which the second call wants.

    Returns the `Animation` (already sent to the viewer when `play=True`) so
    callers can inspect or re-send it.
    """
    build_scene(vis, skeleton, girth_fraction, top_girth_fraction,
                ground=True, force_arrow=ground is not None, root=root,
                pelvis_span=pelvis_span, torso=torso)
    n_frames = int(kin.prox_pos.shape[0])
    step = max(1, int(decimate))
    fps = max(1.0, float(kin.rate) / step * float(realtime_scale))
    if animation is None:
        animation = anim.Animation(default_framerate=fps)

    for k, t in enumerate(range(0, n_frames, step)):
        with animation.at_frame(vis, k) as frame:
            for s, name in enumerate(skeleton.segment_names):
                prox = kin.prox_pos[t, s]
                dist = kin.dist_pos[t, s]
                # the stretch has to ride a separate scale track, so that the
                # quaternion track only ever sees an unscaled rotation.
                _frame_stretch(frame, f"{root}/skeleton/{name}", prox, dist,
                               float(skeleton.length[s]))
                frame[f"{root}/skeleton/{name}_joint"].set_transform(
                    _rigid(kin.r_world[t, s], prox))
            if pelvis_span is not None:
                hip_l, hip_r, l5s1 = _pelvis_span_parts(pelvis_span)
                p_l, p_r = _at(hip_l, t), _at(hip_r, t)
                _frame_stretch(frame, f"{root}/pelvis/band", p_r, p_l)
                frame[f"{root}/pelvis/cap_l"].set_transform(_rigid(np.eye(3), p_l))
                frame[f"{root}/pelvis/cap_r"].set_transform(_rigid(np.eye(3), p_r))
                if l5s1 is not None:
                    _frame_stretch(frame, f"{root}/pelvis/sacrum",
                                   0.5 * (p_l + p_r), _at(l5s1, t))
            if torso is not None:
                t_l5s1, t_top, _ = _torso_parts(torso)
                _frame_stretch(frame, f"{root}/torso/trunk",
                               _at(t_l5s1, t), _at(t_top, t))
            if ground is not None:
                origin = force_application_point(ground, t,
                                                 plane_height)
                f_vec = np.asarray(ground.force[t], dtype=float)
                rot = _frame_from_direction(f_vec)
                length = float(np.linalg.norm(f_vec)) * float(force_scale)
                body = max(length - head_length, 1e-4)
                frame[f"{root}/grf/shaft"].set_transform(
                    _rigid(rot, origin + rot[:, 1] * (0.5 * body)))
                frame[f"{root}/grf/shaft"].set_property(
                    "scale", "vector3", [1.0, body, 1.0])
                frame[f"{root}/grf/head"].set_transform(
                    _rigid(rot, origin + rot[:, 1] * (body + 0.5 * head_length)))

    # a sane initial pose + camera so the static HTML looks right before play
    draw_skeleton(vis, skeleton, kin, 0, build=False, root=root)
    if ground is not None:
        draw_ground_force(vis, ground, 0, force_scale, head_length,
                          plane_height, root=root)
    if camera:
        # aim at the mid-height of everything the motion visits, not frame 0
        mid = np.array([np.mean(kin.prox_pos[..., 0]),
                        np.mean(kin.prox_pos[..., 1]),
                        0.55 * kin.prox_pos[..., 2].max()])
        set_camera(vis, target=mid)
    vis.set_animation(animation, play=play, repetitions=repetitions)
    return animation


# --------------------------------------------------------------------------
# Static export
# --------------------------------------------------------------------------

def render_static_html(vis: meshcat.Visualizer) -> str:
    """Return meshcat's standalone HTML snapshot of the current scene.

    The server replays the whole command history into a self-contained page
    (three.js and the meshcat viewer are inlined, so it is ~800 kB minimum and
    needs no network). Any animation sent with `animate` is part of that
    history, so the snapshot animates — which is the point: drop it into a
    report and the reader gets the moving stick figure.
    """
    return vis.static_html()


def save_html(vis: meshcat.Visualizer, path: str | os.PathLike) -> str:
    """Write `render_static_html(vis)` to `path`; return the path as a str.

    Parent directories are created if needed.
    """
    path = os.fspath(path)
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    html = render_static_html(vis)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html)
    return path


__all__ = [
    "ROOT", "FORCE_SCALE_M_PER_N", "LOOP_REPETITIONS", "BODY_DENSITY",
    "start_viewer", "stop_viewer", "set_camera",
    "build_scene", "clear_scene", "draw_ground",
    "draw_skeleton", "draw_markers", "draw_ground_force",
    "force_application_point", "arrow_transforms",
    "segment_transform", "segment_girth", "translation",
    "build_pelvis_band", "draw_pelvis_band", "pelvis_band_radius",
    "build_torso", "draw_torso", "torso_radius",
    "parse_urdf", "draw_urdf_skeleton",
    "animate", "render_static_html", "save_html",
]
