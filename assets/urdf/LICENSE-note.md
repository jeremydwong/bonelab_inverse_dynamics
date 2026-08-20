# Vendored URDF: human-gazebo `humanSubject01`

## What is here

| File | Bytes | Source |
| --- | --- | --- |
| `humanSubject01_66dof.urdf` | 62064 | <https://raw.githubusercontent.com/robotology/human-gazebo/master/humanSubject01/humanSubject01_66dof.urdf> |
| `humanSubject01_48dof.urdf` | 52070 | <https://raw.githubusercontent.com/robotology/human-gazebo/master/humanSubject01/humanSubject01_48dof.urdf> |
| `UPSTREAM-LICENSE.txt` | 26526 | <https://raw.githubusercontent.com/robotology/human-gazebo/master/LICENSE> |

Upstream project: <https://github.com/robotology/human-gazebo>
Retrieved: 2026-08-19, from `master`. Files are **verbatim**, unmodified.

## License

**GNU LGPL-2.1** (the full text travels alongside, in `UPSTREAM-LICENSE.txt`).
LGPL-2.1 permits redistribution of verbatim and modified copies provided the
license text and copyright notices are kept, which is what this directory does.

Two consequences worth stating plainly:

* These URDFs are **copyleft**. If anyone modifies them, the modified URDF must
  stay LGPL-2.1. `boneid` only *reads* them at runtime, which is the ordinary
  "use as a library" case and does not affect the license of `boneid` itself.
* The upstream repository *also* ships STL meshes under
  `humanSubjectWithMeshes/` and `humanSubjectWithSpinalCordMeshes/` which the
  upstream README places under **CC-BY-SA 2.0** — a different license. Those are
  **not** vendored here and are not needed: the `humanSubjectXX_*.urdf` models
  above reference **zero** mesh files.

## Why this model, and what it contains

`boneid.viz.draw_urdf_skeleton` needs a humanoid whose geometry meshcat can
render without a mesh loader. This model qualifies exactly:

* Geometry is **primitives only** — 12 `<box>`, 11 `<cylinder>`, 2 `<sphere>`
  across 25 visual links, and **no `<mesh>` elements at all**. Each maps 1:1
  onto `meshcat.geometry.Box` / `Cylinder` / `Sphere`.
* Link names are anatomical, so a caller can map them onto our segments:
  `Pelvis, L5, L3, T12, T8, Neck, Head, RightShoulder, RightUpperArm,
  RightForeArm, RightHand, RightHandCOM, LeftShoulder, LeftUpperArm,
  LeftForeArm, LeftHand, LeftHandCOM, RightUpperLeg, RightLowerLeg, RightFoot,
  RightToe, LeftUpperLeg, LeftLowerLeg, LeftFoot, LeftToe`.
* The other 44 of the 69 links are massless `*_f1` / `*_f2` "fake" links that
  split each 3-DOF anatomical joint into three revolute joints. They carry no
  `<visual>`, so the parser simply never sees them.
* Kinematic root is `Pelvis`. Joints are named `j<Anatomy>_rot{x,y,z}`
  (`jRightHip_rotx`, `jLeftKnee_roty`, ...).

The `48dof` variant is the same 25 visual links with fewer DOF per joint; the
renderer ignores joints entirely (it poses named links directly from our own
`SegmentKinematics`), so either file works.

### Gotcha the renderer handles for you

URDF `<cylinder>` runs along its local **+Z**; three.js/meshcat `Cylinder` runs
along its local **+Y**. `parse_urdf` bakes the +90 degrees-about-X correction into each
cylinder visual's origin transform, so callers never see it.
