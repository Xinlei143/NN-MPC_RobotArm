# SO101 detailed MuJoCo collision model

Downloaded from the official [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100)
repository, `Simulation/SO101`, on 2026-08-05:

- `scene.xml` is the display/table scene;
- `so101_new_calib.xml` contains the native SO101 joint and collision model;
- `assets/*.stl` are the CAD meshes referenced by that MJCF.

The collision geoms in `so101_new_calib.xml` are enabled for the articulated
links.  MuJoCo evaluates mesh contacts using convex hulls, so horizontal-table
clearance is conservative.  This model is used only for real-hardware table
safety.  `dynamics_modeling/robots/so101/so101_nominal.xml` remains the
separate lightweight model for the existing dynamics experiments until its
coordinate/frame contract is migrated and validated.

Do not replace an STL or XML in this directory without regenerating the
`mesh_collision.bundle_sha256` in the table safety profile.
