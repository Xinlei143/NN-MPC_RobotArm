# UR5e model provenance

The files under `vendor/` are copied without modification from
`google-deepmind/mujoco_menagerie/universal_robots_ur5e` at commit:

`71f066ad0be9cd271f7ed58c030243ef157af9f4`

Upstream: <https://github.com/google-deepmind/mujoco_menagerie>

The upstream model is distributed under the BSD-3-Clause license included as
`vendor/LICENSE`.

`ur5e_project.xml` is the project adapter derived from the pinned upstream
`ur5e.xml`. It makes only the experiment-contract changes needed here:

- resolves meshes from `vendor/assets`;
- explicitly sets a 2 ms MuJoCo step (five frames per 10 ms control step);
- disables contacts to match the ABB free-space paper plant; and
- adds `ee_site` as an alias of the upstream `attachment_site`.
