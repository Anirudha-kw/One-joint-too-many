# One Joint Too Many

An interactive article on redundancy, dynamics and torque control of the Franka Research 3 (FR3).
Every figure runs MuJoCo in the browser (the official `@mujoco/mujoco` WebAssembly build) and renders
the FR3 meshes from MuJoCo Menagerie with three.js.

## View it

The page loads ES modules and fetches model files, so it has to be served over HTTP. Opening
`index.html` directly from disk won't work.

```bash
python3 -m http.server 8000
```

Then go to http://localhost:8000. MuJoCo (WASM), three.js, KaTeX and the fonts load from CDNs.

## Layout

| Path | What it is |
| --- | --- |
| `index.html` | The article |
| `js/fr3core.js` | Kinematics, damped-least-squares IK, null-space projection, joint/Cartesian impedance and operational-space control. Used by both the page and the Node experiments. |
| `js/main.js` | The six figures |
| `js/viewer.js`, `js/plot.js` | three.js rendering of the FR3; canvas plots |
| `models/fr3_torque.xml`, `models/fr3_position.xml` | Mesh-free FR3 physics models (torque motors / Menagerie position actuators) |
| `assets/fr3_meshes.{bin,json}` | Decimated, quantized visual meshes (0.6 MB) |
| `tools/build_web_assets.py` | Builds `models/` and `assets/` from Menagerie and checks the dynamics are unchanged |
| `tools/experiments.mjs` | Produces every number quoted in the article |
| `tools/check_wasm_vs_python.{py,mjs}` | Checks MuJoCo WASM against MuJoCo Python on the FR3 |

## Reproduce the numbers

```bash
npm install                          # @mujoco/mujoco 4.0.1, for Node
node tools/experiments.mjs           # or: kinematics | drift | dynamics | control | integrators
pip install mujoco                   # 4.0.1 was used
python3 tools/check_wasm_vs_python.py
```

## Rebuild the model assets

```bash
pip install mujoco fast-simplification
git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie.git /tmp/menagerie
python3 tools/build_web_assets.py /tmp/menagerie/franka_fr3
```

## License and attribution

The FR3 model and meshes come from MuJoCo Menagerie (Apache-2.0, `assets/MENAGERIE_LICENSE`),
derived from Franka Robotics' public description. Before publishing, replace the "Your Name"
placeholders, the acknowledgments and the citation URL.
