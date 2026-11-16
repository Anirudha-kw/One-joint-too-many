"""Compare MuJoCo's WebAssembly build (used by the page) with the Python bindings.

    npm install && pip install mujoco
    python tools/check_wasm_vs_python.py
"""

import json
import pathlib
import subprocess

import mujoco
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
res = subprocess.run(["node", "tools/check_wasm_vs_python.mjs"], cwd=ROOT, capture_output=True, text=True, check=True)
w = json.loads(res.stdout)

m = mujoco.MjModel.from_xml_path(str(ROOT / "models" / "fr3_torque.xml"))
d = mujoco.MjData(m)
d.qpos[:] = [0.1, -0.3, 0.2, -2.0, 0.3, 1.8, 0.5]
d.qvel[:] = [0.1, 0.2, -0.1, 0.3, 0.2, -0.2, 0.1]
mujoco.mj_forward(m, d)
M = np.zeros((7, 7))
mujoco.mj_fullM(m, d, M)
jacp = np.zeros((3, 7))
mujoco.mj_jacSite(m, d, jacp, None, m.site("flange").id)
errs = {
    "M": np.abs(M.ravel() - w["M"]).max(),
    "bias": np.abs(d.qfrc_bias - w["bias"]).max(),
    "site": np.abs(d.site_xpos.ravel() - w["site"]).max(),
    "jacp": np.abs(jacp.ravel() - w["jacp"]).max(),
}
for _ in range(500):
    mujoco.mj_step(m, d)
errs["qpos after 1 s"] = np.abs(d.qpos - w["qpos_after"]).max()
for k, e in errs.items():
    print(f"max |python - wasm| {k:>15}: {e:.1e}")
ok = max(errs.values()) < 1e-8
print("PASS" if ok else "FAIL")
raise SystemExit(0 if ok else 1)
