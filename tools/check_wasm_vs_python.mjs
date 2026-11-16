// Checks that MuJoCo's WebAssembly build (the one the page runs) agrees with
// MuJoCo's Python bindings on the FR3 model.
//
//   npm install
//   node tools/check_wasm_vs_python.mjs
//
// Prints the WASM results as JSON. tools/check_wasm_vs_python.py runs this script
// and compares the output with Python.
import load from "@mujoco/mujoco";
import fs from "fs";

const mj = await load();
const m = mj.MjModel.from_xml_string(fs.readFileSync("models/fr3_torque.xml", "utf8"));
const d = new mj.MjData(m);
const q = [0.1, -0.3, 0.2, -2.0, 0.3, 1.8, 0.5];
const v = [0.1, 0.2, -0.1, 0.3, 0.2, -0.2, 0.1];
d.qpos.set(q);
d.qvel.set(v);
mj.mj_forward(m, d);

const M = new mj.DoubleBuffer(49);
mj.mj_fullM(m, d, M);
const jacp = new mj.DoubleBuffer(21);
const jacr = new mj.DoubleBuffer(21);
const site = mj.mj_name2id(m, mj.mjtObj.mjOBJ_SITE.value, "flange");
mj.mj_jacSite(m, d, jacp, jacr, site);
const out = {
  M: Array.from(M.GetView()),
  bias: Array.from(d.qfrc_bias),
  site: Array.from(d.site_xpos),
  jacp: Array.from(jacp.GetView()),
};
for (let i = 0; i < 500; i++) mj.mj_step(m, d);
out.qpos_after = Array.from(d.qpos);
console.log(JSON.stringify(out));
for (const x of [M, jacp, jacr, d, m]) x.delete();
