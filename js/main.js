// main.js — the article's interactive figures. Each figure runs MuJoCo (WebAssembly)
// in the page and renders the FR3 with three.js.
import { Scene3D, Robot, loadMeshes, webglAvailable, setArrow } from "./viewer.js";
import { TimePlot, drawJointBars, drawHeatmap, css } from "./plot.js";
import * as F from "./fr3core.js";

// Pages can set window.MUJOCO_URL to a self-hosted copy (mujoco.wasm must sit next to it).
const MUJOCO_URL = window.MUJOCO_URL
  ? new URL(window.MUJOCO_URL, document.baseURI).href
  : "https://cdn.jsdelivr.net/npm/@mujoco/mujoco@4.0.1/mujoco.js";
const READY = [0, -Math.PI / 4, 0, (-3 * Math.PI) / 4, 0, Math.PI / 2, Math.PI / 4];
const HOME = [0, 0, 0, -1.57079, 0, 1.57079, -0.7853];
const DEG = 180 / Math.PI;
const JOINT_COLORS = ["#e15759", "#f28e2b", "#d4a72c", "#59a14f", "#4e79a7", "#b07aa1", "#5fb0a8"];
const fmt = (x, d = 1) => (Number.isFinite(x) ? x.toFixed(d) : "—");
const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));

let MJ, PARTS, XML_T, XML_P;

// ---------------------------------------------------------------- UI helpers

function seg(root, name, onChange) {
  const el = root.querySelector(`.seg[data-name="${name}"]`);
  const buttons = [...el.querySelectorAll("button")];
  const set = (v) => {
    buttons.forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.value === v)));
    onChange(v);
  };
  buttons.forEach((b) => b.addEventListener("click", () => set(b.dataset.value)));
  return set;
}

function range(root, name, onInput) {
  const input = root.querySelector(`input[name="${name}"]`);
  const out = root.querySelector(`output[for="${name}"]`);
  const update = () => {
    const r = onInput(parseFloat(input.value));
    if (out && r !== undefined) out.textContent = r;
  };
  input.addEventListener("input", update);
  update();
  return { input, update };
}

function readout(root, pairs) {
  root.querySelector(".readout").innerHTML = pairs.map(([k, v]) => `<span>${k} <b>${v}</b></span>`).join("");
}

function newArm(xml = XML_T) {
  return new F.FR3(MJ, MJ.MjModel.from_xml_string(xml));
}

function bodyId(m, name) {
  return MJ.mj_name2id(m, MJ.mjtObj.mjOBJ_BODY.value, name);
}

function jointItems(arm, q, extra = {}) {
  return q.map((x, i) => {
    const margin = Math.min(x - arm.lo[i], arm.hi[i] - x) / (arm.hi[i] - arm.lo[i]);
    return { label: `q${i + 1}`, value: x, lo: arm.lo[i], hi: arm.hi[i], warn: margin < 0.03, color: extra.color, text: `${(x * DEG).toFixed(0)}°` };
  });
}

// ================================================================ Figure 1: the chain
function figChain(root) {
  const arm = newArm();
  const sc = new Scene3D(root.querySelector(".view"), { camera: [1.2, -1.1, 0.85], target: [0.2, 0, 0.4] });
  const robot = new Robot(MJ, arm.m, PARTS);
  sc.scene.add(robot.group);
  const axes = sc.jointAxes(7, JOINT_COLORS);
  const triad = sc.triad(0.1);
  const q = READY.slice();
  const J = new Float64Array(42);

  const box = root.querySelector(".joint-sliders");
  box.innerHTML = q.map((x, i) =>
    `<label><i class="dot" style="background:${JOINT_COLORS[i]}"></i>q${i + 1}
      <input type="range" name="j${i}" min="${Math.ceil(arm.lo[i] * DEG * 2) / 2}" max="${Math.floor(arm.hi[i] * DEG * 2) / 2}" step="0.5" value="${(x * DEG).toFixed(1)}">
      <output for="j${i}"></output></label>`).join("");
  const inputs = q.map((_, i) => range(root, `j${i}`, (v) => { q[i] = v / DEG; update(); return `${v.toFixed(0)}°`; }).input);

  function update() {
    arm.setQ(q); arm.kinematics();
    robot.update(arm.d); axes.set(arm.d); triad.set(arm.pos(), arm.rot());
    arm.jacobian(J);
    const p = arm.pos();
    readout(root, [["flange x", `${fmt(p[0], 3)} m`], ["y", `${fmt(p[1], 3)} m`], ["z", `${fmt(p[2], 3)} m`], ["manipulability", fmt(arm.manipulability(J), 3)]]);
    sc.requestRender();
  }
  const setPose = (pose) => { pose.forEach((x, i) => { q[i] = x; inputs[i].value = (x * DEG).toFixed(1); inputs[i].dispatchEvent(new Event("input")); }); };
  seg(root, "pose", (v) => setPose(v === "ready" ? READY : v === "home" ? HOME : [0, 1.2, 0, -0.5, 0, 1.75, 0.785]));
  root.querySelector('input[name="axes"]').addEventListener("change", (e) => { axes.visible(e.target.checked); sc.requestRender(); });
  root.querySelector('button[name="view"]').addEventListener("click", () => sc.resetView());
  update();
  return { tick() { sc.render(); } };
}

// ================================================================ Figure 2: self-motion
function figSelfMotion(root) {
  const arm = newArm();
  const sc = new Scene3D(root.querySelector(".view"), { camera: [1.35, -1.0, 0.95], target: [0.1, 0, 0.42] });
  const robot = new Robot(MJ, arm.m, PARTS);
  sc.scene.add(robot.group);
  const ghostA = new Robot(MJ, arm.m, PARTS, { ghost: true, opacity: 0.16 });
  const ghostB = new Robot(MJ, arm.m, PARTS, { ghost: true, opacity: 0.16 });
  sc.scene.add(ghostA.group, ghostB.group);
  const triad = sc.triad(0.1);
  const elbowPath = sc.line(css("--accent"), 4000, { opacity: 0.8 });

  // Trace the self-motion curve: follow the null-space tangent of J both ways from READY
  // until a joint limit, correcting the hand pose at every step.
  arm.setQ(READY); arm.kinematics();
  const p0 = arm.pos(), R0 = arm.rot();
  const J = new Float64Array(42);
  const elbow = bodyId(arm.m, "fr3_link4");
  function tangent(prev) {
    arm.kinematics(); arm.jacobian(J);
    const A = F.sandwich(J, 6, 7, null);
    for (let i = 0; i < 6; i++) A[i * 6 + i] += 1e-10;
    F.cholesky(A, 6);
    let best = null, bn = 0;
    for (let k = 0; k < 7; k++) {
      const z = new Array(7).fill(0); z[k] = 1;
      const n = F.nullProject(J, A, z), nn = Math.hypot(...n);
      if (nn > bn) { bn = nn; best = n; }
    }
    let t = best.map((x) => x / bn);
    if (prev && t.reduce((s, x, i) => s + x * prev[i], 0) < 0) t = t.map((x) => -x);
    return t;
  }
  const branches = [];
  for (const dir of [1, -1]) {
    arm.setQ(READY);
    let t = tangent(null).map((x) => x * dir);
    const pts = [];
    for (let k = 0; k < 5000; k++) {
      t = tangent(t);
      arm.kinematics(); arm.jacobian(J);
      const corr = F.ikVelocity(arm, J, p0, R0, { gain: 1, lambda: 1e-4 });
      const next = arm.q().map((x, i) => x + t[i] * 0.002 + corr[i]);
      if (next.some((x, i) => x < arm.lo[i] || x > arm.hi[i])) break;
      arm.setQ(next);
      if (k % 4 === 0) {
        arm.kinematics(); arm.jacobian(J);
        pts.push({ q: next, w: arm.manipulability(J), elbow: Array.from(arm.d.xpos.subarray(3 * elbow, 3 * elbow + 3)) });
      }
    }
    branches.push(pts);
  }
  const path = branches[1].reverse().concat([{ q: READY, w: 0 }], branches[0]);
  arm.setQ(READY); arm.kinematics(); arm.jacobian(J);
  const zeroIdx = branches[1].length;
  path[zeroIdx] = { q: READY, w: arm.manipulability(J), elbow: Array.from(arm.d.xpos.subarray(3 * elbow, 3 * elbow + 3)) };
  elbowPath.set(path.map((p) => p.elbow));
  arm.setQ(path[0].q); arm.kinematics(); ghostA.update(arm.d);
  arm.setQ(path[path.length - 1].q); arm.kinematics(); ghostB.update(arm.d);

  const slider = root.querySelector('input[name="s"]');
  slider.max = String(path.length - 1);
  slider.value = String(zeroIdx);
  const wCanvas = root.querySelector("canvas.manip");
  const barCanvas = root.querySelector("canvas.bars");
  let idx = zeroIdx, playing = false, dirPlay = 1;

  function drawManip() {
    const c = wCanvas, d = window.devicePixelRatio || 1, w = c.clientWidth || 300, h = 120;
    c.style.height = h + "px"; c.width = w * d; c.height = h * d;
    const ctx = c.getContext("2d"); ctx.setTransform(d, 0, 0, d, 0, 0);
    const L = 44, R = 8, T = 22, B = 18, pw = w - L - R, ph = h - T - B;
    const wmax = 0.09;
    ctx.font = "11px Inter, system-ui, sans-serif"; ctx.fillStyle = css("--ink-2");
    ctx.fillText("manipulability along the self-motion", L, 14);
    ctx.strokeStyle = css("--grid");
    ctx.fillStyle = css("--ink-3"); ctx.textAlign = "right";
    for (const v of [0, 0.045, 0.09]) { const y = T + ph * (1 - v / wmax); ctx.beginPath(); ctx.moveTo(L, y); ctx.lineTo(L + pw, y); ctx.stroke(); ctx.fillText(v.toFixed(3), L - 6, y + 4); }
    ctx.textAlign = "left";
    ctx.strokeStyle = css("--accent"); ctx.lineWidth = 1.8; ctx.beginPath();
    path.forEach((p, i) => { const x = L + (pw * i) / (path.length - 1), y = T + ph * (1 - p.w / wmax); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
    ctx.stroke();
    const x = L + (pw * idx) / (path.length - 1);
    ctx.fillStyle = css("--ink"); ctx.beginPath(); ctx.arc(x, T + ph * (1 - path[idx].w / wmax), 4, 0, 2 * Math.PI); ctx.fill();
    ctx.fillStyle = css("--ink-3"); ctx.fillText("one limit", L, h - 4); ctx.textAlign = "right"; ctx.fillText("the other", L + pw, h - 4); ctx.textAlign = "left";
  }

  function update() {
    const q = path[idx].q;
    arm.setQ(q); arm.kinematics(); arm.jacobian(J);
    robot.update(arm.d); triad.set(arm.pos(), arm.rot());
    const e = arm.poseError(p0, R0);
    drawManip();
    drawJointBars(barCanvas, jointItems(arm, q), { title: "joint positions within their limits" });
    readout(root, [["hand position error", `${fmt(Math.hypot(e[0], e[1], e[2]) * 1000, 3)} mm`], ["orientation error", `${fmt(Math.hypot(e[3], e[4], e[5]) * DEG, 3)}°`], ["manipulability", fmt(path[idx].w, 4)]]);
    sc.requestRender();
  }
  slider.addEventListener("input", () => { idx = parseInt(slider.value, 10); playing = false; playBtn.textContent = "Sweep"; update(); });
  const playBtn = root.querySelector('button[name="play"]');
  playBtn.addEventListener("click", () => { playing = !playing; playBtn.textContent = playing ? "Stop" : "Sweep"; });
  update();
  return {
    tick(dt) {
      if (playing) {
        idx += dirPlay * Math.max(1, Math.round(dt * 200));
        if (idx >= path.length - 1) { idx = path.length - 1; dirPlay = -1; }
        if (idx <= 0) { idx = 0; dirPlay = 1; }
        slider.value = String(idx);
        update();
      }
      sc.render();
    },
    redraw: update,
  };
}

// ================================================================ Figure 3: drift under pseudo-inverse IK
function figDrift(root) {
  const OFF = 0.6;
  const sc = new Scene3D(root.querySelector(".view"), { camera: [2.15, -0.25, 0.95], target: [0.2, 0, 0.4] });
  const sims = ["plain", "posture"].map((mode, k) => {
    const arm = newArm();
    const offset = [0, k === 0 ? -OFF : OFF, 0];
    const robot = new Robot(MJ, arm.m, PARTS, { offset });
    sc.scene.add(robot.group);
    const circle = sc.line(css("--ink-3"), 200, { loop: true, dashed: true, opacity: 0.7 });
    const trail = sc.line(k === 0 ? css("--warn") : css("--accent"), 600, { opacity: 0.9 });
    const target = sc.sphere(0.014, css("--ink"));
    return { mode, arm, robot, circle, trail, target, offset, trailPts: [], J: new Float64Array(42) };
  });
  const barsEl = [root.querySelector("canvas.bars-a"), root.querySelector("canvas.bars-b")];
  const r = 0.2;
  let period = 2, speed = 1, t, state;

  function reset() {
    t = 0;
    sims.forEach((s) => {
      s.arm.setQ(READY); s.arm.kinematics();
      s.R = s.arm.rot();
      s.c = [0.307, 0, 0.64];
      s.loopStarts = []; s.err = 0; s.errLoop = 0; s.trailPts = [];
      const pts = [];
      for (let i = 0; i < 100; i++) {
        const a = (2 * Math.PI * i) / 100;
        pts.push([s.c[0] + s.offset[0], s.c[1] + r * Math.sin(a) + s.offset[1], s.c[2] + r * (Math.cos(a) - 1)]);
      }
      s.circle.set(pts);
    });
    state = { loop: 0 };
  }

  function step(dt) {
    const w = (2 * Math.PI) / period;
    const p = (s) => [s.c[0], s.c[1] + r * Math.sin(w * t), s.c[2] + r * (Math.cos(w * t) - 1)];
    const v = [0, r * w * Math.cos(w * t), -r * w * Math.sin(w * t), 0, 0, 0];
    for (const s of sims) {
      const pt = p(s);
      s.arm.kinematics(); s.arm.jacobian(s.J);
      const dq = F.ikVelocity(s.arm, s.J, pt, s.R, { gain: 20, lambda: 0.01, vFeedforward: v, posture: s.mode === "posture" ? READY : null, kNull: 2 });
      s.arm.setQ(s.arm.q().map((x, i) => clamp(x + dq[i] * dt, s.arm.lo[i], s.arm.hi[i])));
      s.arm.kinematics();
      const e = s.arm.poseError(pt, s.R);
      s.err = Math.hypot(e[0], e[1], e[2]);
      s.errLoop = Math.max(s.errLoop, s.err);
      s.pt = pt;
    }
    t += dt;
    const loop = Math.floor(t / period);
    if (loop > state.loop) {
      state.loop = loop;
      for (const s of sims) { s.loopStarts.push(s.arm.q()); s.lastLoopErr = s.errLoop; s.errLoop = 0; }
    }
  }

  function draw() {
    sims.forEach((s, k) => {
      s.robot.update(s.arm.d);
      s.target.position.set(s.pt[0] + s.offset[0], s.pt[1] + s.offset[1], s.pt[2] + s.offset[2]);
      const h = s.arm.pos();
      s.trailPts.push([h[0] + s.offset[0], h[1] + s.offset[1], h[2] + s.offset[2]]);
      if (s.trailPts.length > 600) s.trailPts.shift();
      s.trail.set(s.trailPts);
      drawJointBars(barsEl[k], jointItems(s.arm, s.arm.q()), { title: k === 0 ? "pseudo-inverse only" : "with null-space posture" });
    });
    const drift = (s) => {
      if (s.loopStarts.length < 2) return 0;
      const a = s.loopStarts[0], b = s.loopStarts[s.loopStarts.length - 1];
      return Math.max(...a.map((x, i) => Math.abs(x - b[i]))) * DEG;
    };
    const [a, b] = sims;
    readout(root, [
      ["laps", String(state.loop)],
      ["drift since lap 1", `${fmt(drift(a), 0)}° | ${fmt(drift(b), 0)}°`],
      ["hand error now", `${fmt(a.err * 1000, 1)} | ${fmt(b.err * 1000, 1)} mm`],
      ["worst, last lap", `${fmt((a.lastLoopErr ?? 0) * 1000, 1)} | ${fmt((b.lastLoopErr ?? 0) * 1000, 1)} mm`],
    ]);
    sc.requestRender();
  }

  seg(root, "period", (v) => { period = parseFloat(v); reset(); });
  seg(root, "speed", (v) => { speed = parseFloat(v); });
  root.querySelector('button[name="reset"]').addEventListener("click", reset);
  reset();
  step(0.002);
  draw();
  let acc = 0;
  return {
    tick(dt) {
      acc += dt * speed;
      let n = Math.min(Math.floor(acc / 0.002), 200);
      acc -= n * 0.002;
      while (n-- > 0) step(0.002);
      draw();
      sc.render();
    },
    redraw: draw,
  };
}

// ================================================================ Figure 4: dynamics
function figDynamics(root) {
  const arm = newArm();
  arm.m.opt.timestep = 0.001;
  const sc = new Scene3D(root.querySelector(".view"), { camera: [1.3, -1.25, 0.8], target: [0.3, 0, 0.35] });
  const robot = new Robot(MJ, arm.m, PARTS);
  sc.scene.add(robot.group);
  const load = sc.sphere(0.04, css("--warn"));
  const pid = bodyId(arm.m, "payload");
  const armature = Array.from(arm.m.dof_armature);
  const POSES = { ready: READY, home: HOME, reach: [0, 0.6, 0, -0.9, 0, 1.5, 0.785], stretched: [0, Math.PI / 2 - 1e-4, 0, -0.1169, 0, 1.6, 0.785] };
  const q = READY.slice();
  let withArmature = true;
  const sl2 = root.querySelector('input[name="dq2"]'), sl4 = root.querySelector('input[name="dq4"]');

  function update() {
    arm.setQ(q, [0, 0, 0, 0, 0, 0, 0]); arm.forward();
    robot.update(arm.d);
    const mp = arm.m.body_mass[pid];
    load.visible = mp > 0;
    const pp = arm.d.xipos;
    load.position.set(pp[3 * pid], pp[3 * pid + 1], pp[3 * pid + 2]);
    load.scale.setScalar(0.6 + 0.4 * Math.cbrt(mp));
    const M = arm.massMatrix();
    if (!withArmature) for (let i = 0; i < 7; i++) M[i * 8] -= armature[i];
    drawHeatmap(root.querySelector("canvas.heat"), Array.from(M), 7, { title: `mass matrix M(q), kg·m²${withArmature ? "" : " (links only)"}`, vmax: 3.5 });
    const g = arm.bias();
    drawJointBars(root.querySelector("canvas.grav"), g.map((x, i) => ({
      label: `q${i + 1}`, value: Math.abs(x) / arm.tauMax[i], lo: 0, hi: 1.25, warn: Math.abs(x) > arm.tauMax[i],
      text: `${Math.abs(x).toFixed(1)} / ${arm.tauMax[i]}`,
    })), { title: "gravity torque / limit (N·m)", mode: "fill", limitAt: 1, rightW: 64 });
    const share = Array.from({ length: 7 }, (_, i) => armature[i] / arm.massMatrix()[i * 8]);
    readout(root, [["flange height", `${fmt(arm.pos()[2], 3)} m`], ["wrist motor share (q5–q7)", share.slice(4).map((s) => `${(s * 100).toFixed(0)}%`).join(", ")], ["worst |τ|/limit", `${fmt(Math.max(...g.map((x, i) => Math.abs(x) / arm.tauMax[i])) * 100, 0)}%`]]);
    sc.requestRender();
  }
  seg(root, "pose", (v) => {
    POSES[v].forEach((x, i) => (q[i] = x));
    sl2.value = (q[1] * DEG).toFixed(0); sl4.value = (q[3] * DEG).toFixed(0);
    sl2.dispatchEvent(new Event("input")); sl4.dispatchEvent(new Event("input"));
  });
  range(root, "dq2", (v) => { q[1] = clamp(v / DEG, arm.lo[1], arm.hi[1]); update(); return `${v.toFixed(0)}°`; });
  range(root, "dq4", (v) => { q[3] = clamp(v / DEG, arm.lo[3], arm.hi[3]); update(); return `${v.toFixed(0)}°`; });
  range(root, "payload", (v) => { arm.m.body_mass[pid] = v; update(); return `${v.toFixed(2)} kg`; });
  root.querySelector('input[name="armature"]').addEventListener("change", (e) => { withArmature = e.target.checked; update(); });
  update();
  return { tick() { sc.render(); }, redraw: update };
}

// ================================================================ Figure 5: torque control
function figControl(root) {
  const plant = newArm(), ctrl = newArm(), ik = newArm();
  plant.m.opt.timestep = 0.001;
  const sc = new Scene3D(root.querySelector(".view"), { camera: [1.35, -1.1, 0.85], target: [0.3, 0, 0.38] });
  const robot = new Robot(MJ, plant.m, PARTS);
  sc.scene.add(robot.group);
  const targetBall = sc.sphere(0.016, css("--accent"));
  const trail = sc.line(css("--accent"), 800, { opacity: 0.75 });
  const circleLine = sc.line(css("--ink-3"), 200, { loop: true, dashed: true, opacity: 0.6 });
  const pushArrow = sc.arrow(css("--warn"));
  const pid = bodyId(plant.m, "payload"), link7 = bodyId(plant.m, "fr3_link7"), link4 = bodyId(plant.m, "fr3_link4");
  const errPlot = new TimePlot(root.querySelector("canvas.err"), { title: "hand error (mm)", auto: true, symmetric: false, minSpan: 5, maxSpan: 500, window: 6, height: 130 });
  const tauPlot = new TimePlot(root.querySelector("canvas.tau"), {
    title: "joint torque / limit", min: -1.2, max: 1.2, window: 6, height: 150,
    hlines: [{ v: 1, color: "--warn" }, { v: -1, color: "--warn" }],
    series: JOINT_COLORS.map((c) => ({ color: c, alpha: 0.9 })),
  });
  errPlot.o.series = [{ color: "--warn" }];

  const GAINS = {
    ci: { kp: 600, kr: 30, zeta: 1, posture: READY, kNull: 30, dNull: 3.8, lambda: 0.01 },
    osc: { wp: 20, wr: 20, zeta: 1, posture: READY, kNull: 30, dNull: 3.8, lambda: 0.01 },
    ji: { kq: [600, 600, 600, 600, 250, 150, 50], dq: [50, 50, 50, 50, 30, 25, 15] },
  };
  const C = [0.45, 0, 0.45], RAD = 0.15;
  const cfg = { kind: "osc", motion: "circle", nullspace: true, period: 3 };
  const J = new Float64Array(42), Jik = new Float64Array(42);
  let t, tMotion, Rd, trailPts, push, peakErr, elbowRef;

  const READY_HAND = [0.307, 0, 0.59];
  let ramp;
  function reset() {
    plant.d.qvel.fill(0); plant.setQ(READY, [0, 0, 0, 0, 0, 0, 0]); plant.forward();
    ik.setQ(READY);
    Rd = plant.rot(); t = 0; tMotion = 0; trailPts = []; push = null; peakErr = 0;
    elbowRef = null;
    ramp = { from: READY_HAND, t0: 0 };
    errPlot.clear(); tauPlot.clear();
    const pts = [];
    for (let i = 0; i < 100; i++) { const a = (2 * Math.PI * i) / 100; pts.push([C[0], C[1] + RAD * Math.sin(a), C[2] + RAD * (Math.cos(a) - 1)]); }
    circleLine.set(pts);
  }
  // Switching between Circle and Hold blends from the current target over 1 s.
  function setMotion(v) {
    if (lastTarget) ramp = { from: lastTarget.p.slice(), t0: t };
    cfg.motion = v; tMotion = 0; elbowRef = null; peakErr = 0;
  }
  const ramping = () => t - ramp.t0 < 1;
  function target() {
    const goal = cfg.motion === "hold" ? READY_HAND : circleAt(0).p;
    if (ramping()) {
      const s = t - ramp.t0, b = 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5;
      return { p: ramp.from.map((x, i) => x + (goal[i] - x) * b), R: Rd };
    }
    return cfg.motion === "hold" ? { p: READY_HAND, R: Rd } : circleAt(tMotion);
  }
  function circleAt(tt) {
    const w = (2 * Math.PI) / cfg.period;
    return {
      p: [C[0], C[1] + RAD * Math.sin(w * tt), C[2] + RAD * (Math.cos(w * tt) - 1)],
      v: [0, RAD * w * Math.cos(w * tt), -RAD * w * Math.sin(w * tt), 0, 0, 0],
      a: [0, -RAD * w * w * Math.sin(w * tt), -RAD * w * w * Math.cos(w * tt), 0, 0, 0],
      R: Rd,
    };
  }

  let lastTau = new Array(7).fill(0), lastTarget = null, sample = 0;
  function step() {
    const dt = 0.001;
    const tg = target();
    lastTarget = tg;
    ctrl.setQ(plant.q(), plant.qd()); ctrl.forward(); ctrl.jacobian(J);
    const g = Object.assign({}, GAINS[cfg.kind]);
    if (!cfg.nullspace && g.posture) g.posture = null;
    let tau;
    if (cfg.kind === "ci") tau = F.cartesianImpedance(ctrl, J, tg, g);
    else if (cfg.kind === "osc") tau = F.operationalSpace(ctrl, J, tg, g);
    else {
      ik.kinematics(); ik.jacobian(Jik);
      const dq = F.ikVelocity(ik, Jik, tg.p, Rd, { gain: 50, lambda: 0.01, vFeedforward: tg.v, posture: READY, kNull: 2 });
      ik.setQ(ik.q().map((x, i) => x + dq[i] * dt));
      tau = F.jointImpedance(ctrl, ik.q(), g);
    }
    lastTau = tau;
    plant.d.ctrl.set(F.saturate(tau, plant.tauMax));
    plant.d.xfrc_applied.fill(0);
    if (push && t < push.until) plant.d.xfrc_applied[6 * push.body + 1] = push.f;
    MJ.mj_step(plant.m, plant.d);
    if (!Number.isFinite(plant.d.qpos[0])) reset();
    t += dt;
    if (!ramping() && cfg.motion === "circle") tMotion += dt;
    sample += dt;
    if (sample >= 0.01) {
      sample = 0;
      plant.kinematics();
      const h = plant.pos();
      const err = Math.hypot(h[0] - tg.p[0], h[1] - tg.p[1], h[2] - tg.p[2]) * 1000;
      if (!ramping()) peakErr = Math.max(peakErr * 0.999, err);
      errPlot.push(t, [err]);
      tauPlot.push(t, lastTau.map((x, i) => clamp(x / plant.tauMax[i], -1.3, 1.3)));
      trailPts.push(h);
      if (trailPts.length > 400) trailPts.shift();
    }
  }

  function draw() {
    plant.kinematics();
    robot.update(plant.d);
    if (lastTarget) targetBall.position.set(...lastTarget.p);
    trail.set(trailPts);
    circleLine.obj.visible = cfg.motion === "circle";
    if (push && t < push.until) {
      const b = push.body, x = plant.d.xpos;
      pushArrow.visible = true;
      setArrow(pushArrow, [x[3 * b], x[3 * b + 1] - 0.25, x[3 * b + 2]], [0, 1, 0], 0.22);
    } else pushArrow.visible = false;
    errPlot.draw(t); tauPlot.draw(t);
    const e = plant.d.xpos;
    const el = [e[3 * link4], e[3 * link4 + 1], e[3 * link4 + 2]];
    if (!elbowRef && cfg.motion === "hold" && t - ramp.t0 > 1.5) elbowRef = el;
    if (cfg.motion !== "hold") elbowRef = null;
    const h = plant.pos();
    readout(root, [
      ["hand error", `${fmt(Math.hypot(h[0] - lastTarget.p[0], h[1] - lastTarget.p[1], h[2] - lastTarget.p[2]) * 1000, 1)} mm`],
      ["recent peak", `${fmt(peakErr, 1)} mm`],
      ["elbow moved", elbowRef ? `${fmt(Math.hypot(el[0] - elbowRef[0], el[1] - elbowRef[1], el[2] - elbowRef[2]) * 1000, 0)} mm` : "— (hold mode)"],
      ["worst |τ|/limit", `${fmt(Math.max(...lastTau.map((x, i) => Math.abs(x) / plant.tauMax[i])) * 100, 0)}%`],
      ["sim time", `${fmt(t, 1)} s`],
    ]);
    sc.requestRender();
  }

  seg(root, "kind", (v) => { cfg.kind = v; ik.setQ(plant.q()); peakErr = 0; });
  seg(root, "motion", setMotion);
  range(root, "payload", (v) => { plant.m.body_mass[pid] = v; return `${v.toFixed(2)} kg`; });
  root.querySelector('input[name="nullspace"]').addEventListener("change", (e) => { cfg.nullspace = e.target.checked; });
  root.querySelector('button[name="push-hand"]').addEventListener("click", () => { push = { body: link7, f: 30, until: t + 0.5 }; });
  root.querySelector('button[name="push-elbow"]').addEventListener("click", () => { push = { body: link4, f: 20, until: t + 0.5 }; });
  root.querySelector('button[name="reset"]').addEventListener("click", reset);
  reset();
  step();
  draw();
  let acc = 0;
  return {
    tick(dt) {
      acc += dt;
      let n = Math.min(Math.floor(acc / 0.001), 40);
      acc -= n * 0.001;
      if (acc > 0.05) acc = 0;
      while (n-- > 0) step();
      draw();
      sc.render();
    },
    redraw: draw,
  };
}

// ================================================================ Figure 6: integrators
function figIntegrators(root) {
  const m = MJ.MjModel.from_xml_string(XML_P);
  const d = new MJ.MjData(m);
  const sc = new Scene3D(root.querySelector(".view"), { camera: [1.05, -0.85, 0.8], target: [0.3, 0, 0.42] });
  const robot = new Robot(MJ, m, PARTS);
  sc.scene.add(robot.group);
  const plot = new TimePlot(root.querySelector("canvas.vel"), {
    title: "joint velocities (rad/s)", auto: true, symmetric: true, minSpan: 0.5, maxSpan: 10, window: 3, height: 170,
    series: JOINT_COLORS.map((c) => ({ color: c, alpha: 0.9 })),
  });
  const cfg = { integ: "euler", dt: 0.002 };
  let t, nextStep, sign, hist;

  function reset() {
    MJ.mj_resetData(m, d);
    m.opt.integrator = cfg.integ === "euler" ? MJ.mjtIntegrator.mjINT_EULER.value : MJ.mjtIntegrator.mjINT_IMPLICITFAST.value;
    m.opt.timestep = cfg.dt;
    d.qpos.set(READY); d.ctrl.set(READY);
    t = 0; nextStep = 0.5; sign = 1; hist = [];
    plot.clear();
  }
  function step() {
    if (t >= nextStep) {
      d.ctrl.set(READY.map((x, i) => x + sign * (i === 3 || i === 5 ? -0.2 : 0.2)));
      sign = -sign; nextStep += 1.5;
    }
    const prev = Array.from(d.qvel);
    MJ.mj_step(m, d);
    t += cfg.dt;
    const f = d.actuator_force;
    const atLimit = Array.from({ length: 7 }, (_, j) => Math.abs(f[j]) >= (j < 4 ? 87 : 12) - 1e-6);
    const flips = Array.from({ length: 7 }, (_, j) => Math.sign(d.qvel[j]) !== Math.sign(prev[j]) && Math.abs(d.qvel[j]) > 1e-3);
    hist.push({ t, atLimit, flips });
    while (hist.length && hist[0].t < t - 0.5) hist.shift();
    plot.push(t, Array.from(d.qvel));
  }
  function draw() {
    robot.update(d);
    plot.draw(t);
    const n = hist.length || 1;
    const limitPct = Math.max(...Array.from({ length: 7 }, (_, j) => hist.filter((h) => h.atLimit[j]).length / n)) * 100;
    const flipRate = Math.max(...Array.from({ length: 7 }, (_, j) => hist.filter((h) => h.flips[j]).length)) / 0.5;
    readout(root, [["at a force limit (last 0.5 s)", `${fmt(limitPct, 0)}%`], ["velocity sign flips", `${fmt(flipRate, 0)} /s`], ["sim time", `${fmt(t, 1)} s`]]);
    sc.requestRender();
  }
  seg(root, "integ", (v) => { cfg.integ = v; reset(); });
  seg(root, "dt", (v) => { cfg.dt = parseFloat(v) / 1000; reset(); });
  root.querySelector('button[name="reset"]').addEventListener("click", reset);
  reset();
  draw();
  let acc = 0;
  return {
    tick(dtReal) {
      acc += dtReal;
      let n = Math.min(Math.floor(acc / cfg.dt), 100);
      acc -= n * cfg.dt;
      if (acc > 0.05) acc = 0;
      while (n-- > 0) step();
      draw();
      sc.render();
    },
    redraw: draw,
  };
}

// ---------------------------------------------------------------- boot

const FIGURES = {
  "fig-chain": figChain,
  "fig-selfmotion": figSelfMotion,
  "fig-drift": figDrift,
  "fig-dynamics": figDynamics,
  "fig-control": figControl,
  "fig-integrators": figIntegrators,
};

function setStatus(text, isError = false) {
  document.querySelectorAll(".fig-status").forEach((el) => {
    el.textContent = text;
    el.classList.toggle("error", isError);
    el.hidden = !text;
  });
}

async function boot() {
  if (!webglAvailable()) { setStatus("These figures need WebGL, which this browser has turned off.", true); return; }
  setStatus("Loading MuJoCo and the FR3 model…");
  try {
    const [mod, parts, xt, xp] = await Promise.all([
      import(MUJOCO_URL),
      loadMeshes(),
      fetch("models/fr3_torque.xml").then((r) => r.text()),
      fetch("models/fr3_position.xml").then((r) => r.text()),
    ]);
    MJ = await mod.default({ locateFile: (p) => new URL(p, MUJOCO_URL).href });
    PARTS = parts; XML_T = xt; XML_P = xp;
  } catch (e) {
    console.error(e);
    setStatus(`Could not load the simulator: ${e.message}. If you opened the file directly, serve the folder over HTTP instead.`, true);
    return;
  }
  setStatus("");
  const live = new Map();
  const visible = new Set();
  const io = new IntersectionObserver((entries) => {
    for (const en of entries) {
      const id = en.target.id;
      if (en.isIntersecting) {
        if (!live.has(id)) {
          try { live.set(id, FIGURES[id](en.target)); } catch (e) { console.error(e); en.target.querySelector(".fig-status").hidden = false; en.target.querySelector(".fig-status").textContent = `Figure failed: ${e.message}`; continue; }
        }
        visible.add(id);
      } else visible.delete(id);
    }
  }, { rootMargin: "200px" });
  Object.keys(FIGURES).forEach((id) => { const el = document.getElementById(id); if (el) io.observe(el); });
  const redrawAll = () => live.forEach((f) => f.redraw && f.redraw());
  document.addEventListener("themechange", redrawAll);
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", redrawAll);
  let last = 0;
  const frame = (now) => {
    const dt = last ? Math.min(0.05, (now - last) / 1000) : 0;
    last = now;
    for (const id of visible) live.get(id)?.tick(dt);
    requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);
}

boot();
