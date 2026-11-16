// Experiments behind the numbers quoted in the article. Uses js/fr3core.js, the
// same code the figures run, on MuJoCo's WebAssembly build.
//
//   npm install
//   node tools/experiments.mjs [kinematics|drift|dynamics|control|integrators]
import load from "@mujoco/mujoco";
import fs from "fs";
import * as F from "../js/fr3core.js";

const mj = await load();
const XML = fs.readFileSync(new URL("../models/fr3_torque.xml", import.meta.url), "utf8");
const XMLPOS = fs.readFileSync(new URL("../models/fr3_position.xml", import.meta.url), "utf8");
const HOME = [0, 0, 0, -1.57079, 0, 1.57079, -0.7853];
// Franka's usual "ready" pose (libfranka examples).
const READY = [0, -Math.PI / 4, 0, (-3 * Math.PI) / 4, 0, Math.PI / 2, Math.PI / 4];
const DEG = 180 / Math.PI;
const which = process.argv[2] || "all";
const f3 = (a) => a.map((x) => x.toFixed(3)).join(", ");

function newArm(xml = XML) {
  const m = mj.MjModel.from_xml_string(xml);
  return new F.FR3(mj, m);
}

// ------------------------------------------------------------------ kinematics
if (which === "all" || which === "kinematics") {
  const arm = newArm();
  arm.setQ(READY); arm.kinematics();
  const J = arm.jacobian();
  console.log("home flange pos", f3(arm.pos()), "R", f3(arm.rot()));
  console.log("home manipulability", arm.manipulability(J).toFixed(4));
  console.log("joint limits (deg)", arm.lo.map((l, i) => `[${(l * DEG).toFixed(1)}, ${(arm.hi[i] * DEG).toFixed(1)}]`).join(" "));

  // Self-motion: hold the home flange pose, push q3 toward each limit through the null space.
  const pd = arm.pos(), Rd = arm.rot();
  for (const dir of [+1, -1]) {
    arm.setQ(READY);
    let worstPos = 0, worstRot = 0, mMin = Infinity, mMax = 0, stopped = null;
    const dt = 0.01;
    for (let k = 0; k < 4000; k++) {
      arm.kinematics();
      arm.jacobian(J);
      const nullVel = [0, 0, dir * 0.5, 0, 0, 0, 0];
      const dq = F.ikVelocity(arm, J, pd, Rd, { gain: 5, lambda: 1e-3, nullVel });
      const q = arm.q();
      const next = q.map((x, i) => x + dq[i] * dt);
      const hit = next.findIndex((x, i) => x < arm.lo[i] || x > arm.hi[i]);
      if (hit >= 0) { stopped = { joint: hit + 1, q3: q[2] * DEG, q: q.map((x) => (x * DEG).toFixed(1)) }; break; }
      arm.setQ(next);
      arm.kinematics(); arm.jacobian(J);
      const e = arm.poseError(pd, Rd);
      worstPos = Math.max(worstPos, Math.hypot(e[0], e[1], e[2]));
      worstRot = Math.max(worstRot, Math.hypot(e[3], e[4], e[5]));
      const w = arm.manipulability(J); mMin = Math.min(mMin, w); mMax = Math.max(mMax, w);
    }
    console.log(`self-motion dir ${dir}: stopped by joint ${stopped && stopped.joint} at q3=${stopped && stopped.q3.toFixed(1)} deg, q=${stopped && stopped.q}; ` +
      `max pos err ${(worstPos * 1000).toFixed(3)} mm, max rot err ${(worstRot * DEG).toFixed(3)} deg, manip ${mMin.toFixed(4)}..${mMax.toFixed(4)}`);
  }
}

// ------------------------------------------------------------------ cyclic drift
export function circleTarget(c, r, period, t) {
  const w = (2 * Math.PI) / period;
  return {
    p: [c[0], c[1] + r * Math.sin(w * t), c[2] + r * (Math.cos(w * t) - 1)],
    v: [0, r * w * Math.cos(w * t), -r * w * Math.sin(w * t), 0, 0, 0],
    a: [0, -r * w * w * Math.sin(w * t), -r * w * w * Math.cos(w * t), 0, 0, 0],
  };
}

if (which === "all" || which === "drift") {
  // Same setup as the article's drift figure: 0.2 m circle in the y–z plane, 2 s per lap,
  // IK at 500 Hz, joints clamped at their limits.
  const J = new Float64Array(42);
  for (const mode of ["plain", "posture"]) {
    const arm = newArm();
    arm.setQ(READY); arm.kinematics();
    const Rd = arm.rot(), c = [0.307, 0, 0.64], r = 0.2, period = 2, dt = 0.002;
    const lapStarts = [], contacts = [];
    let worst = 0, firstBad = null;
    for (let k = 0; k < (20 * period) / dt; k++) {
      const t = k * dt, w = (2 * Math.PI) / period;
      const p = [c[0], c[1] + r * Math.sin(w * t), c[2] + r * (Math.cos(w * t) - 1)];
      const v = [0, r * w * Math.cos(w * t), -r * w * Math.sin(w * t), 0, 0, 0];
      arm.kinematics(); arm.jacobian(J);
      const dq = F.ikVelocity(arm, J, p, Rd, { gain: 20, lambda: 0.01, vFeedforward: v, posture: mode === "posture" ? READY : null, kNull: 2 });
      const raw = arm.q().map((x, i) => x + dq[i] * dt);
      raw.forEach((x, i) => {
        if ((x < arm.lo[i] || x > arm.hi[i]) && !contacts.some((c) => c.joint === i + 1)) contacts.push({ joint: i + 1, t: +t.toFixed(2) });
      });
      arm.setQ(raw.map((x, i) => Math.min(arm.hi[i], Math.max(arm.lo[i], x))));
      arm.kinematics();
      const e = arm.poseError(p, Rd), err = Math.hypot(e[0], e[1], e[2]);
      if (t > 0.5) { worst = Math.max(worst, err); if (err > 0.01 && firstBad === null) firstBad = t; }
      if ((k + 1) % Math.round(period / dt) === 0) lapStarts.push(arm.q());
    }
    const dev = (a, b) => Math.max(...a.map((x, i) => Math.abs(x - b[i]))) * DEG;
    console.log(`${mode}: worst error ${(worst * 1000).toFixed(1)} mm; error first > 10 mm at ${firstBad === null ? "never" : firstBad.toFixed(1) + " s (lap " + (Math.floor(firstBad / period) + 1) + ")"}; ` +
      `change in lap-end pose vs lap 1 after laps 2,5,10,20: ${[1, 4, 9, 19].map((i) => dev(lapStarts[i], lapStarts[0]).toFixed(1)).join(", ")} deg; ` +
      `first limit contacts: ${contacts.length ? contacts.map((c) => `q${c.joint}@${c.t}s`).join(" ") : "none"}`);
  }
}

// ------------------------------------------------------------------ dynamics
if (which === "all" || which === "dynamics") {
  const arm = newArm();
  const poses = {
    home: HOME,
    ready: READY,
    reach: [0, 0.6, 0, -0.9, 0, 1.5, 0.785],
    stretched: [0, Math.PI / 2 - 0.0001, 0, -0.1169, 0, 1.6, 0.785],
  };
  const arm0 = arm.m.dof_armature;
  for (const [name, q] of Object.entries(poses)) {
    arm.setQ(q, [0, 0, 0, 0, 0, 0, 0]); arm.forward();
    const M = arm.massMatrix();
    const diag = Array.from({ length: 7 }, (_, i) => M[i * 8]);
    const g = arm.bias();
    console.log(`${name}: flange ${f3(arm.pos())}; diag M ${diag.map((x) => x.toFixed(3)).join(" ")}`);
    console.log(`   armature share of diag: ${diag.map((x, i) => ((arm0[i] / x) * 100).toFixed(0) + "%").join(" ")}`);
    console.log(`   gravity torque ${g.map((x) => x.toFixed(2)).join(" ")} N·m; |τ|/limit ${g.map((x, i) => ((Math.abs(x) / arm.tauMax[i]) * 100).toFixed(0) + "%").join(" ")}`);
    const off = M.reduce((s, x, k) => (Math.floor(k / 7) !== k % 7 ? Math.max(s, Math.abs(x)) : s), 0);
    console.log(`   largest off-diagonal ${off.toFixed(3)}`);
  }
  // Payload check: setting body_mass at runtime should change gravity torques by m·g·lever.
  const pid = mj.mj_name2id(arm.m, mj.mjtObj.mjOBJ_BODY.value, "payload");
  arm.setQ(poses.stretched, [0, 0, 0, 0, 0, 0, 0]); arm.forward();
  const g0 = arm.bias();
  arm.m.body_mass[pid] = 1.0; arm.forward();
  const g1 = arm.bias();
  const pp = Array.from(arm.d.xipos.subarray(3 * pid, 3 * pid + 3));
  console.log(`payload 1 kg at ${f3(pp)}: Δτ2 = ${(g1[1] - g0[1]).toFixed(3)} N·m (expect ≈ g·x = ${(9.81 * pp[0]).toFixed(3)})`);
  // Largest payload the stretched pose can hold statically.
  for (let mp = 0; mp <= 6; mp += 0.25) {
    arm.m.body_mass[pid] = mp; arm.forward();
    const g = arm.bias();
    const worst = Math.max(...g.map((x, i) => Math.abs(x) / arm.tauMax[i]));
    if (worst > 1) { console.log(`stretched pose: torque limit exceeded at payload ${mp} kg (worst joint ${g.map((x, i) => Math.abs(x) / arm.tauMax[i]).indexOf(worst) + 1})`); break; }
  }
  arm.m.body_mass[pid] = 0;
}

// ------------------------------------------------------------------ control
// Shared simulation harness, used by the article's control figure as well.
export function makeSim(xml = XML) {
  const plant = new F.FR3(mj, mj.MjModel.from_xml_string(xml));
  const ctrl = new F.FR3(mj, mj.MjModel.from_xml_string(xml));
  plant.m.opt.timestep = 0.001;
  return { plant, ctrl, payloadId: mj.mj_name2id(plant.m, mj.mjtObj.mjOBJ_BODY.value, "payload") };
}

const GAINS = {
  ci: { kp: 600, kr: 30, zeta: 1, posture: READY, kNull: 30, dNull: 3.8, lambda: 0.01 },
  osc: { wp: 20, wr: 20, zeta: 1, posture: READY, kNull: 30, dNull: 3.8, lambda: 0.01 },
  ji: { kq: [600, 600, 600, 600, 250, 150, 50], dq: [50, 50, 50, 50, 30, 25, 15] },
};

function runControl({ kind, payload = 0, push = null, nullspace = true, seconds = 9, period = 3, r = 0.15 }) {
  const { plant, ctrl, payloadId } = makeSim();
  plant.m.body_mass[payloadId] = payload;
  plant.setQ(READY, [0, 0, 0, 0, 0, 0, 0]); plant.forward();
  const Rd = plant.rot();
  const c = [0.45, 0, 0.45];
  // joint-space reference for joint impedance: kinematic IK with posture, run alongside
  const ik = new F.FR3(mj, mj.MjModel.from_xml_string(XML));
  ik.setQ(READY);
  const J = new Float64Array(42), Jik = new Float64Array(42);
  const g = Object.assign({}, GAINS[kind]);
  if (!nullspace) g.posture = null;
  const dt = plant.m.opt.timestep;
  let errMax = 0, errSum = 0, n = 0, tauPeak = new Array(7).fill(0), devPush = 0, nullDrift = 0;
  const steps = Math.round(seconds / dt);
  for (let k = 0; k < steps; k++) {
    const t = k * dt;
    // ramp onto the circle over the first second, then go round
    const w = (2 * Math.PI) / period, tt = Math.max(0, t - 1);
    const ramp = Math.min(1, t);
    const start = [c[0], c[1], c[2]];
    let target;
    if (t < 1) {
      const s = 10 * ramp ** 3 - 15 * ramp ** 4 + 6 * ramp ** 5;
      const p = [0.307 + (start[0] - 0.307) * s, 0, 0.59 + (start[2] - 0.59) * s];
      target = { p, R: Rd };
    } else {
      target = {
        p: [c[0], c[1] + r * Math.sin(w * tt), c[2] + r * (Math.cos(w * tt) - 1)],
        v: [0, r * w * Math.cos(w * tt), -r * w * Math.sin(w * tt), 0, 0, 0],
        a: [0, -r * w * w * Math.sin(w * tt), -r * w * w * Math.cos(w * tt), 0, 0, 0],
        R: Rd,
      };
    }
    // controller sees the nominal model at the plant's state
    ctrl.setQ(plant.q(), plant.qd()); ctrl.forward(); ctrl.jacobian(J);
    let tau;
    if (kind === "ci") tau = F.cartesianImpedance(ctrl, J, target, g);
    else if (kind === "osc") tau = F.operationalSpace(ctrl, J, target, g);
    else {
      ik.kinematics(); ik.jacobian(Jik);
      const dq = F.ikVelocity(ik, Jik, target.p, Rd, { gain: 50, lambda: 0.01, vFeedforward: target.v, posture: READY, kNull: 2 });
      ik.setQ(ik.q().map((x, i) => x + dq[i] * dt));
      tau = F.jointImpedance(ctrl, ik.q(), g);
    }
    tauPeak = tauPeak.map((x, i) => Math.max(x, Math.abs(tau[i]) / plant.tauMax[i]));
    plant.d.ctrl.set(F.saturate(tau, plant.tauMax));
    plant.d.xfrc_applied.fill(0);
    if (push && t >= push.t && t < push.t + push.dur) {
      const b = mj.mj_name2id(plant.m, mj.mjtObj.mjOBJ_BODY.value, "fr3_link7");
      plant.d.xfrc_applied.set(push.f, 6 * b);
    }
    mj.mj_step(plant.m, plant.d);
    if (t > 1 + period) {
      plant.kinematics();
      const p = plant.pos();
      const e = Math.hypot(p[0] - target.p[0], p[1] - target.p[1], p[2] - target.p[2]);
      if (!push) { errMax = Math.max(errMax, e); errSum += e * e; n++; }
      else if (t >= push.t && t < push.t + push.dur + 1.0) devPush = Math.max(devPush, e);
      nullDrift = Math.max(nullDrift, ...plant.q().map((x, i) => Math.abs(x - READY[i])));
    }
  }
  return { errMax, errRms: Math.sqrt(errSum / Math.max(1, n)), tauPeak, devPush, nullDrift };
}

if (which === "all" || which === "control") {
  const pct = (a) => a.map((x) => (x * 100).toFixed(0) + "%").join(" ");
  for (const kind of ["ji", "ci", "osc"]) {
    for (const payload of [0, 1]) {
      const r = runControl({ kind, payload });
      console.log(`${kind} payload ${payload} kg: tracking max ${(r.errMax * 1000).toFixed(1)} mm, rms ${(r.errRms * 1000).toFixed(1)} mm; peak |τ|/limit ${pct(r.tauPeak)}`);
    }
    const p = runControl({ kind, push: { t: 5, dur: 0.5, f: [0, 30, 0, 0, 0, 0] }, seconds: 7 });
    console.log(`${kind} 30 N push for 0.5 s: peak deviation ${(p.devPush * 1000).toFixed(1)} mm`);
  }
  for (const kind of ["ci", "osc"]) for (const ns of [true, false]) {
    const r = elbowPush({ kind, nullspace: ns });
    console.log(`${kind} null space ${ns ? "on " : "off"}: elbow push 20 N, 0.5 s -> hand peak ${(r.handPeak * 1000).toFixed(1)} mm, ` +
      `elbow peak ${(r.elbowPeak * 1000).toFixed(0)} mm, elbow 3 s later ${(r.elbowAfter * 1000).toFixed(0)} mm`);
  }
}

// Hold the ready pose and push sideways on the elbow (link 4). The hand should hold;
// with null-space control the elbow should come back.
function elbowPush({ kind, nullspace, force = 20 }) {
  const { plant, ctrl } = makeSim();
  plant.setQ(READY, [0, 0, 0, 0, 0, 0, 0]); plant.forward();
  const target = { p: plant.pos(), R: plant.rot() };
  const g = Object.assign({}, GAINS[kind]);
  if (!nullspace) g.posture = null;
  const elbow = mj.mj_name2id(plant.m, mj.mjtObj.mjOBJ_BODY.value, "fr3_link4");
  const e0 = Array.from(plant.d.xpos.subarray(3 * elbow, 3 * elbow + 3));
  const J = new Float64Array(42);
  const dt = plant.m.opt.timestep;
  let handPeak = 0, elbowPeak = 0, elbowAfter = 0;
  for (let k = 0; k < Math.round(5 / dt); k++) {
    const t = k * dt;
    ctrl.setQ(plant.q(), plant.qd()); ctrl.forward(); ctrl.jacobian(J);
    const tau = kind === "ci" ? F.cartesianImpedance(ctrl, J, target, g) : F.operationalSpace(ctrl, J, target, g);
    plant.d.ctrl.set(F.saturate(tau, plant.tauMax));
    plant.d.xfrc_applied.fill(0);
    if (t >= 1 && t < 1.5) plant.d.xfrc_applied[6 * elbow + 1] = force;
    mj.mj_step(plant.m, plant.d);
    plant.kinematics();
    const p = plant.pos();
    handPeak = Math.max(handPeak, Math.hypot(p[0] - target.p[0], p[1] - target.p[1], p[2] - target.p[2]));
    const e = plant.d.xpos.subarray(3 * elbow, 3 * elbow + 3);
    const de = Math.hypot(e[0] - e0[0], e[1] - e0[1], e[2] - e0[2]);
    elbowPeak = Math.max(elbowPeak, de);
    elbowAfter = de;
  }
  return { handPeak, elbowPeak, elbowAfter };
}

// ------------------------------------------------------------------ integrators
// Menagerie's position actuators (kp up to 4500, kv up to 450) under Euler vs implicitfast.
// Step every joint target by +0.2 rad from the ready pose and watch for blow-ups.
export function positionStep(integrator, timestep, seconds = 2) {
  const m = mj.MjModel.from_xml_string(XMLPOS);
  const d = new mj.MjData(m);
  m.opt.integrator = integrator === "euler" ? mj.mjtIntegrator.mjINT_EULER.value : mj.mjtIntegrator.mjINT_IMPLICITFAST.value;
  m.opt.timestep = timestep;
  d.qpos.set(READY);
  d.ctrl.set(READY.map((x, i) => x + (i === 3 ? 0.2 : i === 5 ? -0.2 : 0.2)));
  let maxV = 0, unstable = null, settle = null;
  const goal = Array.from(d.ctrl);
  for (let k = 0; k < Math.round(seconds / timestep); k++) {
    const t0 = d.time;
    mj.mj_step(m, d);
    const v = Math.max(...Array.from(d.qvel).map(Math.abs));
    if (!Number.isFinite(v) || v > 100 || d.time < t0) { unstable = t0; break; }
    maxV = Math.max(maxV, v);
    const err = Math.max(...Array.from(d.qpos).map((x, i) => Math.abs(x - goal[i])));
    if (settle === null && err < 0.01) settle = d.time;
    if (err >= 0.01) settle = null;
  }
  const r = { maxV, unstable, settle, finalErr: Math.max(...Array.from(d.qpos).map((x, i) => Math.abs(x - goal[i]))) };
  d.delete(); m.delete();
  return r;
}

if (which === "all" || which === "integrators") {
  for (const integ of ["euler", "implicitfast"]) for (const ts of [0.0005, 0.001, 0.002, 0.004, 0.008]) {
    const r = positionStep(integ, ts);
    console.log(`${integ} dt=${ts * 1000} ms: ${r.unstable !== null ? `UNSTABLE at t=${r.unstable.toFixed(3)} s` : `stable, max |qvel| ${r.maxV.toFixed(2)} rad/s, settled (<0.01 rad) at ${r.settle && r.settle.toFixed(3)} s, final err ${(r.finalErr * 1000).toFixed(1)} mrad`}`);
  }
}
