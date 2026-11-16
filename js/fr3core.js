// fr3core.js — kinematics, inverse kinematics and torque controllers for the FR3,
// built on MuJoCo's WebAssembly bindings. The page and the Node experiments
// (tools/experiments.mjs) both use this file, so the numbers quoted in the
// article come from the same code that runs the figures.
//
// Conventions: J is the 6×7 flange Jacobian, rows 0–2 translational and 3–5
// rotational (world frame). All matrices are row-major Float64Arrays.

export const N = 7;

// ---------------------------------------------------------------- small dense linear algebra

// In-place Cholesky of an n×n SPD matrix (lower triangle). Returns false if not SPD.
export function cholesky(A, n) {
  for (let j = 0; j < n; j++) {
    let s = A[j * n + j];
    for (let k = 0; k < j; k++) s -= A[j * n + k] ** 2;
    if (s <= 0) return false;
    const ljj = Math.sqrt(s);
    A[j * n + j] = ljj;
    for (let i = j + 1; i < n; i++) {
      let t = A[i * n + j];
      for (let k = 0; k < j; k++) t -= A[i * n + k] * A[j * n + k];
      A[i * n + j] = t / ljj;
    }
  }
  return true;
}

// Solve L Lᵀ x = b in place (b overwritten with x).
export function cholSolve(L, n, b) {
  for (let i = 0; i < n; i++) {
    let t = b[i];
    for (let k = 0; k < i; k++) t -= L[i * n + k] * b[k];
    b[i] = t / L[i * n + i];
  }
  for (let i = n - 1; i >= 0; i--) {
    let t = b[i];
    for (let k = i + 1; k < n; k++) t -= L[k * n + i] * b[k];
    b[i] = t / L[i * n + i];
  }
  return b;
}

// y = A x, A is r×c.
export function mulMV(A, r, c, x, y = new Float64Array(r)) {
  for (let i = 0; i < r; i++) {
    let s = 0;
    for (let j = 0; j < c; j++) s += A[i * c + j] * x[j];
    y[i] = s;
  }
  return y;
}

// y = Aᵀ x, A is r×c.
export function mulMtV(A, r, c, x, y = new Float64Array(c)) {
  y.fill(0);
  for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) y[j] += A[i * c + j] * x[i];
  return y;
}

// C = A W Aᵀ for A r×c and W c×c (W = null means identity). C is r×r.
export function sandwich(A, r, c, W, C = new Float64Array(r * r)) {
  const AW = W ? new Float64Array(r * c) : A;
  if (W) for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) {
    let s = 0;
    for (let k = 0; k < c; k++) s += A[i * c + k] * W[k * c + j];
    AW[i * c + j] = s;
  }
  for (let i = 0; i < r; i++) for (let j = 0; j <= i; j++) {
    let s = 0;
    for (let k = 0; k < c; k++) s += AW[i * c + k] * A[j * c + k];
    C[i * r + j] = s; C[j * r + i] = s;
  }
  return C;
}

// Inverse of an SPD matrix via Cholesky.
export function invSPD(A, n) {
  const L = Float64Array.from(A);
  if (!cholesky(L, n)) return null;
  const inv = new Float64Array(n * n), e = new Float64Array(n);
  for (let j = 0; j < n; j++) {
    e.fill(0); e[j] = 1;
    cholSolve(L, n, e);
    for (let i = 0; i < n; i++) inv[i * n + j] = e[i];
  }
  return inv;
}

// ---------------------------------------------------------------- rotations

export function mat2quat(R) {
  const t = R[0] + R[4] + R[8];
  let w, x, y, z;
  if (t > 0) {
    const s = Math.sqrt(t + 1) * 2;
    w = 0.25 * s; x = (R[7] - R[5]) / s; y = (R[2] - R[6]) / s; z = (R[3] - R[1]) / s;
  } else if (R[0] > R[4] && R[0] > R[8]) {
    const s = Math.sqrt(1 + R[0] - R[4] - R[8]) * 2;
    w = (R[7] - R[5]) / s; x = 0.25 * s; y = (R[1] + R[3]) / s; z = (R[2] + R[6]) / s;
  } else if (R[4] > R[8]) {
    const s = Math.sqrt(1 + R[4] - R[0] - R[8]) * 2;
    w = (R[2] - R[6]) / s; x = (R[1] + R[3]) / s; y = 0.25 * s; z = (R[5] + R[7]) / s;
  } else {
    const s = Math.sqrt(1 + R[8] - R[0] - R[4]) * 2;
    w = (R[3] - R[1]) / s; x = (R[2] + R[6]) / s; y = (R[5] + R[7]) / s; z = 0.25 * s;
  }
  return [w, x, y, z];
}

// Rotation vector (axis·angle, world frame) taking R to Rd: log(Rd Rᵀ).
export function rotError(R, Rd) {
  const E = new Float64Array(9);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) {
    let s = 0;
    for (let k = 0; k < 3; k++) s += Rd[i * 3 + k] * R[j * 3 + k];
    E[i * 3 + j] = s;
  }
  let [w, x, y, z] = mat2quat(E);
  if (w < 0) { w = -w; x = -x; y = -y; z = -z; }
  const sv = Math.hypot(x, y, z);
  if (sv < 1e-12) return [2 * x, 2 * y, 2 * z];
  const angle = 2 * Math.atan2(sv, w);
  return [(x / sv) * angle, (y / sv) * angle, (z / sv) * angle];
}

// ---------------------------------------------------------------- the arm

export class FR3 {
  constructor(mj, model, data) {
    this.mj = mj;
    this.m = model;
    this.d = data || new mj.MjData(model);
    this.site = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE.value, "flange");
    this.jacpBuf = new mj.DoubleBuffer(3 * N);
    this.jacrBuf = new mj.DoubleBuffer(3 * N);
    this.MBuf = new mj.DoubleBuffer(N * N);
    const r = model.jnt_range;
    this.lo = Array.from({ length: N }, (_, i) => r[2 * i]);
    this.hi = Array.from({ length: N }, (_, i) => r[2 * i + 1]);
    const c = model.actuator_ctrlrange;
    this.tauMax = Array.from({ length: N }, (_, i) => c[2 * i + 1]);
  }
  free() { this.jacpBuf.delete(); this.jacrBuf.delete(); this.MBuf.delete(); }

  setQ(q, qd) {
    this.d.qpos.set(q);
    if (qd) this.d.qvel.set(qd);
  }
  forward() { this.mj.mj_forward(this.m, this.d); }
  // Positions only: enough for site poses and Jacobians.
  kinematics() { this.mj.mj_kinematics(this.m, this.d); this.mj.mj_comPos(this.m, this.d); }

  q() { return Array.from(this.d.qpos); }
  qd() { return Array.from(this.d.qvel); }
  pos() { const p = this.d.site_xpos; const s = 3 * this.site; return [p[s], p[s + 1], p[s + 2]]; }
  rot() { const R = this.d.site_xmat; const s = 9 * this.site; return Array.from(R.subarray(s, s + 9)); }

  jacobian(J = new Float64Array(6 * N)) {
    this.mj.mj_jacSite(this.m, this.d, this.jacpBuf, this.jacrBuf, this.site);
    J.set(this.jacpBuf.GetView(), 0);
    J.set(this.jacrBuf.GetView(), 3 * N);
    return J;
  }
  massMatrix(M = new Float64Array(N * N)) {
    this.mj.mj_fullM(this.m, this.d, this.MBuf);
    M.set(this.MBuf.GetView());
    return M;
  }
  // Coriolis + gravity (needs forward() or the velocity stage to have run).
  bias() { return Array.from(this.d.qfrc_bias); }

  // 6-vector pose error from the current flange pose to (pd, Rd).
  poseError(pd, Rd) {
    const p = this.pos(), r = rotError(this.rot(), Rd);
    return [pd[0] - p[0], pd[1] - p[1], pd[2] - p[2], r[0], r[1], r[2]];
  }

  // Yoshikawa manipulability sqrt(det(J Jᵀ)) of the 6×7 Jacobian.
  manipulability(J) {
    const A = sandwich(J, 6, N, null);
    if (!cholesky(A, 6)) return 0;
    let p = 1;
    for (let i = 0; i < 6; i++) p *= A[i * 6 + i];
    return p;
  }

  // Normalised distance of each joint to its nearer limit: 0 at a limit, 1 at mid-range.
  limitMargins(q = this.q()) {
    return q.map((x, i) => {
      const mid = (this.lo[i] + this.hi[i]) / 2, half = (this.hi[i] - this.lo[i]) / 2;
      return 1 - Math.abs(x - mid) / half;
    });
  }
}

// ---------------------------------------------------------------- differential IK

// Damped least squares: returns { dq, A } with dq = Jᵀ (J Jᵀ + λ² I)⁻¹ v and the
// factored A, which nullProject() reuses.
export function dls(J, v, lambda) {
  const A = sandwich(J, 6, N, null);
  for (let i = 0; i < 6; i++) A[i * 6 + i] += lambda * lambda;
  cholesky(A, 6);
  const y = cholSolve(A, 6, Float64Array.from(v));
  return { dq: mulMtV(J, 6, N, y), A };
}

// Project a joint-space vector into the (approximate) null space of J:  (I − J⁺J) z.
export function nullProject(J, A, z) {
  const w = mulMV(J, 6, N, z);
  cholSolve(A, 6, w);
  const corr = mulMtV(J, 6, N, w);
  return z.map((x, i) => x - corr[i]);
}

// One IK step toward (pd, Rd). Options:
//   gain       pose-error feedback (1/s)
//   lambda     DLS damping
//   posture    null-space target posture (array) or null
//   kNull      null-space gain toward the posture (1/s)
//   nullVel    explicit null-space joint velocity to project (array) or null
// Returns joint velocities.
export function ikVelocity(arm, J, pd, Rd, o) {
  const e = arm.poseError(pd, Rd).map((x) => x * o.gain);
  if (o.vFeedforward) for (let i = 0; i < 6; i++) e[i] += o.vFeedforward[i];
  const { dq, A } = dls(J, e, o.lambda);
  let z = null;
  if (o.posture) {
    const q = arm.q();
    z = q.map((x, i) => o.kNull * (o.posture[i] - x));
  }
  if (o.nullVel) z = z ? z.map((x, i) => x + o.nullVel[i]) : o.nullVel.slice();
  if (z) {
    const n = nullProject(J, A, z);
    for (let i = 0; i < N; i++) dq[i] += n[i];
  }
  return dq;
}

// Solve IK to convergence from the arm's current configuration (kinematic only).
// Leaves the arm at the solution; returns { q, err } with err the final position error (m).
export function solveIK(arm, pd, Rd, o = {}) {
  const opts = Object.assign({ gain: 1, lambda: 0.02, iters: 300, dt: 0.1, posture: null, kNull: 0.5 }, o);
  const J = new Float64Array(6 * N);
  for (let it = 0; it < opts.iters; it++) {
    arm.kinematics();
    arm.jacobian(J);
    const dq = ikVelocity(arm, J, pd, Rd, opts);
    const q = arm.q().map((x, i) => Math.min(arm.hi[i], Math.max(arm.lo[i], x + dq[i] * opts.dt)));
    arm.setQ(q);
  }
  arm.kinematics();
  const e = arm.poseError(pd, Rd);
  return { q: arm.q(), err: Math.hypot(e[0], e[1], e[2]), rotErr: Math.hypot(e[3], e[4], e[5]) };
}

// ---------------------------------------------------------------- torque controllers
//
// Each takes the arm (plant state, already forwarded with the controller's model),
// the target, gains, and returns 7 joint torques before saturation.
//   target: { p, R, v? (6), a? (6) }
//   g: { kp, kr, zeta, posture, kNull, dNull, lambda }

function nullspaceTorque(arm, q, qd, g) {
  if (!g.posture) return new Array(N).fill(0);
  return q.map((x, i) => g.kNull * (g.posture[i] - x) - g.dNull * qd[i]);
}

// Cartesian impedance: τ = Jᵀ (K e − D ẋ) + c + g + (I − Jᵀ J⁺ᵀ) τ₀.
// D is chosen for damping ratio ζ against a unit mass, as in libfranka's example.
export function cartesianImpedance(arm, J, target, g) {
  const q = arm.q(), qd = arm.qd();
  const e = arm.poseError(target.p, target.R);
  const xd = mulMV(J, 6, N, qd);
  const v = target.v || [0, 0, 0, 0, 0, 0];
  const F = new Array(6);
  for (let i = 0; i < 6; i++) {
    const k = i < 3 ? g.kp : g.kr;
    F[i] = k * e[i] + 2 * g.zeta * Math.sqrt(k) * (v[i] - xd[i]);
  }
  const tau = mulMtV(J, 6, N, F);
  const bias = arm.bias();
  const t0 = nullspaceTorque(arm, q, qd, g);
  if (g.posture) {
    // (I − Jᵀ J⁺ᵀ) τ₀ with J⁺ = Jᵀ (J Jᵀ + λ²I)⁻¹
    const A = sandwich(J, 6, N, null);
    for (let i = 0; i < 6; i++) A[i * 6 + i] += g.lambda ** 2;
    cholesky(A, 6);
    const w = cholSolve(A, 6, mulMV(J, 6, N, t0));
    const proj = mulMtV(J, 6, N, w);
    for (let i = 0; i < N; i++) tau[i] += t0[i] - proj[i];
  }
  for (let i = 0; i < N; i++) tau[i] += bias[i];
  return tau;
}

// Operational-space control: τ = Jᵀ Λ (a_d + Kp e + Kd ė) + c + g + (I − Jᵀ J̄ᵀ) τ₀,
// with Λ = (J M⁻¹ Jᵀ)⁻¹ and J̄ = M⁻¹ Jᵀ Λ (dynamically consistent null space).
// Ignores the J̇q̇ term, which is small at these speeds.
export function operationalSpace(arm, J, target, g) {
  const q = arm.q(), qd = arm.qd();
  const M = arm.massMatrix();
  const Minv = invSPD(M, N);
  const Li = sandwich(J, 6, N, Minv);
  for (let i = 0; i < 6; i++) Li[i * 6 + i] += g.lambda ** 2;
  const Lam = invSPD(Li, 6);
  const e = arm.poseError(target.p, target.R);
  const xd = mulMV(J, 6, N, qd);
  const v = target.v || [0, 0, 0, 0, 0, 0], a = target.a || [0, 0, 0, 0, 0, 0];
  const acc = new Array(6);
  for (let i = 0; i < 6; i++) {
    const w = i < 3 ? g.wp : g.wr;
    acc[i] = a[i] + w * w * e[i] + 2 * g.zeta * w * (v[i] - xd[i]);
  }
  const F = mulMV(Lam, 6, 6, acc);
  const tau = mulMtV(J, 6, N, F);
  if (g.posture) {
    // Null-space accelerations shaped by M so they are dynamically consistent.
    const t0 = nullspaceTorque(arm, q, qd, g);
    const y = mulMV(Lam, 6, 6, mulMV(J, 6, N, mulMV(Minv, N, N, t0)));
    const proj = mulMtV(J, 6, N, y);
    for (let i = 0; i < N; i++) tau[i] += t0[i] - proj[i];
  }
  const bias = arm.bias();
  for (let i = 0; i < N; i++) tau[i] += bias[i];
  return tau;
}

// Joint impedance to a joint-space goal: τ = K (q* − q) − D q̇ + c + g.
export function jointImpedance(arm, qGoal, g) {
  const q = arm.q(), qd = arm.qd(), bias = arm.bias();
  return q.map((x, i) => g.kq[i] * (qGoal[i] - x) - g.dq[i] * qd[i] + bias[i]);
}

export function saturate(tau, tauMax) {
  return tau.map((t, i) => Math.max(-tauMax[i], Math.min(tauMax[i], t)));
}
