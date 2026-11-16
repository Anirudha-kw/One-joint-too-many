// viewer.js — three.js rendering of the FR3, driven by MuJoCo body poses.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";

const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

// ---------------------------------------------------------------- meshes

let meshPromise = null;

// Loads assets/fr3_meshes.{json,bin} (written by tools/build_web_assets.py) once.
export function loadMeshes(base = "assets/") {
  if (!meshPromise) {
    // The binary either sits next to the manifest (fr3_meshes.bin) or is embedded in it as base64
    // (manifest.binBase64), for hosts that don't serve .bin files.
    meshPromise = fetch(base + "fr3_meshes.json").then((r) => r.json()).then((manifest) =>
      (manifest.binBase64
        ? Promise.resolve(Uint8Array.from(atob(manifest.binBase64), (c) => c.charCodeAt(0)).buffer)
        : fetch(base + "fr3_meshes.bin").then((r) => r.arrayBuffer())
      ).then((buf) => [manifest, buf])
    ).then(([manifest, buf]) =>
      manifest.parts.map((p) => {
        const qv = new Uint16Array(buf, p.positionOffset, p.vertexCount * 3);
        const idx = new Uint16Array(buf, p.indexOffset, p.indexCount);
        const pos = new Float32Array(p.vertexCount * 3);
        for (let i = 0; i < p.vertexCount; i++) {
          for (let k = 0; k < 3; k++) {
            pos[3 * i + k] = p.min[k] + (qv[3 * i + k] / 65535) * (p.max[k] - p.min[k]);
          }
        }
        const g = new THREE.BufferGeometry();
        g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
        g.setIndex(new THREE.BufferAttribute(idx, 1));
        g.computeVertexNormals();
        return { body: p.body, color: p.color, geometry: g };
      })
    );
  }
  return meshPromise;
}

// ---------------------------------------------------------------- robot

// Schematic motor per joint: a small puck centred
// on the real joint axis (the body's z axis through its origin). Menagerie's FR3 has no internal
// geometry, so the sizes are illustrative.
// z moves a motor along its own axis, so motors whose axes cross at one point don't overlap.
const DRIVES = {
  fr3_link1: { r: 0.03, len: 0.04, z: -0.09 }, fr3_link2: { r: 0.03, len: 0.04, z: 0 },
  fr3_link3: { r: 0.028, len: 0.036, z: -0.09 }, fr3_link4: { r: 0.028, len: 0.036, z: 0 },
  fr3_link5: { r: 0.022, len: 0.03, z: -0.15 }, fr3_link6: { r: 0.022, len: 0.03, z: 0 }, fr3_link7: { r: 0.02, len: 0.026, z: 0 },
};

const driveCache = new Map();
function driveGroup(spec, colors, ghost) {
  const key = `${spec.r}|${spec.len}`;
  if (!driveCache.has(key)) {
    const can = new THREE.CylinderGeometry(spec.r, spec.r, spec.len, 40);
    can.rotateX(Math.PI / 2);
    driveCache.set(key, { can });
  }
  const geo = driveCache.get(key);
  const mat = (color, rough) => new THREE.MeshStandardMaterial({
    color, roughness: rough, metalness: 0.1, envMapIntensity: 0.8,
    transparent: ghost, opacity: ghost ? 0.1 : 1, depthWrite: !ghost,
  });
  const g = new THREE.Group();
  const can = new THREE.Mesh(geo.can, mat(colors.motor, 0.5));
  can.position.z = spec.z;
  g.add(can);
  return g;
}

export class Robot {
  // opts.style: "xray" (translucent shell, solid motors; the default) or "solid".
  constructor(mj, model, parts, opts = {}) {
    this.group = new THREE.Group();
    this.bodies = new Map();
    const ghost = !!opts.ghost;
    const xray = (opts.style || "xray") === "xray";
    const ensureBody = (id) => {
      if (!this.bodies.has(id)) {
        const g = new THREE.Group();
        this.bodies.set(id, g);
        this.group.add(g);
      }
      return this.bodies.get(id);
    };
    for (const part of parts) {
      const id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY.value, part.body);
      if (id < 0) continue;
      const color = new THREE.Color().setRGB(...part.color, THREE.SRGBColorSpace);
      if (xray) color.lerp(new THREE.Color(0xb9b7b0), 0.55);
      const translucent = ghost || xray;
      const opacity = ghost ? (opts.opacity ?? 0.1) : 0.22;
      const mat = new THREE.MeshStandardMaterial({
        color, roughness: 0.5, metalness: 0.0, side: THREE.DoubleSide, envMapIntensity: 0.7,
        transparent: translucent, opacity: translucent ? opacity : 1, depthWrite: !translucent,
      });
      mat.userData.base = color.clone();
      mat.userData.opacity = mat.opacity;
      const mesh = new THREE.Mesh(part.geometry, mat);
      mesh.castShadow = !ghost;
      mesh.receiveShadow = false; // self-shadowing on these thin shells causes acne
      mesh.renderOrder = 1;
      ensureBody(id).add(mesh);
    }
    if (xray) {
      const colors = { motor: 0x2a2b2f };
      for (const [name, spec] of Object.entries(DRIVES)) {
        const id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY.value, name);
        if (id >= 0) ensureBody(id).add(driveGroup(spec, colors, ghost));
      }
    }
    if (opts.offset) this.group.position.set(...opts.offset);
  }
  // Copy body poses from MuJoCo data (positions and quaternions, world frame).
  update(data) {
    const xp = data.xpos, xq = data.xquat;
    for (const [id, g] of this.bodies) {
      g.position.set(xp[3 * id], xp[3 * id + 1], xp[3 * id + 2]);
      g.quaternion.set(xq[4 * id + 1], xq[4 * id + 2], xq[4 * id + 3], xq[4 * id]);
    }
  }
  // Copy poses from a saved snapshot { xpos, xquat } (Float64Arrays).
  updateFrom(snap) { this.update(snap); }
  // Tint one body's meshes toward `color` (null restores the original colors).
  highlight(bodyId, color, amount = 0.75) {
    const g = this.bodies.get(bodyId);
    if (!g) return;
    g.traverse((o) => {
      if (!o.isMesh) return;
      const m = o.material;
      if (!m.userData.base) return;
      m.color.copy(m.userData.base);
      m.emissive.setRGB(0, 0, 0);
      m.opacity = m.userData.opacity;
      if (color) {
        const c = new THREE.Color(color);
        m.color.lerp(c, amount);
        m.emissive.copy(c).multiplyScalar(0.25);
        if (m.transparent) m.opacity = Math.max(m.opacity, 0.45);
      }
    });
  }
}

// ---------------------------------------------------------------- scene

export class Scene3D {
  constructor(container, opts = {}) {
    this.container = container;
    // Transparent canvas: the robot sits directly on the page background.
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setClearColor(0x000000, 0);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.NeutralToneMapping;
    this.renderer.toneMappingExposure = 0.8;
    container.appendChild(this.renderer.domElement);

    this.scene = new THREE.Scene();
    // Soft studio lighting from a generated room environment.
    const pmrem = new THREE.PMREMGenerator(this.renderer);
    this.scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
    pmrem.dispose();
    THREE.Object3D.DEFAULT_UP.set(0, 0, 1);
    this.camera = new THREE.PerspectiveCamera(opts.fov || 35, 1, 0.01, 50);
    this.camera.up.set(0, 0, 1);
    this.camera.position.set(...(opts.camera || [1.9, -1.6, 1.2]));
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.target.set(...(opts.target || [0.25, 0, 0.4]));
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.12;
    this.controls.minDistance = 0.4;
    this.controls.maxDistance = 6;
    this.controls.maxPolarAngle = Math.PI * 0.495;
    this.controls.addEventListener("change", () => this.requestRender());
    this.home = { camera: this.camera.position.clone(), target: this.controls.target.clone() };

    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x8a8880, 0.25));
    const sun = new THREE.DirectionalLight(0xffffff, 1.5);
    sun.position.set(0.6, -0.4, 3.0);
    sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    sun.shadow.radius = 6;
    Object.assign(sun.shadow.camera, { left: -1.5, right: 1.5, top: 1.5, bottom: -1.5, near: 0.5, far: 8 });
    sun.shadow.bias = -0.0005;
    sun.shadow.normalBias = 0.02;
    this.scene.add(sun);

    // No floor, only the shadow it would catch.
    this.floor = new THREE.Mesh(new THREE.PlaneGeometry(8, 8), new THREE.ShadowMaterial({ opacity: 0.12 }));
    this.floor.receiveShadow = true;
    this.floor.position.z = -0.001;
    this.scene.add(this.floor);

    this.applyTheme();
    document.addEventListener("themechange", () => this.applyTheme());
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => this.applyTheme());

    this.needsRender = true;
    new ResizeObserver(() => this.resize()).observe(container);
    this.resize();
  }
  applyTheme() {
    const bg = new THREE.Color(css("--bg"));
    const dark = bg.r + bg.g + bg.b < 1.2;
    this.floor.material.opacity = dark ? 0.28 : 0.13;
    this.requestRender();
  }
  resize() {
    const w = this.container.clientWidth, h = this.container.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.renderer.domElement.style.width = "100%";
    this.renderer.domElement.style.height = "100%";
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.requestRender();
  }
  resetView() {
    this.camera.position.copy(this.home.camera);
    this.controls.target.copy(this.home.target);
    this.controls.update();
    this.requestRender();
  }
  requestRender() { this.needsRender = true; }
  render(force = false) {
    this.controls.update();
    if (this.needsRender || force) {
      this.renderer.render(this.scene, this.camera);
      this.needsRender = false;
    }
  }

  // ---- overlays
  sphere(radius, color, opacity = 1) {
    const m = new THREE.Mesh(
      new THREE.SphereGeometry(radius, 24, 16),
      new THREE.MeshBasicMaterial({ color, transparent: opacity < 1, opacity, depthTest: opacity >= 1 })
    );
    this.scene.add(m);
    return m;
  }
  // A polyline with a fixed capacity; call set(points) with an array of [x,y,z].
  line(color, capacity = 2000, opts = {}) {
    const geom = new THREE.BufferGeometry();
    const pos = new Float32Array(capacity * 3);
    geom.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    geom.setDrawRange(0, 0);
    const mat = opts.dashed
      ? new THREE.LineDashedMaterial({ color, dashSize: 0.015, gapSize: 0.012, transparent: true, opacity: opts.opacity ?? 1 })
      : new THREE.LineBasicMaterial({ color, transparent: true, opacity: opts.opacity ?? 1 });
    const l = opts.loop ? new THREE.LineLoop(geom, mat) : new THREE.Line(geom, mat);
    this.scene.add(l);
    return {
      obj: l,
      set(points) {
        const n = Math.min(points.length, capacity);
        for (let i = 0; i < n; i++) pos.set(points[i], 3 * i);
        geom.attributes.position.needsUpdate = true;
        geom.setDrawRange(0, n);
        geom.computeBoundingSphere();
        if (opts.dashed) l.computeLineDistances();
      },
    };
  }
  // A coordinate triad (x red, y green, z blue).
  triad(size = 0.08) {
    const g = new THREE.Group();
    const cols = [0xd1495b, 0x2a9d8f, 0x3a7bd5];
    for (let k = 0; k < 3; k++) {
      const dir = new THREE.Vector3(k === 0 ? 1 : 0, k === 1 ? 1 : 0, k === 2 ? 1 : 0);
      g.add(new THREE.ArrowHelper(dir, new THREE.Vector3(), size, cols[k], size * 0.28, size * 0.16));
    }
    this.scene.add(g);
    return {
      obj: g,
      set(p, R) {
        g.position.set(p[0], p[1], p[2]);
        const m = new THREE.Matrix4().set(R[0], R[1], R[2], 0, R[3], R[4], R[5], 0, R[6], R[7], R[8], 0, 0, 0, 0, 1);
        g.quaternion.setFromRotationMatrix(m);
      },
    };
  }
  arrow(color) {
    const a = new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), new THREE.Vector3(), 0.2, color, 0.05, 0.03);
    a.visible = false;
    this.scene.add(a);
    return a;
  }
  // Joint-axis markers: a short arrow per joint along its axis.
  jointAxes(n, colors, length = 0.14) {
    const arrows = [];
    for (let i = 0; i < n; i++) {
      const a = new THREE.ArrowHelper(new THREE.Vector3(0, 0, 1), new THREE.Vector3(), length, colors[i], 0.035, 0.022);
      a.renderOrder = 10;
      a.line.material.depthTest = false; a.cone.material.depthTest = false;
      a.line.material.linewidth = 2;
      this.scene.add(a);
      arrows.push(a);
    }
    return {
      arrows,
      set(data) {
        const p = data.xanchor, ax = data.xaxis;
        arrows.forEach((a, i) => {
          const d = new THREE.Vector3(ax[3 * i], ax[3 * i + 1], ax[3 * i + 2]);
          a.position.set(p[3 * i] - d.x * length / 2, p[3 * i + 1] - d.y * length / 2, p[3 * i + 2] - d.z * length / 2);
          a.setDirection(d);
        });
      },
      visible(v) { arrows.forEach((a) => (a.visible = v)); },
    };
  }
}

// Point an ArrowHelper from `pos` along `dir` (need not be unit length).
export function setArrow(a, pos, dir, length) {
  const d = new THREE.Vector3(...dir).normalize();
  a.position.set(...pos);
  a.setDirection(d);
  a.setLength(length, Math.min(0.05, length * 0.3), 0.03);
}

export function webglAvailable() {
  try {
    const c = document.createElement("canvas");
    return !!(window.WebGLRenderingContext && (c.getContext("webgl2") || c.getContext("webgl")));
  } catch (e) {
    return false;
  }
}
