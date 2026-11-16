// viewer.js — three.js rendering of the FR3, driven by MuJoCo body poses.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

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

export class Robot {
  constructor(mj, model, parts, opts = {}) {
    this.group = new THREE.Group();
    this.bodies = new Map();
    const ghost = !!opts.ghost;
    for (const part of parts) {
      const id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY.value, part.body);
      if (id < 0) continue;
      if (!this.bodies.has(id)) {
        const g = new THREE.Group();
        g.matrixAutoUpdate = true;
        this.bodies.set(id, g);
        this.group.add(g);
      }
      const color = new THREE.Color().setRGB(...part.color, THREE.SRGBColorSpace);
      if (opts.tint) color.lerp(new THREE.Color(opts.tint), opts.tintAmount ?? 0.5);
      const mat = new THREE.MeshStandardMaterial({
        color, roughness: 0.6, metalness: 0.0, side: THREE.DoubleSide,
        transparent: ghost, opacity: ghost ? opts.opacity ?? 0.22 : 1, depthWrite: !ghost,
      });
      const mesh = new THREE.Mesh(part.geometry, mat);
      mesh.castShadow = !ghost;
      mesh.receiveShadow = false; // self-shadowing on these thin shells causes acne
      this.bodies.get(id).add(mesh);
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
}

// ---------------------------------------------------------------- scene

export class Scene3D {
  constructor(container, opts = {}) {
    this.container = container;
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    container.appendChild(this.renderer.domElement);

    this.scene = new THREE.Scene();
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

    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x6f6d66, 1.1));
    const sun = new THREE.DirectionalLight(0xffffff, 2.0);
    const fill = new THREE.DirectionalLight(0xffffff, 0.5);
    fill.position.set(-1.5, 1.2, 1.0);
    this.scene.add(fill);
    sun.position.set(1.5, -1.0, 3.0);
    sun.castShadow = true;
    sun.shadow.mapSize.set(1024, 1024);
    Object.assign(sun.shadow.camera, { left: -1.5, right: 1.5, top: 1.5, bottom: -1.5, near: 0.5, far: 8 });
    sun.shadow.bias = -0.0005;
    sun.shadow.normalBias = 0.02;
    this.scene.add(sun);

    this.floor = new THREE.Mesh(
      new THREE.CircleGeometry(opts.floorRadius || 6, 96),
      new THREE.MeshStandardMaterial({ roughness: 1, metalness: 0 })
    );
    this.floor.receiveShadow = true;
    this.floor.position.z = -0.001;
    this.scene.add(this.floor);
    this.grid = new THREE.PolarGridHelper(1.5, 8, 6, 64);
    this.grid.rotation.x = Math.PI / 2;
    this.scene.add(this.grid);

    this.applyTheme();
    document.addEventListener("themechange", () => this.applyTheme());
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => this.applyTheme());

    this.needsRender = true;
    new ResizeObserver(() => this.resize()).observe(container);
    this.resize();
  }
  applyTheme() {
    this.scene.background = new THREE.Color(css("--bg-sunken"));
    this.floor.material.color = new THREE.Color(css("--floor"));
    const gc = new THREE.Color(css("--grid"));
    this.grid.material.color = gc;
    this.grid.material.vertexColors = false;
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
