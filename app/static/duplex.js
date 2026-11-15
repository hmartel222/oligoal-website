import * as THREE from "./vendor/three.module.js";
import {OrbitControls} from "./vendor/OrbitControls.js";

const strandColors = {wing: 0x68dbb0, core: 0x8db9ff, rna: 0xf7b06d};
const elementColors = {N: 0x659cff, O: 0xff6d79, P: 0xffa350, S: 0xffd779};
const radii = {C: 1.70, N: 1.55, O: 1.52, P: 1.80, S: 1.80};
const up = new THREE.Vector3(0, 1, 0);

function parsePdb(source) {
  const atoms = [];
  const bySerial = new Map();
  const links = [];
  const pairs = [];
  for (const line of source.split("\n")) {
    if (line.startsWith("ATOM  ")) {
      const atom = {
        name: line.slice(12, 16).trim(),
        kind: line.slice(17, 20).trim(),
        residue: Number(line.slice(22, 26)),
        chain: line[21],
        element: line.slice(76, 78).trim(),
        xyz: new THREE.Vector3(...[30, 38, 46].map((i) => Number(line.slice(i, i + 8)))),
      };
      bySerial.set(Number(line.slice(6, 11)), atoms.length);
      atoms.push(atom);
    } else if (line.startsWith("CONECT")) {
      links.push([Number(line.slice(6, 11)), Number(line.slice(11, 16))]);
    } else if (line.startsWith("REMARK   3 PAIR")) {
      pairs.push(line.slice(15).trim().split(/ +/).map(Number));
    }
  }
  const indices = ([a, b]) => [bySerial.get(a), bySerial.get(b)];
  if (!atoms.length || !links.length) throw new Error("The duplex has no atomic coordinates.");
  return {atoms, bonds: links.map(indices), pairs: pairs.map(indices)};
}

function dispose(group) {
  const geometries = new Set(), materials = new Set();
  group.traverse((object) => {
    if (object.geometry) geometries.add(object.geometry);
    if (object.material) materials.add(object.material);
    if (object.isInstancedMesh) object.dispose();
  });
  geometries.forEach((geometry) => geometry.dispose());
  materials.forEach((material) => material.dispose());
  group.clear();
}

function tracePoint(edge, index, count, direction) {
  const t = index / Math.max(1, count);
  const stretch = Math.min(108, 7 + Math.sqrt(count) * 1.1);
  const along = t * stretch;
  const angle = t * Math.PI * (count < 60 ? 1.1 : 3.2);
  const spread = Math.sin(Math.PI * t / 2) * Math.min(49, 4 + Math.sqrt(count) * .65);
  return new THREE.Vector3(
    edge.x + Math.sin(angle) * spread,
    edge.y + direction * Math.sin(angle * .9) * spread * .72,
    edge.z + direction * (along + Math.cos(angle) * spread * .32),
  );
}

export class DuplexView {
  constructor(canvas, onPick) {
    this.canvas = canvas;
    this.onPick = onPick;
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(34, 1, .1, 4000);
    this.camera.up.set(0, .89, .45).normalize();
    this.duplex = new THREE.Group();
    this.trace = new THREE.Group();
    this.scene.add(this.trace, this.duplex, new THREE.HemisphereLight(0xd9f7ff, 0x50546d, 2.9));
    for (const [position, color, intensity] of [
      [[-30, 30, 70], 0xfff3dd, 3],
      [[-30, -35, -15], 0x7fbfff, 1.8],
    ]) {
      const light = new THREE.DirectionalLight(color, intensity);
      light.position.set(...position);
      this.scene.add(light);
    }
    this.renderer = new THREE.WebGLRenderer({canvas, alpha: true, antialias: true});
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.8));
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.minDistance = 3;
    this.controls.maxDistance = 1500;
    this.controls.zoomSpeed = .9;
    this.controls.target.set(0, 0, 0);
    this.controls.addEventListener("change", () => this.draw());
    this.observer = new ResizeObserver(() => this.resize());
    this.observer.observe(canvas);
    this.animation = 0;
    this.setMotion(!matchMedia("(prefers-reduced-motion: reduce)").matches);
    this.mode = "binding";
    this.style = "sticks";
    this.transform = new THREE.Object3D();
    this.pick = new THREE.Raycaster();
    this.pointer = new THREE.Vector2();
    this.drag = null;
    canvas.addEventListener("pointerdown", (event) => {
      this.drag = {x: event.clientX, y: event.clientY};
    });
    canvas.addEventListener("pointerup", (event) => {
      if (event.button === 0 && this.drag && Math.hypot(event.clientX - this.drag.x, event.clientY - this.drag.y) < 5) {
        this.selectAtom(event);
      }
      this.drag = null;
    });
    canvas.addEventListener("keydown", (event) => this.keyboard(event));
  }

  resize() {
    const {clientWidth, clientHeight} = this.canvas;
    if (!clientWidth || !clientHeight) return;
    this.renderer.setSize(clientWidth, clientHeight, false);
    this.camera.aspect = clientWidth / clientHeight;
    this.camera.updateProjectionMatrix();
    this.draw();
  }

  clear() {
    this.structure = null;
    this.atoms = null;
    dispose(this.duplex);
    dispose(this.trace);
    this.draw();
  }

  load(pdb, candidate, length) {
    this.clear();
    this.structure = parsePdb(pdb);
    this.start = candidate.start;
    this.end = candidate.end;
    this.length = Math.max(this.end, length);
    this.mode = "binding";
    this.buildDuplex();
    this.buildTrace();
    this.resize();
    this.reset();
  }

  color(atom) {
    return new THREE.Color(elementColors[atom.element] ?? (
      atom.chain === "B" ? strandColors.rna : ["MOE", "OME"].includes(atom.kind) ? strandColors.wing : strandColors.core
    ));
  }

  buildDuplex() {
    dispose(this.duplex);
    const {atoms, bonds, pairs} = this.structure;
    const material = new THREE.MeshStandardMaterial({roughness: .52, metalness: .04});
    this.atoms = new THREE.InstancedMesh(new THREE.SphereGeometry(1, 14, 10), material, atoms.length);
    this.atoms.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.atoms.frustumCulled = false;
    const color = (index) => this.color(atoms[index]);

    atoms.forEach((atom, index) => {
      this.atoms.setColorAt(index, color(index));
    });
    this.duplex.add(this.atoms);
    this.bonds = new THREE.InstancedMesh(new THREE.CylinderGeometry(1, 1, 1, 10), material, bonds.length * 2);
    this.bonds.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.bonds.frustumCulled = false;
    bonds.forEach(([a, b], i) => {
      this.bonds.setColorAt(i * 2, color(a));
      this.bonds.setColorAt(i * 2 + 1, color(b));
    });
    this.duplex.add(this.bonds);

    const segments = [];
    for (const [a, b] of pairs) {
      if (atoms[a].xyz.distanceTo(atoms[b].xyz) <= 3.85) segments.push(atoms[a].xyz, atoms[b].xyz);
    }
    const lines = new THREE.BufferGeometry().setFromPoints(segments);
    this.pairs = new THREE.LineSegments(lines, new THREE.LineDashedMaterial({
      color: 0xded8ad, dashSize: .24, gapSize: .22, transparent: true, opacity: .53,
    }));
    this.pairs.computeLineDistances();
    this.duplex.add(this.pairs);
    this.updateGeometry(0);
  }

  buildTrace() {
    dispose(this.trace);
    const rna = this.structure.atoms.filter((atom) => atom.chain === "B");
    const first = rna.find((atom) => atom.residue === 1 && atom.name === "O5'").xyz;
    const last = rna.find((atom) => atom.residue === this.end - this.start && atom.name === "O3'").xyz;
    const overall = [...rna.map((atom) => atom.xyz)];
    for (const [edge, count, direction] of [[first, this.start, -1], [last, this.length - this.end, 1]]) {
      if (!count) continue;
      const positions = new Float32Array((count + 1) * 3);
      for (let i = 0; i <= count; i++) {
        const position = tracePoint(edge, i, count, direction);
        position.toArray(positions, i * 3);
        if (i % Math.max(1, Math.floor(count / 100)) === 0 || i === count) overall.push(position);
      }
      const path = new THREE.BufferGeometry();
      path.setAttribute("position", new THREE.BufferAttribute(positions, 3));
      this.trace.add(new THREE.Line(path, new THREE.LineBasicMaterial({
        color: strandColors.rna, transparent: true, opacity: .74,
      })));
      const samples = Math.min(130, count);
      const points = Array.from({length: samples + 1}, (_, i) => tracePoint(edge, i * count / samples, count, direction));
      const tubing = new THREE.TubeGeometry(new THREE.CatmullRomCurve3(points), Math.min(200, count + 5), .44, 7);
      this.trace.add(new THREE.Mesh(tubing, new THREE.MeshStandardMaterial({color: strandColors.rna, roughness: .61})));
      const marker = new THREE.Mesh(new THREE.SphereGeometry(1.22, 12, 8),
        new THREE.MeshStandardMaterial({color: strandColors.rna}));
      marker.position.copy(points.at(-1));
      this.trace.add(marker);
    }
    this.overall = overall;
  }

  updateGeometry(time) {
    const {atoms, bonds} = this.structure;
    const transform = this.transform;
    const offset = (atom) => {
      const {xyz} = atom, z = xyz.z;
      const wave = (z + 22) * .17;
      return new THREE.Vector3(
        xyz.x + Math.sin(wave + time * 4.2) * .18 * Math.sin(time * 3.9),
        xyz.y + Math.cos(wave + time * 3.1) * .18 * Math.sin(time * 4.3),
        z + Math.sin(wave + time * 4.6) * .08 * Math.sin(time * 3.7),
      );
    };
    const points = atoms.map(offset);
    const size = this.style === "space" ? .94 : .25;
    this.bonds.visible = this.style !== "space";
    this.pairs.visible = this.style !== "space";
    this.atoms.visible = true;
    for (const [index, atom] of atoms.entries()) {
      transform.position.copy(points[index]);
      transform.quaternion.identity();
      transform.scale.setScalar(size * (radii[atom.element] ?? 1.65));
      transform.updateMatrix();
      this.atoms.setMatrixAt(index, transform.matrix);
    }
    this.atoms.instanceMatrix.needsUpdate = true;
    if (this.bonds.visible) {
      const position = new THREE.Vector3(), direction = new THREE.Vector3();
      const segment = (a, b, index) => {
        direction.subVectors(b, a);
        const length = direction.length();
        transform.position.copy(a).addScaledVector(direction, .5);
        transform.quaternion.setFromUnitVectors(up, direction.normalize());
        transform.scale.set(.119, length, .119);
        transform.updateMatrix();
        this.bonds.setMatrixAt(index, transform.matrix);
      };
      bonds.forEach(([a, b], index) => {
        position.copy(points[a]).lerp(points[b], .5);
        segment(points[a], position, index * 2);
        segment(position, points[b], index * 2 + 1);
      });
      this.bonds.instanceMatrix.needsUpdate = true;
    }
  }

  setStyle(style) {
    this.style = style;
    if (this.structure) this.updateGeometry(0);
    this.draw();
  }

  setMotion(enabled) {
    this.motion = enabled;
    if (enabled && !this.animation) {
      const animate = (time) => {
        if (!this.motion) {
          this.animation = 0;
          return;
        }
        if (this.structure && this.canvas.getClientRects().length && !document.hidden && time - this.lastFrame >= 32) {
          this.lastFrame = time;
          this.updateGeometry(time / 1000);
          this.draw();
        }
        this.animation = requestAnimationFrame(animate);
      };
      this.lastFrame = 0;
      this.animation = requestAnimationFrame(animate);
    } else if (!enabled && this.structure) {
      this.updateGeometry(0);
      this.draw();
    }
  }

  setMode(mode) {
    this.mode = mode;
    this.reset();
  }

  reset() {
    if (!this.structure) return;
    const points = this.mode === "transcript" ? this.overall : this.structure.atoms.map((atom) => atom.xyz);
    const center = new THREE.Box3().setFromPoints(points).getCenter(this.controls.target);
    const direction = new THREE.Vector3(1, .35, .04).normalize();
    const right = new THREE.Vector3().crossVectors(this.camera.up, direction).normalize();
    const vertical = new THREE.Vector3().crossVectors(direction, right);
    const tanVertical = Math.tan(THREE.MathUtils.degToRad(this.camera.fov / 2));
    const tanHorizontal = tanVertical * this.camera.aspect;
    const delta = new THREE.Vector3();
    let radius = 9;
    for (const point of points) {
      delta.subVectors(point, center);
      const projected = Math.max(Math.abs(right.dot(delta)) / tanHorizontal, Math.abs(vertical.dot(delta)) / tanVertical);
      radius = Math.max(radius, projected + delta.dot(direction));
    }
    this.camera.position.copy(center).addScaledVector(direction, radius * 1.20 + 4);
    this.controls.update();
    this.draw();
  }

  draw() {
    if (this.camera.aspect && this.canvas.clientHeight) this.renderer.render(this.scene, this.camera);
  }

  zoom(factor) {
    this.camera.position.sub(this.controls.target).multiplyScalar(factor).add(this.controls.target);
    this.controls.update();
    this.draw();
  }

  keyboard(event) {
    if (["+", "=", "-"].includes(event.key)) {
      event.preventDefault();
      this.zoom(event.key === "-" ? 1.23 : .81);
      return;
    }
    const angles = {ArrowLeft: [-.14, 0], ArrowRight: [.14, 0], ArrowUp: [0, -.14], ArrowDown: [0, .14]};
    const movement = angles[event.key];
    if (!movement) return;
    event.preventDefault();
    const delta = this.camera.position.clone().sub(this.controls.target);
    const spherical = new THREE.Spherical().setFromVector3(delta);
    spherical.theta += movement[0];
    spherical.phi += movement[1];
    this.camera.position.setFromSpherical(spherical.makeSafe()).add(this.controls.target);
    this.controls.update();
  }

  selectAtom(event) {
    if (!this.atoms) return;
    const rect = this.canvas.getBoundingClientRect();
    this.pointer.set(2 * (event.clientX - rect.left) / rect.width - 1, 1 - 2 * (event.clientY - rect.top) / rect.height);
    this.pick.setFromCamera(this.pointer, this.camera);
    this.atoms.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 29);
    const [hit] = this.pick.intersectObject(this.atoms);
    if (!hit) return;
    const atom = this.structure.atoms[hit.instanceId];
    this.onPick(atom);
  }
}
