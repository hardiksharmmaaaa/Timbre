/* Timbre landing experience: spatial signal field + scroll choreography. */
(function () {
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const nav = document.getElementById("siteNav");

  function animateFavicon() {
    const favicon = document.getElementById("favicon");
    if (!favicon || reducedMotion) return;
    const canvas = document.createElement("canvas");
    canvas.width = 64;
    canvas.height = 64;
    const context = canvas.getContext("2d");
    if (!context) return;
    let frame = 0;
    const barX = [15, 24, 33, 42];
    const phases = [0, 1.7, 3.1, 4.5];

    const roundRect = (x, y, width, height, radius) => {
      context.beginPath();
      context.roundRect(x, y, width, height, radius);
      context.fill();
    };

    const renderIcon = () => {
      context.clearRect(0, 0, 64, 64);
      context.fillStyle = "#080808";
      roundRect(0, 0, 64, 64, 15);
      context.beginPath();
      context.arc(32, 32, 22 + Math.sin(frame * 0.16) * 2, 0, Math.PI * 2);
      context.strokeStyle = `rgba(255,92,26,${0.12 + Math.sin(frame * 0.16) * 0.04})`;
      context.lineWidth = 1;
      context.stroke();
      context.fillStyle = "#ff5c1a";
      barX.forEach((x, index) => {
        const height = 9 + (Math.sin(frame * 0.34 + phases[index]) + 1) * (7 + index % 2 * 3);
        roundRect(x, 32 - height / 2, 5, height, 2.5);
      });
      favicon.type = "image/png";
      favicon.href = canvas.toDataURL("image/png");
      frame += 1;
    };

    renderIcon();
    window.setInterval(renderIcon, 280);
  }

  animateFavicon();

  window.addEventListener("scroll", () => nav?.classList.toggle("scrolled", scrollY > 40), {
    passive: true,
  });

  const kineticStates = [
    { word: "tone.", pattern: "STEADY", confidence: 64, phrase: "“I’m fine.”" },
    { word: "rhythm.", pattern: "HESITANT", confidence: 58, phrase: "“Yeah, sure.”" },
    { word: "energy.", pattern: "BRIGHT", confidence: 71, phrase: "“Let’s do it.”" },
    { word: "doubt.", pattern: "UNCERTAIN", confidence: 52, phrase: "“Maybe.”" },
  ];
  const kineticWord = document.getElementById("kineticWord");
  const kineticGhost = document.querySelector(".kinetic-ghost");
  const scannerTone = document.getElementById("scannerTone");
  const scannerConfidence = document.getElementById("scannerConfidence");
  const scannerPhrase = document.getElementById("scannerPhrase");
  const scannerBar = document.getElementById("scannerBar");
  let kineticIndex = 0;

  function showKineticState(index) {
    const state = kineticStates[index];
    kineticWord?.classList.remove("is-switching");
    void kineticWord?.offsetWidth;
    kineticWord?.classList.add("is-switching");
    window.setTimeout(() => {
      if (kineticWord) kineticWord.textContent = state.word;
      if (kineticGhost) kineticGhost.textContent = state.word;
      if (scannerTone) scannerTone.textContent = state.pattern;
      if (scannerConfidence) scannerConfidence.textContent = `${state.confidence}% / ESTIMATE`;
      if (scannerPhrase) scannerPhrase.textContent = state.phrase;
      if (scannerBar) scannerBar.style.width = `${state.confidence}%`;
    }, reducedMotion ? 0 : 220);
  }

  if (!reducedMotion) {
    window.setInterval(() => {
      kineticIndex = (kineticIndex + 1) % kineticStates.length;
      showKineticState(kineticIndex);
    }, 2600);
  }

  const tones = ["steady?", "hesitant?", "bright?", "strained?"];
  const tone = document.getElementById("quoteTone");
  let toneIndex = 0;
  window.setInterval(() => {
    if (!tone) return;
    toneIndex = (toneIndex + 1) % tones.length;
    tone.textContent = tones[toneIndex];
  }, 2400);

  function createScannerWave() {
    const canvas = document.getElementById("scannerCanvas");
    if (!canvas) return;
    const context = canvas.getContext("2d");
    let width = 0;
    let height = 0;
    let frame = 0;
    let scannerVisible = true;
    const resize = () => {
      const ratio = Math.min(devicePixelRatio, 2);
      width = canvas.width = Math.max(1, canvas.clientWidth * ratio);
      height = canvas.height = Math.max(1, canvas.clientHeight * ratio);
    };
    resize();
    new IntersectionObserver(([entry]) => { scannerVisible = entry.isIntersecting; }, {
      threshold: 0.01,
    }).observe(canvas);
    function renderWave() {
      if (!scannerVisible || document.hidden) {
        requestAnimationFrame(renderWave);
        return;
      }
      context.clearRect(0, 0, width, height);
      const mid = height * 0.46;
      for (let layer = 0; layer < 3; layer += 1) {
        context.beginPath();
        for (let x = 0; x <= width; x += 4) {
          const envelope = Math.sin(Math.PI * x / width);
          const signal = Math.sin(x * (0.025 + layer * 0.006) + frame * (0.025 + layer * 0.004));
          const detail = Math.sin(x * 0.09 - frame * 0.017) * 0.28;
          const y = mid + (signal + detail) * height * (0.12 - layer * 0.022) * envelope;
          if (x === 0) context.moveTo(x, y); else context.lineTo(x, y);
        }
        context.strokeStyle = layer === 0 ? "rgba(255,92,26,.88)" : `rgba(255,145,88,${0.28 - layer * 0.07})`;
        context.lineWidth = Math.max(1, devicePixelRatio * (layer === 0 ? 1.1 : 0.6));
        context.stroke();
      }
      frame += 1;
      if (!reducedMotion) requestAnimationFrame(renderWave);
    }
    renderWave();
    window.addEventListener("resize", resize);
  }

  if (window.gsap) {
    window.gsap.registerPlugin(window.ScrollTrigger);
    window.gsap.set(".hero-reveal", { y: 30, opacity: 0 });
    window.gsap.to(".hero-reveal", {
      y: 0,
      opacity: 1,
      duration: 1.15,
      stagger: 0.1,
      ease: "power4.out",
      delay: 0.18,
    });
    window.gsap.utils.toArray(".reveal").forEach((element) => {
      window.gsap.from(element, {
        y: reducedMotion ? 0 : 36,
        opacity: 0,
        duration: 1,
        ease: "power3.out",
        scrollTrigger: { trigger: element, start: "top 88%", once: true },
      });
    });
    if (!reducedMotion) {
      window.gsap.fromTo(".quote-wave i", { scaleY: 0.1 }, {
        scaleY: 1,
        stagger: 0.04,
        ease: "sine.inOut",
        scrollTrigger: { trigger: ".voice-stage", start: "top 80%", end: "bottom 30%", scrub: 1 },
      });
    }
  }

  function createSignalScene() {
    if (!window.THREE || reducedMotion) return;
    const canvas = document.getElementById("timbreScene");
    if (!canvas) return;
    const THREE = window.THREE;
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(42, innerWidth / innerHeight, 0.1, 100);
    camera.position.z = 7;
    const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });
    renderer.setPixelRatio(Math.min(devicePixelRatio, 1.8));
    renderer.setSize(innerWidth, innerHeight);

    const geometry = new THREE.TorusKnotGeometry(1.52, 0.5, 320, 24, 2, 3);
    const base = geometry.attributes.position.array.slice();
    const material = new THREE.PointsMaterial({
      color: 0xff6b2c,
      size: 0.016,
      transparent: true,
      opacity: 0.84,
      blending: THREE.AdditiveBlending,
    });
    const signal = new THREE.Points(geometry, material);
    signal.position.x = innerWidth < 800 ? 1.2 : 2.55;
    signal.rotation.x = -0.35;
    scene.add(signal);

    const shell = new THREE.Mesh(
      new THREE.TorusKnotGeometry(1.54, 0.51, 128, 10, 2, 3),
      new THREE.MeshBasicMaterial({ color: 0x9b72ff, wireframe: true, transparent: true, opacity: 0.11 })
    );
    shell.position.copy(signal.position);
    shell.rotation.copy(signal.rotation);
    scene.add(shell);

    const haloGroup = new THREE.Group();
    haloGroup.position.copy(signal.position);
    [2.4, 2.75, 3.1].forEach((radius, index) => {
      const halo = new THREE.Mesh(
        new THREE.TorusGeometry(radius, 0.008, 4, 120),
        new THREE.MeshBasicMaterial({
          color: index === 1 ? 0xff6b2c : 0x9b72ff,
          transparent: true,
          opacity: 0.16 - index * 0.025,
        })
      );
      halo.rotation.x = Math.PI / 2 + index * 0.34;
      halo.rotation.y = index * 0.42;
      haloGroup.add(halo);
    });
    scene.add(haloGroup);

    const starsGeometry = new THREE.BufferGeometry();
    const stars = new Float32Array(260 * 3);
    for (let i = 0; i < stars.length; i += 3) {
      stars[i] = (Math.random() - 0.5) * 16;
      stars[i + 1] = (Math.random() - 0.5) * 10;
      stars[i + 2] = (Math.random() - 0.5) * 8;
    }
    starsGeometry.setAttribute("position", new THREE.BufferAttribute(stars, 3));
    scene.add(new THREE.Points(starsGeometry, new THREE.PointsMaterial({
      color: 0xeeeade, size: 0.012, transparent: true, opacity: 0.28,
    })));

    const pointer = { x: 0, y: 0 };
    window.addEventListener("pointermove", (event) => {
      pointer.x = event.clientX / innerWidth - 0.5;
      pointer.y = event.clientY / innerHeight - 0.5;
    }, { passive: true });
    const clock = new THREE.Clock();
    let heroVisible = true;
    const visibilityObserver = new IntersectionObserver(([entry]) => {
      heroVisible = entry.isIntersecting;
    }, { threshold: 0.01 });
    visibilityObserver.observe(canvas);
    function render() {
      if (!heroVisible || document.hidden) {
        requestAnimationFrame(render);
        return;
      }
      const time = clock.getElapsedTime();
      const positions = geometry.attributes.position.array;
      for (let i = 0; i < positions.length; i += 3) {
        const bx = base[i]; const by = base[i + 1]; const bz = base[i + 2];
        const wave = 1 + 0.045 * Math.sin((bx + by) * 3.2 + time * 2.1)
          + 0.018 * Math.sin(bz * 7 - time * 1.4);
        positions[i] = bx * wave; positions[i + 1] = by * wave; positions[i + 2] = bz * wave;
      }
      geometry.attributes.position.needsUpdate = true;
      signal.rotation.y += 0.0026;
      signal.rotation.x += (pointer.y * 0.35 - signal.rotation.x) * 0.025;
      signal.rotation.z += (pointer.x * 0.25 - signal.rotation.z) * 0.025;
      shell.rotation.y -= 0.0015;
      shell.rotation.x = signal.rotation.x * 0.72;
      haloGroup.rotation.z = time * 0.055;
      haloGroup.rotation.y = Math.sin(time * 0.3) * 0.12;
      haloGroup.children.forEach((halo, index) => {
        const pulse = 1 + Math.sin(time * 1.15 - index * 0.8) * 0.025;
        halo.scale.setScalar(pulse);
      });
      camera.position.x += (pointer.x * 0.22 - camera.position.x) * 0.018;
      camera.position.y += (-pointer.y * 0.2 - camera.position.y) * 0.018;
      renderer.render(scene, camera);
      requestAnimationFrame(render);
    }
    render();
    window.addEventListener("resize", () => {
      camera.aspect = innerWidth / innerHeight;
      camera.updateProjectionMatrix();
      signal.position.x = innerWidth < 800 ? 1.2 : 2.55;
      shell.position.copy(signal.position);
      haloGroup.position.copy(signal.position);
      renderer.setSize(innerWidth, innerHeight);
    });
  }

  createScannerWave();
  createSignalScene();
}());
