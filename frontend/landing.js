/* Timbre landing experience: spatial signal field + scroll choreography. */
(function () {
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const nav = document.getElementById("siteNav");
  const glow = document.getElementById("cursorGlow");

  window.addEventListener("scroll", () => nav?.classList.toggle("scrolled", scrollY > 40), {
    passive: true,
  });
  window.addEventListener("pointermove", (event) => {
    if (!glow) return;
    glow.style.left = `${event.clientX}px`;
    glow.style.top = `${event.clientY}px`;
  }, { passive: true });

  const tones = ["steady?", "hesitant?", "bright?", "strained?"];
  const tone = document.getElementById("quoteTone");
  let toneIndex = 0;
  window.setInterval(() => {
    if (!tone) return;
    toneIndex = (toneIndex + 1) % tones.length;
    tone.textContent = tones[toneIndex];
  }, 1700);

  if (window.gsap) {
    window.gsap.registerPlugin(window.ScrollTrigger);
    window.gsap.set(".hero-reveal", { y: 55, opacity: 0 });
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
        y: reducedMotion ? 0 : 70,
        opacity: 0,
        duration: 1,
        ease: "power3.out",
        scrollTrigger: { trigger: element, start: "top 88%", once: true },
      });
    });
    if (!reducedMotion) {
      window.gsap.to(".hero-copy", {
        yPercent: 22,
        opacity: 0.2,
        ease: "none",
        scrollTrigger: { trigger: ".hero", start: "top top", end: "bottom top", scrub: true },
      });
      window.gsap.fromTo(".quote-wave i", { scaleY: 0.1 }, {
        scaleY: 1,
        stagger: 0.04,
        ease: "sine.inOut",
        scrollTrigger: { trigger: ".voice-quote", start: "top 80%", end: "bottom 30%", scrub: 1 },
      });
    }
  }

  if (!reducedMotion) {
    document.querySelectorAll(".magnetic").forEach((element) => {
      element.addEventListener("pointermove", (event) => {
        const box = element.getBoundingClientRect();
        const x = (event.clientX - box.left - box.width / 2) * 0.12;
        const y = (event.clientY - box.top - box.height / 2) * 0.12;
        element.style.transform = `translate(${x}px, ${y}px)`;
      });
      element.addEventListener("pointerleave", () => { element.style.transform = ""; });
    });
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

    const geometry = new THREE.IcosahedronGeometry(2.05, 5);
    const base = geometry.attributes.position.array.slice();
    const material = new THREE.PointsMaterial({
      color: 0xff6b2c,
      size: 0.02,
      transparent: true,
      opacity: 0.78,
      blending: THREE.AdditiveBlending,
    });
    const signal = new THREE.Points(geometry, material);
    signal.position.x = innerWidth < 800 ? 1.4 : 2.45;
    scene.add(signal);

    const shell = new THREE.Mesh(
      new THREE.IcosahedronGeometry(2.08, 2),
      new THREE.MeshBasicMaterial({ color: 0x8b5cff, wireframe: true, transparent: true, opacity: 0.12 })
    );
    shell.position.copy(signal.position);
    scene.add(shell);

    const starsGeometry = new THREE.BufferGeometry();
    const stars = new Float32Array(450 * 3);
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
    function render() {
      const time = clock.getElapsedTime();
      const positions = geometry.attributes.position.array;
      for (let i = 0; i < positions.length; i += 3) {
        const bx = base[i]; const by = base[i + 1]; const bz = base[i + 2];
        const wave = 1 + 0.055 * Math.sin(by * 3.2 + time * 1.7) + 0.025 * Math.sin(bx * 5 - time);
        positions[i] = bx * wave; positions[i + 1] = by * wave; positions[i + 2] = bz * wave;
      }
      geometry.attributes.position.needsUpdate = true;
      signal.rotation.y += 0.0015;
      signal.rotation.x += (pointer.y * 0.35 - signal.rotation.x) * 0.025;
      signal.rotation.z += (pointer.x * 0.25 - signal.rotation.z) * 0.025;
      shell.rotation.y -= 0.001;
      shell.rotation.x = signal.rotation.x * 0.7;
      camera.position.x += (pointer.x * 0.22 - camera.position.x) * 0.018;
      camera.position.y += (-pointer.y * 0.2 - camera.position.y) * 0.018;
      renderer.render(scene, camera);
      requestAnimationFrame(render);
    }
    render();
    window.addEventListener("resize", () => {
      camera.aspect = innerWidth / innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(innerWidth, innerHeight);
    });
  }

  function drawClosingSignal() {
    const canvas = document.getElementById("closingScene");
    if (!canvas) return;
    const context = canvas.getContext("2d");
    let width; let height; let frame = 0;
    const resize = () => {
      width = canvas.width = canvas.clientWidth * Math.min(devicePixelRatio, 2);
      height = canvas.height = canvas.clientHeight * Math.min(devicePixelRatio, 2);
    };
    resize();
    function render() {
      context.clearRect(0, 0, width, height);
      context.beginPath();
      for (let x = 0; x <= width; x += 5) {
        const envelope = Math.sin(Math.PI * x / width);
        const y = height * 0.52 + Math.sin(x * 0.018 + frame * 0.018) * height * 0.08 * envelope
          + Math.sin(x * 0.006 - frame * 0.011) * height * 0.12 * envelope;
        if (x === 0) context.moveTo(x, y); else context.lineTo(x, y);
      }
      const gradient = context.createLinearGradient(0, 0, width, 0);
      gradient.addColorStop(0, "rgba(123,97,255,0)");
      gradient.addColorStop(0.42, "rgba(123,97,255,.20)");
      gradient.addColorStop(0.7, "rgba(255,92,26,.32)");
      gradient.addColorStop(1, "rgba(255,92,26,0)");
      context.strokeStyle = gradient;
      context.lineWidth = Math.max(1, devicePixelRatio);
      context.stroke();
      frame += 1;
      if (!reducedMotion) requestAnimationFrame(render);
    }
    render();
    window.addEventListener("resize", resize);
  }

  createSignalScene();
  drawClosingSignal();
}());
