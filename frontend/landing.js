/* Timbre landing experience: spatial signal field + scroll choreography. */
(function () {
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const nav = document.getElementById("siteNav");

  window.addEventListener("scroll", () => nav?.classList.toggle("scrolled", scrollY > 40), {
    passive: true,
  });

  const tones = ["steady?", "hesitant?", "bright?", "strained?"];
  const tone = document.getElementById("quoteTone");
  let toneIndex = 0;
  window.setInterval(() => {
    if (!tone) return;
    toneIndex = (toneIndex + 1) % tones.length;
    tone.textContent = tones[toneIndex];
  }, 2400);

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
      signal.position.x = innerWidth < 800 ? 1.4 : 2.45;
      shell.position.copy(signal.position);
      renderer.setSize(innerWidth, innerHeight);
    });
  }

  createSignalScene();
}());
