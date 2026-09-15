// ==========================================================================
// Minimal Interactive Grid Background for RetinaScan AI
// - Fixed grid structure visible in both dark and light modes
// - Positioned in the bottom layer of all website elements (z-index: 0, pointer-events: none)
// - Glows only in a subtle, minimal way when clicked
// ==========================================================================

class InteractiveGridBackground {
  constructor() {
    this.canvas = null;
    this.ctx = null;
    this.gridSize = 36;
    this.pulses = [];
    this.isAnimating = false;
    this.isDarkMode = this.checkDarkMode();

    this.init();
    this.setupEventListeners();
    this.draw();
  }

  init() {
    let existing = document.getElementById('grid-background-canvas');
    if (existing) {
      this.canvas = existing;
    } else {
      this.canvas = document.createElement('canvas');
      this.canvas.id = 'grid-background-canvas';
      this.canvas.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        z-index: 0;
        pointer-events: none;
      `;
      document.body.prepend(this.canvas);
    }

    this.ctx = this.canvas.getContext('2d');
    this.resizeCanvas();
  }

  checkDarkMode() {
    try {
      const savedTheme = localStorage.getItem('theme');
      if (savedTheme === 'dark') return true;
      if (savedTheme === 'light') return false;
    } catch (e) {}
    return document.documentElement.classList.contains('dark') ||
           document.documentElement.getAttribute('data-theme') === 'dark' ||
           (!document.documentElement.hasAttribute('data-theme') &&
            window.matchMedia('(prefers-color-scheme: dark)').matches);
  }

  getGridColor() {
    // Subtly visible minimal grid lines for both dark and light modes
    return this.isDarkMode
      ? 'rgba(129, 140, 248, 0.085)'
      : 'rgba(99, 102, 241, 0.065)';
  }

  resizeCanvas() {
    const dpr = window.devicePixelRatio || 1;
    this.width = window.innerWidth;
    this.height = window.innerHeight;
    this.canvas.width = Math.floor(this.width * dpr);
    this.canvas.height = Math.floor(this.height * dpr);
    this.canvas.style.width = this.width + 'px';
    this.canvas.style.height = this.height + 'px';
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    if (!this.isAnimating) {
      this.draw();
    }
  }

  setupEventListeners() {
    window.addEventListener('resize', () => this.resizeCanvas(), { passive: true });

    // Handle theme changes (Tailwind dark class or data-theme)
    const observer = new MutationObserver(() => {
      const wasDark = this.isDarkMode;
      this.isDarkMode = this.checkDarkMode();
      if (wasDark !== this.isDarkMode) {
        this.draw();
      }
    });
    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ['class', 'data-theme']
    });

    if (window.matchMedia) {
      const mql = window.matchMedia('(prefers-color-scheme: dark)');
      const onThemeChange = () => {
        this.isDarkMode = this.checkDarkMode();
        this.draw();
      };
      if (mql.addEventListener) {
        mql.addEventListener('change', onThemeChange);
      } else if (mql.addListener) {
        mql.addListener(onThemeChange);
      }
    }

    // Glow triggers ONLY when clicked (anywhere on the page)
    window.addEventListener('pointerdown', (e) => {
      if (e.button !== undefined && e.button !== 0) return;
      this.triggerGlow(e.clientX, e.clientY);
    }, { passive: true });
  }

  triggerGlow(x, y) {
    this.pulses.push({
      x,
      y,
      radius: 6,
      maxRadius: 170, // Contained subtle radius
      alpha: 1.0,
      speed: 5.5,
      decay: 0.034    // Smooth ~550ms decay
    });

    if (this.pulses.length > 4) {
      this.pulses.shift();
    }

    if (!this.isAnimating) {
      this.isAnimating = true;
      requestAnimationFrame(() => this.animate());
    }
  }

  drawBaseGrid() {
    const color = this.getGridColor();
    this.ctx.strokeStyle = color;
    this.ctx.lineWidth = 1;

    this.ctx.beginPath();
    for (let x = 0; x <= this.width; x += this.gridSize) {
      this.ctx.moveTo(x, 0);
      this.ctx.lineTo(x, this.height);
    }
    for (let y = 0; y <= this.height; y += this.gridSize) {
      this.ctx.moveTo(0, y);
      this.ctx.lineTo(this.width, y);
    }
    this.ctx.stroke();
  }

  draw() {
    this.ctx.clearRect(0, 0, this.width, this.height);
    this.drawBaseGrid();
  }

  animate() {
    if (this.pulses.length === 0) {
      this.draw();
      this.isAnimating = false;
      return;
    }

    this.ctx.clearRect(0, 0, this.width, this.height);
    this.drawBaseGrid();

    // Render subtle glowing wave on grid lines around each click point
    for (let i = this.pulses.length - 1; i >= 0; i--) {
      const pulse = this.pulses[i];
      pulse.radius += pulse.speed;
      pulse.speed = Math.max(3.0, pulse.speed * 0.96);
      pulse.alpha -= pulse.decay;

      if (pulse.alpha <= 0.01 || pulse.radius >= pulse.maxRadius) {
        this.pulses.splice(i, 1);
        continue;
      }

      const r = pulse.radius;
      const currentAlpha = Math.max(0, pulse.alpha);

      // 1. Soft, very subtle radial illumination wash
      const grad = this.ctx.createRadialGradient(pulse.x, pulse.y, 0, pulse.x, pulse.y, r);
      if (this.isDarkMode) {
        grad.addColorStop(0, `rgba(99, 102, 241, ${currentAlpha * 0.12})`);
        grad.addColorStop(0.55, `rgba(6, 182, 212, ${currentAlpha * 0.05})`);
        grad.addColorStop(1, 'rgba(99, 102, 241, 0)');
      } else {
        grad.addColorStop(0, `rgba(79, 70, 229, ${currentAlpha * 0.09})`);
        grad.addColorStop(0.55, `rgba(99, 102, 241, ${currentAlpha * 0.04})`);
        grad.addColorStop(1, 'rgba(79, 70, 229, 0)');
      }

      this.ctx.fillStyle = grad;
      this.ctx.beginPath();
      this.ctx.arc(pulse.x, pulse.y, r, 0, Math.PI * 2);
      this.ctx.fill();

      // 2. Glow along intersecting grid lines within pulse radius
      const minX = Math.max(0, Math.floor((pulse.x - r) / this.gridSize) * this.gridSize);
      const maxX = Math.min(this.width, Math.ceil((pulse.x + r) / this.gridSize) * this.gridSize);
      const minY = Math.max(0, Math.floor((pulse.y - r) / this.gridSize) * this.gridSize);
      const maxY = Math.min(this.height, Math.ceil((pulse.y + r) / this.gridSize) * this.gridSize);

      this.ctx.lineWidth = 1.1;

      // Vertical glowing line segments
      for (let x = minX; x <= maxX; x += this.gridSize) {
        const dx = Math.abs(x - pulse.x);
        if (dx <= r) {
          const dy = Math.sqrt(r * r - dx * dx);
          const y1 = Math.max(0, pulse.y - dy);
          const y2 = Math.min(this.height, pulse.y + dy);
          const lineAlpha = currentAlpha * (1 - dx / r) * (this.isDarkMode ? 0.32 : 0.26);

          this.ctx.strokeStyle = this.isDarkMode
            ? `rgba(129, 140, 248, ${lineAlpha})`
            : `rgba(79, 70, 229, ${lineAlpha})`;

          this.ctx.beginPath();
          this.ctx.moveTo(x, y1);
          this.ctx.lineTo(x, y2);
          this.ctx.stroke();
        }
      }

      // Horizontal glowing line segments
      for (let y = minY; y <= maxY; y += this.gridSize) {
        const dy = Math.abs(y - pulse.y);
        if (dy <= r) {
          const dx = Math.sqrt(r * r - dy * dy);
          const x1 = Math.max(0, pulse.x - dx);
          const x2 = Math.min(this.width, pulse.x + dx);
          const lineAlpha = currentAlpha * (1 - dy / r) * (this.isDarkMode ? 0.32 : 0.26);

          this.ctx.strokeStyle = this.isDarkMode
            ? `rgba(129, 140, 248, ${lineAlpha})`
            : `rgba(79, 70, 229, ${lineAlpha})`;

          this.ctx.beginPath();
          this.ctx.moveTo(x1, y);
          this.ctx.lineTo(x2, y);
          this.ctx.stroke();
        }
      }

      // 3. Delicate intersection node accents
      for (let x = minX; x <= maxX; x += this.gridSize) {
        for (let y = minY; y <= maxY; y += this.gridSize) {
          const dist = Math.hypot(x - pulse.x, y - pulse.y);
          if (dist <= r) {
            const nodeAlpha = currentAlpha * (1 - dist / r) * (this.isDarkMode ? 0.38 : 0.30);
            if (nodeAlpha > 0.04) {
              this.ctx.fillStyle = this.isDarkMode
                ? `rgba(165, 180, 252, ${nodeAlpha})`
                : `rgba(79, 70, 229, ${nodeAlpha})`;
              this.ctx.beginPath();
              this.ctx.arc(x, y, 1.2, 0, Math.PI * 2);
              this.ctx.fill();
            }
          }
        }
      }
    }

    requestAnimationFrame(() => this.animate());
  }
}

// Global initialization
if (typeof window !== 'undefined') {
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => new InteractiveGridBackground());
  } else {
    new InteractiveGridBackground();
  }
}
