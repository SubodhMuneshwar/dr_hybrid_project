// Minimal Interactive Grid Background
// Grid glows subtly when clicked, visible in both dark and light modes

class InteractiveGridBackground {
  constructor() {
    this.canvas = null;
    this.ctx = null;
    this.gridSize = 40;
    this.glowIntensity = 0;
    this.glowDecay = 0.02;
    this.isDarkMode = this.checkDarkMode();
    this.clickPositions = [];
    this.maxClickPositions = 5;

    this.init();
    this.setupEventListeners();
    this.animate();
  }

  init() {
    // Create canvas element
    this.canvas = document.createElement('canvas');
    this.canvas.id = 'grid-background-canvas';
    this.canvas.style.cssText = `
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      z-index: -1;
      pointer-events: none;
    `;
    document.body.insertBefore(this.canvas, document.body.firstChild);

    this.ctx = this.canvas.getContext('2d');
    this.resizeCanvas();
  }

  resizeCanvas() {
    this.canvas.width = window.innerWidth;
    this.canvas.height = window.innerHeight;
  }

  checkDarkMode() {
    return document.documentElement.getAttribute('data-theme') === 'dark' ||
           (!document.documentElement.hasAttribute('data-theme') &&
            window.matchMedia('(prefers-color-scheme: dark)').matches);
  }

  setupEventListeners() {
    // Handle window resize
    window.addEventListener('resize', () => this.resizeCanvas());

    // Handle theme changes
    const observer = new MutationObserver(() => {
      this.isDarkMode = this.checkDarkMode();
    });
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });

    // Handle click anywhere on page to trigger grid glow
    document.addEventListener('click', (e) => {
      // Only trigger for meaningful clicks (not on buttons/links that handle their own events)
      if (e.target.tagName === 'BODY' || e.target === this.canvas) {
        this.triggerGlow(e.clientX, e.clientY);
      }
    });

    // Listen for theme toggle button clicks
    const themeToggle = document.getElementById('themeToggle');
    if (themeToggle) {
      themeToggle.addEventListener('click', (e) => {
        setTimeout(() => {
          this.isDarkMode = this.checkDarkMode();
          this.triggerGlow(e.clientX, e.clientY);
        }, 100);
      });
    }
  }

  triggerGlow(x, y) {
    this.clickPositions.push({ x, y, intensity: 1 });
    if (this.clickPositions.length > this.maxClickPositions) {
      this.clickPositions.shift();
    }
    this.glowIntensity = Math.min(this.glowIntensity + 0.5, 1);
  }

  drawGrid() {
    const gridColor = this.isDarkMode
      ? 'rgba(148, 163, 184, 0.08)' // slate-400 with low opacity for dark mode
      : 'rgba(203, 213, 225, 0.12)'; // slate-300 with low opacity for light mode

    this.ctx.strokeStyle = gridColor;
    this.ctx.lineWidth = 1;

    // Draw vertical lines
    for (let x = 0; x < this.canvas.width; x += this.gridSize) {
      this.ctx.beginPath();
      this.ctx.moveTo(x, 0);
      this.ctx.lineTo(x, this.canvas.height);
      this.ctx.stroke();
    }

    // Draw horizontal lines
    for (let y = 0; y < this.canvas.height; y += this.gridSize) {
      this.ctx.beginPath();
      this.ctx.moveTo(0, y);
      this.ctx.lineTo(this.canvas.width, y);
      this.ctx.stroke();
    }
  }

  drawGlowEffect() {
    if (this.clickPositions.length === 0) return;

    this.clickPositions = this.clickPositions.filter(pos => pos.intensity > 0.01);

    this.clickPositions.forEach((pos, index) => {
      const glowRadius = 150 * (1 - pos.intensity);
      const glowOpacity = pos.intensity * 0.15;

      // Create radial gradient for subtle glow
      const gradient = this.ctx.createRadialGradient(pos.x, pos.y, 0, pos.x, pos.y, glowRadius);

      const glowColor = this.isDarkMode
        ? `rgba(99, 102, 241, ${glowOpacity})` // indigo glow for dark mode
        : `rgba(79, 70, 229, ${glowOpacity})`; // darker indigo for light mode

      gradient.addColorStop(0, glowColor);
      gradient.addColorStop(1, 'rgba(0, 0, 0, 0)');

      this.ctx.fillStyle = gradient;
      this.ctx.fillRect(pos.x - glowRadius, pos.y - glowRadius, glowRadius * 2, glowRadius * 2);

      // Decay the glow
      pos.intensity -= this.glowDecay;
    });
  }

  animate() {
    // Clear canvas with transparent background (preserves page content)
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

    // Draw grid
    this.drawGrid();

    // Draw glow effects
    this.drawGlowEffect();

    // Continue animation loop
    requestAnimationFrame(() => this.animate());
  }
}

// Initialize grid background when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => {
    new InteractiveGridBackground();
  });
} else {
  new InteractiveGridBackground();
}
