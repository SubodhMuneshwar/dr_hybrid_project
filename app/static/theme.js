// ==========================================================================
// RetinaScan AI - Interactive Engine
// - Unified dark/light theme switching with localStorage persistence
// - Interactive Before/After image comparison slider
// - One-click clinical sample image loader
// - Real-time dashboard search, filtering, and CSV export
// - Responsive mobile drawer navigation
// ==========================================================================

(function () {
  const html = document.documentElement;

  function getSystemTheme() {
    return (window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches)
      ? "dark"
      : "light";
  }

  function getPreferredTheme() {
    try {
      const saved = localStorage.getItem("theme");
      if (saved === "dark" || saved === "light") {
        return saved;
      }
    } catch (e) {}
    return getSystemTheme();
  }

  function applyTheme(theme, persist = false) {
    const isDark = theme === "dark";
    html.classList.toggle("dark", isDark);
    html.setAttribute("data-theme", isDark ? "dark" : "light");

    if (persist) {
      try {
        localStorage.setItem("theme", theme);
      } catch (e) {}
    }

    // Sync all ocular toggle buttons on page
    document.querySelectorAll("[data-theme-toggle], #themeToggle").forEach((toggle) => {
      toggle.setAttribute("aria-pressed", String(isDark));
      toggle.setAttribute(
        "title",
        isDark ? "Dark mode active (click to switch to daylight mode)" : "Daylight mode active (click to switch to night mode)"
      );
      toggle.setAttribute(
        "aria-label",
        isDark ? "Dark mode active" : "Light mode active"
      );
    });
  }

  function currentTheme() {
    return html.classList.contains("dark") ? "dark" : "light";
  }

  // Apply theme according to saved user choice or system configuration
  applyTheme(getPreferredTheme());

  // Listen in real-time to OS/system preference changes only if user hasn't set a manual override
  if (window.matchMedia) {
    const mediaQuery = window.matchMedia("(prefers-color-scheme: dark)");
    const onSystemThemeChange = (e) => {
      try {
        if (!localStorage.getItem("theme")) {
          applyTheme(e.matches ? "dark" : "light");
        }
      } catch (err) {
        applyTheme(e.matches ? "dark" : "light");
      }
    };
    if (mediaQuery.addEventListener) {
      mediaQuery.addEventListener("change", onSystemThemeChange);
    } else if (mediaQuery.addListener) {
      mediaQuery.addListener(onSystemThemeChange);
    }
  }

  // ==========================================================================
  // Neural Heatmap Inspector (Split Slider, Opacity Blend, Optic Loupe, Auto-Scan, Hotspots)
  // ==========================================================================
  function initNeuralInspector() {
    const container = document.getElementById("comparisonContainer");
    if (!container) return;

    const heatmapLayer = document.getElementById("comparisonHeatmapLayer") || document.getElementById("comparisonOverlay");
    const handle = document.getElementById("comparisonHandle");
    const percentBadge = document.getElementById("sliderPercentBadge");
    const modeBtns = document.querySelectorAll(".inspector-mode-btn");

    const controlsSplit = document.getElementById("controlsSplitBar");
    const controlsBlend = document.getElementById("controlsBlendBar");
    const controlsLens = document.getElementById("controlsLensBar");
    const controlsAuto = document.getElementById("controlsAutoBar");

    const blendRange = document.getElementById("blendOpacityRange");
    const blendLabel = document.getElementById("blendPercentLabel");

    const loupe = document.getElementById("inspectorLoupe");
    const loupeImg = document.getElementById("loupeHeatmapImg");

    const hotspotsGroup = document.getElementById("lesionHotspotsGroup");
    const toggleHotspotsBtn = document.getElementById("toggleHotspotsBtn");
    const hotspotsBtnLabel = document.getElementById("hotspotsBtnLabel");
    const snapCenterBtn = document.getElementById("snapCenterBtn");

    const autoPlayPauseBtn = document.getElementById("autoPlayPauseBtn");
    const autoPlayIcon = document.getElementById("autoPlayIcon");
    const autoPlayLabel = document.getElementById("autoPlayLabel");

    let currentMode = "split"; // "split", "blend", "lens", "auto"
    let currentSplitPct = 50;
    let isDragging = false;
    let autoRafId = null;
    let isAutoPlaying = true;
    let hotspotsVisible = true;

    function applySplit(pct) {
      if (pct < 0) pct = 0;
      if (pct > 100) pct = 100;
      currentSplitPct = pct;

      container.style.setProperty("--split-x", pct + "%");
      if (heatmapLayer) {
        heatmapLayer.style.clipPath = `polygon(0 0, ${pct}% 0, ${pct}% 100%, 0 100%)`;
      }
      if (handle) {
        handle.style.left = pct + "%";
      }
      if (percentBadge) {
        percentBadge.textContent = Math.round(pct) + "%";
      }
    }

    function setPositionFromClientX(clientX) {
      const rect = container.getBoundingClientRect();
      let pos = (clientX - rect.left) / rect.width;
      let pct = pos * 100;
      applySplit(pct);
    }

    // Mode Switching
    function setMode(mode) {
      currentMode = mode;

      // Update mode switcher buttons active UI
      modeBtns.forEach((btn) => {
        const isActive = btn.getAttribute("data-mode") === mode;
        btn.classList.toggle("active", isActive);
        if (isActive) {
          btn.classList.add("text-indigo-600", "dark:text-indigo-400", "bg-white", "dark:bg-slate-700/80", "shadow-xs");
          btn.classList.remove("text-slate-600", "dark:text-slate-400");
        } else {
          btn.classList.remove("text-indigo-600", "dark:text-indigo-400", "bg-white", "dark:bg-slate-700/80", "shadow-xs");
          btn.classList.add("text-slate-600", "dark:text-slate-400");
        }
      });

      // Toggle secondary control bars
      if (controlsSplit) controlsSplit.classList.toggle("hidden", mode !== "split");
      if (controlsBlend) controlsBlend.classList.toggle("hidden", mode !== "blend");
      if (controlsLens) controlsLens.classList.toggle("hidden", mode !== "lens");
      if (controlsAuto) controlsAuto.classList.toggle("hidden", mode !== "auto");

      // Mode-specific canvas resets
      if (autoRafId) {
        cancelAnimationFrame(autoRafId);
        autoRafId = null;
      }

      if (loupe) loupe.classList.add("hidden");

      if (mode === "split") {
        container.style.cursor = "ew-resize";
        if (handle) handle.classList.remove("hidden");
        if (heatmapLayer) {
          heatmapLayer.style.opacity = "1";
          applySplit(currentSplitPct);
        }
      } else if (mode === "blend") {
        container.style.cursor = "default";
        if (handle) handle.classList.add("hidden");
        if (heatmapLayer) {
          heatmapLayer.style.clipPath = "none";
          const val = blendRange ? blendRange.value : 70;
          heatmapLayer.style.opacity = String(val / 100);
        }
      } else if (mode === "lens") {
        container.style.cursor = "crosshair";
        if (handle) handle.classList.add("hidden");
        if (heatmapLayer) {
          heatmapLayer.style.clipPath = "none";
          heatmapLayer.style.opacity = "0";
        }
        if (loupe) loupe.classList.remove("hidden");
        // Center loupe by default
        updateLoupePosition(container.clientWidth / 2, container.clientHeight / 2);
      } else if (mode === "auto") {
        container.style.cursor = "pointer";
        if (handle) handle.classList.remove("hidden");
        if (heatmapLayer) {
          heatmapLayer.style.opacity = "1";
        }
        isAutoPlaying = true;
        updateAutoPlayUI();
        startAutoScanLoop();
      }
    }

    modeBtns.forEach((btn) => {
      btn.addEventListener("click", () => {
        const mode = btn.getAttribute("data-mode");
        if (mode) setMode(mode);
      });
    });

    // Split Mode Interactions (Mouse, Touch, Keyboard)
    function onPointerDown(e) {
      if (currentMode !== "split") return;
      isDragging = true;
      if (handle) handle.classList.add("is-dragging");
      const clientX = e.clientX || (e.touches && e.touches[0].clientX);
      if (clientX !== undefined) setPositionFromClientX(clientX);
    }

    function onPointerMove(e) {
      if (!isDragging || currentMode !== "split") return;
      const clientX = e.clientX || (e.touches && e.touches[0].clientX);
      if (clientX !== undefined) setPositionFromClientX(clientX);
    }

    function onPointerUp() {
      isDragging = false;
      if (handle) handle.classList.remove("is-dragging");
    }

    container.addEventListener("mousedown", onPointerDown);
    window.addEventListener("mousemove", onPointerMove);
    window.addEventListener("mouseup", onPointerUp);

    container.addEventListener("touchstart", onPointerDown, { passive: true });
    window.addEventListener("touchmove", onPointerMove, { passive: true });
    window.addEventListener("touchend", onPointerUp);

    // Keyboard navigation
    container.setAttribute("tabindex", "0");
    container.addEventListener("keydown", (e) => {
      if (currentMode !== "split") return;
      const step = e.shiftKey ? 10 : 2;
      if (e.key === "ArrowLeft") {
        e.preventDefault();
        applySplit(currentSplitPct - step);
      } else if (e.key === "ArrowRight") {
        e.preventDefault();
        applySplit(currentSplitPct + step);
      } else if (e.key === "Home") {
        e.preventDefault();
        applySplit(0);
      } else if (e.key === "End") {
        e.preventDefault();
        applySplit(100);
      }
    });

    // Blend Mode Slider
    if (blendRange) {
      blendRange.addEventListener("input", () => {
        const val = blendRange.value;
        if (blendLabel) blendLabel.textContent = val + "%";
        if (currentMode === "blend" && heatmapLayer) {
          heatmapLayer.style.opacity = String(val / 100);
        }
      });
    }

    // Optic Loupe Spotlight Handler with Pixel-Perfect Alignment
    let currentLoupeZoom = 1.0;
    let lastLoupeX = null;
    let lastLoupeY = null;

    function updateLoupePosition(x, y) {
      if (!loupe || !loupeImg) return;
      const w = container.clientWidth;
      const h = container.clientHeight;
      if (!w || !h) return;

      lastLoupeX = x;
      lastLoupeY = y;

      // Loupe radius (half of width/height)
      const loupeRadius = (loupe.offsetWidth || 140) / 2;

      // Position loupe centered at cursor (x, y)
      loupe.style.left = x + "px";
      loupe.style.top = y + "px";

      // Pixel-Perfect Geometric Alignment:
      // The loupe-optic-ring top-left is positioned at (x - loupeRadius, y - loupeRadius).
      // Inside loupe-optic-ring, a pixel at image coordinate (x, y) must appear
      // precisely at the loupe's center (loupeRadius, loupeRadius).
      // When scaled by currentLoupeZoom:
      // imageLeft = loupeRadius - (x * currentLoupeZoom)
      // imageTop = loupeRadius - (y * currentLoupeZoom)
      const scaledW = w * currentLoupeZoom;
      const scaledH = h * currentLoupeZoom;

      loupeImg.style.width = scaledW + "px";
      loupeImg.style.height = scaledH + "px";
      loupeImg.style.left = (loupeRadius - (x * currentLoupeZoom)) + "px";
      loupeImg.style.top = (loupeRadius - (y * currentLoupeZoom)) + "px";
    }

    // Loupe Zoom Preset Buttons
    const loupeZoomBtns = document.querySelectorAll(".loupe-zoom-btn");
    const loupeBadge = loupe ? loupe.querySelector(".loupe-hud-badge") : null;

    loupeZoomBtns.forEach((btn) => {
      btn.addEventListener("click", () => {
        const zoom = parseFloat(btn.getAttribute("data-zoom") || "1.0");
        currentLoupeZoom = zoom;

        loupeZoomBtns.forEach((b) => {
          b.classList.remove("text-indigo-600", "dark:text-cyan-400", "bg-white", "dark:bg-slate-700", "shadow-2xs");
          b.classList.add("text-slate-500", "dark:text-slate-400");
        });
        btn.classList.add("text-indigo-600", "dark:text-cyan-400", "bg-white", "dark:bg-slate-700", "shadow-2xs");
        btn.classList.remove("text-slate-500", "dark:text-slate-400");

        if (loupeBadge) {
          loupeBadge.textContent = zoom === 1.0 ? "1:1 ALIGNED" : "1.5x ZOOM";
        }

        if (lastLoupeX !== null && lastLoupeY !== null) {
          updateLoupePosition(lastLoupeX, lastLoupeY);
        }
      });
    });

    container.addEventListener("mousemove", (e) => {
      if (currentMode !== "lens") return;
      const rect = container.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      updateLoupePosition(x, y);
    });

    container.addEventListener("touchmove", (e) => {
      if (currentMode !== "lens" || !e.touches[0]) return;
      const rect = container.getBoundingClientRect();
      const x = e.touches[0].clientX - rect.left;
      const y = e.touches[0].clientY - rect.top;
      updateLoupePosition(x, y);
    }, { passive: true });

    container.addEventListener("mouseenter", () => {
      if (currentMode === "lens" && loupe) loupe.classList.remove("hidden");
    });

    container.addEventListener("mouseleave", () => {
      if (currentMode === "lens" && loupe) loupe.classList.add("hidden");
    });

    // Auto-Scan Radar Loop
    let autoStartTime = performance.now();
    function startAutoScanLoop() {
      function tick(now) {
        if (currentMode !== "auto") return;
        if (isAutoPlaying) {
          const elapsed = (now - autoStartTime) * 0.0018;
          // Smooth sine sweep between 8% and 92%
          const sweepPct = 50 + 40 * Math.sin(elapsed);
          applySplit(sweepPct);
        }
        autoRafId = requestAnimationFrame(tick);
      }
      autoRafId = requestAnimationFrame(tick);
    }

    function updateAutoPlayUI() {
      if (!autoPlayLabel || !autoPlayIcon) return;
      if (isAutoPlaying) {
        autoPlayIcon.className = "fas fa-pause text-[10px]";
        autoPlayLabel.textContent = "Pause Scan";
      } else {
        autoPlayIcon.className = "fas fa-play text-[10px]";
        autoPlayLabel.textContent = "Resume Scan";
      }
    }

    if (autoPlayPauseBtn) {
      autoPlayPauseBtn.addEventListener("click", () => {
        isAutoPlaying = !isAutoPlaying;
        if (isAutoPlaying) autoStartTime = performance.now();
        updateAutoPlayUI();
      });
    }

    // Reset to 50%
    if (snapCenterBtn) {
      snapCenterBtn.addEventListener("click", () => {
        applySplit(50);
      });
    }

    // Hotspot Toggle & Focus Jump
    if (toggleHotspotsBtn && hotspotsGroup) {
      toggleHotspotsBtn.addEventListener("click", () => {
        hotspotsVisible = !hotspotsVisible;
        hotspotsGroup.style.opacity = hotspotsVisible ? "1" : "0";
        hotspotsGroup.style.pointerEvents = hotspotsVisible ? "auto" : "none";
        if (hotspotsBtnLabel) {
          hotspotsBtnLabel.textContent = hotspotsVisible ? "Hotspots: ON" : "Hotspots: OFF";
        }
      });
    }

    document.querySelectorAll(".hotspot-node").forEach((node) => {
      node.addEventListener("click", () => {
        const pos = node.getAttribute("data-pos");
        if (pos && (currentMode === "split" || currentMode === "auto")) {
          applySplit(parseFloat(pos));
        }
        document.querySelectorAll(".hotspot-node").forEach((n) => {
          if (n !== node) n.classList.remove("is-active");
        });
        node.classList.toggle("is-active");
      });
    });

    // Initialize in Split Mode at 50%
    applySplit(50);
  }

  // One-Click Sample Image Loader for Scanner Page
  function initSampleLoader() {
    const sampleButtons = document.querySelectorAll("[data-sample-url]");
    const fileInput = document.getElementById("image-upload");
    const patientInput = document.getElementById("patient_name");
    const uploadPrompt = document.getElementById("uploadPrompt");
    const previewContainer = document.getElementById("previewContainer");
    const previewImg = document.getElementById("previewImg");
    const previewFilename = document.getElementById("previewFilename");
    const previewFilesize = document.getElementById("previewFilesize");
    const scanSubmitBtn = document.getElementById("scanSubmitBtn");

    if (!sampleButtons.length || !fileInput) return;

    sampleButtons.forEach((btn) => {
      btn.addEventListener("click", async (e) => {
        e.preventDefault();
        const url = btn.getAttribute("data-sample-url");
        const name = btn.getAttribute("data-sample-filename") || btn.getAttribute("data-sample-name") || "sample_retina.jpg";
        const patientName = btn.getAttribute("data-sample-patient") || btn.getAttribute("data-patient-default") || "Jane Doe";

        // Reset visual state on other preset buttons
        sampleButtons.forEach((b) => {
          b.classList.remove("is-selected");
          const s = b.querySelector(".preset-status");
          if (s) s.innerHTML = '<i class="fas fa-circle text-[5px]"></i>';
        });

        // Set active selection state on clicked button
        btn.classList.add("is-selected");
        const statusEl = btn.querySelector(".preset-status");
        if (statusEl) {
          statusEl.innerHTML = '<i class="fas fa-circle-notch fa-spin text-[10px]"></i>';
        }

        try {
          const res = await fetch(url);
          if (!res.ok) throw new Error(`HTTP ${res.status}`);
          const blob = await res.blob();
          const file = new File([blob], name, { type: blob.type || "image/jpeg" });

          // Create DataTransfer to populate native file input
          const dt = new DataTransfer();
          dt.items.add(file);
          fileInput.files = dt.files;

          if (patientInput) {
            patientInput.value = patientName;
            try {
              localStorage.setItem("last_patient", patientName);
            } catch (err) {}
          }

          // Trigger preview
          if (previewImg && previewContainer && uploadPrompt) {
            const reader = new FileReader();
            reader.onload = (re) => {
              previewImg.src = re.target.result;
              previewContainer.classList.remove("hidden");
              uploadPrompt.classList.add("hidden");
              if (previewFilename) previewFilename.textContent = name;
              if (previewFilesize) previewFilesize.textContent = (file.size / 1024).toFixed(1) + " KB";
            };
            reader.readAsDataURL(file);
          }

          // Update status indicator to success checkmark (preserves button markup!)
          if (statusEl) {
            statusEl.innerHTML = '<i class="fas fa-check text-[10px] text-emerald-500"></i>';
          }

          // Pulse submit button to guide user
          if (scanSubmitBtn) {
            scanSubmitBtn.classList.remove("btn-attention-pulse");
            void scanSubmitBtn.offsetWidth;
            scanSubmitBtn.classList.add("btn-attention-pulse");
          }
        } catch (err) {
          console.error("Failed to load sample retina:", err);
          if (statusEl) {
            statusEl.innerHTML = '<i class="fas fa-exclamation-triangle text-[10px] text-rose-500"></i>';
          }
        }
      });
    });
  }

  // Dashboard Live Search, Filter & Static Client-Side Pagination
  function initDashboardFilters() {
    const searchInput = document.getElementById("patientSearchInput") || document.getElementById("patientSearch");
    const filterButtons = document.querySelectorAll("[data-filter-risk]");
    const patientCards = Array.from(document.querySelectorAll("[data-patient-card], .patient-record-card"));
    const exportBtn = document.getElementById("exportCsvBtn");
    const noResults = document.getElementById("noFilterResults");
    const paginationControls = document.getElementById("paginationControls");
    const prevBtn = document.getElementById("prevPageBtn");
    const nextBtn = document.getElementById("nextPageBtn");
    const pageIndicator = document.getElementById("pageIndicator");

    if (!patientCards.length) return;

    const PAGE_SIZE = 3;
    let currentRiskFilter = "all";
    let searchQuery = "";
    let currentPage = 1;

    function getFilteredCards() {
      return patientCards.filter((card) => {
        const name = (card.getAttribute("data-patient-name") || "").toLowerCase();
        const risk = (card.getAttribute("data-patient-risk") || "").toLowerCase();
        const diag = (card.getAttribute("data-patient-diag") || "").toLowerCase();
        const sev = (card.getAttribute("data-patient-sev") || "").toLowerCase();

        const matchesSearch = !searchQuery || name.includes(searchQuery) || diag.includes(searchQuery);

        let matchesRisk = currentRiskFilter === "all";
        if (!matchesRisk) {
          if (currentRiskFilter === "healthy") {
            matchesRisk = risk === "healthy" || risk === "low" || sev.includes("none") || sev.includes("healthy") || diag.includes("no dr");
          } else if (currentRiskFilter === "moderate") {
            matchesRisk = risk === "moderate" || risk === "mild" || sev.includes("moderate") || sev.includes("mild");
          } else if (currentRiskFilter === "severe") {
            matchesRisk = risk === "severe" || risk === "high" || risk === "critical" || sev.includes("severe") || sev.includes("proliferative");
          }
        }

        return matchesSearch && matchesRisk;
      });
    }

    function render() {
      const filtered = getFilteredCards();
      const totalPages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));

      if (currentPage > totalPages) currentPage = totalPages;
      if (currentPage < 1) currentPage = 1;

      const startIndex = (currentPage - 1) * PAGE_SIZE;
      const endIndex = startIndex + PAGE_SIZE;

      // Hide all cards first
      patientCards.forEach((card) => card.classList.add("hidden"));

      // Show cards in current slice
      const currentSlice = filtered.slice(startIndex, endIndex);
      currentSlice.forEach((card) => card.classList.remove("hidden"));

      // Toggle no results box
      if (noResults) {
        noResults.classList.toggle("hidden", filtered.length > 0);
      }

      // Update pagination UI
      if (paginationControls) {
        paginationControls.classList.toggle("hidden", filtered.length === 0);
      }
      if (pageIndicator) {
        pageIndicator.textContent = `Page ${currentPage} of ${totalPages}`;
      }
      if (prevBtn) {
        prevBtn.disabled = (currentPage <= 1);
      }
      if (nextBtn) {
        nextBtn.disabled = (currentPage >= totalPages);
      }
    }

    // Filter Buttons Click
    filterButtons.forEach((btn) => {
      btn.addEventListener("click", (e) => {
        e.preventDefault();
        filterButtons.forEach((b) => {
          b.classList.remove("bg-indigo-600", "text-white", "font-bold");
          b.classList.add("glass-card", "text-slate-600", "dark:text-slate-300", "font-semibold");
        });
        btn.classList.add("bg-indigo-600", "text-white", "font-bold");
        btn.classList.remove("glass-card", "text-slate-600", "dark:text-slate-300", "font-semibold");

        currentRiskFilter = btn.getAttribute("data-filter-risk").toLowerCase();
        currentPage = 1;
        render();
      });
    });

    // Search Input
    if (searchInput) {
      searchInput.addEventListener("input", (e) => {
        searchQuery = e.target.value.trim().toLowerCase();
        currentPage = 1;
        render();
      });
    }

    // Pagination Buttons
    if (prevBtn) {
      prevBtn.addEventListener("click", (e) => {
        e.preventDefault();
        if (currentPage > 1) {
          currentPage--;
          render();
        }
      });
    }

    if (nextBtn) {
      nextBtn.addEventListener("click", (e) => {
        e.preventDefault();
        const filtered = getFilteredCards();
        const totalPages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
        if (currentPage < totalPages) {
          currentPage++;
          render();
        }
      });
    }

    // Export CSV
    if (exportBtn) {
      exportBtn.addEventListener("click", (e) => {
        e.preventDefault();
        const filtered = getFilteredCards();
        const rows = [["Patient Name", "Scan Count", "Diagnosis", "Severity", "Confidence (%)", "Captured At"]];
        filtered.forEach((card) => {
          const name = card.getAttribute("data-patient-name") || "";
          const count = card.getAttribute("data-patient-scans") || "1";
          const diag = card.getAttribute("data-patient-diag") || "";
          const sev = card.getAttribute("data-patient-sev") || "";
          const conf = card.getAttribute("data-patient-conf") || "";
          const time = card.getAttribute("data-patient-time") || "";
          rows.push([name, count, diag, sev, conf, time]);
        });

        const csvContent = "data:text/csv;charset=utf-8," + rows.map((e) => e.map((val) => `"${val}"`).join(",")).join("\n");
        const encodedUri = encodeURI(csvContent);
        const link = document.createElement("a");
        link.setAttribute("href", encodedUri);
        link.setAttribute("download", `retinascan_patient_registry_${new Date().toISOString().slice(0, 10)}.csv`);
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
      });
    }

    // Initial render
    render();
  }

  // Delegated Theme Toggle Event Handler (Guaranteed to work regardless of DOM ready state or navigation)
  function handleThemeToggle(e) {
    const toggle = e.target.closest("#themeToggle, [data-theme-toggle]");
    if (!toggle) return;
    // Prevent double toggle when both direct + delegated handlers fire for same click,
    // or when script is loaded twice (scanner.html duplicate include). The same
    // Event object bubbles from target to document, so a flag on the event dedupes.
    if (e.__themeToggleHandled) return;
    e.__themeToggleHandled = true;
    e.preventDefault();
    // Stop further propagation so second handler for same click does not re-toggle
    try { e.stopPropagation(); } catch (_) {}

    // Trigger organic ocular blink animation
    const eyeWrapper = toggle.querySelector("#themeEyeWrapper, .theme-eye-wrapper, #themeIconBox, .theme-icon-box");
    if (eyeWrapper) {
      eyeWrapper.classList.remove("eye-blink-anim", "theme-spin-burst");
      void eyeWrapper.offsetWidth; // Force reflow
      eyeWrapper.classList.add("eye-blink-anim");
    }

    // Trigger dynamic ripple burst
    const ripple = document.createElement("span");
    ripple.className = "theme-ripple";
    toggle.appendChild(ripple);
    setTimeout(() => ripple.remove(), 550);

    const isCurrentlyDark = html.classList.contains("dark") || html.getAttribute("data-theme") === "dark";
    const next = isCurrentlyDark ? "light" : "dark";
    applyTheme(next, true);
  }

  // Attach immediate document delegation
  document.addEventListener("click", handleThemeToggle);

  // Core Application Initializer
  function initApp() {
    applyTheme(currentTheme());

    // Direct binding for theme toggle buttons
    document.querySelectorAll("[data-theme-toggle], #themeToggle").forEach((toggle) => {
      if (toggle.__themeBound) return;
      toggle.__themeBound = true;
      toggle.addEventListener("click", handleThemeToggle);
    });

    // Mobile nav toggle handler with smooth icon flip and outside-click close
    const mobileBtn = document.getElementById("mobileMenuBtn");
    const mobileMenu = document.getElementById("mobileNavMenu");
    if (mobileBtn && mobileMenu && !mobileBtn.__navBound) {
      mobileBtn.__navBound = true;
      mobileBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        const isHidden = mobileMenu.classList.toggle("hidden");
        const icon = mobileBtn.querySelector("i");
        if (icon) {
          if (isHidden) {
            icon.className = "fas fa-bars text-lg";
          } else {
            icon.className = "fas fa-times text-lg text-indigo-600 dark:text-indigo-400";
          }
        }
      });

      // Close mobile menu on click outside
      document.addEventListener("click", (e) => {
        if (!mobileMenu.classList.contains("hidden") && !mobileMenu.contains(e.target) && !mobileBtn.contains(e.target)) {
          mobileMenu.classList.add("hidden");
          const icon = mobileBtn.querySelector("i");
          if (icon) icon.className = "fas fa-bars text-lg";
        }
      });

      // Close mobile menu on link click
      mobileMenu.querySelectorAll("a").forEach(link => {
        link.addEventListener("click", () => {
          mobileMenu.classList.add("hidden");
          const icon = mobileBtn.querySelector("i");
          if (icon) icon.className = "fas fa-bars text-lg";
        });
      });
    }

    // Initialize interactive subsystems
    initInteractiveClickGrid();
    initNeuralInspector();
    initSampleLoader();
    initDashboardFilters();
    initScrollReveal();
    initAnimatedCounters();
    initPresetPulse();
    initCopyBadges();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initApp);
  } else {
    initApp();
  }

  // ==========================================================================
  // Scroll-Reveal Observer (IntersectionObserver-based)
  // ==========================================================================
  function initScrollReveal() {
    const elements = document.querySelectorAll('.reveal-on-scroll');
    if (!elements.length) return;

    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.classList.add('is-visible');
          observer.unobserve(entry.target);
        }
      });
    }, {
      threshold: 0.12,
      rootMargin: '0px 0px -40px 0px'
    });

    elements.forEach(el => observer.observe(el));
  }

  // ==========================================================================
  // Animated Counter Utility
  // ==========================================================================
  function animateCounter(el, target, duration = 1200) {
    const start = 0;
    const startTime = performance.now();

    // If target is a string like "< 2.5s", extract number
    let numericTarget = parseFloat(target);
    if (isNaN(numericTarget)) {
      el.textContent = target;
      return;
    }

    const isFloat = String(target).includes('.');
    const prefix = String(target).replace(/[\d.]+/, '').split(/[\d.]/)[0] || '';
    const suffix = String(target).replace(/^[^0-9]*[\d.]+/, '') || '';

    function tick(now) {
      const elapsed = now - startTime;
      const progress = Math.min(elapsed / duration, 1);
      // Ease-out cubic
      const eased = 1 - Math.pow(1 - progress, 3);
      const current = start + (numericTarget - start) * eased;

      if (isFloat) {
        el.textContent = prefix + current.toFixed(1) + suffix;
      } else {
        el.textContent = prefix + Math.round(current) + suffix;
      }

      if (progress < 1) {
        requestAnimationFrame(tick);
      }
    }

    requestAnimationFrame(tick);
  }

  function initAnimatedCounters() {
    const counters = document.querySelectorAll('[data-counter]');
    if (!counters.length) return;

    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          const el = entry.target;
          const target = el.getAttribute('data-counter');
          const duration = parseInt(el.getAttribute('data-counter-duration') || '1200', 10);
          animateCounter(el, target, duration);
          observer.unobserve(el);
        }
      });
    }, {
      threshold: 0.3
    });

    counters.forEach(el => observer.observe(el));
  }

  // ==========================================================================
  // Preset Button Click Micro-Animation
  // ==========================================================================
  function initPresetPulse() {
    document.querySelectorAll('[data-sample-url]').forEach(btn => {
      btn.addEventListener('click', () => {
        btn.classList.remove('preset-pulse');
        void btn.offsetWidth; // Force reflow to restart animation
        btn.classList.add('preset-pulse');
      });
    });
  }

  // ==========================================================================
  // One-Click Copy-to-Clipboard with Feedback Tooltip
  // ==========================================================================
  function initCopyBadges() {
    document.querySelectorAll('[data-copy]').forEach((btn) => {
      if (btn.__copyBound) return;
      btn.__copyBound = true;
      btn.addEventListener('click', async (e) => {
        e.preventDefault();
        const textToCopy = btn.getAttribute('data-copy');
        if (!textToCopy) return;
        try {
          if (navigator.clipboard && navigator.clipboard.writeText) {
            await navigator.clipboard.writeText(textToCopy);
          } else {
            const temp = document.createElement('input');
            temp.value = textToCopy;
            document.body.appendChild(temp);
            temp.select();
            document.execCommand('copy');
            document.body.removeChild(temp);
          }
          btn.classList.add('is-copied');
          setTimeout(() => btn.classList.remove('is-copied'), 2000);
        } catch (err) {
          console.warn('Copy failed:', err);
        }
      });
    });
  }

  // ==========================================================================
  // ==========================================================================
  // Interactive Grid Background Delegation
  // ==========================================================================
  function initInteractiveClickGrid() {
    if (typeof InteractiveGridBackground !== "undefined" && !window.__gridBackgroundInstance) {
      window.__gridBackgroundInstance = new InteractiveGridBackground();
    }
  }

  window.__retinascan = {
    applyTheme,
    currentTheme,
    initInteractiveClickGrid,
    initNeuralInspector,
    initSampleLoader,
    initDashboardFilters,
    initScrollReveal,
    initAnimatedCounters,
    animateCounter
  };
})();
