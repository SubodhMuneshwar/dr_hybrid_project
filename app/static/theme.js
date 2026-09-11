// ==========================================================================
// RetinaScan AI - Interactive Engine
// - Unified dark/light theme switching with localStorage persistence
// - Interactive Before/After image comparison slider
// - One-click clinical sample image loader
// - Real-time dashboard search, filtering, and CSV export
// - Responsive mobile drawer navigation
// ==========================================================================

(function () {
  const STORAGE_KEY = "theme";
  const html = document.documentElement;

  function applyTheme(theme) {
    const isDark = theme === "dark";
    html.classList.toggle("dark", isDark);

    // Sync all ocular buttons on page
    document.querySelectorAll("[data-theme-toggle], #themeToggle").forEach((toggle) => {
      toggle.setAttribute("aria-pressed", String(isDark));
      toggle.setAttribute(
        "title",
        isDark ? "Open eye: switch to daylight mode" : "Close eye: switch to night mode"
      );
      toggle.setAttribute(
        "aria-label",
        isDark ? "Switch to light mode (open eye)" : "Switch to dark mode (close eye)"
      );
    });
  }

  function currentTheme() {
    return html.classList.contains("dark") ? "dark" : "light";
  }

  // Load saved preference or system default
  let saved = null;
  try {
    saved = localStorage.getItem(STORAGE_KEY);
  } catch (e) {}

  if (saved === "dark" || saved === "light") {
    applyTheme(saved);
  } else if (window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches) {
    applyTheme("dark");
  } else {
    applyTheme("light");
  }

  // Before/After Comparison Slider Initializer
  function initComparisonSlider() {
    const container = document.getElementById("comparisonContainer");
    const overlay = document.getElementById("comparisonOverlay");
    const handle = document.getElementById("comparisonHandle");
    if (!container || !overlay || !handle) return;

    let isDragging = false;

    function setPosition(x) {
      const rect = container.getBoundingClientRect();
      let pos = (x - rect.left) / rect.width;
      if (pos < 0.05) pos = 0.05;
      if (pos > 0.95) pos = 0.95;
      const pct = pos * 100;
      overlay.style.width = pct + "%";
      handle.style.left = pct + "%";
    }

    function onPointerDown(e) {
      isDragging = true;
      setPosition(e.clientX || (e.touches && e.touches[0].clientX));
    }

    function onPointerMove(e) {
      if (!isDragging) return;
      setPosition(e.clientX || (e.touches && e.touches[0].clientX));
    }

    function onPointerUp() {
      isDragging = false;
    }

    container.addEventListener("mousedown", onPointerDown);
    window.addEventListener("mousemove", onPointerMove);
    window.addEventListener("mouseup", onPointerUp);

    container.addEventListener("touchstart", onPointerDown, { passive: true });
    window.addEventListener("touchmove", onPointerMove, { passive: true });
    window.addEventListener("touchend", onPointerUp);
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
    const diagnoseBtn = document.getElementById("diagnoseBtn");

    if (!sampleButtons.length || !fileInput) return;

    sampleButtons.forEach((btn) => {
      btn.addEventListener("click", async (e) => {
        e.preventDefault();
        const url = btn.getAttribute("data-sample-url");
        const name = btn.getAttribute("data-sample-name") || "sample_retina.jpg";
        const patientName = btn.getAttribute("data-patient-default") || "Jane Doe";

        try {
          btn.innerHTML = '<i class="fas fa-spinner fa-spin text-xs"></i> Loading...';
          const res = await fetch(url);
          const blob = await res.blob();
          const file = new File([blob], name, { type: blob.type || "image/jpeg" });

          // Create DataTransfer to populate native file input
          const dt = new DataTransfer();
          dt.items.add(file);
          fileInput.files = dt.files;

          if (patientInput && !patientInput.value) {
            patientInput.value = patientName;
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
              if (diagnoseBtn) diagnoseBtn.disabled = false;
            };
            reader.readAsDataURL(file);
          }

          btn.innerHTML = '<i class="fas fa-check text-xs text-emerald-500"></i> Loaded';
          setTimeout(() => {
            btn.innerHTML = `<i class="fas fa-magic text-xs text-indigo-500"></i> ${name.replace('.jpg', '')}`;
          }, 1800);
        } catch (err) {
          console.error("Failed to load sample retina:", err);
          btn.innerHTML = '<i class="fas fa-exclamation-triangle text-xs text-rose-500"></i> Error';
        }
      });
    });
  }

  // Dashboard Live Search & Filter
  function initDashboardFilters() {
    const searchInput = document.getElementById("patientSearch");
    const filterButtons = document.querySelectorAll("[data-filter-risk]");
    const patientCards = document.querySelectorAll(".patient-record-card");
    const exportBtn = document.getElementById("exportCsvBtn");
    const recordCountEl = document.getElementById("visibleRecordCount");

    if (!patientCards.length) return;

    let currentRiskFilter = "all";
    let searchQuery = "";

    function filterCards() {
      let visible = 0;
      patientCards.forEach((card) => {
        const name = (card.getAttribute("data-patient-name") || "").toLowerCase();
        const risk = (card.getAttribute("data-patient-risk") || "").toLowerCase();
        const diag = (card.getAttribute("data-patient-diag") || "").toLowerCase();

        const matchesSearch = !searchQuery || name.includes(searchQuery) || diag.includes(searchQuery);
        const matchesRisk = currentRiskFilter === "all" || risk === currentRiskFilter;

        if (matchesSearch && matchesRisk) {
          card.classList.remove("hidden");
          visible++;
        } else {
          card.classList.add("hidden");
        }
      });

      if (recordCountEl) {
        recordCountEl.textContent = `${visible} records`;
      }
    }

    if (searchInput) {
      searchInput.addEventListener("input", (e) => {
        searchQuery = e.target.value.trim().toLowerCase();
        filterCards();
      });
    }

    filterButtons.forEach((btn) => {
      btn.addEventListener("click", () => {
        filterButtons.forEach((b) => {
          b.classList.remove("bg-indigo-600", "text-white");
          b.classList.add("glass-card", "text-slate-600", "dark:text-slate-300");
        });
        btn.classList.add("bg-indigo-600", "text-white");
        btn.classList.remove("glass-card", "text-slate-600", "dark:text-slate-300");

        currentRiskFilter = btn.getAttribute("data-filter-risk").toLowerCase();
        filterCards();
      });
    });

    if (exportBtn) {
      exportBtn.addEventListener("click", () => {
        const rows = [["Patient Name", "Scan Count", "Diagnosis", "Severity", "Confidence (%)", "Captured At"]];
        patientCards.forEach((card) => {
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
  }

  // Global DOM Loaded Handler
  document.addEventListener("DOMContentLoaded", () => {
    applyTheme(currentTheme());

    // Theme toggle buttons (Interactive Ocular Eye: Open vs Closed Eye)
    document.querySelectorAll("[data-theme-toggle], #themeToggle").forEach((toggle) => {
      toggle.addEventListener("click", () => {
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

        const next = currentTheme() === "dark" ? "light" : "dark";
        try {
          localStorage.setItem(STORAGE_KEY, next);
        } catch (e) {}
        applyTheme(next);
      });
    });

    // Mobile nav toggle handler
    const mobileBtn = document.getElementById("mobileMenuBtn");
    const mobileMenu = document.getElementById("mobileNavMenu");
    if (mobileBtn && mobileMenu) {
      mobileBtn.addEventListener("click", () => {
        mobileMenu.classList.toggle("hidden");
      });
    }

    // Initialize interactive subsystems
    initComparisonSlider();
    initSampleLoader();
    initDashboardFilters();
  });

  window.__retinascan = {
    applyTheme,
    currentTheme,
    initComparisonSlider,
    initSampleLoader,
    initDashboardFilters
  };
})();
