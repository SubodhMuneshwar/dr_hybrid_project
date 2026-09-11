// Shared light/dark theme — used by index, scanner, dashboard
(function () {
  const STORAGE_KEY = "retinascan-theme";
  const html = document.documentElement;

  function applyTheme(theme) {
    const isDark = theme === "dark";
    html.classList.toggle("dark", isDark);
    // sync all toggles on the page
    document.querySelectorAll("[data-theme-toggle]").forEach((toggle) => {
      const circle = toggle.querySelector("[data-toggle-circle]");
      const icon = toggle.querySelector("[data-toggle-icon]");
      if (circle) circle.style.transform = isDark ? "translateX(36px)" : "translateX(0px)";
      if (icon) icon.textContent = isDark ? "\u263E" : "\u2600";
      toggle.setAttribute("aria-pressed", String(isDark));
      toggle.classList.toggle("from-yellow-200", !isDark);
      toggle.classList.toggle("to-slate-300", !isDark);
      toggle.classList.toggle("from-slate-700", isDark);
      toggle.classList.toggle("to-slate-900", isDark);
    });
  }

  function currentTheme() {
    return html.classList.contains("dark") ? "dark" : "light";
  }

  // init from localStorage or prefers-color-scheme
  let saved = null;
  try { saved = localStorage.getItem(STORAGE_KEY); } catch (e) {}
  if (saved === "dark" || saved === "light") {
    applyTheme(saved);
  } else if (window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches) {
    applyTheme("dark");
  } else {
    applyTheme("light");
  }

  // delegate click on any toggle
  document.addEventListener("click", function (e) {
    const t = e.target.closest("[data-theme-toggle]");
    if (!t) return;
    const next = currentTheme() === "dark" ? "light" : "dark";
    applyTheme(next);
    try { localStorage.setItem(STORAGE_KEY, next); } catch (e) {}
  });

  // expose for inline scripts if needed
  window.__retinascanTheme = { applyTheme, currentTheme };
})();
