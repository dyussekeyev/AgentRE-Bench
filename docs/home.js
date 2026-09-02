(() => {
  const root = document.documentElement;
  const toggle = document.querySelector(".theme-toggle");
  const label = document.querySelector(".theme-label");
  const storageKey = "agentre-theme-v2";

  let savedTheme = null;
  try {
    savedTheme = window.localStorage.getItem(storageKey);
  } catch (_) {
    savedTheme = null;
  }

  const applyTheme = (theme) => {
    root.dataset.theme = theme;
    const isDark = theme === "dark";
    toggle?.setAttribute("aria-label", `Switch to ${isDark ? "light" : "dark"} theme`);
    toggle?.setAttribute("aria-pressed", String(!isDark));
    if (label) label.textContent = isDark ? "Light" : "Dark";
  };

  applyTheme(savedTheme === "light" ? "light" : "dark");

  toggle?.addEventListener("click", () => {
    const nextTheme = root.dataset.theme === "dark" ? "light" : "dark";
    applyTheme(nextTheme);
    try {
      window.localStorage.setItem(storageKey, nextTheme);
    } catch (_) {
      // Theme switching still works when storage is unavailable.
    }
  });

  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const revealItems = [...document.querySelectorAll(".reveal")];

  if (!reduceMotion && "IntersectionObserver" in window) {
    root.classList.add("motion-ready");
    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        entry.target.classList.add("is-visible");
        observer.unobserve(entry.target);
      });
    }, { rootMargin: "0px 0px -8%", threshold: 0.12 });

    revealItems.forEach((item) => observer.observe(item));
  } else {
    revealItems.forEach((item) => item.classList.add("is-visible"));
  }
})();
