import { useEffect, useState } from "react";
import { NavLink } from "react-router-dom";

const NAV = [
  {
    group: "Operate",
    items: [
      { to: "/", label: "Command Center", end: true, icon: "M2 3.5A1.5 1.5 0 013.5 2h9A1.5 1.5 0 0114 3.5v9a1.5 1.5 0 01-1.5 1.5h-9A1.5 1.5 0 012 12.5v-9zM3 7h4V3H3.5a.5.5 0 00-.5.5V7zm5 0h5V3.5a.5.5 0 00-.5-.5H8v4zM3 8v4.5a.5.5 0 00.5.5H7V8H3zm5 5h4.5a.5.5 0 00.5-.5V8H8v5z" },
      { to: "/actions", label: "Action List", icon: "M2.5 4h11a.5.5 0 010 1h-11a.5.5 0 010-1zm0 3.5h11a.5.5 0 010 1h-11a.5.5 0 010-1zM2.5 11h7a.5.5 0 010 1h-7a.5.5 0 010-1z" },
      { to: "/customer", label: "Customer 360", icon: "M8 8a3 3 0 100-6 3 3 0 000 6zm-5.5 6.5C2.5 11.5 5 10 8 10s5.5 1.5 5.5 4.5a.5.5 0 01-.5.5H3a.5.5 0 01-.5-.5z" },
      { to: "/budget", label: "Budget Optimizer", icon: "M8 1.5a.5.5 0 01.5.5v1.05a2.6 2.6 0 011.9 1.2.5.5 0 11-.85.52A1.7 1.7 0 008 4c-1 0-1.6.5-1.6 1.1 0 .7.6 1 1.9 1.3 1.6.4 2.7 1 2.7 2.4 0 1.2-.9 2-2.2 2.2V12a.5.5 0 01-1 0v-1c-1-.1-1.8-.6-2.2-1.4a.5.5 0 01.88-.48c.3.6.9.9 1.8.9 1.1 0 1.74-.45 1.74-1.15 0-.72-.6-1.03-1.98-1.36C6.4 7.15 5.4 6.5 5.4 5.15c0-1.1.85-1.9 2.1-2.1V2a.5.5 0 01.5-.5zM2 14.5a.5.5 0 01.5-.5h11a.5.5 0 010 1h-11a.5.5 0 01-.5-.5z" },
    ],
  },
  {
    group: "Verify",
    items: [
      { to: "/models", label: "Model Lab", icon: "M2 13.5a.5.5 0 01.5-.5H3V9a.5.5 0 011 0v4h1.5V6.5a.5.5 0 011 0V13H8V4a.5.5 0 011 0v9h1.5V7.5a.5.5 0 011 0V13h.5V10a.5.5 0 011 0v3h.5a.5.5 0 010 1h-11a.5.5 0 01-.5-.5z" },
      { to: "/responsible-ai", label: "Responsible AI", icon: "M8 1.2l5.3 2v4.2c0 3.2-2.2 6.1-5.3 7.4-3.1-1.3-5.3-4.2-5.3-7.4V3.2l5.3-2zm2.6 4.4a.5.5 0 01.05.7l-3 3.5a.5.5 0 01-.73.03L5.3 8.6a.5.5 0 11.7-.7l1.24 1.23 2.65-3.1a.5.5 0 01.71-.05z" },
    ],
  },
  {
    group: "Ask",
    items: [
      { to: "/copilot", label: "Churn Copilot", icon: "M8 2c3.3 0 6 2.2 6 5s-2.7 5-6 5c-.6 0-1.2-.07-1.75-.2L3.3 13.4a.4.4 0 01-.58-.43l.44-2.2C2.63 9.9 2 8.5 2 7c0-2.8 2.7-5 6-5zm-2.2 5a.95.95 0 100 1.9.95.95 0 000-1.9zm2.2 0a.95.95 0 100 1.9.95.95 0 000-1.9zm2.2 0a.95.95 0 100 1.9.95.95 0 000-1.9z" },
    ],
  },
];

const STACK = ["XGBoost", "EconML", "DoWhy", "SHAP", "Fairlearn"];

function useTheme() {
  const [theme, setTheme] = useState(() => {
    try {
      return localStorage.getItem("cs-theme") || "";
    } catch {
      return "";
    }
  });
  useEffect(() => {
    if (theme) document.documentElement.setAttribute("data-theme", theme);
    else document.documentElement.removeAttribute("data-theme");
    try {
      if (theme) localStorage.setItem("cs-theme", theme);
    } catch {}
  }, [theme]);
  const toggle = () => {
    const isDark = theme
      ? theme === "dark"
      : window.matchMedia("(prefers-color-scheme: dark)").matches;
    setTheme(isDark ? "light" : "dark");
  };
  return toggle;
}

export default function Shell({ title, crumb, actions, summary, children }) {
  const toggleTheme = useTheme();

  return (
    <div className="grid min-h-screen" style={{ gridTemplateColumns: "232px 1fr" }}>
      <nav
        className="sticky top-0 flex h-screen flex-col overflow-y-auto"
        style={{ background: "var(--card)", borderRight: "1px solid var(--stroke)" }}
      >
        <div className="flex items-center gap-2.5 px-4 pb-3.5 pt-4">
          <div className="grid h-[22px] w-[22px] shrink-0 grid-cols-2 gap-[2px]" aria-hidden>
            <i style={{ background: "#F25022" }} />
            <i style={{ background: "#7FBA00" }} />
            <i style={{ background: "#00A4EF" }} />
            <i style={{ background: "#FFB900" }} />
          </div>
          <div>
            <div className="text-[15px] font-semibold tracking-[-0.01em]" style={{ fontFamily: "var(--font-d)" }}>
              ChurnShield
            </div>
            <div className="eyebrow">Retention Console</div>
          </div>
        </div>

        <div className="flex flex-col gap-px px-2 pb-2">
          {NAV.map((g) => (
            <div key={g.group} className="flex flex-col gap-px">
              <div className="eyebrow px-2.5 pb-1 pt-3">{g.group}</div>
              {g.items.map((it) => (
                <NavLink
                  key={it.to}
                  to={it.to}
                  end={it.end}
                  className="relative flex w-full items-center gap-2.5 rounded px-2.5 py-2 text-[13.5px]"
                  style={({ isActive }) => ({
                    background: isActive ? "var(--brand-soft)" : "transparent",
                    color: isActive ? "var(--brand-ink)" : "var(--ink-2)",
                    fontWeight: isActive ? 600 : 400,
                    textDecoration: "none",
                  })}
                >
                  {({ isActive }) => (
                    <>
                      {isActive && (
                        <span
                          className="absolute bottom-[7px] left-0 top-[7px] w-[3px] rounded"
                          style={{ background: "var(--brand)" }}
                        />
                      )}
                      <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor" className="shrink-0">
                        <path d={it.icon} />
                      </svg>
                      {it.label}
                    </>
                  )}
                </NavLink>
              ))}
            </div>
          ))}
        </div>

        <div
          className="mt-auto flex flex-col gap-1.5 px-4 py-3.5 text-[11.5px]"
          style={{ borderTop: "1px solid var(--stroke)", color: "var(--ink-3)" }}
        >
          <div className="flex flex-wrap gap-1.5">
            {STACK.map((s) => (
              <b key={s} className="tag text-[10.5px]">
                {s}
              </b>
            ))}
          </div>
          <div>
            {summary?.data?.source ?? "not loaded"}
            <br />
            {summary ? `${summary.data.rows_clean.toLocaleString()} customers` : ""}
            {summary?.generated_at ? ` · scored ${summary.generated_at.replace("T", " ")}` : ""}
          </div>
          {summary?.data?.synthetic && (
            <div className="flag no mt-1">Synthetic data</div>
          )}
        </div>
      </nav>

      <div className="flex min-w-0 flex-col">
        <div
          className="sticky top-0 z-20 flex items-center gap-3.5 px-6 py-3"
          style={{ background: "var(--card)", borderBottom: "1px solid var(--stroke)" }}
        >
          <div>
            <h1 className="text-[18px] tracking-[-0.015em]">{title}</h1>
            <div className="text-[12px]" style={{ color: "var(--ink-3)" }}>
              {crumb}
            </div>
          </div>
          <div className="flex-1" />
          {actions}
          <button className="btn ghost" onClick={toggleTheme}>
            Theme
          </button>
        </div>
        <div className="flex max-w-[1500px] flex-col gap-4 px-6 pb-11 pt-5">{children}</div>
      </div>
    </div>
  );
}
