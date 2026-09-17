import { quadClass, QUADRANT_MEANING } from "../lib/format";

export function Card({ title, subtitle, children, right, className = "" }) {
  return (
    <section className={`card ${className}`}>
      {(title || right) && (
        <header className="flex items-start gap-3 px-4 pt-3.5">
          <div className="min-w-0">
            {title && <h3 className="text-[14.5px]">{title}</h3>}
            {subtitle && (
              <p className="mt-0.5 text-[12px] max-w-[64ch]" style={{ color: "var(--ink-3)" }}>
                {subtitle}
              </p>
            )}
          </div>
          {right && <div className="ml-auto shrink-0">{right}</div>}
        </header>
      )}
      <div className="px-4 pb-4 pt-3.5">{children}</div>
    </section>
  );
}

export function Kpi({ label, value, detail, accent, badge }) {
  const bar =
    accent === "bad" ? "var(--bad)" : accent === "good" ? "var(--good)" : null;
  return (
    <div
      className="card relative overflow-hidden px-4 pt-3.5 pb-3 flex flex-col gap-0.5"
      style={
        accent === "hero"
          ? { background: "linear-gradient(180deg, var(--brand-soft), var(--card) 78%)" }
          : undefined
      }
    >
      {bar && (
        <span className="absolute left-0 top-0 bottom-0 w-[3px]" style={{ background: bar }} />
      )}
      <div className="eyebrow flex items-center gap-2">
        {label}
        {badge}
      </div>
      <div
        className="mono text-[29px] font-semibold leading-tight tracking-[-0.025em]"
        style={{
          fontFamily: "var(--font-d)",
          color: accent === "hero" ? "var(--brand-ink)" : undefined,
        }}
      >
        {value}
      </div>
      {detail && (
        <div className="text-[11.5px]" style={{ color: "var(--ink-2)" }}>
          {detail}
        </div>
      )}
    </div>
  );
}

export function Quadrant({ value, title = true }) {
  return (
    <span className={quadClass(value)} title={title ? QUADRANT_MEANING[value] : undefined}>
      {value}
    </span>
  );
}

export function Note({ children, caution = false }) {
  return <div className={`note ${caution ? "caution" : ""}`}>{children}</div>;
}

export function Segmented({ options, value, onChange }) {
  return (
    <div
      className="inline-flex gap-0.5 rounded-[5px] p-0.5"
      style={{ background: "var(--card-sunk)", border: "1px solid var(--stroke)" }}
    >
      {options.map((o) => {
        const on = o.value === value;
        return (
          <button
            key={o.value}
            aria-pressed={on}
            onClick={() => onChange(o.value)}
            className="rounded-[3px] px-3 py-1 text-[12.5px] cursor-pointer border-0"
            style={{
              background: on ? "var(--card)" : "transparent",
              color: on ? "var(--ink)" : "var(--ink-2)",
              fontWeight: on ? 600 : 400,
              boxShadow: on ? "var(--shadow)" : "none",
            }}
          >
            {o.label}
          </button>
        );
      })}
    </div>
  );
}

export function Slider({ label, display, min, max, step, value, onChange }) {
  return (
    <div>
      <label className="eyebrow">{label}</label>
      <div
        className="mb-1.5 mt-1 text-[26px] font-semibold tracking-[-0.02em]"
        style={{ fontFamily: "var(--font-d)" }}
      >
        {display}
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </div>
  );
}

export function Legend({ items }) {
  return (
    <div className="mt-2.5 flex flex-wrap items-center gap-x-4 gap-y-1.5 text-[12px]" style={{ color: "var(--ink-2)" }}>
      {items.map((i) => (
        <span key={i.label} className="inline-flex items-center gap-1.5">
          <i
            className="inline-block shrink-0"
            style={{
              background: i.color,
              width: i.line ? 16 : 11,
              height: i.line ? 3 : 11,
              borderRadius: 2,
            }}
          />
          {i.label}
        </span>
      ))}
    </div>
  );
}

export function DefinitionRow({ label, value }) {
  return (
    <div
      className="flex justify-between gap-3 py-1.5 text-[12.5px]"
      style={{ borderBottom: "1px solid var(--grid)" }}
    >
      <span style={{ color: "var(--ink-2)" }}>{label}</span>
      <span className="mono font-semibold">{value}</span>
    </div>
  );
}

export function Loading({ what = "data" }) {
  return (
    <div className="p-10 text-center text-[13px]" style={{ color: "var(--ink-3)" }}>
      Loading {what}...
    </div>
  );
}

export function ErrorPanel({ error }) {
  const notBuilt = String(error).includes("Artifacts not built");
  return (
    <div className="card p-6">
      <h3 className="mb-2" style={{ color: "var(--bad-ink)" }}>
        {notBuilt ? "Pipeline has not been run yet" : "Could not reach the API"}
      </h3>
      <p className="text-[13px]" style={{ color: "var(--ink-2)" }}>
        {String(error)}
      </p>
      <pre
        className="mono mt-3 overflow-x-auto rounded p-3 text-[12px]"
        style={{ background: "var(--card-sunk)", color: "var(--ink-2)" }}
      >
{notBuilt
  ? `cd backend
.venv\\Scripts\\python -m src.build_artifacts`
  : `cd backend
.venv\\Scripts\\python -m uvicorn app.main:app --reload --port 8000`}
      </pre>
    </div>
  );
}

export function ChartTooltip({ active, payload, label, rows }) {
  if (!active || !payload?.length) return null;
  return (
    <div
      className="rounded px-2.5 py-2 text-[12px]"
      style={{
        background: "var(--card)",
        border: "1px solid var(--stroke-strong)",
        boxShadow: "var(--shadow-8)",
      }}
    >
      {label !== undefined && <b className="mb-1 block text-[12.5px]">{label}</b>}
      {(rows ? rows(payload) : payload.map((p) => ({ k: p.name, v: p.value }))).map((r, i) => (
        <div key={i} className="flex justify-between gap-4">
          <span style={{ color: "var(--ink-2)" }}>{r.k}</span>
          <span className="mono">{r.v}</span>
        </div>
      ))}
    </div>
  );
}
