import { useEffect, useState } from "react";
import { useSearchParams } from "react-router-dom";
import {
  Bar, BarChart, CartesianGrid, Cell, ErrorBar, ReferenceLine,
  ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis,
} from "recharts";
import { api } from "../lib/api";
import { Card, Kpi, Loading, Note, Quadrant } from "../components/ui";
import { money, pct, pp } from "../lib/format";

// Which controls the simulator shows is decided by the backend, from the
// active dataset's real model features. The UI never hardcodes a dataset's
// vocabulary, and a field the model never saw can't appear as a slider.

/** The customer's current values for every editable field. */
function baselineOf(d) {
  const out = {};
  for (const f of d.editable ?? []) {
    if (d.profile[f.field] !== undefined) out[f.field] = d.profile[f.field];
  }
  return out;
}

export default function Customer360({ summary }) {
  const [params, setParams] = useSearchParams();
  const id = params.get("id");
  const [list, setList] = useState([]);
  const [c, setC] = useState(null);
  const [whatif, setWhatif] = useState({});
  const [sim, setSim] = useState(null);

  useEffect(() => {
    api.customers({ limit: 120 }).then((d) => {
      setList(d.rows);
      if (!id && d.rows.length) setParams({ id: d.rows[0].id }, { replace: true });
    });
  }, []);

  useEffect(() => {
    if (!id) return;
    setC(null);
    setSim(null);
    api.customer(id).then((d) => {
      setC(d);
      setWhatif(baselineOf(d));
    });
  }, [id]);

  useEffect(() => {
    if (!c || !Object.keys(whatif).length) return;
    const t = setTimeout(() => {
      api.simulate({ customer_id: c.id, changes: whatif }).then(setSim).catch(() => {});
    }, 200);
    return () => clearTimeout(t);
  }, [c, whatif]);

  if (!c) return <Loading what="customer" />;

  const shap = [...(c.shap ?? [])].reverse();
  const cate = c.treatments.map((t) => ({
    ...t,
    err: [t.effect - t.lo, t.hi - t.effect],
  }));
  const best = c.treatments[0];

  const jumpToSleeper = () => {
    api.customers({ quadrant: "Sleeping dog", limit: 40 }).then((d) => {
      const pool = d.rows.filter((r) => r.id !== c.id);
      if (pool.length) setParams({ id: pool[Math.floor(Math.random() * pool.length)].id });
    });
  };

  return (
    <>
      <div className="flex flex-wrap items-center gap-2.5">
        <select value={id ?? ""} onChange={(e) => setParams({ id: e.target.value })} style={{ minWidth: 300 }}>
          {list.map((r) => (
            <option key={r.id} value={r.id}>
              {r.id} — {pct(r.churn_prob, 0)} risk · {r.quadrant}
            </option>
          ))}
          {!list.find((r) => r.id === c.id) && <option value={c.id}>{c.id}</option>}
        </select>
        <button className="btn ghost" onClick={jumpToSleeper}>
          Jump to a sleeping dog
        </button>
      </div>

      <div className="grid gap-3.5 md:grid-cols-2 xl:grid-cols-4">
        <Kpi
          label="Customer"
          value={<span className="text-[22px]">{c.id}</span>}
          detail={`${Math.round(c.tenure)} months · ${money(c.monthly)}/mo`}
        />
        <Kpi
          label="Churn probability"
          value={pct(c.churn_prob, 0)}
          detail={`Base rate for this portfolio is ${pct(summary.base_rate)}`}
          accent={c.churn_prob > 0.6 ? "bad" : undefined}
        />
        <Kpi
          label="Best available uplift"
          value={<span style={{ color: c.best_uplift > 0 ? "var(--good-ink)" : "var(--bad-ink)" }}>{pp(c.best_uplift)}</span>}
          detail={`via ${best.label}`}
        />
        <Kpi
          label="Classification"
          value={<span className="text-[18px]"><Quadrant value={c.quadrant} /></span>}
          detail={
            c.quadrant === "Sleeping dog"
              ? "Contacting this customer is estimated to increase churn"
              : c.quadrant === "Persuadable"
              ? `Expected net value ${money(c.net)}`
              : c.quadrant === "Lost cause"
              ? "High risk, but no offer moves the needle"
              : "Likely to stay without intervention"
          }
        />
      </div>

      <div className="grid gap-3.5 xl:grid-cols-2">
        <Card
          title="Why the model says this"
          subtitle={`Local SHAP contribution: how each feature pushed this one prediction away from the ${pct(
            summary.base_rate
          )} base rate.`}
        >
          <div style={{ width: "100%", height: 330 }}>
            <ResponsiveContainer>
              <BarChart data={shap} layout="vertical" margin={{ top: 8, right: 44, bottom: 4, left: 150 }}>
                <CartesianGrid stroke="var(--grid)" horizontal={false} />
                <XAxis type="number" tickFormatter={(v) => v.toFixed(2)} />
                <YAxis type="category" dataKey="feature" width={148} tick={{ fontSize: 11 }} />
                <ReferenceLine x={0} stroke="var(--stroke-strong)" />
                <Tooltip
                  cursor={{ fill: "var(--card-alt)" }}
                  contentStyle={{ background: "var(--card)", border: "1px solid var(--stroke-strong)", borderRadius: 4, fontSize: 12 }}
                  formatter={(v) => [v.toFixed(3), v > 0 ? "Increases churn risk" : "Reduces churn risk"]}
                />
                <Bar dataKey="value" radius={3} barSize={13}>
                  {shap.map((d, i) => (
                    <Cell key={i} fill={d.value > 0 ? "var(--bad)" : "var(--good)"} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="mt-2 flex justify-between text-[11px]" style={{ color: "var(--ink-3)" }}>
            <span style={{ color: "var(--good)" }}>← reduces churn risk</span>
            <span style={{ color: "var(--bad)" }}>increases churn risk →</span>
          </div>
        </Card>

        <Card
          title="Would an offer change the outcome?"
          subtitle={`Estimated causal effect of each intervention on retention, with 90% confidence intervals from ${summary.causal_estimator}.`}
        >
          <div style={{ width: "100%", height: 260 }}>
            <ResponsiveContainer>
              <ScatterChart margin={{ top: 12, right: 24, bottom: 24, left: 118 }}>
                <CartesianGrid stroke="var(--grid)" />
                <XAxis
                  type="number"
                  dataKey="effect"
                  tickFormatter={(v) => `${v > 0 ? "+" : ""}${(v * 100).toFixed(0)}`}
                  label={{ value: "Change in retention (percentage points)", position: "insideBottom", offset: -14, fill: "var(--ink-3)", fontSize: 11 }}
                />
                <YAxis type="category" dataKey="label" width={116} tick={{ fontSize: 11.5 }} />
                <ReferenceLine x={0} stroke="var(--ink-3)" strokeDasharray="4 3" />
                <Tooltip
                  cursor={{ strokeDasharray: "3 3" }}
                  contentStyle={{ background: "var(--card)", border: "1px solid var(--stroke-strong)", borderRadius: 4, fontSize: 12 }}
                  formatter={(v, n, p) => {
                    const d = p.payload;
                    return [
                      `${pp(d.effect)}  (90% CI ${pp(d.lo)} to ${pp(d.hi)}) · cost ${money(
                        d.cost
                      )} · net ${money(d.net)}`,
                      d.label,
                    ];
                  }}
                />
                <Scatter data={cate}>
                  {cate.map((t) => (
                    <Cell
                      key={t.key}
                      fill={t.effect <= -0.012 ? "var(--bad)" : t.significant ? "var(--s4)" : "var(--ink-3)"}
                    />
                  ))}
                  <ErrorBar dataKey="err" direction="x" width={5} strokeWidth={2} stroke="var(--stroke-strong)" />
                </Scatter>
              </ScatterChart>
            </ResponsiveContainer>
          </div>
          <div className="mt-3">
            {c.quadrant === "Sleeping dog" ? (
              <Note caution>
                <b>Sleeping dog — suppress from all campaigns.</b> The best estimated effect for{" "}
                {c.id} is {pp(c.best_uplift)}: contacting them with an offer is estimated to make
                churn <em>worse</em>. A risk-ranked list would place them near the top, because
                their churn probability is {pct(c.churn_prob, 0)}.
              </Note>
            ) : c.quadrant === "Persuadable" ? (
              <Note>
                <b>Act on this one.</b> {best.label} has a confidence interval clear of zero. At{" "}
                {money(best.cost)} against {money(c.clv)} of margin at risk, expected net value is{" "}
                <b>{money(best.net)}</b>.
              </Note>
            ) : c.quadrant === "Lost cause" ? (
              <Note caution>
                <b>High risk, low savability.</b> Churn probability is {pct(c.churn_prob, 0)}, but
                no offer's interval clears zero by a useful margin. This is exactly the customer a
                risk-ranked campaign wastes money on.
              </Note>
            ) : (
              <Note>
                <b>Likely to stay anyway.</b> At {pct(c.churn_prob, 0)} churn probability, a
                discount here is margin given away to a customer who was not leaving.
              </Note>
            )}
          </div>
        </Card>
      </div>

      <Card
        title="What-if simulator"
        subtitle="Every control below moves a feature the model was actually trained on. There is no invented discount field, because the model never saw one."
      >
        <div className="grid gap-5 xl:grid-cols-[1fr_1.6fr]">
          <div className="flex flex-col gap-3.5">
            {(c.editable ?? []).map((f) =>
              f.type === "choice" ? (
                <div key={f.field}>
                  <label className="eyebrow">{f.label}</label>
                  <select
                    className="mt-1 w-full"
                    value={whatif[f.field] ?? ""}
                    onChange={(e) => setWhatif({ ...whatif, [f.field]: e.target.value })}
                  >
                    {f.options.map((o) => (
                      <option key={o}>{o}</option>
                    ))}
                  </select>
                </div>
              ) : (
                <div key={f.field}>
                  <label className="eyebrow">
                    {f.label} — {money(whatif[f.field] ?? 0)}
                  </label>
                  <input
                    className="mt-2"
                    type="range"
                    min={Math.floor(f.min)}
                    max={Math.ceil(f.max)}
                    step={1}
                    value={whatif[f.field] ?? f.min}
                    onChange={(e) =>
                      setWhatif({ ...whatif, [f.field]: Number(e.target.value) })
                    }
                  />
                </div>
              )
            )}
            <button className="btn ghost self-start" onClick={() => setWhatif(baselineOf(c))}>
              Reset to current
            </button>
          </div>

          <div>
            {sim && (
              <>
                <div style={{ width: "100%", height: 230 }}>
                  <ResponsiveContainer>
                    <BarChart
                      data={[
                        { name: "Current prediction", v: sim.before },
                        { name: "Simulated prediction", v: sim.after },
                      ]}
                      margin={{ top: 24, right: 16, bottom: 8, left: 8 }}
                    >
                      <CartesianGrid stroke="var(--grid)" vertical={false} />
                      <XAxis dataKey="name" tick={{ fontSize: 12 }} />
                      <YAxis domain={[0, 1]} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
                      <Tooltip
                        cursor={{ fill: "var(--card-alt)" }}
                        contentStyle={{ background: "var(--card)", border: "1px solid var(--stroke-strong)", borderRadius: 4, fontSize: 12 }}
                        formatter={(v) => [pct(v), "Churn probability"]}
                      />
                      <Bar dataKey="v" radius={[4, 4, 0, 0]} barSize={86}>
                        <Cell fill="var(--s2)" />
                        <Cell fill="var(--s1)" />
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
                <div className="mt-1 flex items-baseline justify-end gap-3">
                  <span
                    className="mono text-[24px] font-bold"
                    style={{
                      color: sim.delta < 0 ? "var(--good)" : sim.delta > 0 ? "var(--bad)" : "var(--ink-3)",
                    }}
                  >
                    {sim.delta < 0 ? "-" : "+"}
                    {Math.abs(sim.delta * 100).toFixed(1)} pp
                  </span>
                  <span className="text-[12px]" style={{ color: "var(--ink-3)" }}>
                    {sim.delta < 0 ? "predicted risk falls" : sim.delta > 0 ? "predicted risk rises" : "no change"}
                  </span>
                </div>
                <div className="mt-2">
                  <Note caution>{sim.note}</Note>
                </div>
              </>
            )}
          </div>
        </div>
      </Card>
    </>
  );
}
