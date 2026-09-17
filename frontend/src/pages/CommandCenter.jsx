import { useNavigate } from "react-router-dom";
import {
  Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Scatter,
  ScatterChart, Tooltip, XAxis, YAxis, ZAxis, ReferenceLine,
} from "recharts";
import { Card, Kpi, Legend, Note, Quadrant, DefinitionRow } from "../components/ui";
import { money, num, pct, pp, QUADRANT_COLOR, QUADRANTS } from "../lib/format";

function ScatterTip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div
      className="rounded px-2.5 py-2 text-[12px]"
      style={{ background: "var(--card)", border: "1px solid var(--stroke-strong)", boxShadow: "var(--shadow-8)" }}
    >
      <b className="mb-1 block">{d.id}</b>
      <div className="flex justify-between gap-4">
        <span style={{ color: "var(--ink-2)" }}>Quadrant</span>
        <span>{d.quadrant}</span>
      </div>
      <div className="flex justify-between gap-4">
        <span style={{ color: "var(--ink-2)" }}>Churn probability</span>
        <span className="mono">{pct(d.churn, 0)}</span>
      </div>
      <div className="flex justify-between gap-4">
        <span style={{ color: "var(--ink-2)" }}>Best uplift</span>
        <span className="mono">{pp(d.uplift)}</span>
      </div>
      <div className="mt-1.5 text-[11px]" style={{ color: "var(--ink-3)" }}>
        Click to open this customer
      </div>
    </div>
  );
}

export default function CommandCenter({ summary }) {
  const nav = useNavigate();
  const s = summary;

  // The offer a conventional campaign would blast at the high-risk tail, and
  // the number of customers it is estimated to actively harm.
  const worstOffer = [...(s.harmful_by_offer ?? [])].sort(
    (a, b) => b.customers - a.customers
  )[0];

  const shap = s.global_shap.slice(0, 12).map((d) => ({ ...d }));

  return (
    <>
      <div className="grid gap-3.5 md:grid-cols-2 xl:grid-cols-4">
        <Kpi
          label="Revenue at risk / yr"
          value={money(s.revenue_at_risk, true)}
          detail={`${num(s.customers_at_risk)} customers above 50% churn probability`}
          accent="bad"
        />
        <Kpi
          label="Savable revenue"
          value={money(s.savable_revenue, true)}
          detail="What causal uplift says you can actually recover"
          accent="hero"
        />
        <Kpi
          label="Persuadables found"
          value={num(s.quadrants.find((q) => q.quadrant === "Persuadable")?.customers)}
          detail="Customers whose decision an offer would change"
        />
        <Kpi
          label="Would be harmed by an offer"
          value={num(worstOffer?.customers)}
          detail={`${worstOffer?.label ?? "The default offer"} is estimated to increase churn for these customers`}
          accent="bad"
        />
      </div>

      <Note>
        <b>Why those are two different numbers.</b> Revenue at risk counts everyone likely to
        leave. Savable revenue counts only those an intervention would actually change,
        measured as a causal treatment effect rather than a correlation. The gap between them
        is the money a conventional churn dashboard cannot see.
      </Note>

      <div className="grid gap-3.5 xl:grid-cols-[1.55fr_1fr]">
        <Card
          title="Risk is not savability"
          subtitle="Every customer plotted by predicted churn probability against the estimated causal uplift of their best-fit offer. The crowd targets the right edge. The money is at the top."
        >
          <div style={{ width: "100%", height: 380 }}>
            <ResponsiveContainer>
              <ScatterChart margin={{ top: 12, right: 16, bottom: 28, left: 8 }}>
                <CartesianGrid stroke="var(--grid)" />
                <XAxis
                  type="number"
                  dataKey="churn"
                  domain={[0, 1]}
                  tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
                  label={{ value: "Predicted churn probability", position: "insideBottom", offset: -16, fill: "var(--ink-3)", fontSize: 11 }}
                />
                <YAxis
                  type="number"
                  dataKey="uplift"
                  tickFormatter={(v) => `${v > 0 ? "+" : ""}${(v * 100).toFixed(0)} pp`}
                  label={{ value: "Estimated uplift", angle: -90, position: "insideLeft", fill: "var(--ink-3)", fontSize: 11 }}
                />
                <ZAxis range={[26, 26]} />
                <ReferenceLine y={0} stroke="var(--ink-3)" strokeDasharray="4 3" />
                <Tooltip content={<ScatterTip />} cursor={{ strokeDasharray: "3 3" }} />
                {QUADRANTS.map((q) => (
                  <Scatter
                    key={q}
                    name={q}
                    data={s.scatter.filter((d) => d.quadrant === q)}
                    fill={QUADRANT_COLOR[q]}
                    fillOpacity={q === "Persuadable" ? 0.85 : 0.55}
                    onClick={(d) => nav(`/customer?id=${encodeURIComponent(d.id)}`)}
                    style={{ cursor: "pointer" }}
                  />
                ))}
              </ScatterChart>
            </ResponsiveContainer>
          </div>
          <Legend items={QUADRANTS.map((q) => ({ label: q, color: QUADRANT_COLOR[q] }))} />
        </Card>

        <Card
          title="Portfolio by quadrant"
          subtitle="Share of the base, with the annual revenue each quadrant carries."
        >
          <div className="flex flex-col gap-3.5">
            {s.quadrants.map((q) => (
              <div key={q.quadrant}>
                <div className="mb-1 flex items-center justify-between gap-2">
                  <Quadrant value={q.quadrant} />
                  <span className="mono text-[12.5px] font-semibold">{pct(q.share, 0)}</span>
                </div>
                <div className="h-[10px] w-full overflow-hidden rounded" style={{ background: "var(--card-sunk)" }}>
                  <div
                    className="h-full rounded"
                    style={{
                      width: `${Math.max(1.5, q.share * 100)}%`,
                      background: QUADRANT_COLOR[q.quadrant],
                    }}
                  />
                </div>
                <div className="mt-1 text-[11px]" style={{ color: "var(--ink-3)" }}>
                  {num(q.customers)} customers · {money(q.annual_revenue, true)} of annual revenue
                </div>
              </div>
            ))}
          </div>
        </Card>
      </div>

      <div className="grid gap-3.5 xl:grid-cols-[1.55fr_1fr]">
        <Card
          title="Churn drivers across the base"
          subtitle="Mean absolute SHAP value: how much each feature moves the prediction on average. Global importance, not a per-customer explanation."
        >
          <div style={{ width: "100%", height: 340 }}>
            <ResponsiveContainer>
              <BarChart data={shap} layout="vertical" margin={{ top: 4, right: 52, bottom: 4, left: 132 }}>
                <CartesianGrid stroke="var(--grid)" horizontal={false} />
                <XAxis type="number" tickFormatter={(v) => v.toFixed(2)} />
                <YAxis type="category" dataKey="feature" width={130} tick={{ fontSize: 11.5 }} />
                <Tooltip
                  cursor={{ fill: "var(--card-alt)" }}
                  contentStyle={{
                    background: "var(--card)", border: "1px solid var(--stroke-strong)",
                    borderRadius: 4, fontSize: 12, color: "var(--ink)",
                  }}
                  formatter={(v) => [v.toFixed(3), "Mean |SHAP|"]}
                />
                <Bar dataKey="value" radius={[0, 3, 3, 0]} barSize={14}>
                  {shap.map((d) => (
                    <Cell
                      key={d.feature}
                      fill={d.feature.toLowerCase() === "gender" ? "var(--mute)" : "var(--s1)"}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>

        <Card title="Last pipeline run" subtitle="Every stage that produced what you are looking at.">
          <DefinitionRow label="Dataset" value={s.data.source} />
          <DefinitionRow label="Rows after cleaning" value={num(s.data.rows_clean)} />
          <DefinitionRow label="Dropped (blank TotalCharges)" value={num(s.data.dropped_total_charges)} />
          <DefinitionRow label="Base churn rate" value={pct(s.base_rate)} />
          <DefinitionRow label="Churn model" value={s.selected_model} />
          <DefinitionRow label="ROC-AUC" value={s.roc_auc.toFixed(3)} />
          <DefinitionRow label="Brier score" value={s.brier.toFixed(3)} />
          <DefinitionRow label="Causal estimator" value={s.causal_estimator} />
          <DefinitionRow label="Qini coefficient" value={s.qini.toFixed(3)} />
          <DefinitionRow
            label="Refutation tests"
            value={`${s.refutations_passed} / ${s.refutations_total} passed`}
          />
          <DefinitionRow label="Fairness gate" value={s.fairness_passed ? "Passed" : "FAILED"} />
          <DefinitionRow label="Runtime" value={`${s.runtime_seconds}s`} />
        </Card>
      </div>
    </>
  );
}
