import { useEffect, useState } from "react";
import {
  Area, AreaChart, CartesianGrid, Legend as RLegend, Line, ResponsiveContainer,
  Tooltip, XAxis, YAxis,
} from "recharts";
import { api } from "../lib/api";
import { Card, DefinitionRow, Kpi, Legend, Loading, Note, Slider } from "../components/ui";
import { money, num, pct } from "../lib/format";

const SERIES = [
  { key: "uplift", label: "Ranked by causal uplift", color: "var(--s1)" },
  { key: "risk", label: "Ranked by churn risk", color: "var(--s2)" },
  { key: "random", label: "Random targeting", color: "var(--ink-3)" },
];

export default function BudgetOptimizer({ summary }) {
  const d = summary.assumptions;
  const [budget, setBudget] = useState(d.budget);
  const [margin, setMargin] = useState(d.gross_margin * 100);
  const [horizon, setHorizon] = useState(d.horizon_months);
  const [res, setRes] = useState(null);
  const [busy, setBusy] = useState(true);

  useEffect(() => {
    let live = true;
    setBusy(true);
    const t = setTimeout(() => {
      api
        .allocate({ budget, margin: margin / 100, horizon })
        .then((r) => live && setRes(r))
        .finally(() => live && setBusy(false));
    }, 220);
    return () => {
      live = false;
      clearTimeout(t);
    };
  }, [budget, margin, horizon]);

  const curve = (res?.curves ?? []).map((p) => ({
    share: p.share,
    uplift: p.uplift,
    risk: p.risk,
    random: p.random,
    gap: p.uplift - p.risk,
  }));

  return (
    <>
      <Card
        title="Spend allocation"
        subtitle="A knapsack over expected net value: buy the most retained margin per unit of budget. Costs and margin are assumptions you set here, not constants baked into the model."
      >
        <div className="grid gap-6 md:grid-cols-3">
          <Slider
            label="Retention budget"
            display={money(budget)}
            min={5000}
            max={250000}
            step={5000}
            value={budget}
            onChange={setBudget}
          />
          <Slider
            label="Gross margin on retained revenue"
            display={`${margin}%`}
            min={15}
            max={80}
            step={5}
            value={margin}
            onChange={setMargin}
          />
          <Slider
            label="Value horizon"
            display={`${horizon} months`}
            min={6}
            max={36}
            step={6}
            value={horizon}
            onChange={setHorizon}
          />
        </div>

        {res && (
          <div className="mt-5 grid gap-3.5 md:grid-cols-2 xl:grid-cols-4" style={{ opacity: busy ? 0.6 : 1 }}>
            <Kpi
              label="Customers targeted"
              value={num(res.targeted)}
              detail={`${money(res.spend, true)} committed of ${money(res.budget, true)}`}
            />
            <Kpi
              label="Expected retained margin"
              value={money(res.expected_margin_retained, true)}
              detail={`${res.expected_customers_retained.toFixed(0)} customers expected to stay who otherwise would not`}
              accent="hero"
            />
            <Kpi
              label="Return on spend"
              value={`${res.roi.toFixed(1)}x`}
              detail="Margin recovered per unit committed"
              accent="good"
            />
            <Kpi
              label="Lift over risk targeting"
              value={
                <span style={{ color: res.lift_vs_risk_targeting >= 0 ? "var(--good-ink)" : "var(--bad-ink)" }}>
                  {res.lift_vs_risk_targeting >= 0 ? "+" : ""}
                  {(res.lift_vs_risk_targeting * 100).toFixed(0)}%
                </span>
              }
              detail={`Same budget ranked by churn probability retains ${res.naive.expected_customers_retained.toFixed(
                0
              )}`}
              accent="good"
            />
          </div>
        )}
      </Card>

      <div className="grid gap-3.5 xl:grid-cols-[1.55fr_1fr]">
        <Card
          title="Uplift targeting vs. risk targeting"
          subtitle="Cumulative customers retained as you work down each ranking. The shaded gap between the two lines is the entire argument for this project."
        >
          {!res ? (
            <Loading what="curves" />
          ) : (
            <>
              <div style={{ width: "100%", height: 350 }}>
                <ResponsiveContainer>
                  <AreaChart data={curve} margin={{ top: 10, right: 16, bottom: 26, left: 6 }}>
                    <defs>
                      <linearGradient id="gapFill" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="var(--s1)" stopOpacity={0.22} />
                        <stop offset="100%" stopColor="var(--s1)" stopOpacity={0.05} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid stroke="var(--grid)" />
                    <XAxis
                      dataKey="share"
                      type="number"
                      domain={[0, 1]}
                      tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
                      label={{ value: "Share of base contacted", position: "insideBottom", offset: -14, fill: "var(--ink-3)", fontSize: 11 }}
                    />
                    <YAxis
                      tickFormatter={(v) => v.toFixed(0)}
                      label={{ value: "Customers retained", angle: -90, position: "insideLeft", fill: "var(--ink-3)", fontSize: 11 }}
                    />
                    <Tooltip
                      contentStyle={{ background: "var(--card)", border: "1px solid var(--stroke-strong)", borderRadius: 4, fontSize: 12 }}
                      labelFormatter={(v) => `${(v * 100).toFixed(0)}% of base contacted`}
                      formatter={(v, n) => [
                        v.toFixed(1),
                        SERIES.find((s) => s.key === n)?.label ?? n,
                      ]}
                    />
                    <Area type="monotone" dataKey="uplift" stroke="var(--s1)" strokeWidth={2.5} fill="url(#gapFill)" dot={false} />
                    <Line type="monotone" dataKey="risk" stroke="var(--s2)" strokeWidth={2} dot={false} />
                    <Line type="monotone" dataKey="random" stroke="var(--ink-3)" strokeWidth={1.5} strokeDasharray="5 4" dot={false} />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
              <Legend items={SERIES.map((s) => ({ label: s.label, color: s.color, line: true }))} />
            </>
          )}
        </Card>

        <Card title="Where the budget goes" subtitle="Allocation by offer type under the current assumptions.">
          {!res ? (
            <Loading what="allocation" />
          ) : res.offer_mix.length === 0 ? (
            <p className="text-[13px]" style={{ color: "var(--ink-3)" }}>
              No offer clears its own cost at this margin and horizon. Raise the margin, extend the
              horizon, or accept that this base is not worth an outbound campaign — which is itself
              a legitimate finding.
            </p>
          ) : (
            <div className="flex flex-col gap-3">
              <div className="flex h-[26px] w-full gap-0.5 overflow-hidden rounded">
                {res.offer_mix.map((m, i) => (
                  <div
                    key={m.offer}
                    style={{
                      width: `${(m.spend / res.spend) * 100}%`,
                      background: ["var(--s1)", "var(--s2)", "var(--s3)", "var(--s4)"][i % 4],
                    }}
                    title={`${m.label}: ${money(m.spend)}`}
                  />
                ))}
              </div>
              {res.offer_mix.map((m, i) => (
                <div key={m.offer} className="flex items-start gap-2.5">
                  <i
                    className="mt-1 block h-[11px] w-[11px] shrink-0 rounded-sm"
                    style={{ background: ["var(--s1)", "var(--s2)", "var(--s3)", "var(--s4)"][i % 4] }}
                  />
                  <div className="min-w-0 flex-1">
                    <div className="flex justify-between gap-3">
                      <span className="text-[12.5px] font-semibold">{m.label}</span>
                      <span className="mono text-[12.5px] font-semibold">{money(m.spend, true)}</span>
                    </div>
                    <div className="text-[11px]" style={{ color: "var(--ink-3)" }}>
                      {num(m.customers)} customers · {pct(m.spend / res.spend, 0)} of budget
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </Card>
      </div>

      <Card
        title="Stated assumptions"
        subtitle="None of this is derived from the data. Change any of it and every number on this page changes, which is exactly the point."
      >
        <div className="grid gap-x-8 md:grid-cols-2">
          <DefinitionRow label="Gross margin on retained revenue" value={`${margin}%`} />
          <DefinitionRow label="Value horizon" value={`${horizon} months of billed revenue`} />
          <DefinitionRow label="Contract offer cost" value={`15% of monthly charge x ${horizon} months`} />
          <DefinitionRow label="Tech support cost to serve" value={`${money(d.support_cost_per_month)} / month`} />
          <DefinitionRow label="Security bundle cost to serve" value={`${money(d.security_cost_per_month)} / month`} />
          <DefinitionRow label="Auto-pay incentive" value={`${money(d.autopay_incentive)} one-time`} />
          <DefinitionRow label="Offer acceptance" value="Assumed 100% — real uptake will be lower" />
          <DefinitionRow label="Sleeping dogs" value="Structurally excluded, unreachable by any budget" />
        </div>
        <div className="mt-3">
          <Note caution>
            <b>Read the acceptance assumption.</b> Every return figure here assumes the customer
            takes the offer. A real campaign converts a fraction of those contacted, so treat these
            as an upper bound and scale by your observed uptake rate.
          </Note>
        </div>
      </Card>
    </>
  );
}
