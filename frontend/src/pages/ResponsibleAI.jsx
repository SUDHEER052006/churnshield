import { useEffect, useState } from "react";
import {
  Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { api } from "../lib/api";
import { Card, Kpi, Legend, Loading, Note } from "../components/ui";
import { num, pct } from "../lib/format";

const LIMITS = [
  ["Causal identification", "Observational data. Estimates assume no unmeasured confounding — tested here, never proven."],
  ["Offer uptake", "Modelled as certain. Real acceptance rates reduce every return figure proportionally."],
  ["Drift", "Scored nightly. Retraining is triggered when PSI on any top-five feature exceeds 0.25."],
  ["Automation", "No offer is sent automatically. Every row is a recommendation for a human agent."],
  ["Personal data", "No name, address, or contact detail enters the model or this interface."],
  ["Suppression", "Sleeping dogs and flagged error cohorts are excluded from all outbound campaigns."],
];

export default function ResponsibleAI() {
  const [r, setR] = useState(null);
  useEffect(() => {
    api.rai().then(setR);
  }, []);
  if (!r) return <Loading what="audit" />;

  const groups = r.fairness.groups.map((g) => ({
    ...g,
    sel: g.selection_rate,
    fnr: g.false_negative_rate,
  }));
  const worstCohort = r.errors.cohorts[0];
  const passed = r.refutation.tests.filter((t) => t.passed).length;

  return (
    <>
      <div className="grid gap-3.5 md:grid-cols-3">
        <Kpi
          label="Fairness gate"
          value={
            <span style={{ color: r.fairness.passed ? "var(--good-ink)" : "var(--bad-ink)" }}>
              {r.fairness.passed ? "Pass" : "Fail"}
            </span>
          }
          detail={`Max demographic parity difference ${r.fairness.demographic_parity_difference.toFixed(
            3
          )} against a ${r.fairness.threshold.toFixed(2)} threshold`}
          accent={r.fairness.passed ? "good" : "bad"}
        />
        <Kpi
          label="Refutation tests"
          value={`${passed} / ${r.refutation.tests.length}`}
          detail={`${r.refutation.library} — placebo, random common cause, data subset`}
          accent={passed === r.refutation.tests.length ? "good" : "bad"}
        />
        <Kpi
          label="Worst cohort error"
          value={pct(worstCohort.error_rate)}
          detail={`${worstCohort.cohort} (${num(worstCohort.n)} customers)`}
          accent="bad"
        />
      </div>

      <div className="grid gap-3.5 xl:grid-cols-2">
        <Card
          title={`${r.fairness.library} — model performance by group`}
          subtitle="False negative rate is the one that matters ethically: a missed churner is a customer nobody tried to save. It must not be systematically higher for any group."
        >
          <div style={{ width: "100%", height: 320 }}>
            <ResponsiveContainer>
              <BarChart data={groups} layout="vertical" margin={{ top: 4, right: 40, bottom: 20, left: 90 }}>
                <CartesianGrid stroke="var(--grid)" horizontal={false} />
                <XAxis type="number" tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
                <YAxis type="category" dataKey="group" width={88} tick={{ fontSize: 11.5 }} />
                <Tooltip
                  cursor={{ fill: "var(--card-alt)" }}
                  contentStyle={{ background: "var(--card)", border: "1px solid var(--stroke-strong)", borderRadius: 4, fontSize: 12 }}
                  formatter={(v, n) => [pct(v), n === "sel" ? "Selection rate" : "False negative rate"]}
                />
                <Bar dataKey="sel" fill="var(--s1)" radius={[0, 2, 2, 0]} barSize={9} />
                <Bar dataKey="fnr" fill="var(--s2)" radius={[0, 2, 2, 0]} barSize={9} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <Legend
            items={[
              { label: "Selection rate", color: "var(--s1)" },
              { label: "False negative rate", color: "var(--s2)" },
            ]}
          />
          <div className="mt-2 text-[11.5px]" style={{ color: "var(--ink-3)" }}>
            Equalised odds difference {r.fairness.equalized_odds_difference.toFixed(3)}
          </div>
        </Card>

        <Card
          title={`${r.refutation.library} — refutation tests`}
          subtitle="Each test attacks the causal estimate. Passing means the finding survived an attack, which is a stronger claim than a good fit."
        >
          <div className="overflow-x-auto">
            <table>
              <thead>
                <tr>
                  <th>Test</th>
                  <th>What it does</th>
                  <th className="num">Expected</th>
                  <th className="num">Observed</th>
                  <th>Result</th>
                </tr>
              </thead>
              <tbody>
                {r.refutation.tests.map((t) => (
                  <tr key={t.test}>
                    <td>
                      <b>{t.test}</b>
                    </td>
                    <td className="whitespace-normal text-[12px]" style={{ color: "var(--ink-2)" }}>
                      {t.description}
                    </td>
                    <td className="num">{t.expected?.toFixed(3) ?? "--"}</td>
                    <td className="num">{t.observed?.toFixed(3) ?? "--"}</td>
                    <td>
                      <span className={`flag ${t.passed ? "ok" : "no"}`}>
                        {t.passed ? "Survived" : "Failed"}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="mt-3">
            <Note>
              <b>Why the placebo test matters most.</b> It replaces the real offer with a random
              fake one. If the method still "finds" an effect, it is finding structure in noise. A
              good result here is an effect of approximately zero.
            </Note>
          </div>
        </Card>
      </div>

      <Card
        title="Error analysis — where the model is weakest"
        subtitle={`Errors are not spread evenly. Cohorts are discovered automatically (${r.errors.method}), not hand-picked. Overall error rate is ${pct(
          r.errors.overall_error_rate
        )}.`}
      >
        <div className="flex flex-col gap-3">
          {r.errors.cohorts.map((c) => {
            const max = Math.max(...r.errors.cohorts.map((x) => x.error_rate));
            const color = c.lift >= 1.5 ? "var(--bad)" : c.lift >= 1.15 ? "var(--warn)" : "var(--s4)";
            return (
              <div key={c.cohort}>
                <div className="mb-1 flex items-baseline justify-between gap-3">
                  <span className="text-[12.5px]">{c.cohort}</span>
                  <span className="mono shrink-0 text-[12px] font-semibold">
                    {pct(c.error_rate)} · {c.lift.toFixed(1)}x
                  </span>
                </div>
                <div className="h-[16px] w-full overflow-hidden rounded" style={{ background: "var(--card-sunk)" }}>
                  <div className="h-full rounded" style={{ width: `${(c.error_rate / max) * 100}%`, background: color }} />
                </div>
                <div className="mt-0.5 text-[10.5px]" style={{ color: "var(--ink-3)" }}>
                  n = {num(c.n)}
                </div>
              </div>
            );
          })}
        </div>
      </Card>

      <Card title="Operating limits" subtitle="What this system does not claim, stated before anyone asks.">
        <div className="grid gap-x-8 gap-y-3 md:grid-cols-2">
          {LIMITS.map(([k, v]) => (
            <div key={k}>
              <div className="mb-0.5 text-[12.5px] font-semibold">{k}</div>
              <div className="text-[12px]" style={{ color: "var(--ink-2)" }}>
                {v}
              </div>
            </div>
          ))}
        </div>
      </Card>
    </>
  );
}
