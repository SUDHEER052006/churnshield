import { useEffect, useState } from "react";
import {
  CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer,
  Tooltip, XAxis, YAxis,
} from "recharts";
import { api } from "../lib/api";
import { Card, Legend, Loading, Note } from "../components/ui";
import { num, pct } from "../lib/format";

function Verdict({ v }) {
  const cls = v === "selected" ? "ok" : v === "rejected" ? "no" : v === "failed" ? "no" : "mid";
  const label = v === "selected" ? "Selected" : v === "rejected" ? "Rejected" : v === "failed" ? "Failed" : "Candidate";
  return <span className={`flag ${cls}`}>{label}</span>;
}

export default function ModelLab() {
  const [m, setM] = useState(null);
  useEffect(() => {
    api.models().then(setM);
  }, []);
  if (!m) return <Loading what="model report" />;

  const cm = m.confusion;
  const total = cm.tn + cm.fp + cm.fn + cm.tp;
  const cells = [
    { l: "True negative", v: cm.tn, d: "Predicted stay, stayed", good: true },
    { l: "False positive", v: cm.fp, d: "Offer sent unnecessarily", good: false },
    { l: "False negative", v: cm.fn, d: "Churner nobody tried to save", good: false },
    { l: "True positive", v: cm.tp, d: "Churner correctly flagged", good: true },
  ];
  const maxCell = Math.max(...cells.map((c) => c.v));

  const calib = m.calibration.calibrated.map((p, i) => ({
    predicted: p.predicted,
    calibrated: p.observed,
    raw: m.calibration.raw[i]?.observed ?? null,
    perfect: p.predicted,
  }));

  return (
    <>
      <Card
        title="Stage 1 — churn classifiers"
        subtitle={`Every candidate trained on the same stratified split (${num(m.n_train)} train / ${num(
          m.n_test
        )} test). Selected on PR-AUC and calibration, not accuracy, because the base rate makes accuracy reward predicting "stays" every time.`}
      >
        <div className="overflow-x-auto">
          <table>
            <thead>
              <tr>
                <th>Model</th>
                <th className="num">Accuracy</th>
                <th className="num">Precision</th>
                <th className="num">Recall</th>
                <th className="num">F1</th>
                <th className="num">ROC-AUC</th>
                <th className="num">PR-AUC</th>
                <th className="num">Brier</th>
                <th>Verdict</th>
              </tr>
            </thead>
            <tbody>
              {m.classifiers.map((r) => (
                <tr key={r.model} style={r.verdict === "selected" ? { background: "var(--good-soft)" } : undefined}>
                  <td>
                    <b>{r.model}</b>
                    <span className="block text-[11px]" style={{ color: "var(--ink-3)" }}>
                      {r.note}
                    </span>
                  </td>
                  <td className="num">{pct(r.accuracy)}</td>
                  <td className="num">{r.precision ? pct(r.precision) : "--"}</td>
                  <td className="num">{r.recall ? pct(r.recall) : "--"}</td>
                  <td className="num">{r.f1 ? r.f1.toFixed(3) : "--"}</td>
                  <td className="num">{r.roc_auc.toFixed(3)}</td>
                  <td className="num">{r.pr_auc.toFixed(3)}</td>
                  <td className="num font-semibold">{r.brier.toFixed(3)}</td>
                  <td>
                    <Verdict v={r.verdict} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="mt-3">
          <Note>
            <b>Read the Brier column, not the accuracy column.</b> The dummy classifier scores a
            respectable accuracy by never predicting churn at all. Accuracy alone would have picked
            a model that is useless. Every downstream currency figure multiplies by the predicted
            probability, so calibration decides the pick.
          </Note>
        </div>
      </Card>

      <div className="grid gap-3.5 xl:grid-cols-2">
        <Card
          title="Calibration"
          subtitle="Predicted probability against observed churn rate, ten quantile bins. A model on the diagonal means 80% really means 80%."
        >
          <div style={{ width: "100%", height: 300 }}>
            <ResponsiveContainer>
              <LineChart data={calib} margin={{ top: 10, right: 16, bottom: 24, left: 6 }}>
                <CartesianGrid stroke="var(--grid)" />
                <XAxis
                  dataKey="predicted"
                  type="number"
                  domain={[0, 1]}
                  tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
                  label={{ value: "Predicted probability", position: "insideBottom", offset: -14, fill: "var(--ink-3)", fontSize: 11 }}
                />
                <YAxis domain={[0, 1]} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
                <Tooltip
                  contentStyle={{ background: "var(--card)", border: "1px solid var(--stroke-strong)", borderRadius: 4, fontSize: 12 }}
                  formatter={(v, n) => [pct(v), n === "calibrated" ? m.calibration.calibrated_label : n === "raw" ? m.calibration.raw_label : "Perfect"]}
                  labelFormatter={(v) => `Predicted ${pct(v, 0)}`}
                />
                <Line type="monotone" dataKey="perfect" stroke="var(--ink-3)" strokeWidth={1.5} strokeDasharray="5 4" dot={false} />
                <Line type="monotone" dataKey="raw" stroke="var(--s2)" strokeWidth={2} dot={{ r: 3 }} />
                <Line type="monotone" dataKey="calibrated" stroke="var(--s1)" strokeWidth={2.5} dot={{ r: 3.5 }} />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <Legend
            items={[
              { label: m.calibration.calibrated_label, color: "var(--s1)", line: true },
              { label: m.calibration.raw_label, color: "var(--s2)", line: true },
              { label: "Perfect calibration", color: "var(--ink-3)", line: true },
            ]}
          />
        </Card>

        <Card
          title="Confusion matrix — selected model"
          subtitle={`Threshold ${m.threshold}, tuned to favour recall: missing a churner costs more than an unnecessary offer.`}
        >
          <div className="grid grid-cols-2 gap-1.5">
            {cells.map((c) => (
              <div
                key={c.l}
                className="rounded p-4 text-center"
                style={{
                  background: `color-mix(in srgb, ${c.good ? "var(--s1)" : "var(--bad)"} ${(
                    10 + 62 * (c.v / maxCell)
                  ).toFixed(0)}%, var(--card))`,
                  border: "1px solid var(--stroke)",
                }}
              >
                <div className="mono text-[26px] font-bold">{num(c.v)}</div>
                <div className="text-[11.5px]" style={{ color: "var(--ink-2)" }}>
                  {c.l}
                </div>
                <div className="mt-0.5 text-[10.5px]" style={{ color: "var(--ink-3)" }}>
                  {c.d}
                </div>
              </div>
            ))}
          </div>
          <div className="mt-3 text-[11.5px]" style={{ color: "var(--ink-3)" }}>
            The {num(cm.fn)} false negatives ({pct(cm.fn / total)} of the holdout) are the ethical
            cost centre: customers who left while the system said they were safe.
          </div>
        </Card>
      </div>

      <Card
        title="Stage 2 — causal uplift learners"
        subtitle="Five estimators of the same quantity: the effect of an offer on each customer. Scored on Qini and AUUC, because there is no ground-truth uplift label to be accurate against."
      >
        <div className="overflow-x-auto">
          <table>
            <thead>
              <tr>
                <th>Estimator</th>
                <th>Library</th>
                <th className="num">Qini</th>
                <th className="num">AUUC</th>
                <th className="num">Uplift @ 20%</th>
                <th className="num">Mean CI width</th>
                <th>Verdict</th>
              </tr>
            </thead>
            <tbody>
              {m.causal.map((r) => (
                <tr key={r.model} style={r.verdict === "selected" ? { background: "var(--good-soft)" } : undefined}>
                  <td>
                    <b>{r.model}</b>
                    <span className="block text-[11px]" style={{ color: "var(--ink-3)" }}>
                      {r.note}
                    </span>
                  </td>
                  <td>
                    <span className="tag">{r.library}</span>
                  </td>
                  <td className="num">{r.qini.toFixed(3)}</td>
                  <td className="num">{r.auuc.toFixed(3)}</td>
                  <td className="num">{pct(r.uplift_at_20)}</td>
                  <td className="num">{r.ci_width ? pct(r.ci_width) : "--"}</td>
                  <td>
                    <Verdict v={r.verdict} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="mt-3">
          <Note caution>
            <b>Honest framing.</b> The Telco data is observational: customers chose their own
            contracts, nobody randomised them. These estimates assume no unmeasured confounding.
            That assumption is tested rather than asserted — see the refutation panel under
            Responsible AI.
          </Note>
        </div>
      </Card>
    </>
  );
}
