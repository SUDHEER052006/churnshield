import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "../lib/api";
import { Card, Loading, Note, Quadrant, Segmented } from "../components/ui";
import { money, num, pct, pp } from "../lib/format";

const QUAD_OPTIONS = [
  { value: "all", label: "All actionable" },
  { value: "Persuadable", label: "Persuadables" },
  { value: "Lost cause", label: "Lost causes" },
  { value: "Sleeping dog", label: "Sleeping dogs" },
];

export default function ActionList({ summary }) {
  const nav = useNavigate();
  const [quadrant, setQuadrant] = useState("all");
  const [offer, setOffer] = useState("all");
  const [q, setQ] = useState("");
  const [data, setData] = useState(null);
  const [busy, setBusy] = useState(true);

  useEffect(() => {
    let live = true;
    setBusy(true);
    const t = setTimeout(() => {
      api
        .customers({ quadrant, offer, q, limit: 80 })
        .then((d) => live && setData(d))
        .finally(() => live && setBusy(false));
    }, 180);
    return () => {
      live = false;
      clearTimeout(t);
    };
  }, [quadrant, offer, q]);

  const offers = summary?.treatments ?? {};

  return (
    <Card
      title="Today's worklist"
      subtitle="Not a list of who is at risk, but a list of who to call, with which offer, ranked by expected net value after the cost of that offer. Sleeping dogs are excluded from the default view by design."
    >
      <div className="mb-3 flex flex-wrap items-center gap-2.5">
        <Segmented options={QUAD_OPTIONS} value={quadrant} onChange={setQuadrant} />
        <select value={offer} onChange={(e) => setOffer(e.target.value)}>
          <option value="all">Every offer type</option>
          {Object.entries(offers).map(([k, v]) => (
            <option key={k} value={k}>
              {v.label}
            </option>
          ))}
        </select>
        <input
          type="search"
          placeholder="Find customer ID..."
          value={q}
          onChange={(e) => setQ(e.target.value)}
          style={{ width: 190 }}
        />
        <div className="flex-1" />
        {data && (
          <span className="tag">
            {num(data.total)} match · showing top {data.rows.length}
          </span>
        )}
      </div>

      {busy && !data ? (
        <Loading what="worklist" />
      ) : (
        <div className="max-h-[620px] overflow-auto" style={{ opacity: busy ? 0.55 : 1 }}>
          <table>
            <thead>
              <tr>
                <th style={{ width: 40 }}>#</th>
                <th>Customer</th>
                <th>Quadrant</th>
                <th className="num">Churn P</th>
                <th className="num">Uplift</th>
                <th>Recommended offer</th>
                <th className="num">Cost</th>
                <th className="num">Expected net</th>
                <th>Profile</th>
              </tr>
            </thead>
            <tbody>
              {data?.rows.map((r, i) => (
                <tr
                  key={r.id}
                  onClick={() => nav(`/customer?id=${encodeURIComponent(r.id)}`)}
                  className="cursor-pointer"
                >
                  <td className="mono text-[11.5px]" style={{ color: "var(--ink-3)" }}>
                    {i + 1}
                  </td>
                  <td className="mono text-[12px] font-semibold" style={{ color: "var(--brand-ink)" }}>
                    {r.id}
                  </td>
                  <td>
                    <Quadrant value={r.quadrant} />
                  </td>
                  <td className="num">{pct(r.churn_prob, 0)}</td>
                  <td
                    className="num"
                    style={{ color: r.offer_effect > 0 ? "var(--good-ink)" : "var(--bad-ink)" }}
                  >
                    {pp(r.offer_effect)}
                  </td>
                  <td className="text-[12.5px]">
                    {r.offer_label}
                    <i className="block not-italic text-[11px]" style={{ color: "var(--ink-3)" }}>
                      {offers[r.offer]?.detail}
                    </i>
                  </td>
                  <td className="num">{money(r.cost)}</td>
                  <td className="num">
                    <span
                      className="mono font-semibold"
                      style={{ color: r.net >= 0 ? "var(--good-ink)" : "var(--bad-ink)" }}
                    >
                      {r.net >= 0 ? "+" : "-"}
                      {money(Math.abs(r.net))}
                    </span>
                  </td>
                  <td>
                    <span className="flex gap-1.5">
                      <span className="tag">{r.contract === "Month-to-month" ? "M2M" : r.contract}</span>
                      <span className="tag">{r.internet}</span>
                      <span className="tag">{r.tenure}mo</span>
                    </span>
                  </td>
                </tr>
              ))}
              {data?.rows.length === 0 && (
                <tr>
                  <td colSpan={9} className="py-8 text-center" style={{ color: "var(--ink-3)" }}>
                    No customers match these filters.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      )}

      <div className="mt-3">
        <Note caution>
          <b>Human review required.</b> Every row is a model-estimated effect with a confidence
          interval, not an instruction. Offers are held for agent confirmation before any
          customer is contacted.
        </Note>
      </div>
    </Card>
  );
}
