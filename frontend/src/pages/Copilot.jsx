import { useEffect, useRef, useState } from "react";
import { api } from "../lib/api";
import { Card } from "../components/ui";

const SUGGESTIONS = [
  "Who should I spend my next 20000 on?",
  "Which customers should we NOT contact?",
  "What are the main churn factors?",
  "Show high-risk customers with expensive plans",
  "How many are on month-to-month?",
  "Is the model fair?",
  "How good is the model?",
];

function Answer({ msg }) {
  return (
    <div
      className="max-w-[80%] self-start rounded-[7px] rounded-bl-[2px] px-3.5 py-2.5 text-[13px] leading-relaxed"
      style={{ background: "var(--card-alt)", border: "1px solid var(--stroke)" }}
    >
      <div className="whitespace-pre-line">{msg.text}</div>

      {msg.table?.length > 0 && (
        <div className="mt-2.5 overflow-x-auto">
          <table className="text-[12px]">
            <thead>
              <tr>
                {Object.keys(msg.table[0]).map((h) => (
                  <th key={h} className="px-2 py-1">
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {msg.table.map((row, i) => (
                <tr key={i}>
                  {Object.entries(row).map(([k, v]) => (
                    <td key={k} className="px-2 py-1">
                      {v}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {msg.intent && (
        <div
          className="mono mt-2.5 pt-1.5 text-[11px]"
          style={{ color: "var(--ink-3)", borderTop: "1px dashed var(--stroke-strong)" }}
        >
          intent: {msg.intent} · params: {JSON.stringify(msg.params)}
        </div>
      )}
    </div>
  );
}

export default function Copilot() {
  const [msgs, setMsgs] = useState([
    {
      role: "a",
      text:
        "Scored base loaded. I answer from the scored dataset using a fixed set of query " +
        "templates — I do not write queries against production. Every answer shows which " +
        "template ran and with what arguments.",
    },
  ]);
  const [q, setQ] = useState("");
  const [busy, setBusy] = useState(false);
  const endRef = useRef(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [msgs]);

  const ask = async (question) => {
    const text = (question ?? q).trim();
    if (!text || busy) return;
    setQ("");
    setMsgs((m) => [...m, { role: "u", text }]);
    setBusy(true);
    try {
      const res = await api.copilot(text);
      setMsgs((m) => [...m, { role: "a", ...res }]);
    } catch (e) {
      setMsgs((m) => [...m, { role: "a", text: `Could not answer: ${e.message}` }]);
    } finally {
      setBusy(false);
    }
  };

  return (
    <Card
      title="Churn Copilot"
      subtitle="Questions are routed to a small set of parameterised queries over the scored dataset. The resolved intent and its arguments are printed under every answer, so the behaviour is auditable."
    >
      <div className="flex max-h-[540px] flex-col gap-3 overflow-y-auto pr-1">
        {msgs.map((m, i) =>
          m.role === "u" ? (
            <div
              key={i}
              className="max-w-[78%] self-end rounded-[7px] rounded-br-[2px] px-3.5 py-2.5 text-[13px]"
              style={{ background: "var(--brand)", color: "#fff" }}
            >
              {m.text}
            </div>
          ) : (
            <Answer key={i} msg={m} />
          )
        )}
        {busy && (
          <div className="self-start text-[12px]" style={{ color: "var(--ink-3)" }}>
            Running query...
          </div>
        )}
        <div ref={endRef} />
      </div>

      <div className="mt-3.5 flex flex-wrap items-center gap-2.5">
        <input
          type="text"
          className="min-w-[200px] flex-1"
          placeholder="Ask about the customer base..."
          value={q}
          onChange={(e) => setQ(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && ask()}
        />
        <button className="btn" onClick={() => ask()} disabled={busy}>
          Ask
        </button>
      </div>

      <div className="mt-3 flex flex-wrap gap-2">
        {SUGGESTIONS.map((s) => (
          <button
            key={s}
            onClick={() => ask(s)}
            className="cursor-pointer rounded-full px-3 py-1 text-[12px]"
            style={{
              border: "1px solid var(--stroke-strong)",
              background: "var(--card)",
              color: "var(--ink-2)",
            }}
          >
            {s}
          </button>
        ))}
      </div>
    </Card>
  );
}
