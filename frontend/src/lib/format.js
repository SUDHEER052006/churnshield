export const CURRENCY = "$";

export function money(n, compact = false) {
  if (n === null || n === undefined || Number.isNaN(n)) return "--";
  const sign = n < 0 ? "-" : "";
  const v = Math.abs(n);
  if (compact) {
    if (v >= 1_000_000) return `${sign}${CURRENCY}${(v / 1_000_000).toFixed(2)}M`;
    if (v >= 1_000) return `${sign}${CURRENCY}${(v / 1_000).toFixed(1)}k`;
  }
  return `${sign}${CURRENCY}${Math.round(v).toLocaleString()}`;
}

export const pct = (x, d = 1) =>
  x === null || x === undefined ? "--" : `${(x * 100).toFixed(d)}%`;

/** Treatment effects are on retention: positive = more likely to stay. */
export const pp = (x, d = 1) =>
  x === null || x === undefined ? "--" : `${x >= 0 ? "+" : "-"}${Math.abs(x * 100).toFixed(d)} pp`;

export const num = (n) => (n ?? 0).toLocaleString();

export const QUADRANTS = ["Persuadable", "Sure thing", "Lost cause", "Sleeping dog"];

export const QUADRANT_COLOR = {
  Persuadable: "var(--good)",
  "Sure thing": "var(--brand)",
  "Lost cause": "var(--mute)",
  "Sleeping dog": "var(--bad)",
};

export const quadClass = (q) => `pill q-${(q || "").replace(/\s+/g, "-")}`;

export const QUADRANT_MEANING = {
  Persuadable: "An offer changes their decision. This is where the budget goes.",
  "Sure thing": "Stays either way. Discounting them is margin given away.",
  "Lost cause": "Leaves either way. No offer moves the needle.",
  "Sleeping dog": "Contacting them is estimated to make churn worse.",
};
