const BASE = "/api";

async function get(path, params) {
  const qs = params
    ? "?" +
      new URLSearchParams(
        Object.entries(params).filter(([, v]) => v !== undefined && v !== null && v !== "")
      )
    : "";
  const res = await fetch(`${BASE}${path}${qs}`);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `${res.status} ${res.statusText}`);
  }
  return res.json();
}

async function post(path, body) {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const b = await res.json().catch(() => ({}));
    throw new Error(b.detail || `${res.status} ${res.statusText}`);
  }
  return res.json();
}

export const api = {
  health: () => get("/health"),
  summary: () => get("/summary"),
  models: () => get("/models"),
  rai: () => get("/rai"),
  customers: (p) => get("/customers", p),
  customer: (id, p) => get(`/customers/${encodeURIComponent(id)}`, p),
  allocate: (p) => get("/allocate", p),
  simulate: (body) => post("/simulate", body),
  copilot: (question) => post("/copilot", { question }),
};
