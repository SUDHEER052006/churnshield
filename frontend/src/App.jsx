import { useEffect, useState } from "react";
import { Route, Routes, useLocation } from "react-router-dom";
import Shell from "./components/Shell";
import { ErrorPanel, Loading } from "./components/ui";
import { api } from "./lib/api";
import ActionList from "./pages/ActionList";
import BudgetOptimizer from "./pages/BudgetOptimizer";
import CommandCenter from "./pages/CommandCenter";
import Copilot from "./pages/Copilot";
import Customer360 from "./pages/Customer360";
import ModelLab from "./pages/ModelLab";
import ResponsibleAI from "./pages/ResponsibleAI";

const META = {
  "/": ["Command Center", "Retention operations"],
  "/actions": ["Action List", "Ranked worklist — who to call, with what"],
  "/customer": ["Customer 360", "Prediction, explanation and causal effect for one customer"],
  "/budget": ["Budget Optimizer", "Constrained allocation across the scored base"],
  "/models": ["Model Lab", "Every candidate, and why each was kept or cut"],
  "/responsible-ai": ["Responsible AI", "Fairness, refutation and error analysis"],
  "/copilot": ["Churn Copilot", "Natural-language queries over the scored base"],
};

export default function App() {
  const { pathname } = useLocation();
  const [summary, setSummary] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    api.summary().then(setSummary).catch((e) => setError(e.message));
  }, []);

  const [title, crumb] = META[pathname] ?? ["ChurnShield", ""];

  return (
    <Shell title={title} crumb={crumb} summary={summary}>
      {error ? (
        <ErrorPanel error={error} />
      ) : !summary ? (
        <Loading what="scored base" />
      ) : (
        <Routes>
          <Route path="/" element={<CommandCenter summary={summary} />} />
          <Route path="/actions" element={<ActionList summary={summary} />} />
          <Route path="/customer" element={<Customer360 summary={summary} />} />
          <Route path="/budget" element={<BudgetOptimizer summary={summary} />} />
          <Route path="/models" element={<ModelLab />} />
          <Route path="/responsible-ai" element={<ResponsibleAI />} />
          <Route path="/copilot" element={<Copilot />} />
        </Routes>
      )}
    </Shell>
  );
}
