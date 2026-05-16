import { useEffect, useState } from "react";

import { API_BASE_URL, fetchHealth, runMockInference } from "./api/client";
import HealthStatus from "./components/HealthStatus";
import MockInferencePanel from "./components/MockInferencePanel";
import PredictionCard from "./components/PredictionCard";

export default function App() {
  const [health, setHealth] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [healthError, setHealthError] = useState("");
  const [inferenceError, setInferenceError] = useState("");
  const [isHealthLoading, setIsHealthLoading] = useState(true);
  const [isInferenceLoading, setIsInferenceLoading] = useState(false);

  async function loadHealth() {
    setIsHealthLoading(true);
    setHealthError("");

    try {
      const response = await fetchHealth();
      setHealth(response);
    } catch (error) {
      setHealth(null);
      setHealthError(
        error instanceof Error
          ? error.message
          : "Unable to reach the backend health endpoint.",
      );
    } finally {
      setIsHealthLoading(false);
    }
  }

  async function handleRunInference() {
    setIsInferenceLoading(true);
    setInferenceError("");

    try {
      const response = await runMockInference();
      setPrediction(response);
    } catch (error) {
      setPrediction(null);
      setInferenceError(
        error instanceof Error
          ? error.message
          : "Unable to run mock inference.",
      );
    } finally {
      setIsInferenceLoading(false);
    }
  }

  useEffect(() => {
    loadHealth();
  }, []);

  return (
    <div className="app-shell">
      <div className="background-orb background-orb--left" />
      <div className="background-orb background-orb--right" />

      <main className="dashboard">
        <section className="hero">
          <p className="eyebrow">Phase 2 frontend scaffold</p>
          <h1>ASL AI Platform</h1>
          <p className="hero-copy">
            A clean React dashboard for validating backend availability and
            testing the mock ASL inference flow before real webcam and model
            integration arrive in Phase 3.
          </p>

          <div className="hero-meta">
            <div>
              <span className="hero-label">Backend URL</span>
              <code>{API_BASE_URL}</code>
            </div>
            <div>
              <span className="hero-label">Current mode</span>
              <strong>Mock inference only</strong>
            </div>
          </div>
        </section>

        <section className="dashboard-grid">
          <HealthStatus
            health={health}
            isLoading={isHealthLoading}
            error={healthError}
            onRefresh={loadHealth}
          />

          <MockInferencePanel
            isLoading={isInferenceLoading}
            error={inferenceError}
            onRunInference={handleRunInference}
          />

          <PredictionCard prediction={prediction} />
        </section>
      </main>
    </div>
  );
}
