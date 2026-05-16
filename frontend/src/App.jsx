import { useEffect, useState } from "react";

import { API_BASE_URL, fetchHealth, runMockInference } from "./api/client";
import HealthStatus from "./components/HealthStatus";
import MockInferencePanel from "./components/MockInferencePanel";
import PredictionCard from "./components/PredictionCard";
import WebcamPanel from "./components/WebcamPanel";

export default function App() {
  const [health, setHealth] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [healthError, setHealthError] = useState("");
  const [manualInferenceError, setManualInferenceError] = useState("");
  const [webcamInferenceError, setWebcamInferenceError] = useState("");
  const [isHealthLoading, setIsHealthLoading] = useState(true);
  const [isManualInferenceLoading, setIsManualInferenceLoading] = useState(false);
  const [isWebcamInferenceLoading, setIsWebcamInferenceLoading] = useState(false);

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
    setIsManualInferenceLoading(true);
    setManualInferenceError("");

    try {
      const response = await runMockInference();
      setPrediction(response);
    } catch (error) {
      setPrediction(null);
      setManualInferenceError(
        error instanceof Error
          ? error.message
          : "Unable to run mock inference.",
      );
    } finally {
      setIsManualInferenceLoading(false);
    }
  }

  async function handleCaptureAndRunInference(imageBase64) {
    setIsWebcamInferenceLoading(true);
    setWebcamInferenceError("");

    try {
      const response = await runMockInference(imageBase64);
      setPrediction(response);
    } catch (error) {
      setPrediction(null);
      setWebcamInferenceError(
        error instanceof Error
          ? error.message
          : "Unable to run webcam mock inference.",
      );
      throw error;
    } finally {
      setIsWebcamInferenceLoading(false);
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
          <p className="eyebrow">Phase 3 webcam scaffold</p>
          <h1>ASL AI Platform</h1>
          <p className="hero-copy">
            A browser-first dashboard for validating backend availability,
            capturing a webcam frame, and testing the mock ASL inference flow
            before real model integration arrives later.
          </p>

          <div className="hero-meta">
            <div>
              <span className="hero-label">Backend URL</span>
              <code>{API_BASE_URL}</code>
            </div>
            <div>
              <span className="hero-label">Current mode</span>
              <strong>Webcam capture with mock inference</strong>
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

          <WebcamPanel
            isLoading={isWebcamInferenceLoading}
            error={webcamInferenceError}
            onCaptureAndInfer={handleCaptureAndRunInference}
          />

          <MockInferencePanel
            isLoading={isManualInferenceLoading}
            error={manualInferenceError}
            onRunInference={handleRunInference}
          />

          <PredictionCard prediction={prediction} />
        </section>
      </main>
    </div>
  );
}
