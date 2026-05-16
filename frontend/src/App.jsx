import { useEffect, useState } from "react";

import {
  API_BASE_URL,
  fetchHealth,
  resetRealInferenceSession,
  runMockInference,
  runRealInference,
} from "./api/client";
import HealthStatus from "./components/HealthStatus";
import MockInferencePanel from "./components/MockInferencePanel";
import PredictionCard from "./components/PredictionCard";
import WebcamPanel from "./components/WebcamPanel";

const REAL_INFERENCE_SESSION_ID = "default";

export default function App() {
  const [health, setHealth] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [healthError, setHealthError] = useState("");
  const [manualInferenceError, setManualInferenceError] = useState("");
  const [realInferenceError, setRealInferenceError] = useState("");
  const [isHealthLoading, setIsHealthLoading] = useState(true);
  const [isManualInferenceLoading, setIsManualInferenceLoading] = useState(false);
  const [isRealInferenceLoading, setIsRealInferenceLoading] = useState(false);
  const [isSessionResetting, setIsSessionResetting] = useState(false);

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
      setPrediction({
        ...response,
        status: "mock",
        frames_collected: 0,
        sequence_length: 30,
        top_k: [],
      });
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

  async function handleCaptureAndRunRealInference(imageBase64) {
    setIsRealInferenceLoading(true);
    setRealInferenceError("");

    try {
      const response = await runRealInference(
        imageBase64,
        REAL_INFERENCE_SESSION_ID,
      );
      setPrediction(response);
      return response;
    } catch (error) {
      setPrediction(null);
      setRealInferenceError(
        error instanceof Error
          ? error.message
          : "Unable to run raw real inference.",
      );
      throw error;
    } finally {
      setIsRealInferenceLoading(false);
    }
  }

  async function handleResetRealInferenceSession() {
    setIsSessionResetting(true);
    setRealInferenceError("");

    try {
      const response = await resetRealInferenceSession(REAL_INFERENCE_SESSION_ID);
      setPrediction({
        prediction: null,
        confidence: 0,
        top_k: [],
        raw_prediction: null,
        raw_confidence: 0,
        stable_prediction: null,
        stable_confidence: 0,
        stabilization_status: "raw_only",
        vote_count: 0,
        vote_window_size: 10,
        model_source: "asl_wlasl300_realtime",
        status: response.status,
        frames_collected: 0,
        sequence_length: 30,
        note: response.note,
      });
      return response;
    } catch (error) {
      setRealInferenceError(
        error instanceof Error
          ? error.message
          : "Unable to reset the real inference session.",
      );
    } finally {
      setIsSessionResetting(false);
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
          <p className="eyebrow">Phase 4E stabilized real inference</p>
          <h1>ASL AI Platform</h1>
          <p className="hero-copy">
            A browser-first dashboard for validating backend availability,
            capturing webcam frames continuously, filling a 30-frame rolling
            buffer automatically, and testing raw plus stabilized ASL model
            inference during live webcam use.
          </p>

          <div className="hero-meta">
            <div>
              <span className="hero-label">Backend URL</span>
              <code>{API_BASE_URL}</code>
            </div>
            <div>
              <span className="hero-label">Current mode</span>
              <strong>Continuous stabilized 30-frame inference</strong>
            </div>
            <div>
              <span className="hero-label">Session ID</span>
              <strong>{REAL_INFERENCE_SESSION_ID}</strong>
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
            isLoading={isRealInferenceLoading}
            isResetting={isSessionResetting}
            error={realInferenceError}
            inferenceResult={prediction}
            onCaptureAndRunRealInference={handleCaptureAndRunRealInference}
            onResetRealInferenceSession={handleResetRealInferenceSession}
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
