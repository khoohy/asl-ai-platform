import { useEffect, useRef, useState } from "react";

import { getHandLandmarker } from "../lib/handLandmarker";

const CAPTURE_INTERVAL_MS = 100;
const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17],
];
const OVERLAY_TARGET_FPS = 30;
const OVERLAY_FRAME_INTERVAL_MS = Math.round(1000 / OVERLAY_TARGET_FPS);

export default function WebcamPanel({
  isLoading,
  isResetting,
  error,
  inferenceResult,
  onCaptureAndRunRealInference,
  onResetRealInferenceSession,
  onRuntimeStatsChange,
}) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const overlayCanvasRef = useRef(null);
  const streamRef = useRef(null);
  const intervalRef = useRef(null);
  const requestInFlightRef = useRef(false);
  const runRealInferenceRef = useRef(onCaptureAndRunRealInference);
  const handLandmarkerRef = useRef(null);
  const animationFrameRef = useRef(null);
  const lastOverlayTimestampRef = useRef(0);
  const overlayStatusRef = useRef("Keypoints: loading");
  const loggedVideoReadyRef = useRef(false);
  const overlayCanvasMetricsRef = useRef({
    stageWidth: 0,
    stageHeight: 0,
    dpr: 0,
    canvasWidth: 0,
    canvasHeight: 0,
  });
  const [cameraStatus, setCameraStatus] = useState(
    "Camera is idle. Start the camera to enable capture.",
  );
  const [captureStatus, setCaptureStatus] = useState(
    "No frame captured yet.",
  );
  const [isStartingCamera, setIsStartingCamera] = useState(false);
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [isInferenceSessionRunning, setIsInferenceSessionRunning] = useState(false);
  const [isRequestInFlight, setIsRequestInFlight] = useState(false);
  const [cameraError, setCameraError] = useState("");
  const [framesSent, setFramesSent] = useState(0);
  const [successfulResponses, setSuccessfulResponses] = useState(0);
  const [failedResponses, setFailedResponses] = useState(0);
  const [skippedTicks, setSkippedTicks] = useState(0);
  const [showKeypoints, setShowKeypoints] = useState(true);
  const [overlayStatus, setOverlayStatus] = useState("Keypoints: loading");
  const setOverlayStatusValue = (nextStatus) => {
    if (overlayStatusRef.current === nextStatus) {
      return;
    }

    overlayStatusRef.current = nextStatus;
    setOverlayStatus(nextStatus);
  };

  useEffect(() => {
    runRealInferenceRef.current = onCaptureAndRunRealInference;
  }, [onCaptureAndRunRealInference]);

  useEffect(() => {
    onRuntimeStatsChange?.({
      framesSent,
      successfulResponses,
      failedResponses,
      skippedTicks,
      captureIntervalMs: CAPTURE_INTERVAL_MS,
      isRequestInFlight,
      isCameraActive,
      isInferenceSessionRunning,
    });
  }, [
    failedResponses,
    framesSent,
    isCameraActive,
    isInferenceSessionRunning,
    isRequestInFlight,
    onRuntimeStatsChange,
    skippedTicks,
    successfulResponses,
  ]);

  useEffect(() => {
    let isMounted = true;

    getHandLandmarker()
      .then((landmarker) => {
        if (!isMounted) {
          return;
        }
        handLandmarkerRef.current = landmarker;
        setOverlayStatusValue("Keypoints: no hands");
      })
      .catch((landmarkerError) => {
        if (!isMounted) {
          return;
        }
        console.error("[keypoints] detector load failed", landmarkerError);
        setOverlayStatusValue("Keypoints: error");
      });

    return () => {
      isMounted = false;
    };
  }, []);

  useEffect(() => {
    if (!showKeypoints) {
      setOverlayStatusValue("Keypoints: off");
      clearOverlayCanvas();
      return;
    }

    if (!isCameraActive) {
      setOverlayStatusValue(
        handLandmarkerRef.current ? "Keypoints: no hands" : "Keypoints: loading",
      );
      return;
    }

    setOverlayStatusValue(
      handLandmarkerRef.current ? "Keypoints: no hands" : "Keypoints: loading",
    );
  }, [isCameraActive, showKeypoints]);

  useEffect(() => {
    if (showKeypoints && isCameraActive) {
      startOverlayLoop();
    } else {
      stopOverlayLoop();
      clearOverlayCanvas();
    }

    return () => {
      stopOverlayLoop();
    };
  }, [isCameraActive, showKeypoints]);

  useEffect(() => {
    function handleResize() {
      clearOverlayCanvas();
    }

    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  useEffect(() => {
    return () => {
      releaseCamera(false);
      stopOverlayLoop();
      if (handLandmarkerRef.current) {
        handLandmarkerRef.current.close();
        handLandmarkerRef.current = null;
      }
    };
  }, []);

  function stopInferenceSession(options = {}) {
    const { updateStatus = true } = options;

    if (intervalRef.current !== null) {
      window.clearInterval(intervalRef.current);
      intervalRef.current = null;
    }

    requestInFlightRef.current = false;
    setIsRequestInFlight(false);
    setIsInferenceSessionRunning(false);

    if (updateStatus) {
      setCaptureStatus("Recognition stopped.");
    }
  }

  function releaseCamera(shouldUpdateState = true) {
    stopInferenceSession({ updateStatus: false });
    stopOverlayLoop();

    const stream = streamRef.current;
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.pause();
      videoRef.current.srcObject = null;
    }

    if (shouldUpdateState) {
      setIsCameraActive(false);
      setCameraError("");
      setCameraStatus("Camera stopped. You can start it again at any time.");
      loggedVideoReadyRef.current = false;
      setOverlayStatusValue(
        showKeypoints ? "Keypoints: no hands" : "Keypoints: off",
      );
      clearOverlayCanvas();
    }
  }

  function stopCamera() {
    releaseCamera(true);
  }

  async function startCamera() {
    setIsStartingCamera(true);
    setCameraError("");
    setCameraStatus("Requesting webcam access...");

    try {
      releaseCamera(false);

      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error("This browser does not support webcam access.");
      }

      const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: false,
      });

      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      setIsCameraActive(true);
      setCameraStatus("Camera ready.");
      setCaptureStatus("Recognition is ready.");
      setOverlayStatusValue(
        handLandmarkerRef.current ? "Keypoints: no hands" : "Keypoints: loading",
      );
      if (showKeypoints) {
        startOverlayLoop();
      }
    } catch (cameraStartError) {
      const message = getCameraErrorMessage(cameraStartError);
      setCameraError(message);
      setCameraStatus(message);
      setIsCameraActive(false);
    } finally {
      setIsStartingCamera(false);
    }
  }

  async function captureAndSendFrame({ source }) {
    const imageBase64 = captureFrameToBase64();
    setFramesSent((count) => count + 1);

    try {
      const response = await runRealInferenceRef.current(imageBase64);
      setSuccessfulResponses((count) => count + 1);

      if (source === "manual") {
        setCaptureStatus("Single recognition frame submitted.");
      } else {
        setCaptureStatus(
          `Recognition is running at ${CAPTURE_INTERVAL_MS}ms intervals.`,
        );
      }

      return response;
    } catch (captureError) {
      setFailedResponses((count) => count + 1);
      const message =
        captureError instanceof Error ? captureError.message : "Capture failed.";
      setCaptureStatus(message);
      throw captureError;
    }
  }

  async function handleManualRealInference() {
    if (isInferenceSessionRunning || requestInFlightRef.current) {
      return;
    }

    requestInFlightRef.current = true;
    setIsRequestInFlight(true);

    try {
      await captureAndSendFrame({ source: "manual" });
    } catch (captureError) {
      return;
    } finally {
      requestInFlightRef.current = false;
      setIsRequestInFlight(false);
    }
  }

  function handleStartInferenceSession() {
    if (requestInFlightRef.current || isInferenceSessionRunning) {
      return;
    }

    if (!isCameraActive) {
      setCaptureStatus("Start the camera before recognition.");
      return;
    }

    if (intervalRef.current !== null) {
      setCaptureStatus("Recognition is already running.");
      return;
    }

    setCaptureStatus(
      `Recognition started at ${CAPTURE_INTERVAL_MS}ms intervals.`,
    );
    setIsInferenceSessionRunning(true);

    intervalRef.current = window.setInterval(async () => {
      if (requestInFlightRef.current) {
        setSkippedTicks((count) => count + 1);
        return;
      }

      requestInFlightRef.current = true;
      setIsRequestInFlight(true);

      try {
        await captureAndSendFrame({ source: "loop" });
      } catch (captureError) {
        return;
      } finally {
        requestInFlightRef.current = false;
        setIsRequestInFlight(false);
      }
    }, CAPTURE_INTERVAL_MS);
  }

  async function handleResetSession() {
    if (isInferenceSessionRunning || isResetting) {
      return;
    }

    const response = await onResetRealInferenceSession();
    if (response?.status === "reset") {
      setFramesSent(0);
      setSuccessfulResponses(0);
      setFailedResponses(0);
      setSkippedTicks(0);
      setCaptureStatus("Recognition state reset.");
    }
  }

  function captureFrameToBase64() {
    if (!videoRef.current || !canvasRef.current || !isCameraActive) {
      throw new Error("Camera is not ready yet.");
    }

    const video = videoRef.current;
    const canvas = canvasRef.current;

    if (!video.videoWidth || !video.videoHeight) {
      throw new Error("Capture failed because the video frame is not ready.");
    }

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const context = canvas.getContext("2d");
    if (!context) {
      throw new Error("Canvas rendering is unavailable in this browser.");
    }

    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL("image/jpeg", 0.92);
  }

  function startOverlayLoop() {
    if (animationFrameRef.current !== null) {
      return;
    }

    lastOverlayTimestampRef.current = 0;
    setOverlayStatusValue(
      handLandmarkerRef.current ? "Keypoints: no hands" : "Keypoints: loading",
    );

    const step = (timestamp) => {
      if (!showKeypoints || !isCameraActive || !videoRef.current) {
        animationFrameRef.current = null;
        return;
      }

      if (
        timestamp - lastOverlayTimestampRef.current >= OVERLAY_FRAME_INTERVAL_MS
      ) {
        lastOverlayTimestampRef.current = timestamp;
        updateOverlayFromVideo(timestamp);
      }

      animationFrameRef.current = window.requestAnimationFrame(step);
    };

    animationFrameRef.current = window.requestAnimationFrame(step);
  }

  function stopOverlayLoop() {
    if (animationFrameRef.current !== null) {
      window.cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }
  }

  async function updateOverlayFromVideo(timestamp) {
    const video = videoRef.current;
    const canvas = overlayCanvasRef.current;
    const handLandmarker = handLandmarkerRef.current;

    if (!video || !canvas || !showKeypoints || !isCameraActive) {
      return;
    }

    const overlayContext = prepareOverlayCanvas(canvas, overlayCanvasMetricsRef.current);
    if (!overlayContext) {
      return;
    }

    if (!video.videoWidth || !video.videoHeight || video.readyState < 2) {
      setOverlayStatusValue("Keypoints: loading");
      return;
    }

    if (!loggedVideoReadyRef.current) {
      console.info("[keypoints] video ready", {
        width: video.videoWidth,
        height: video.videoHeight,
        readyState: video.readyState,
      });
      loggedVideoReadyRef.current = true;
    }

    if (!handLandmarker) {
      setOverlayStatusValue("Keypoints: loading");
      return;
    }

    try {
      const result = handLandmarker.detectForVideo(video, timestamp);
      setOverlayStatusValue("Keypoints: detecting");
      drawOverlay(result?.landmarks ?? [], overlayContext);
      setOverlayStatusValue(
        result?.landmarks?.length
          ? "Keypoints: detecting"
          : "Keypoints: no hands",
      );
    } catch (detectionError) {
      console.error("[keypoints] detection error", detectionError);
      setOverlayStatusValue("Keypoints: error");
    }
  }

  function clearOverlayCanvas() {
    const canvas = overlayCanvasRef.current;
    if (!canvas) {
      return;
    }

    const context = canvas.getContext("2d");
    if (!context) {
      return;
    }

    context.clearRect(0, 0, canvas.width, canvas.height);
  }

  function drawOverlay(handLandmarks, context) {
    const video = videoRef.current;
    if (!video || !context) {
      return;
    }

    const drawRect = getCoverRect(
      context.stageWidth,
      context.stageHeight,
      video.videoWidth || 640,
      video.videoHeight || 480,
    );
    handLandmarks.forEach((landmarks) => {
      drawLineGroup(
        context.ctx,
        landmarks,
        HAND_CONNECTIONS,
        drawRect,
        "rgba(77, 232, 170, 0.88)",
        2,
      );
      drawPointGroup(
        context.ctx,
        landmarks,
        drawRect,
        "rgba(77, 232, 170, 1)",
        3,
      );
    });
  }

  const recognitionLabel = isInferenceSessionRunning
    ? "Recognition live"
    : "Recognition ready";
  const recognitionStateText = isInferenceSessionRunning
    ? "Recognition is running."
    : "Recognition is stopped.";
  const overlayToggleLabel = showKeypoints ? "Keypoints on" : "Keypoints off";

  return (
    <section className="panel panel--webcam">
      <div className="panel-header">
        <div>
          <p className="eyebrow">Live camera</p>
          <h2>Recognition view</h2>
        </div>
        <div
          className={`live-indicator ${isInferenceSessionRunning ? "live-indicator--active" : ""}`}
        >
          <span className="live-indicator__dot" />
          <span className="metric-code">
            {isInferenceSessionRunning ? "recognition_live" : "recognition_idle"}
          </span>
        </div>
      </div>

      <p className="panel-copy">
        Start the camera, keep your upper body in frame, and watch stabilized
        sign recognition update in real time.
      </p>

      <div className="control-toolbar">
        <div className="toolbar-group toolbar-group--camera">
          <span className="toolbar-group__label">Camera</span>
          <div className="toolbar-actions">
            <button
              type="button"
              className="secondary-button secondary-button--ghost control-button control-button--compact"
              onClick={startCamera}
              disabled={isStartingCamera || isCameraActive}
            >
              <span className="control-button__label">Start Camera</span>
              <span className="control-button__state metric-code">
                {isStartingCamera ? "booting" : "ready"}
              </span>
            </button>

            <button
              type="button"
              className="secondary-button secondary-button--ghost control-button control-button--compact"
              onClick={stopCamera}
              disabled={!isCameraActive}
            >
              <span className="control-button__label">Stop Camera</span>
              <span className="control-button__state metric-code">
                {isCameraActive ? "online" : "idle"}
              </span>
            </button>

            <button
              type="button"
              className={`secondary-button secondary-button--ghost control-button control-button--compact ${showKeypoints ? "control-button--active" : ""}`}
              onClick={() => setShowKeypoints((value) => !value)}
              disabled={!isCameraActive}
            >
              <span className="control-button__label">Show Keypoints</span>
              <span className="control-button__state metric-code">
                {overlayToggleLabel}
              </span>
            </button>
          </div>
        </div>

        <div className="toolbar-group toolbar-group--inference">
          <span className="toolbar-group__label">Recognition</span>
          <div className="toolbar-actions">
            <button
              type="button"
              className="primary-button primary-button--accent control-button control-button--compact"
              onClick={handleStartInferenceSession}
              disabled={
                !isCameraActive ||
                isInferenceSessionRunning ||
                isRequestInFlight ||
                isLoading
              }
            >
              <span className="control-button__label">Start Recognition</span>
              <span className="control-button__state metric-code">
                {isInferenceSessionRunning ? "live" : "armed"}
              </span>
            </button>

            <button
              type="button"
              className="secondary-button secondary-button--danger control-button control-button--compact"
              onClick={() => stopInferenceSession({ updateStatus: true })}
              disabled={!isInferenceSessionRunning}
            >
              <span className="control-button__label">Stop Recognition</span>
              <span className="control-button__state metric-code">halt</span>
            </button>

            <button
              type="button"
              className="secondary-button secondary-button--danger control-button control-button--compact"
              onClick={handleResetSession}
              disabled={isResetting || isInferenceSessionRunning}
            >
              <span className="control-button__label">Reset</span>
              <span className="control-button__state metric-code">
                {isResetting ? "resetting" : "clear"}
              </span>
            </button>
          </div>
        </div>
      </div>

      <div className={`webcam-stage ${isCameraActive ? "webcam-stage--live" : ""}`}>
        <video
          ref={videoRef}
          className="webcam-video"
          autoPlay
          muted
          playsInline
        />
        <canvas
          ref={overlayCanvasRef}
          className={`webcam-overlay ${showKeypoints ? "webcam-overlay--visible" : ""}`}
          aria-hidden="true"
        />
        {!isCameraActive ? (
          <div className="webcam-placeholder">
            <strong>Camera preview</strong>
            <span>Start the camera to see the live feed here.</span>
          </div>
        ) : null}
      </div>

      <div className="webcam-bottom-bar">
        <div className="webcam-status-stack">
          <div className="status-message-box">
            <p className="status-message-box__line metric-code">{cameraStatus}</p>
            <p className="status-message-box__line metric-code">{recognitionStateText}</p>
            <p className="status-message-box__line metric-code">{captureStatus}</p>
          </div>

          <p
            className={`inline-message inline-message--compact helper-message ${
              cameraError || error ? "inline-message--error" : ""
            }`}
          >
            {cameraError ||
              error ||
              (inferenceResult?.status ? inferenceResult.note : "") ||
              "Frontend keypoints are for live visual feedback only. Backend MediaPipe remains the source of truth for recognition."}
          </p>

          <p className="inline-message inline-message--compact helper-message helper-message--secondary">
            {showKeypoints
              ? overlayStatus
              : "Keypoints: off"}
          </p>
        </div>

        <div className="secondary-actions-panel">
          <div className="release-summary">
            <span className="release-summary__label">Mode</span>
            <strong className="metric-code">{recognitionLabel}</strong>
            <span className="release-summary__meta metric-code">
              {CAPTURE_INTERVAL_MS}ms | about 10 FPS
            </span>
          </div>
        </div>
      </div>

      <dl className="meta-grid meta-grid--camera">
        <div>
          <dt>Camera status</dt>
          <dd>{cameraStatus}</dd>
        </div>
        <div>
          <dt>Overlay</dt>
          <dd>{showKeypoints ? overlayStatus : "Hidden"}</dd>
        </div>
        <div>
          <dt>Live session state</dt>
          <dd className="metric-code">
            {inferenceResult?.status ?? "idle"} |{" "}
            {typeof inferenceResult?.frames_collected === "number" &&
            typeof inferenceResult?.sequence_length === "number"
              ? `${inferenceResult.frames_collected}/${inferenceResult.sequence_length}`
              : "0/30"}
          </dd>
        </div>
        <div>
          <dt>Stabilization state</dt>
          <dd className="metric-code">
            {inferenceResult?.stabilization_status ?? "raw_only"} |{" "}
            {typeof inferenceResult?.vote_count === "number" &&
            typeof inferenceResult?.vote_window_size === "number"
              ? `${inferenceResult.vote_count}/${inferenceResult.vote_window_size}`
              : "0/10"}
          </dd>
        </div>
        <div className="meta-grid__span-full">
          <dt>Session progress</dt>
          <dd className="metric-code">
            {typeof inferenceResult?.frames_collected === "number" &&
            typeof inferenceResult?.sequence_length === "number"
              ? `${inferenceResult.frames_collected}/${inferenceResult.sequence_length} frames with ${inferenceResult?.vote_count ?? 0}/${inferenceResult?.vote_window_size ?? 10} stabilization votes`
              : "Waiting for frames to enter the rolling buffer."}
          </dd>
        </div>
      </dl>

      <details className="runtime-details runtime-details--webcam">
        <summary>Advanced details</summary>
        <div className="advanced-actions">
          <button
            type="button"
            className="secondary-button secondary-button--emphasis control-button control-button--manual"
            onClick={handleManualRealInference}
            disabled={
              !isCameraActive ||
              isLoading ||
              isRequestInFlight ||
              isInferenceSessionRunning
            }
          >
            <span className="control-button__label">Capture single frame</span>
            <span className="control-button__state metric-code">
              {isRequestInFlight && !isInferenceSessionRunning ? "sending" : "manual"}
            </span>
          </button>
        </div>
      </details>

      <canvas ref={canvasRef} className="hidden-canvas" aria-hidden="true" />
    </section>
  );
}

function prepareOverlayCanvas(canvas, metrics) {
  const stage = canvas.parentElement;
  if (!stage) {
    return null;
  }

  const stageWidth = stage.clientWidth;
  const stageHeight = stage.clientHeight;
  const dpr = window.devicePixelRatio || 1;
  const canvasWidth = Math.max(1, Math.round(stageWidth * dpr));
  const canvasHeight = Math.max(1, Math.round(stageHeight * dpr));

  if (
    !metrics ||
    metrics.stageWidth !== stageWidth ||
    metrics.stageHeight !== stageHeight ||
    metrics.dpr !== dpr ||
    metrics.canvasWidth !== canvasWidth ||
    metrics.canvasHeight !== canvasHeight
  ) {
    canvas.width = canvasWidth;
    canvas.height = canvasHeight;
    canvas.style.width = `${stageWidth}px`;
    canvas.style.height = `${stageHeight}px`;

    if (metrics) {
      metrics.stageWidth = stageWidth;
      metrics.stageHeight = stageHeight;
      metrics.dpr = dpr;
      metrics.canvasWidth = canvasWidth;
      metrics.canvasHeight = canvasHeight;
    }
  }

  const ctx = canvas.getContext("2d");
  if (!ctx) {
    return null;
  }

  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, stageWidth, stageHeight);

  return { ctx, stageWidth, stageHeight };
}

function getCameraErrorMessage(error) {
  if (!error || typeof error !== "object") {
    return "Unable to access the webcam.";
  }

  if ("name" in error) {
    switch (error.name) {
      case "NotAllowedError":
      case "PermissionDeniedError":
        return "Camera permission denied. Allow access in the browser and try again.";
      case "NotFoundError":
      case "DevicesNotFoundError":
        return "No camera found on this device.";
      case "NotReadableError":
      case "TrackStartError":
        return "The camera is already in use by another application.";
      case "OverconstrainedError":
        return "Requested camera settings are not supported on this device.";
      default:
        break;
    }
  }

  if (error instanceof Error && error.message) {
    return error.message;
  }

  return "Unable to access the webcam.";
}

function getCoverRect(containerWidth, containerHeight, mediaWidth, mediaHeight) {
  const mediaAspect = mediaWidth / mediaHeight;
  const containerAspect = containerWidth / containerHeight;

  if (mediaAspect > containerAspect) {
    const height = containerHeight;
    const width = height * mediaAspect;
    return {
      offsetX: (containerWidth - width) / 2,
      offsetY: 0,
      width,
      height,
    };
  }

  const width = containerWidth;
  const height = width / mediaAspect;
  return {
    offsetX: 0,
    offsetY: (containerHeight - height) / 2,
    width,
    height,
  };
}

function drawLineGroup(context, points, connections, rect, color, lineWidth) {
  if (!Array.isArray(points) || points.length === 0) {
    return;
  }

  context.strokeStyle = color;
  context.lineWidth = lineWidth;
  context.lineCap = "round";
  context.beginPath();

  connections.forEach(([startIndex, endIndex]) => {
    const start = points[startIndex];
    const end = points[endIndex];
    if (!start || !end) {
      return;
    }
    context.moveTo(
      rect.offsetX + start.x * rect.width,
      rect.offsetY + start.y * rect.height,
    );
    context.lineTo(
      rect.offsetX + end.x * rect.width,
      rect.offsetY + end.y * rect.height,
    );
  });

  context.stroke();
}

function drawPointGroup(context, points, rect, color, radius) {
  if (!Array.isArray(points) || points.length === 0) {
    return;
  }

  context.fillStyle = color;
  points.forEach((point) => {
    context.beginPath();
    context.arc(
      rect.offsetX + point.x * rect.width,
      rect.offsetY + point.y * rect.height,
      radius,
      0,
      Math.PI * 2,
    );
    context.fill();
  });
}
