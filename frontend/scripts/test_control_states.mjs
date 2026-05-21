import {
  getRecognitionStartTransition,
  getWebcamControlState,
} from "../src/components/webcamControlState.js";
import {
  createHandOverlayState,
  getVisibleHandDetections,
} from "../src/lib/handOverlayFilter.js";

function assert(condition, message) {
  if (!condition) {
    throw new Error(message);
  }
}

function main() {
  const idle = getWebcamControlState({
    isStartingCamera: false,
    isStoppingCamera: false,
    isCameraActive: false,
    isRecognitionActive: false,
    isResetting: false,
  });
  assert(idle.startCameraDisabled === false, "Start Camera should be enabled when camera is off.");
  assert(idle.stopCameraDisabled === true, "Stop Camera should be disabled when camera is off.");
  assert(idle.startRecognitionDisabled === true, "Start Recognition should be disabled when camera is off.");

  const cameraReady = getWebcamControlState({
    isStartingCamera: false,
    isStoppingCamera: false,
    isCameraActive: true,
    isRecognitionActive: false,
    isResetting: false,
  });
  assert(cameraReady.startRecognitionDisabled === false, "Start Recognition should be enabled when the camera is on.");
  assert(cameraReady.stopRecognitionDisabled === true, "Stop Recognition should be disabled before recognition starts.");
  assert(cameraReady.resetDisabled === false, "Reset should stay enabled while recognition is stopped.");

  const firstClickWhileWarmupRequestInFlight = getRecognitionStartTransition({
    isCameraActive: true,
    isRecognitionActive: false,
    isRequestInFlight: true,
    isStoppingCamera: false,
  });
  assert(firstClickWhileWarmupRequestInFlight.shouldStart === true, "The first Start Recognition click should still be accepted while a warmup request is in flight.");
  assert(firstClickWhileWarmupRequestInFlight.nextRecognitionActive === true, "A single Start Recognition click should flip recognition on.");
  assert(firstClickWhileWarmupRequestInFlight.shouldKickImmediateCapture === false, "Recognition should queue for the next live frame instead of dropping the click.");

  const recognitionLive = getWebcamControlState({
    isStartingCamera: false,
    isStoppingCamera: false,
    isCameraActive: true,
    isRecognitionActive: true,
    isResetting: false,
  });
  assert(recognitionLive.startRecognitionDisabled === true, "Start Recognition should be disabled while recognition is already running.");
  assert(recognitionLive.stopRecognitionDisabled === false, "Stop Recognition should be enabled while recognition is running.");
  assert(recognitionLive.resetDisabled === true, "Reset should stay disabled while recognition is running.");

  const cameraStopping = getWebcamControlState({
    isStartingCamera: false,
    isStoppingCamera: true,
    isCameraActive: true,
    isRecognitionActive: false,
    isResetting: false,
  });
  assert(cameraStopping.startCameraDisabled === true, "Start Camera should be disabled during camera shutdown.");
  assert(cameraStopping.stopCameraDisabled === true, "Stop Camera should be disabled during camera shutdown.");

  const overlayFrameOne = getVisibleHandDetections(
    {
      landmarks: [createSyntheticHand()],
      handedness: [[{ categoryName: "Left", score: 0.92, index: 0 }]],
    },
    createHandOverlayState(),
  );
  assert(overlayFrameOne.visibleHands.length === 0, "A single-frame hand candidate should not render immediately.");
  assert(overlayFrameOne.debug.pendingDetections === 1, "A first-frame hand candidate should remain pending until it persists.");

  const overlayFrameTwo = getVisibleHandDetections(
    {
      landmarks: [createSyntheticHand()],
      handedness: [[{ categoryName: "Left", score: 0.93, index: 0 }]],
    },
    overlayFrameOne.nextState,
  );
  assert(overlayFrameTwo.visibleHands.length === 1, "A confident hand candidate should render after persisting across two frames.");

  const weakOverlayFrame = getVisibleHandDetections(
    {
      landmarks: [createSyntheticHand()],
      handedness: [[{ categoryName: "Right", score: 0.45, index: 1 }]],
    },
    createHandOverlayState(),
  );
  assert(weakOverlayFrame.visibleHands.length === 0, "Weak one-frame detections should be suppressed.");
  assert(weakOverlayFrame.debug.weakDetectionCount === 1, "Weak detections should be counted as suppressed.");

  console.log("test_control_states: all checks passed");
}

function createSyntheticHand() {
  return [
    { x: 0.45, y: 0.7 },
    { x: 0.47, y: 0.62 },
    { x: 0.49, y: 0.55 },
    { x: 0.51, y: 0.48 },
    { x: 0.54, y: 0.42 },
    { x: 0.42, y: 0.6 },
    { x: 0.39, y: 0.51 },
    { x: 0.37, y: 0.42 },
    { x: 0.35, y: 0.34 },
    { x: 0.46, y: 0.58 },
    { x: 0.46, y: 0.48 },
    { x: 0.46, y: 0.39 },
    { x: 0.46, y: 0.31 },
    { x: 0.5, y: 0.6 },
    { x: 0.52, y: 0.51 },
    { x: 0.54, y: 0.43 },
    { x: 0.56, y: 0.36 },
    { x: 0.54, y: 0.64 },
    { x: 0.58, y: 0.58 },
    { x: 0.61, y: 0.53 },
    { x: 0.64, y: 0.48 },
  ];
}

main();
