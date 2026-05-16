import { getWebcamControlState } from "../src/components/webcamControlState.js";

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

  console.log("test_control_states: all checks passed");
}

main();
