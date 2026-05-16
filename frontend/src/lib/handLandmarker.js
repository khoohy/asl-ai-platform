import { FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision";

export const HAND_LANDMARKER_WASM_ROOT =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm";
export const HAND_LANDMARKER_MODEL_ASSET_PATH =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

let handLandmarkerPromise = null;

export async function getHandLandmarker() {
  if (handLandmarkerPromise) {
    return handLandmarkerPromise;
  }

  handLandmarkerPromise = (async () => {
    console.info("[keypoints] hand landmarker init start", {
      wasmRoot: HAND_LANDMARKER_WASM_ROOT,
      modelAssetPath: HAND_LANDMARKER_MODEL_ASSET_PATH,
    });

    try {
      const vision = await FilesetResolver.forVisionTasks(
        HAND_LANDMARKER_WASM_ROOT,
      );
      const landmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: HAND_LANDMARKER_MODEL_ASSET_PATH,
        },
        runningMode: "VIDEO",
        numHands: 2,
        minHandDetectionConfidence: 0.5,
        minHandPresenceConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });

      console.info("[keypoints] hand landmarker init success", {
        wasmRoot: HAND_LANDMARKER_WASM_ROOT,
        modelAssetPath: HAND_LANDMARKER_MODEL_ASSET_PATH,
      });
      return landmarker;
    } catch (error) {
      console.error("[keypoints] hand landmarker init error", error);
      handLandmarkerPromise = null;
      throw error;
    }
  })();

  return handLandmarkerPromise;
}
