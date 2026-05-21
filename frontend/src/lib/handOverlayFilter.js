export const HAND_OVERLAY_SETTINGS = Object.freeze({
  minHandDetectionConfidence: 0.72,
  minHandPresenceConfidence: 0.72,
  minTrackingConfidence: 0.7,
  minHandednessScore: 0.8,
  minPersistentFrames: 2,
  stableTrackRetentionFrames: 1,
});

export function createHandOverlayState() {
  return {
    tracks: {},
  };
}

export function getVisibleHandDetections(
  result,
  previousState = createHandOverlayState(),
  settings = HAND_OVERLAY_SETTINGS,
) {
  const previousTracks =
    previousState && typeof previousState === "object" && previousState.tracks
      ? previousState.tracks
      : {};
  const nextTracks = {};
  const visibleHands = [];
  const rawLandmarks = Array.isArray(result?.landmarks) ? result.landmarks : [];
  const rawHandedness = getHandednessGroups(result);
  let candidateCount = 0;
  let weakDetectionCount = 0;

  rawLandmarks.forEach((landmarks, index) => {
    if (!Array.isArray(landmarks) || landmarks.length === 0) {
      return;
    }

    const handednessCategory = getBestCategory(rawHandedness[index]);
    const handednessScore = handednessCategory?.score ?? 0;
    if (handednessScore < settings.minHandednessScore) {
      weakDetectionCount += 1;
      return;
    }

    const trackKey = getTrackKey(handednessCategory?.categoryName, index);
    const previousTrack = previousTracks[trackKey];
    const stableFrames = previousTrack?.isStable
      ? previousTrack.stableFrames + 1
      : (previousTrack?.stableFrames ?? 0) + 1;
    const isStable = stableFrames >= settings.minPersistentFrames;

    candidateCount += 1;
    nextTracks[trackKey] = {
      trackKey,
      handedness: handednessCategory?.categoryName ?? "Unknown",
      handednessScore,
      landmarks,
      stableFrames,
      missedFrames: 0,
      isStable,
    };

    if (isStable) {
      visibleHands.push({
        trackKey,
        handedness: handednessCategory?.categoryName ?? "Unknown",
        handednessScore,
        landmarks,
      });
    }
  });

  Object.entries(previousTracks).forEach(([trackKey, track]) => {
    if (nextTracks[trackKey] || !track?.isStable) {
      return;
    }

    const missedFrames = (track.missedFrames ?? 0) + 1;
    if (missedFrames > settings.stableTrackRetentionFrames) {
      return;
    }

    nextTracks[trackKey] = {
      ...track,
      missedFrames,
    };
  });

  return {
    visibleHands,
    nextState: {
      tracks: nextTracks,
    },
    debug: {
      rawDetections: rawLandmarks.length,
      candidateDetections: candidateCount,
      weakDetectionCount,
      pendingDetections: Math.max(candidateCount - visibleHands.length, 0),
      visibleDetections: visibleHands.length,
    },
  };
}

function getHandednessGroups(result) {
  if (Array.isArray(result?.handedness) && result.handedness.length > 0) {
    return result.handedness;
  }

  if (Array.isArray(result?.handednesses) && result.handednesses.length > 0) {
    return result.handednesses;
  }

  return [];
}

function getBestCategory(categories) {
  if (!Array.isArray(categories) || categories.length === 0) {
    return null;
  }

  return categories.reduce((bestCategory, currentCategory) => {
    if (!bestCategory) {
      return currentCategory;
    }

    return (currentCategory?.score ?? 0) > (bestCategory?.score ?? 0)
      ? currentCategory
      : bestCategory;
  }, null);
}

function getTrackKey(categoryName, index) {
  const normalizedName = String(categoryName ?? "")
    .trim()
    .toLowerCase();

  if (normalizedName === "left") {
    return "left";
  }

  if (normalizedName === "right") {
    return "right";
  }

  return `unknown-${index}`;
}
