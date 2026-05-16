"""Portable runtime stabilization logic for raw ASL model predictions."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from app.ml.runtime_config import (
    ADAPTIVE_CONFIDENCE_FLOOR,
    BASE_CONFIDENCE_THRESHOLD,
    CONFUSION_PAIRS,
    MIN_VOTE_COUNT,
    MOTION_REQUIRED_SIGNS,
    PEAK_HISTORY_WINDOW,
    PEAK_MARGIN,
    PEAK_MIN_COUNT,
    PEAK_SIGN_CONFIDENCE_OVERRIDES,
    RUNNER_UP_MARGIN,
    SIGN_CONFIDENCE_OVERRIDES,
    STABILIZATION_WINDOW,
)


@dataclass
class StabilizationResult:
    raw_prediction: str | None
    raw_confidence: float
    stable_prediction: str | None
    stable_confidence: float
    stabilization_status: str
    vote_count: int
    vote_window_size: int
    note: str
    prediction: str | None
    confidence: float


class ASLStabilizer:
    """Apply portable runtime stabilization rules to raw Top-K model output."""

    def stabilize(
        self,
        session,
        top_k_predictions: list[dict[str, float | str]],
        motion_delta: float,
    ) -> StabilizationResult:
        raw_prediction = None
        raw_confidence = 0.0
        if top_k_predictions:
            raw_prediction = str(top_k_predictions[0]["label"])
            raw_confidence = float(top_k_predictions[0]["confidence"])

        session.last_raw_prediction = raw_prediction
        session.last_raw_confidence = raw_confidence
        session.last_motion_score = motion_delta

        if not top_k_predictions or raw_prediction is None:
            session.prediction_history.append(None)
            session.peak_history.append(None)
            session.stabilization_status = "raw_only"
            return self._result(
                session=session,
                raw_prediction=raw_prediction,
                raw_confidence=raw_confidence,
                stable_prediction=None,
                stable_confidence=0.0,
                stabilization_status="raw_only",
                vote_count=0,
                note="Raw prediction unavailable for stabilization.",
            )

        accept_decision = self._should_accept_candidate(top_k_predictions, motion_delta)
        candidate = raw_prediction if accept_decision.accepted else None
        session.prediction_history.append(candidate)

        peak_candidate = raw_prediction if self._is_peak_candidate(top_k_predictions, motion_delta) else None
        session.peak_history.append(peak_candidate)

        vote_count = self._get_vote_count(session, candidate)
        peak_sign = self._get_peak_stabilized_sign(session)

        if not accept_decision.accepted:
            if peak_sign is not None:
                peak_confidence = self._lookup_confidence(top_k_predictions, peak_sign)
                session.last_stable_prediction = peak_sign
                session.last_stable_confidence = peak_confidence
                session.stabilization_status = "peak_accepted"
                return self._result(
                    session=session,
                    raw_prediction=raw_prediction,
                    raw_confidence=raw_confidence,
                    stable_prediction=peak_sign,
                    stable_confidence=peak_confidence,
                    stabilization_status="peak_accepted",
                    vote_count=self._get_vote_count(session, peak_sign),
                    note=(
                        "Peak detection accepted a short high-confidence sign before "
                        "the regular vote window stabilized."
                    ),
                )

            session.stabilization_status = accept_decision.status
            return self._result(
                session=session,
                raw_prediction=raw_prediction,
                raw_confidence=raw_confidence,
                stable_prediction=None,
                stable_confidence=0.0,
                stabilization_status=accept_decision.status,
                vote_count=0,
                note=accept_decision.note,
            )

        if peak_sign is not None:
            peak_confidence = self._lookup_confidence(top_k_predictions, peak_sign)
            session.last_stable_prediction = peak_sign
            session.last_stable_confidence = peak_confidence
            session.stabilization_status = "peak_accepted"
            return self._result(
                session=session,
                raw_prediction=raw_prediction,
                raw_confidence=raw_confidence,
                stable_prediction=peak_sign,
                stable_confidence=peak_confidence,
                stabilization_status="peak_accepted",
                vote_count=self._get_vote_count(session, peak_sign),
                note=(
                    "Peak detection accepted a short high-confidence sign before "
                    "the regular vote window stabilized."
                ),
            )

        if vote_count < MIN_VOTE_COUNT:
            session.stabilization_status = "collecting_votes"
            return self._result(
                session=session,
                raw_prediction=raw_prediction,
                raw_confidence=raw_confidence,
                stable_prediction=None,
                stable_confidence=0.0,
                stabilization_status="collecting_votes",
                vote_count=vote_count,
                note=(
                    f"Collecting stabilization votes for {raw_prediction}: "
                    f"{vote_count}/{MIN_VOTE_COUNT} in the latest {STABILIZATION_WINDOW} predictions."
                ),
            )

        stable_confidence = self._lookup_confidence(top_k_predictions, raw_prediction)
        session.last_stable_prediction = raw_prediction
        session.last_stable_confidence = stable_confidence
        session.stabilization_status = "stable"
        return self._result(
            session=session,
            raw_prediction=raw_prediction,
            raw_confidence=raw_confidence,
            stable_prediction=raw_prediction,
            stable_confidence=stable_confidence,
            stabilization_status="stable",
            vote_count=vote_count,
            note="Stable prediction accepted after vote history.",
        )

    def _should_accept_candidate(
        self,
        top_k_predictions: list[dict[str, float | str]],
        motion_delta: float,
    ) -> "_CandidateDecision":
        top_sign = str(top_k_predictions[0]["label"])
        top_sign_key = top_sign.upper()
        top_confidence = float(top_k_predictions[0]["confidence"])
        runner_up_confidence = (
            float(top_k_predictions[1]["confidence"])
            if len(top_k_predictions) > 1
            else 0.0
        )
        confidence_margin = top_confidence - runner_up_confidence

        required_motion = MOTION_REQUIRED_SIGNS.get(top_sign_key)
        if required_motion is not None and motion_delta < required_motion:
            return _CandidateDecision(
                accepted=False,
                status="motion_required",
                note=(
                    f"{top_sign} needs more motion before stabilization. "
                    f"Observed motion {motion_delta:.4f}, required {required_motion:.4f}."
                ),
            )

        rivals = CONFUSION_PAIRS.get(top_sign_key, set())
        if rivals:
            for rival in top_k_predictions[1:]:
                rival_sign = str(rival["label"])
                rival_confidence = float(rival["confidence"])
                if rival_sign.upper() in rivals and (top_confidence - rival_confidence) < 0.08:
                    return _CandidateDecision(
                        accepted=False,
                        status="held_confusion",
                        note=(
                            f"Holding {top_sign} because rival {rival_sign} remains too close "
                            f"({top_confidence:.2f} vs {rival_confidence:.2f})."
                        ),
                    )

        minimum_confidence = SIGN_CONFIDENCE_OVERRIDES.get(
            top_sign_key,
            BASE_CONFIDENCE_THRESHOLD,
        )
        adaptive_floor = max(
            ADAPTIVE_CONFIDENCE_FLOOR,
            minimum_confidence - 0.10,
        )

        if top_confidence >= minimum_confidence:
            return _CandidateDecision(
                accepted=True,
                status="raw_predicted",
                note=f"Raw prediction {top_sign} exceeded the base confidence threshold.",
            )

        if top_confidence >= adaptive_floor and confidence_margin >= RUNNER_UP_MARGIN:
            return _CandidateDecision(
                accepted=True,
                status="raw_predicted",
                note=(
                    f"Adaptive fallback accepted {top_sign} below the base threshold because "
                    f"the runner-up margin remained strong."
                ),
            )

        return _CandidateDecision(
            accepted=False,
            status="low_confidence",
            note=(
                f"Raw prediction {top_sign} remained below stabilization confidence thresholds "
                f"({top_confidence:.2f}, margin {confidence_margin:.2f})."
            ),
        )

    def _is_peak_candidate(
        self,
        top_k_predictions: list[dict[str, float | str]],
        motion_delta: float,
    ) -> bool:
        top_sign = str(top_k_predictions[0]["label"])
        top_sign_key = top_sign.upper()
        top_confidence = float(top_k_predictions[0]["confidence"])

        peak_threshold = PEAK_SIGN_CONFIDENCE_OVERRIDES.get(top_sign_key)
        if peak_threshold is None or top_confidence < peak_threshold:
            return False

        runner_up_confidence = (
            float(top_k_predictions[1]["confidence"])
            if len(top_k_predictions) > 1
            else 0.0
        )
        if (top_confidence - runner_up_confidence) < PEAK_MARGIN:
            return False

        required_motion = MOTION_REQUIRED_SIGNS.get(top_sign_key)
        if required_motion is not None and motion_delta < required_motion:
            return False

        rivals = CONFUSION_PAIRS.get(top_sign_key, set())
        for rival in top_k_predictions[1:]:
            rival_sign = str(rival["label"])
            rival_confidence = float(rival["confidence"])
            if rival_sign.upper() in rivals and (top_confidence - rival_confidence) < 0.08:
                return False

        return True

    def _get_peak_stabilized_sign(self, session) -> str | None:
        counts = Counter(sign for sign in session.peak_history if sign)
        if not counts:
            return None
        peak_sign, peak_count = counts.most_common(1)[0]
        if peak_count >= PEAK_MIN_COUNT:
            return peak_sign
        return None

    @staticmethod
    def _get_vote_count(session, candidate: str | None) -> int:
        if not candidate:
            return 0
        return sum(1 for sign in session.prediction_history if sign == candidate)

    @staticmethod
    def _lookup_confidence(
        top_k_predictions: list[dict[str, float | str]],
        sign: str | None,
    ) -> float:
        if sign is None:
            return 0.0
        for item in top_k_predictions:
            if str(item["label"]) == sign:
                return float(item["confidence"])
        return 0.0

    @staticmethod
    def _result(
        session,
        raw_prediction: str | None,
        raw_confidence: float,
        stable_prediction: str | None,
        stable_confidence: float,
        stabilization_status: str,
        vote_count: int,
        note: str,
    ) -> StabilizationResult:
        prediction = stable_prediction if stable_prediction is not None else raw_prediction
        confidence = stable_confidence if stable_prediction is not None else raw_confidence
        return StabilizationResult(
            raw_prediction=raw_prediction,
            raw_confidence=raw_confidence,
            stable_prediction=stable_prediction,
            stable_confidence=stable_confidence,
            stabilization_status=stabilization_status,
            vote_count=vote_count,
            vote_window_size=STABILIZATION_WINDOW,
            note=note,
            prediction=prediction,
            confidence=confidence,
        )


@dataclass
class _CandidateDecision:
    accepted: bool
    status: str
    note: str
