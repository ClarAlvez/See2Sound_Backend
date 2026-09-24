from typing import Any, Dict, List, Optional, Tuple


class PauseScheduler:
    def __init__(
        self,
        min_pause_duration: float = 0.5,
        padding_seconds: float = 0.10,
        max_delay_seconds: float = 4.0,
    ):
        self.min_pause_duration = (
            min_pause_duration
        )

        self.padding_seconds = (
            padding_seconds
        )

        self.max_delay_seconds = (
            max_delay_seconds
        )

    def build_available_pauses(
        self,
        speech_segments: List[
            Dict[str, Any]
        ],
        total_duration: float,
    ) -> List[Dict[str, float]]:
        if total_duration <= 0:
            return []

        segments = []

        for segment in speech_segments:
            try:
                start = float(
                    segment["start"]
                )

                end = float(
                    segment["end"]
                )

            except (
                KeyError,
                TypeError,
                ValueError,
            ):
                continue

            if end <= start:
                continue

            segments.append(
                {
                    "start": max(
                        0.0,
                        start,
                    ),
                    "end": min(
                        total_duration,
                        end,
                    ),
                }
            )

        segments.sort(
            key=lambda item: item["start"]
        )

        pauses = []
        cursor = 0.0

        for segment in segments:
            start = segment["start"]
            end = segment["end"]

            if start > cursor:
                self._append_pause(
                    pauses=pauses,
                    start=cursor,
                    end=start,
                )

            cursor = max(
                cursor,
                end,
            )

        if cursor < total_duration:
            self._append_pause(
                pauses=pauses,
                start=cursor,
                end=total_duration,
            )

        return pauses

    def find_slot(
        self,
        pauses: List[
            Dict[str, float]
        ],
        occupied_intervals: List[
            Tuple[float, float]
        ],
        narration_duration: float,
        scene_start: Optional[float],
        scene_end: Optional[float],
    ) -> Optional[
        Dict[str, float]
    ]:
        if narration_duration <= 0:
            return None

        needed_duration = (
            narration_duration
            + self.padding_seconds * 2
        )

        free_slots = (
            self._build_free_slots(
                pauses=pauses,
                occupied_intervals=(
                    occupied_intervals
                ),
            )
        )

        candidates = []

        for slot in free_slots:
            if (
                slot["duration"]
                < needed_duration
            ):
                continue

            score = self._score_slot(
                slot=slot,
                scene_start=scene_start,
                scene_end=scene_end,
            )

            if score is None:
                continue

            playback_start = (
                self._choose_playback_start(
                    slot=slot,
                    narration_duration=(
                        narration_duration
                    ),
                    scene_start=scene_start,
                    scene_end=scene_end,
                )
            )

            playback_end = (
                playback_start
                + narration_duration
            )

            occupied_start = max(
                slot["start"],
                playback_start
                - self.padding_seconds,
            )

            occupied_end = min(
                slot["end"],
                playback_end
                + self.padding_seconds,
            )

            candidates.append(
                {
                    "score": score,
                    "pause_start": (
                        slot["pause_start"]
                    ),
                    "pause_end": (
                        slot["pause_end"]
                    ),
                    "slot_start": (
                        slot["start"]
                    ),
                    "slot_end": (
                        slot["end"]
                    ),
                    "playback_start": (
                        playback_start
                    ),
                    "playback_end": (
                        playback_end
                    ),
                    "occupied_start": (
                        occupied_start
                    ),
                    "occupied_end": (
                        occupied_end
                    ),
                }
            )

        if not candidates:
            return None

        candidates.sort(
            key=lambda item: item["score"]
        )

        return candidates[0]

    def _build_free_slots(
        self,
        pauses: List[
            Dict[str, float]
        ],
        occupied_intervals: List[
            Tuple[float, float]
        ],
    ) -> List[Dict[str, float]]:
        slots = []

        occupied = sorted(
            occupied_intervals,
            key=lambda item: item[0],
        )

        for pause in pauses:
            cursor = pause["start"]

            for occupied_start, occupied_end in occupied:
                if occupied_end <= pause["start"]:
                    continue

                if occupied_start >= pause["end"]:
                    break

                clipped_start = max(
                    pause["start"],
                    occupied_start,
                )

                clipped_end = min(
                    pause["end"],
                    occupied_end,
                )

                if clipped_start > cursor:
                    slots.append(
                        {
                            "start": cursor,
                            "end": clipped_start,
                            "duration": (
                                clipped_start
                                - cursor
                            ),
                            "pause_start": (
                                pause["start"]
                            ),
                            "pause_end": (
                                pause["end"]
                            ),
                        }
                    )

                cursor = max(
                    cursor,
                    clipped_end,
                )

            if cursor < pause["end"]:
                slots.append(
                    {
                        "start": cursor,
                        "end": pause["end"],
                        "duration": (
                            pause["end"]
                            - cursor
                        ),
                        "pause_start": (
                            pause["start"]
                        ),
                        "pause_end": (
                            pause["end"]
                        ),
                    }
                )

        return slots

    def _score_slot(
        self,
        slot: Dict[str, float],
        scene_start: Optional[float],
        scene_end: Optional[float],
    ):
        if scene_start is None:
            return (
                3,
                slot["start"],
            )

        if scene_end is None:
            scene_end = scene_start

        overlap_start = max(
            slot["start"],
            scene_start,
        )

        overlap_end = min(
            slot["end"],
            scene_end,
        )

        overlap_duration = (
            overlap_end
            - overlap_start
        )

        if overlap_duration > 0:
            return (
                0,
                -overlap_duration,
                abs(
                    slot["start"]
                    - scene_start
                ),
            )

        if (
            slot["start"]
            >= scene_start
            and slot["start"]
            <= (
                scene_end
                + self.max_delay_seconds
            )
        ):
            return (
                1,
                slot["start"]
                - scene_start,
            )

        if slot["end"] <= scene_start:
            return (
                2,
                scene_start
                - slot["end"],
            )

        return None

    def _choose_playback_start(
        self,
        slot: Dict[str, float],
        narration_duration: float,
        scene_start: Optional[float],
        scene_end: Optional[float],
    ) -> float:
        earliest = (
            slot["start"]
            + self.padding_seconds
        )

        latest = (
            slot["end"]
            - self.padding_seconds
            - narration_duration
        )

        if latest < earliest:
            return earliest

        if scene_start is None:
            return earliest

        preferred = max(
            earliest,
            scene_start
            + self.padding_seconds,
        )

        if scene_end is not None:
            if (
                preferred
                + narration_duration
                > scene_end
            ):
                preferred = max(
                    earliest,
                    scene_end
                    - narration_duration
                    - self.padding_seconds,
                )

        return min(
            preferred,
            latest,
        )

    def _append_pause(
        self,
        pauses: List[
            Dict[str, float]
        ],
        start: float,
        end: float,
    ):
        duration = end - start

        if (
            duration
            < self.min_pause_duration
        ):
            return

        pauses.append(
            {
                "start": round(
                    start,
                    3,
                ),
                "end": round(
                    end,
                    3,
                ),
                "duration": round(
                    duration,
                    3,
                ),
            }
        )