from typing import Any, Dict, List, Optional


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
        """
        Cria todas as janelas sem fala.

        Inclui:
        - silêncio antes da primeira fala;
        - silêncio entre falas;
        - silêncio depois da última fala;
        - vídeo inteiro, caso não exista fala.
        """

        if total_duration <= 0:
            return []

        segments = sorted(
            speech_segments,
            key=lambda item: float(
                item.get(
                    "start",
                    0.0,
                )
            ),
        )

        pauses = []

        cursor = 0.0

        for segment in segments:
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

    def find_pause(
        self,
        pauses: List[Dict[str, float]],
        used_pause_indexes: set,
        narration_duration: float,
        scene_start: Optional[float],
        scene_end: Optional[float],
    ) -> Optional[
        Dict[str, float]
    ]:
        needed_duration = (
            narration_duration
            + self.padding_seconds * 2
        )

        candidates = []

        for index, pause in enumerate(
            pauses
        ):
            if index in used_pause_indexes:
                continue

            if (
                pause["duration"]
                < needed_duration
            ):
                continue

            score = self._score_pause(
                pause=pause,
                scene_start=scene_start,
                scene_end=scene_end,
            )

            if score is None:
                continue

            candidates.append(
                (
                    score,
                    index,
                    pause,
                )
            )

        if not candidates:
            return None

        candidates.sort(
            key=lambda item: item[0]
        )

        _, pause_index, pause = (
            candidates[0]
        )

        free_space = (
            pause["duration"]
            - narration_duration
        )

        playback_start = (
            pause["start"]
            + free_space / 2
        )

        playback_end = (
            playback_start
            + narration_duration
        )

        return {
            **pause,
            "pause_index": (
                pause_index
            ),
            "playback_start": (
                playback_start
            ),
            "playback_end": (
                playback_end
            ),
        }

    def _score_pause(
        self,
        pause: Dict[str, float],
        scene_start: Optional[float],
        scene_end: Optional[float],
    ):
        if scene_start is None:
            return (
                2,
                pause["start"],
            )

        if scene_end is None:
            scene_end = scene_start

        overlap_start = max(
            pause["start"],
            scene_start,
        )

        overlap_end = min(
            pause["end"],
            scene_end,
        )

        overlap = (
            overlap_end
            - overlap_start
        )

        if overlap > 0:
            return (
                0,
                -overlap,
                abs(
                    pause["start"]
                    - scene_start
                ),
            )

        if (
            pause["start"]
            >= scene_start
            and pause["start"]
            <= (
                scene_end
                + self.max_delay_seconds
            )
        ):
            return (
                1,
                pause["start"]
                - scene_start,
            )

        return None

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