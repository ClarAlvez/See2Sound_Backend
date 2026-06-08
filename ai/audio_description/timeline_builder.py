from typing import Any, Dict, List

from ai.audio_description.data_models import AudioDescriptionCue


class TimelineBuilder:
    """
    Converte saídas do módulo narrativo em cues de audiodescrição.

    Entrada esperada do módulo narrativo:
    {
        "start_time": 0.0,
        "end_time": 4.0,
        "description": "Uma pessoa corre pela rua durante a noite."
    }
    """

    def build_from_narrative_dicts(
        self,
        narrative_outputs: List[Dict[str, Any]],
    ) -> List[AudioDescriptionCue]:
        cues = []

        for item in narrative_outputs:
            if item.get("skipped", False):
                continue

            text = item.get("description") or item.get("text") or ""

            if not text:
                continue

            start_time = item.get("start_time", 0.0)

            if start_time is None:
                start_time = 0.0

            cue = AudioDescriptionCue(
                start_time=float(start_time),
                end_time=item.get("end_time"),
                text=text,
                audio_path=item.get("audio_path"),
            )

            cues.append(cue)

        return cues

    def build_from_narrative_outputs(
        self,
        narrative_outputs: List[Any],
    ) -> List[AudioDescriptionCue]:
        cues = []

        for output in narrative_outputs:
            if getattr(output, "skipped", False):
                continue

            text = getattr(output, "description", "")

            if not text:
                continue

            start_time = getattr(output, "start_time", 0.0)

            if start_time is None:
                start_time = 0.0

            cue = AudioDescriptionCue(
                start_time=float(start_time),
                end_time=getattr(output, "end_time", None),
                text=text,
                audio_path=getattr(output, "audio_path", None),
            )

            cues.append(cue)

        return cues
