from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SpectraScene:
    """
    Representa uma cena/intervalo reconhecido pela Spectra.

    Essa classe serve como ponte entre o pipeline de visão e o módulo narrativo.
    """

    labels: List[str]
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    confidence: Dict[str, float] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SceneContext:
    """
    Representa a organização semântica das labels.

    A Spectra pode entregar labels cruas como:
    ["person", "running", "street", "night"]

    Este contexto separa essas labels em grupos para ajudar o LLM.
    """

    raw_labels: List[str] = field(default_factory=list)
    subjects: List[str] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)
    objects: List[str] = field(default_factory=list)
    environment: List[str] = field(default_factory=list)
    time: List[str] = field(default_factory=list)
    attributes: List[str] = field(default_factory=list)
    unknown: List[str] = field(default_factory=list)


@dataclass
class NarrativeInput:
    """
    Entrada principal do módulo narrativo.

    Normalmente será criada a partir de uma saída da Spectra.
    """

    labels: List[str]
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    previous_description: Optional[str] = None
    confidence: Dict[str, float] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NarrativeOutput:
    """
    Saída textual pronta para ser usada pelo módulo de áudio.

    O módulo audio_description poderá receber:
    start_time, end_time e description.
    """

    description: str
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    labels: List[str] = field(default_factory=list)
    scene_context: Dict[str, Any] = field(default_factory=dict)
    skipped: bool = False
    skip_reason: Optional[str] = None
    fidelity_warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "description": self.description,
            "labels": self.labels,
            "scene_context": self.scene_context,
            "skipped": self.skipped,
            "skip_reason": self.skip_reason,
            "fidelity_warnings": self.fidelity_warnings,
        }
