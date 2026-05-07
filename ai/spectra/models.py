from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class FrameVisualAnalysis:
    frame_path : str
    timestamp  : float
    caption    : str
    objects    : List[str]
    actions    : List[str]
    scenarios  : List[str]
    confidence : float
    
@dataclass
class SceneVisualAnalysis:
    scene_id        : int
    start_time      : float
    end_time        : float
    frames_analyzed : List[FrameVisualAnalysis]
    scene_sumary    : Dict[str, Any]