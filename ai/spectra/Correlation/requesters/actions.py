"""Delegacao ao associador de Actions, que exige identidade individual."""
from __future__ import annotations
from ai.spectra.Correlation.action_adapter import CorrelationActionAdapter


class ActionsRequester:
    def __init__(self, min_score=0.30, min_iou=0.20, min_margin=0.07):
        self.adapter = CorrelationActionAdapter(min_score, min_iou, min_margin)

    def request(self, frame: dict, correlated_people: list[dict]) -> list[dict]:
        return self.adapter.process_result(frame, correlated_people)
