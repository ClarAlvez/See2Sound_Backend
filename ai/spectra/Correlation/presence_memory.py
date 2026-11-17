"""Eventos de visibilidade, sem inferir entrada fisica a partir de corte de camera."""
from __future__ import annotations


class PresenceMemory:
    def __init__(self, gap_seconds=1.5):
        self.gap_seconds = float(gap_seconds)
        self._people = {}
        self.events = []

    @staticmethod
    def display_name(person_id: str):
        tail = str(person_id).split('_')[-1]
        return f'Pessoa {int(tail)}' if tail.isdigit() else f'Pessoa {tail}'

    def observe(self, frame_id: int, timestamp: float, scene_id: str, people: list[dict], explicit_scene_change=False):
        present_ids = set()
        per_frame = []
        for person in people:
            entity_id = person['entity_id']
            if entity_id in present_ids:
                continue
            present_ids.add(entity_id)
            previous = self._people.get(entity_id)
            event = None
            if previous is None:
                event = 'first_appearance'
            elif explicit_scene_change and previous['scene_id'] != scene_id:
                event = 'visible_after_cut'
            elif float(timestamp) - previous['last_seen'] > self.gap_seconds:
                event = 'reappearance'
            self._people[entity_id] = {'last_seen': float(timestamp), 'scene_id': scene_id,
                                      'frame_id': int(frame_id), 'introduced': True}
            if event is not None:
                payload = {'event': event, 'entity_id': entity_id,
                           'display_name': self.display_name(entity_id),
                           'frame_id': int(frame_id), 'timestamp': float(timestamp),
                           'scene_id': scene_id}
                self.events.append(payload)
                per_frame.append(payload)
        return per_frame

    def remap_entities(self, remap):
        if not remap:
            return
        def canonical(eid):
            visited = set()
            while eid in remap and eid not in visited:
                visited.add(eid)
                eid = remap[eid]
            return eid
        new_people = {}
        for eid, rec in self._people.items():
            target = canonical(eid)
            if target not in new_people or new_people[target]['last_seen'] < rec['last_seen']:
                new_people[target] = rec
        self._people = new_people
        seen = set()
        new_events = []
        for event in self.events:
            event = dict(event)
            event['entity_id'] = canonical(event['entity_id'])
            event['display_name'] = self.display_name(event['entity_id'])
            key = (event['frame_id'], event['entity_id'], event['event'])
            if key not in seen:
                seen.add(key)
                new_events.append(event)
        self.events = new_events

    def to_dict(self):
        return {'events': list(self.events), 'people': {eid: {'display_name': self.display_name(eid), **rec}
                                                         for eid, rec in self._people.items()}}
