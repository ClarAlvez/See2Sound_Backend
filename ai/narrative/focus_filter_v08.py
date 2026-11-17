"""Filtro leve de relevancia (opcional) para Narrative v0.7.

Nao modifica fatos, texto ou timestamps. Escolhe quais descricoes sao
reproduzidas, priorizando pessoas recorrentes e acoes concretas, com limites
de ritmo por minuto. Pode ser desligado a qualquer momento.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from copy import deepcopy
import math

NOTABLE_ACTIONS = {'walking', 'running', 'driving', 'dancing', 'writing', 'reading',
                   'swimming', 'eating', 'drinking', 'phone_use', 'computer_use',
                   'cooking', 'cleaning', 'fighting', 'holding', 'cycling'}
LOW_INFORMATION = {'sitting', 'standing', 'talking', 'moving', 'still'}


def _time(value, fallback=0.):
    try:
        val = float(value)
        return val if math.isfinite(val) else float(fallback)
    except (TypeError, ValueError):
        return float(fallback)


def select_focused_descriptions(timeline, *, correlation_result=None,
                                max_per_minute=4, min_gap_seconds=10.):
    """Subconjunto cronologico, sem alterar descricoes, para o TTS.

    Limites sao aproximacoes por janela [60*k,60*(k+1)); preserva uma
    descricao por no minimo se houver candidatas. Uma escolha priorizada
    pode ocorrer perto da borda de uma janela, respeitando o min_gap.
    """
    if not timeline:
        return []
    if max_per_minute <= 0:
        raise ValueError('max_per_minute deve ser maior que zero')
    recurrences = Counter()
    scenes = defaultdict(set)
    for frame in (correlation_result or {}).get('frames') or []:
        for person in frame.get('people') or []:
            eid=person.get('entity_id')
            if eid:
                recurrences[eid]+=1
                scenes[eid].add(str(frame.get('scene_id')))
    recurring = {eid for eid,ss in scenes.items() if len(ss)>=2 or recurrences[eid]>=6}
    ranked = []
    for idx, item in enumerate(timeline):
        if not isinstance(item, dict) or not str(item.get('description') or '').strip():
            continue
        start = _time(item.get('start_time'))
        facts = item.get('scene_context') or {}
        people = facts.get('correlated_people') or []
        score = 0.
        reasons = []
        if people:
            score += 2
            reasons.append('pessoa_visivel')
        else:
            score -= 2.5
            reasons.append('contexto_geral')
        if any(p.get('confirmed_name') for p in people):
            score += 4
            reasons.append('nome_confirmado')
        if any(p.get('entity_id') in recurring for p in people):
            score += 2
            reasons.append('personagem_recorrente')
        if any(p.get('action') in NOTABLE_ACTIONS for p in people):
            score += 3
            reasons.append('acao_relevante')
        if any(p.get('relation') for p in people):
            score += 3
            reasons.append('interacao_objeto')
        if people and all(p.get('action') in (None, *LOW_INFORMATION)
                          and not p.get('relation') for p in people):
            score -= 1
        # Não escolher apenas pela pontuação de clima/cenário.
        if not people and (facts.get('atmosphere') or facts.get('environment')):
            score -= 0.25
        ranked.append((idx, start, score, reasons))

    selected = []
    counts = Counter()
    # Primeiro as melhores evidencias, nao apenas as cenas do inicio.
    for idx,start,score,reasons in sorted(ranked,key=lambda r:(-r[2],r[1])):
        minute = max(0, int(start//60))
        if counts[minute] >= max_per_minute:
            continue
        if any(abs(start - prior_start) < min_gap_seconds for _,prior_start,_,_ in selected):
            continue
        selected.append((idx,start,score,reasons))
        counts[minute] += 1
    if not selected and ranked:
        selected.append(max(ranked,key=lambda r:r[2]))
    selected.sort(key=lambda row: (row[1],row[0]))
    out = []
    for idx,start,score,reasons in selected:
        item = deepcopy(timeline[idx])
        context = dict(item.get('scene_context') or {})
        context['focus_v08'] = {'score':round(score,2),'reasons':reasons,
                                'mode':'character_relevance','max_per_minute':max_per_minute}
        item['scene_context'] = context
        out.append(item)
    return out
