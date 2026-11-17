"""Reconciliacao legada limitada e incremental.

Nao substitui CharacterGraph. A pre-selecao de pares e propositalmente
conservadora: pode deixar de unir um par que a busca exaustiva uniria.
"""
from __future__ import annotations

import heapq
import math
import time
from collections import defaultdict
from typing import Any

import torch
import torch.nn.functional as F

from ai.spectra.Correlation.reconciler import EntityReconciler as _LegacyReconciler


class EntityReconciler(_LegacyReconciler):
    """Preseleciona vizinhos por centroides e atualiza so os pares afetados.

    Parametros adicionais opcionais no config (via getattr):
      reconcile_fast_top_k (8)
      reconcile_fast_min_cosine (0.78)
      reconcile_fast_max_seconds (150)
      reconcile_fast_max_entities (2500)
      reconcile_fast_max_pairs (12000)
      reconcile_fast_max_merges (256)
    """

    def reconcile(self, processed_frames):
        if not self.config.enable_reconciliation:
            return {
                "enabled": False, "completed": True, "merge_count": 0,
                "merges": [], "entity_remap": {}, "reason": "disabled_by_config",
            }

        cfg = self.config
        k = max(1, int(getattr(cfg, "reconcile_fast_top_k", 8)))
        min_cosine = float(getattr(cfg, "reconcile_fast_min_cosine", 0.78))
        max_seconds = max(1.0, float(getattr(cfg, "reconcile_fast_max_seconds", 150.0)))
        max_entities = max(1, int(getattr(cfg, "reconcile_fast_max_entities", 2500)))
        max_pairs = max(1, int(getattr(cfg, "reconcile_fast_max_pairs", 12000)))
        max_merges = max(0, int(getattr(cfg, "reconcile_fast_max_merges", 256)))

        entities = self.memory.entities
        count = len(entities)
        started = time.monotonic()
        last_report = started
        scored = 0
        considered = 0
        merge_records = []
        remap = {}
        versions = defaultdict(int)
        heap = []
        seen_versions = set()
        reason = "completed"
        embeddings = {}
        embedding_dimension = None

        def elapsed():
            return time.monotonic() - started

        def progress(force=False):
            nonlocal last_report
            now = time.monotonic()
            if force or now - last_report >= 12.0:
                print(
                    f"[Reconciliacao rapida] entidades={len(entities)} "
                    f"pares_avaliados={scored}/{max_pairs} "
                    f"fusoes={len(merge_records)} tempo={elapsed():.1f}s",
                    flush=True,
                )
                last_report = now

        def normalized(embedding):
            nonlocal embedding_dimension
            if embedding is None:
                return None
            try:
                vector = embedding.detach().float().flatten().cpu()
                if not vector.numel() or not torch.isfinite(vector).all():
                    return None
                if embedding_dimension is None:
                    embedding_dimension = int(vector.numel())
                if vector.numel() != embedding_dimension:
                    return None
                magnitude = torch.linalg.vector_norm(vector)
                if magnitude <= 1e-8:
                    return None
                return F.normalize(vector, p=2, dim=0)
            except (AttributeError, RuntimeError, ValueError, TypeError):
                return None

        def add_pair(a, b):
            nonlocal scored, considered, reason
            if a == b or a not in entities or b not in entities:
                return
            x, y = sorted((a, b))
            key = (x, y, versions[x], versions[y])
            if key in seen_versions:
                return
            seen_versions.add(key)
            considered += 1
            if self._cooccur_in_same_frame(x, y):
                return
            if scored >= max_pairs:
                reason = "pair_budget_exceeded"
                return
            if elapsed() >= max_seconds:
                reason = "time_budget_exceeded"
                return
            score, details = self._score_pair(x, y)
            scored += 1
            if score is not None and math.isfinite(float(score)) and score >= cfg.reconcile_min_score:
                heapq.heappush(
                    heap,
                    (-float(score), x, y, versions[x], versions[y], details),
                )
            progress()

        if count < 2:
            return {
                "enabled": True, "completed": True, "merge_count": 0,
                "merges": [], "entity_remap": {}, "reason": "too_few_entities",
                "diagnostics": {"entities_initial": count, "pairs_scored": 0},
            }
        if count > max_entities:
            print(f"[Reconciliacao rapida] ignorada: {count} entidades > limite {max_entities}", flush=True)
            return {
                "enabled": False, "completed": False, "merge_count": 0,
                "merges": [], "entity_remap": {}, "reason": "entity_limit_exceeded",
                "diagnostics": {"entities_initial": count, "max_entities": max_entities},
            }

        # A matriz de centroides e calculada por blocos. Nenhum par passa
        # pelo calculo caro de galeria se nao estiver no top-k de pelo menos
        # um dos dois centroides.
        for entity_id, entity in list(entities.items()):
            vec = normalized(entity.embedding)
            if vec is not None:
                embeddings[entity_id] = vec
        ids = list(embeddings)
        if len(ids) >= 2 and max_merges > 0:
            matrix = torch.stack([embeddings[x] for x in ids], dim=0)
            seen_initial = set()
            with torch.no_grad():
                for start in range(0, len(ids), 96):
                    if elapsed() >= max_seconds:
                        reason = "time_budget_exceeded"
                        break
                    sim = matrix[start:start + 96] @ matrix.T
                    neighbors = min(k + 1, len(ids))
                    for local_i in range(sim.shape[0]):
                        i = start + local_i
                        sim[local_i, i] = -1.0
                        scores, indexes = torch.topk(sim[local_i], k=neighbors)
                        for cosine, j in zip(scores.tolist(), indexes.tolist()):
                            if cosine < min_cosine:
                                break
                            a, b = sorted((ids[i], ids[j]))
                            if a == b or (a, b) in seen_initial:
                                continue
                            seen_initial.add((a, b))
                            add_pair(a, b)
                            if reason != "completed":
                                break
                        if reason != "completed":
                            break
                    if reason != "completed":
                        break

        # Se a pre-selecao foi interrompida, nao usar um conjunto parcial
        # de candidatos: isso criaria favorecimento dos IDs iniciais.
        if reason == "completed":
            while heap and len(merge_records) < max_merges:
                if elapsed() >= max_seconds:
                    reason = "time_budget_exceeded"
                    break
                negative_score, a, b, va, vb, details = heapq.heappop(heap)
                if a not in entities or b not in entities:
                    continue
                if versions[a] != va or versions[b] != vb:
                    continue
                if self._cooccur_in_same_frame(a, b):
                    continue
                entity_a, entity_b = entities[a], entities[b]
                if entity_a.first_seen <= entity_b.first_seen:
                    target, source = a, b
                else:
                    target, source = b, a
                score = -negative_score
                self.memory.merge_entities(target_entity_id=target, source_entity_id=source)
                remap[source] = target
                merge_records.append({
                    "source_entity_id": source,
                    "target_entity_id": target,
                    "score": round(score, 4),
                    "details": self._serialize_details(details),
                })
                embeddings.pop(source, None)
                if target in entities:
                    new_vec = normalized(entities[target].embedding)
                    if new_vec is not None:
                        embeddings[target] = new_vec
                    else:
                        embeddings.pop(target, None)
                versions[target] += 1
                versions[source] += 1
                progress()

                # Recalcular somente pares envolvendo o alvo que mudou,
                # sem rescoring dos pares entre as outras entidades.
                vec = embeddings.get(target)
                if vec is not None and len(embeddings) >= 2:
                    candidate_ids = [key for key in embeddings if key != target]
                    with torch.no_grad():
                        mat = torch.stack([embeddings[key] for key in candidate_ids])
                        sims = torch.mv(mat, vec)
                        near = min(k, len(candidate_ids))
                        values, indexes = torch.topk(sims, k=near)
                    for cosine, j in zip(values.tolist(), indexes.tolist()):
                        if cosine < min_cosine:
                            break
                        add_pair(target, candidate_ids[j])
                        if reason != "completed":
                            break
                if reason != "completed":
                    break

            if reason == "completed" and len(merge_records) >= max_merges and heap:
                reason = "merge_budget_exceeded"

        final_remap = self._resolve_entity_remap(remap)
        self._apply_remap_to_frames(processed_frames, final_remap)
        progress(force=True)
        return {
            "enabled": True,
            "completed": reason == "completed",
            "reason": reason,
            "merge_count": len(merge_records),
            "merges": merge_records,
            "entity_remap": final_remap,
            "diagnostics": {
                "policy": "bounded_centroid_topk_incremental_v1",
                "entities_initial": count,
                "centroids_valid": len(ids),
                "shortlist_top_k": k,
                "minimum_centroid_cosine": min_cosine,
                "pairs_considered": considered,
                "pairs_scored": scored,
                "elapsed_seconds": round(elapsed(), 3),
                "max_seconds": max_seconds,
                "max_pairs": max_pairs,
                "max_merges": max_merges,
            },
        }
