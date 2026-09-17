import copy

from experiments.native_enhancement_oral_20260915.prepare_capture import (
    _correct_evidence,
    _stable_groups,
    annotate_capture_queries,
    intervention_subset,
    layer_quartiles,
)


def _row(task, length, world, condition, query):
    if task == "native_binding":
        records = [("A", "B"), ("C", "D"), ("X", "Y")]
        answer = "B" if query == "a" else "D"
        query_node = "A" if query == "a" else "C"
        evidence = [("A", "B", 2), ("C", "D", 8)]
        intervention = {"layout": condition, "query": query}
    else:
        records = [("A", "B"), ("B", "C"), ("C", "D"),
                   ("W", "X"), ("X", "Y"), ("Y", "Z")]
        if condition == "rewired":
            records[1] = ("B", "Y")
            records[4] = ("X", "C")
        query_node = "A" if query == "a" else "W"
        if condition == "base":
            answer = "D" if query == "a" else "Z"
        else:
            answer = "Z" if query == "a" else "D"
        evidence = [(left, right, 2 + 6 * index)
                    for index, (left, right) in enumerate(records)]
        intervention = {"graph": condition, "query": query}
    positions = [{
        "source": left,
        "destination": right,
        "token_start": start,
        "token_end": start + 4,
        "destination_token_start": start + 2,
        "destination_token_end": start + 3,
    } for left, right, start in evidence]
    group = f"{task}:{length}:{world}"
    return {
        "row_id": f"{group}:{condition}:{query}",
        "group_id": group,
        "task": task,
        "length_cap": length,
        "input_tokens": 100,
        "query_node": query_node,
        "references": [answer],
        "source_records": records,
        "evidence_positions": positions,
        "intervention": intervention,
        "context_id": f"{group}:{condition}",
    }


def _panel():
    rows = []
    for task in ("native_binding", "native_chain"):
        conditions = ("near", "far") if task == "native_binding" else ("base", "rewired")
        for length in (1024, 2048, 4096):
            for world in range(6):
                for condition in conditions:
                    for query in "ab":
                        rows.append(_row(task, length, world, condition, query))
    return rows


def test_capture_selection_is_balanced_and_output_blind():
    rows = _panel()
    first = _stable_groups(rows, 4)
    second = _stable_groups(list(reversed(copy.deepcopy(rows))), 4)
    assert [row["row_id"] for row in first] == [row["row_id"] for row in second]
    assert len(first) == 96
    intervention = intervention_subset(first)
    assert len(intervention) == 64
    counts = {}
    for row in intervention:
        cell = row["task"], row["length_cap"]
        counts[cell] = counts.get(cell, 0) + 1
    assert max(counts.values()) - min(counts.values()) <= 1


def test_binding_and_chain_queries_have_causal_evidence():
    binding = annotate_capture_queries(_row("native_binding", 4096, 0, "near", "a"))
    assert [query["role"] for query in binding["capture_queries"]] == [
        "final_readout", "relation_write_1",
    ]
    chain = annotate_capture_queries(_row("native_chain", 4096, 0, "base", "a"))
    assert len(_correct_evidence(chain)) == 3
    assert [query["role"] for query in chain["capture_queries"]] == [
        "final_readout", "relation_write_1", "relation_write_2", "relation_write_3",
    ]
    for query in chain["capture_queries"]:
        assert max(query["evidence_token_indices"]) <= query["position"]


def test_quartile_layers_use_one_based_quarters_then_zero_based_indices():
    assert layer_quartiles(16) == [3, 7, 11, 15]
