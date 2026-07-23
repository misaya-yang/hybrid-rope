#!/usr/bin/env python3
"""Frozen task and prompt protocol for verified SFT data generation.

The program owns every task world, fact, solver, and oracle answer.  The
teacher may write only surface-form framing, unrelated distractors, and an
equivalent question.  Canonical fact sentences are inserted by the program
after the API response, which makes fact preservation directly verifiable.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


GENERATOR_VERSION = "evq-sft-distill-1.2.0"
PROMPT_VERSION = "deepseek-naturalizer-v3"
TOKENIZER_REPO = "EleutherAI/gpt-neox-20b"
TOKENIZER_REVISION = "c292233c833e336628618a88a648727eb3dff0a7"
DEFAULT_SEED = 20260724

TASK_TYPES = (
    "information_retrieval",
    "relation_state_tracking",
    "aggregation_statistics",
    "multi_hop_qa",
)

TASK_WEIGHTS = {
    "information_retrieval": 30,
    "relation_state_tracking": 25,
    "aggregation_statistics": 20,
    "multi_hop_qa": 25,
}

# Bins are half-open except for the final inclusive endpoint.  This removes
# ambiguity at exactly 1K/2K/3K while preserving the requested percentages.
LENGTH_BINS = (
    ("512_1k", 512, 1024, 20),
    ("1k_2k", 1024, 2048, 30),
    ("2k_3k", 2048, 3072, 25),
    ("3k_4k", 3072, 3990, 25),
)

PILOT_SPLIT_COUNTS = {"train": 3000, "validation": 400, "test": 400}
AUDIT_COUNT = 100
MAX_SFT_TOKENS = 4096

BANNED_MARKERS = (
    "ruler",
    "niah",
    "passkey",
    "paul graham",
    "squad",
    "hotpotqa",
    "a points to b",
    "points to",
    "needle in a haystack",
    "common word extraction",
    "frequent word extraction",
    "variable tracking",
)

SPLIT_TEMPLATE_FAMILIES: Mapping[str, Mapping[str, Tuple[str, ...]]] = {
    "audit": {
        "information_retrieval": (
            "audit_craft_inventory",
            "audit_field_station_digest",
            "audit_civic_record",
        ),
        "relation_state_tracking": (
            "audit_conservation_custody",
            "audit_workshop_schedule",
            "audit_greenhouse_status",
        ),
        "aggregation_statistics": (
            "audit_cooperative_shipments",
            "audit_restoration_hours",
            "audit_observation_counts",
        ),
        "multi_hop_qa": (
            "audit_curator_collection_site",
            "audit_coordinator_project_sponsor",
            "audit_vessel_sample_laboratory",
        ),
    },
    "train": {
        "information_retrieval": (
            "train_maritime_bulletin",
            "train_conservatory_catalog",
            "train_workshop_minutes",
        ),
        "relation_state_tracking": (
            "train_gear_handoffs",
            "train_orchard_treatment",
            "train_manuscript_custody",
        ),
        "aggregation_statistics": (
            "train_cooperative_shipments",
            "train_energy_meter_summary",
            "train_clinic_supply_use",
        ),
        "multi_hop_qa": (
            "train_curator_collection_building",
            "train_coordinator_project_sponsor",
            "train_vessel_sample_laboratory",
        ),
    },
    "validation": {
        "information_retrieval": (
            "validation_regional_archive",
            "validation_wildlife_survey",
            "validation_restoration_registry",
        ),
        "relation_state_tracking": (
            "validation_radio_assignments",
            "validation_sample_processing",
            "validation_venue_changes",
        ),
        "aggregation_statistics": (
            "validation_river_readings",
            "validation_library_acquisitions",
            "validation_volunteer_hours",
        ),
        "multi_hop_qa": (
            "validation_translator_manuscript_archive",
            "validation_ranger_trail_station",
            "validation_designer_installation_gallery",
        ),
    },
    "test": {
        "information_retrieval": (
            "test_astronomy_log",
            "test_community_grants",
            "test_geological_inventory",
        ),
        "relation_state_tracking": (
            "test_habitat_permits",
            "test_maintenance_tickets",
            "test_theater_prop_custody",
        ),
        "aggregation_statistics": (
            "test_observatory_exposures",
            "test_harvest_crates",
            "test_transit_delays",
        ),
        "multi_hop_qa": (
            "test_researcher_specimen_vault",
            "test_musician_ensemble_venue",
            "test_engineer_device_facility",
        ),
    },
}

FIRST_NAMES = (
    "Mara",
    "Ilan",
    "Tessa",
    "Niko",
    "Selene",
    "Oren",
    "Leona",
    "Davin",
    "Mira",
    "Cassian",
    "Elara",
    "Bram",
    "Noemi",
    "Ravi",
    "Linnea",
    "Tobin",
    "Amaya",
    "Kellan",
    "Soraya",
    "Eamon",
)
LAST_NAMES = (
    "Vale",
    "Morrow",
    "Kestrel",
    "Alder",
    "Rowan",
    "Sayer",
    "Voss",
    "Merrin",
    "Calder",
    "Fenwick",
    "Ormond",
    "Bell",
    "Thorne",
    "Ives",
    "Hart",
    "Linden",
    "Reeve",
    "Quill",
    "Sorrell",
    "Dale",
)
ADJECTIVES = (
    "Amber",
    "Quiet",
    "North",
    "Silver",
    "Cedar",
    "Willow",
    "Granite",
    "Blue",
    "Juniper",
    "Harbor",
    "Meadow",
    "Copper",
    "Autumn",
    "Lunar",
    "River",
    "Moss",
)
NOUNS = (
    "House",
    "Station",
    "Annex",
    "Gallery",
    "Workshop",
    "Archive",
    "Garden",
    "Depot",
    "Studio",
    "Laboratory",
    "Observatory",
    "Pavilion",
    "Registry",
    "Vault",
    "Terrace",
    "Foundry",
)
VALUES = (
    "alder resin",
    "cobalt glaze",
    "linen backing",
    "cedar oil",
    "slate pigment",
    "bronze mesh",
    "moss fiber",
    "quartz powder",
    "willow paper",
    "amber varnish",
    "hemp cord",
    "clay sealant",
)
STATUS_VALUES = (
    "awaiting review",
    "cleared for display",
    "held for calibration",
    "ready for transfer",
    "returned to storage",
    "scheduled for inspection",
    "released for cataloging",
)
DAYS = (
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
)


@dataclass(frozen=True)
class GenerationJob:
    sample_id: str
    stage: str
    split: str
    task_type: str
    length_bin: str
    min_context_tokens: int
    max_context_tokens: int
    target_context_tokens: int
    template_family: str
    world_id: str
    seed: int

    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class TaskSkeleton:
    sample_id: str
    task_type: str
    template_family: str
    world_id: str
    structured_facts: Dict[str, Any]
    canonical_facts: Tuple[str, ...]
    required_evidence: Tuple[str, ...]
    oracle_answer: str
    question_intent: str
    difficulty: Dict[str, Any]
    reserved_terms: Tuple[str, ...]

    def to_prompt_payload(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "task_type": self.task_type,
            "template_family": self.template_family,
            "world_id": self.world_id,
            "canonical_fact_sentences": list(self.canonical_facts),
            "required_evidence_sentences": list(self.required_evidence),
            "oracle_answer": self.oracle_answer,
            "question_intent": self.question_intent,
            "reserved_terms": list(self.reserved_terms),
        }


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _exact_labels(total: int, weights: Mapping[str, int], seed: int) -> List[str]:
    if total <= 0:
        raise ValueError("total must be positive")
    names = list(weights)
    raw = {name: total * float(weights[name]) / 100.0 for name in names}
    counts = {name: int(raw[name]) for name in names}
    remainder = total - sum(counts.values())
    order = sorted(
        names,
        key=lambda name: (raw[name] - counts[name], name),
        reverse=True,
    )
    for name in order[:remainder]:
        counts[name] += 1
    labels = [name for name in names for _ in range(counts[name])]
    random.Random(seed).shuffle(labels)
    return labels


def _length_labels(total: int, seed: int) -> List[Tuple[str, int, int]]:
    weights = {name: weight for name, _, _, weight in LENGTH_BINS}
    labels = _exact_labels(total, weights, seed)
    lookup = {name: (low, high) for name, low, high, _ in LENGTH_BINS}
    return [(name, lookup[name][0], lookup[name][1]) for name in labels]


def _split_counts(stage: str, num_samples: int) -> Dict[str, int]:
    if stage == "audit":
        return {"audit": num_samples}
    if stage != "pilot":
        raise ValueError("stage must be audit or pilot")
    if num_samples == sum(PILOT_SPLIT_COUNTS.values()):
        return dict(PILOT_SPLIT_COUNTS)
    weights = {
        "train": 100 * PILOT_SPLIT_COUNTS["train"] // 3800,
        "validation": 100 * PILOT_SPLIT_COUNTS["validation"] // 3800,
        "test": 100
        - 100 * PILOT_SPLIT_COUNTS["train"] // 3800
        - 100 * PILOT_SPLIT_COUNTS["validation"] // 3800,
    }
    labels = _exact_labels(num_samples, weights, DEFAULT_SEED + 91)
    return {split: labels.count(split) for split in weights}


def build_jobs(
    *,
    stage: str,
    num_samples: int,
    seed: int = DEFAULT_SEED,
) -> List[GenerationJob]:
    """Build deterministic jobs with exact requested task/length marginals."""
    split_counts = _split_counts(stage, num_samples)
    jobs: List[GenerationJob] = []
    global_index = 0
    for split_index, (split, count) in enumerate(split_counts.items()):
        task_labels = _exact_labels(count, TASK_WEIGHTS, seed + 101 * split_index)
        length_labels = _length_labels(count, seed + 211 * split_index)
        rng = random.Random(seed + 307 * split_index)
        rng.shuffle(length_labels)
        family_counters: Dict[Tuple[str, str], int] = {}
        for local_index, (task_type, length_info) in enumerate(
            zip(task_labels, length_labels)
        ):
            length_name, low, high = length_info
            families = SPLIT_TEMPLATE_FAMILIES[split][task_type]
            family = families[local_index % len(families)]
            key = (task_type, family)
            family_counter = family_counters.get(key, 0)
            family_counters[key] = family_counter + 1
            # Four related samples share a world identifier; worlds never cross
            # split or template family.
            world_group = family_counter // 4
            world_id = f"{split}:{family}:world-{world_group:05d}"
            job_seed = seed + 1_000_003 * global_index
            target = rng.randint(
                low + max(16, (high - low) // 4),
                high - max(24, (high - low) // 8),
            )
            sample_id = f"{stage}-{split}-{global_index:05d}"
            jobs.append(
                GenerationJob(
                    sample_id=sample_id,
                    stage=stage,
                    split=split,
                    task_type=task_type,
                    length_bin=length_name,
                    min_context_tokens=low,
                    max_context_tokens=high,
                    target_context_tokens=target,
                    template_family=family,
                    world_id=world_id,
                    seed=job_seed,
                )
            )
            global_index += 1
    if len(jobs) != num_samples:
        raise RuntimeError(f"job allocation drift: {len(jobs)} != {num_samples}")
    return jobs


def _unique_people(rng: random.Random, count: int) -> List[str]:
    candidates = [f"{a} {b}" for a in FIRST_NAMES for b in LAST_NAMES]
    return rng.sample(candidates, count)


def _unique_places(rng: random.Random, count: int) -> List[str]:
    candidates = [f"{a} {b}" for a in ADJECTIVES for b in NOUNS]
    return rng.sample(candidates, count)


def _fact_id(sample_id: str, index: int) -> str:
    return f"{sample_id}:fact-{index:02d}"


def _retrieval_skeleton(job: GenerationJob, rng: random.Random) -> TaskSkeleton:
    entities = _unique_places(rng, 10)
    values = rng.sample(list(VALUES), len(entities))
    records = []
    facts = []
    for index, (entity, value) in enumerate(zip(entities, values)):
        text = (
            f"The signed registry entry for {entity} identifies its approved "
            f"restoration material as {value}."
        )
        records.append(
            {
                "fact_id": _fact_id(job.sample_id, index),
                "entity": entity,
                "field": "approved restoration material",
                "value": value,
                "text": text,
            }
        )
        facts.append(text)
    target_index = rng.randrange(len(records))
    target = records[target_index]
    question = (
        "Ask for the approved restoration material recorded for "
        f"{target['entity']}. Request only the material name."
    )
    return TaskSkeleton(
        sample_id=job.sample_id,
        task_type=job.task_type,
        template_family=job.template_family,
        world_id=job.world_id,
        structured_facts={
            "record_type": "registry",
            "records": records,
            "query": {
                "entity": target["entity"],
                "field": target["field"],
            },
        },
        canonical_facts=tuple(facts),
        required_evidence=(target["text"],),
        oracle_answer=str(target["value"]),
        question_intent=question,
        difficulty={
            "fact_count": len(facts),
            "reasoning_hops": 1,
            "distractor_record_count": len(facts) - 1,
        },
        reserved_terms=tuple(entities + values),
    )


def _tracking_skeleton(job: GenerationJob, rng: random.Random) -> TaskSkeleton:
    items = [f"{ADJECTIVES[i]} folio" for i in rng.sample(range(len(ADJECTIVES)), 4)]
    places = _unique_places(rng, 9)
    people = _unique_people(rng, 9)
    states: Dict[str, str] = {}
    records: List[Dict[str, Any]] = []
    facts: List[str] = []
    evidence_by_item: Dict[str, List[str]] = {item: [] for item in items}
    fact_index = 0
    for item, place in zip(items, places[: len(items)]):
        states[item] = place
        text = (
            f"At the opening inventory, the {item} was logged in {place}."
        )
        records.append(
            {
                "fact_id": _fact_id(job.sample_id, fact_index),
                "order": fact_index,
                "item": item,
                "event": "initial_location",
                "destination": place,
                "text": text,
            }
        )
        fact_index += 1
        facts.append(text)
        evidence_by_item[item].append(text)
    target_item = rng.choice(items)
    event_items = [target_item, target_item, target_item] + [
        rng.choice(items) for _ in range(7)
    ]
    rng.shuffle(event_items)
    for event_order, item in enumerate(event_items, start=1):
        origin = states[item]
        destination = rng.choice([place for place in places if place != origin])
        actor = people[event_order % len(people)]
        day = DAYS[event_order % len(DAYS)]
        text = (
            f"In transfer entry {event_order}, dated {day}, {actor} moved the "
            f"{item} from {origin} to "
            f"{destination} and recorded the transfer in the custody log."
        )
        states[item] = destination
        records.append(
            {
                "fact_id": _fact_id(job.sample_id, fact_index),
                "order": fact_index,
                "item": item,
                "event": "move",
                "origin": origin,
                "destination": destination,
                "actor": actor,
                "text": text,
            }
        )
        fact_index += 1
        facts.append(text)
        evidence_by_item[item].append(text)
    oracle = states[target_item]
    return TaskSkeleton(
        sample_id=job.sample_id,
        task_type=job.task_type,
        template_family=job.template_family,
        world_id=job.world_id,
        structured_facts={
            "record_type": "ordered custody log",
            "events": records,
            "query": {"item": target_item, "field": "final_location"},
        },
        canonical_facts=tuple(facts),
        required_evidence=tuple(evidence_by_item[target_item]),
        oracle_answer=oracle,
        question_intent=(
            "Ask where the "
            f"{target_item} was located after every dated transfer had occurred. "
            "Request only the final location name."
        ),
        difficulty={
            "fact_count": len(facts),
            "reasoning_hops": len(evidence_by_item[target_item]),
            "state_updates_for_target": len(evidence_by_item[target_item]) - 1,
            "distractor_event_count": len(facts)
            - len(evidence_by_item[target_item]),
        },
        reserved_terms=tuple(items + places + people),
    )


def _aggregation_skeleton(job: GenerationJob, rng: random.Random) -> TaskSkeleton:
    sites = _unique_places(rng, 5)
    days = list(DAYS[:5])
    records: List[Dict[str, Any]] = []
    facts: List[str] = []
    fact_index = 0
    for site in sites:
        chosen_days = rng.sample(days, 3)
        for day in chosen_days:
            count = rng.randint(7, 38)
            text = (
                f"The {day} ledger for {site} records {count} completed "
                "inspection units."
            )
            records.append(
                {
                    "fact_id": _fact_id(job.sample_id, fact_index),
                    "site": site,
                    "day": day,
                    "completed_units": count,
                    "text": text,
                }
            )
            fact_index += 1
            facts.append(text)
    target_site = rng.choice(sites)
    selected = [record for record in records if record["site"] == target_site]
    total = sum(int(record["completed_units"]) for record in selected)
    return TaskSkeleton(
        sample_id=job.sample_id,
        task_type=job.task_type,
        template_family=job.template_family,
        world_id=job.world_id,
        structured_facts={
            "record_type": "inspection ledger",
            "records": records,
            "query": {
                "operation": "sum",
                "site": target_site,
                "field": "completed_units",
            },
        },
        canonical_facts=tuple(facts),
        required_evidence=tuple(record["text"] for record in selected),
        oracle_answer=f"{total} units",
        question_intent=(
            f"Ask for the total completed inspection units recorded for "
            f"{target_site} across all of its listed days. Request a number "
            "followed by the word units."
        ),
        difficulty={
            "fact_count": len(facts),
            "reasoning_hops": len(selected),
            "aggregation_operation": "integer_sum",
            "relevant_record_count": len(selected),
            "distractor_record_count": len(records) - len(selected),
        },
        reserved_terms=tuple(sites),
    )


def _multihop_skeleton(job: GenerationJob, rng: random.Random) -> TaskSkeleton:
    people = _unique_people(rng, 4)
    projects = [f"{ADJECTIVES[i]} {NOUNS[j]} study" for i, j in zip(
        rng.sample(range(len(ADJECTIVES)), 4),
        rng.sample(range(len(NOUNS)), 4),
    )]
    collections = [
        f"{ADJECTIVES[i]} collection"
        for i in rng.sample(range(len(ADJECTIVES)), 4)
    ]
    sites = _unique_places(rng, 4)
    statuses = rng.sample(list(STATUS_VALUES), 4)
    records: List[Dict[str, Any]] = []
    facts: List[str] = []
    chain_evidence: Dict[str, List[str]] = {}
    fact_index = 0
    for person, project, collection, site, status in zip(
        people, projects, collections, sites, statuses
    ):
        chain = [
            (
                f"{person} is the recorded lead for the {project}.",
                {"subject": person, "relation": "leads", "object": project},
            ),
            (
                f"The {project} draws its reference material from the "
                f"{collection}.",
                {
                    "subject": project,
                    "relation": "uses_collection",
                    "object": collection,
                },
            ),
            (
                f"The {collection} is currently housed at {site}.",
                {
                    "subject": collection,
                    "relation": "housed_at",
                    "object": site,
                },
            ),
            (
                f"The current access notice for {site} marks the site as "
                f"{status}.",
                {
                    "subject": site,
                    "relation": "access_status",
                    "object": status,
                },
            ),
        ]
        chain_evidence[person] = []
        for text, triple in chain:
            record = {
                "fact_id": _fact_id(job.sample_id, fact_index),
                "text": text,
                **triple,
            }
            records.append(record)
            facts.append(text)
            chain_evidence[person].append(text)
            fact_index += 1
    target_person = rng.choice(people)
    target_index = people.index(target_person)
    oracle = statuses[target_index]
    return TaskSkeleton(
        sample_id=job.sample_id,
        task_type=job.task_type,
        template_family=job.template_family,
        world_id=job.world_id,
        structured_facts={
            "record_type": "linked institutional notes",
            "relations": records,
            "query": {
                "start_person": target_person,
                "path": [
                    "leads",
                    "uses_collection",
                    "housed_at",
                    "access_status",
                ],
            },
        },
        canonical_facts=tuple(facts),
        required_evidence=tuple(chain_evidence[target_person]),
        oracle_answer=oracle,
        question_intent=(
            f"Ask for the current access status of the site reached by "
            f"following {target_person}'s project, its reference collection, "
            "and the collection's location. Request only the status phrase."
        ),
        difficulty={
            "fact_count": len(facts),
            "reasoning_hops": 4,
            "parallel_chain_count": len(people),
            "distractor_fact_count": len(facts) - 4,
        },
        reserved_terms=tuple(people + projects + collections + sites + statuses),
    )


def build_skeleton(job: GenerationJob) -> TaskSkeleton:
    rng = random.Random(job.seed)
    builders = {
        "information_retrieval": _retrieval_skeleton,
        "relation_state_tracking": _tracking_skeleton,
        "aggregation_statistics": _aggregation_skeleton,
        "multi_hop_qa": _multihop_skeleton,
    }
    skeleton = builders[job.task_type](job, rng)
    solved = solve_skeleton(skeleton)
    if solved != skeleton.oracle_answer:
        raise RuntimeError(
            f"internal oracle mismatch for {job.sample_id}: "
            f"{solved!r} != {skeleton.oracle_answer!r}"
        )
    return skeleton


def solve_skeleton(skeleton: TaskSkeleton) -> str:
    facts = skeleton.structured_facts
    if skeleton.task_type == "information_retrieval":
        query = facts["query"]
        matches = [
            row["value"]
            for row in facts["records"]
            if row["entity"] == query["entity"] and row["field"] == query["field"]
        ]
        if len(matches) != 1:
            raise ValueError("retrieval oracle is not unique")
        return str(matches[0])
    if skeleton.task_type == "relation_state_tracking":
        target = facts["query"]["item"]
        state: Optional[str] = None
        for event in sorted(facts["events"], key=lambda row: int(row["order"])):
            if event["item"] != target:
                continue
            if event["event"] == "move" and state != event["origin"]:
                raise ValueError("tracking event origin does not match prior state")
            state = str(event["destination"])
        if state is None:
            raise ValueError("tracking oracle has no state")
        return state
    if skeleton.task_type == "aggregation_statistics":
        query = facts["query"]
        values = [
            int(row[query["field"]])
            for row in facts["records"]
            if row["site"] == query["site"]
        ]
        if not values:
            raise ValueError("aggregation oracle has no relevant records")
        return f"{sum(values)} units"
    if skeleton.task_type == "multi_hop_qa":
        current = facts["query"]["start_person"]
        for relation in facts["query"]["path"]:
            matches = [
                row["object"]
                for row in facts["relations"]
                if row["subject"] == current and row["relation"] == relation
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"multi-hop oracle is not unique at {current!r}/{relation!r}"
                )
            current = str(matches[0])
        return current
    raise ValueError(f"unknown task type {skeleton.task_type!r}")


SYSTEM_PROMPT = """You are a constrained surface-form editor for a scientific data generator.
The program has already created the task facts and the unique oracle answer.
You MUST NOT solve a new task, alter a fact, invent a claim about a reserved
entity, or copy benchmark language. Output one JSON object and nothing else.

Your allowed work:
1. Write a natural title and opening for a fictional report.
2. Write unrelated but realistic distractor paragraphs. Distractors must not
   mention any reserved term, canonical entity, canonical value, or answer.
3. Write short transition paragraphs that can sit between exact fact sentences.
4. Rewrite the requested question without changing its answer.
5. Copy the oracle answer exactly as teacher_answer.
6. Copy every required evidence sentence exactly into the evidence array.

Forbidden:
- RULER or any official benchmark template/sample;
- NIAH, passkeys, needles, key-value lookup prompts, or “A points to B” language;
- Paul Graham essays, SQuAD, HotpotQA, variable-tracking templates;
- instructions to ignore context, hidden codes, passwords, or secret keys;
- adding a second possible answer or contradicting any canonical fact.
- real organizations, public figures, real cities/countries, historical events,
  published works, public datasets, or claims about the actual world;
- meta-language such as "the answer", "the query", or "the question".

Every distractor must be plainly fictional and generic. Do not use any proper
noun in distractors or transitions. Use phrases such as "the regional office",
"a maintenance team", or "the internal committee", never a named institution.

JSON schema:
{
  "title": "short fictional report title",
  "opening": "natural opening paragraph",
  "transitions": ["paragraph", "..."],
  "distractor_paragraphs": ["paragraph", "..."],
  "instruction": "equivalent question, with concise answer-format request",
  "teacher_answer": "exact oracle string",
  "evidence": ["exact required evidence sentence", "..."]
}
"""


def prompt_hash() -> str:
    return sha256_text(f"{PROMPT_VERSION}\n{SYSTEM_PROMPT}")


def build_teacher_messages(
    job: GenerationJob,
    skeleton: TaskSkeleton,
    *,
    attempt: int,
    prior_errors: Sequence[str] = (),
    prior_token_count: Optional[int] = None,
) -> List[Dict[str, str]]:
    # English prose averages roughly 1.25-1.4 GPT-NeoX tokens per word.  The
    # validator, not this estimate, is authoritative.
    fact_words = sum(len(text.split()) for text in skeleton.canonical_facts)
    target_words = max(260, int(job.target_context_tokens * 0.75))
    adaptive_scale = 1.0
    if prior_token_count is not None and prior_token_count > 0:
        desired = max(job.min_context_tokens + 64, job.target_context_tokens)
        adaptive_scale = min(
            2.5,
            max(1.15, 1.20 * desired / float(prior_token_count)),
        )
        target_words = int(target_words * adaptive_scale)
    filler_words = max(180, target_words - fact_words)
    paragraph_count = max(6, min(48, filler_words // 85))
    payload = {
        "generator_version": GENERATOR_VERSION,
        "prompt_version": PROMPT_VERSION,
        "attempt": attempt,
        "target_context_tokens_with_gpt_neox_tokenizer": {
            "minimum": job.min_context_tokens,
            "maximum_exclusive": job.max_context_tokens,
            "target": job.target_context_tokens,
        },
        "requested_surface_form": {
            "language": "English",
            "approximate_total_context_words": target_words,
            "approximate_distractor_and_transition_words": filler_words,
            "minimum_distractor_paragraphs": paragraph_count,
            "minimum_transitions": max(3, len(skeleton.canonical_facts) // 3),
            "adaptive_retry_length_multiplier": adaptive_scale,
            "style": (
                "a coherent fictional institutional report with varied paragraph "
                "lengths, no benchmark framing, and no lists of opaque key-value pairs"
            ),
        },
        "task_contract": skeleton.to_prompt_payload(),
        "retry_feedback": {
            "previous_validation_errors": list(prior_errors),
            "previous_context_token_count": prior_token_count,
            "instruction": (
                "Correct every listed error. If the context was short, add fresh "
                "unrelated paragraphs; if long, shorten only distractors."
            )
            if prior_errors
            else "none",
        },
    }
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Return valid json for this immutable task contract:\n"
                + json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
            ),
        },
    ]


def _clean_piece(value: Any) -> str:
    text = str(value).strip()
    return re.sub(r"\n{3,}", "\n\n", text)


def assemble_context(
    skeleton: TaskSkeleton,
    teacher: Mapping[str, Any],
    *,
    seed: int,
) -> str:
    """Assemble teacher prose and immutable fact sentences deterministically."""
    title = _clean_piece(teacher.get("title", ""))
    opening = _clean_piece(teacher.get("opening", ""))
    transitions = [
        _clean_piece(value)
        for value in teacher.get("transitions", [])
        if _clean_piece(value)
    ]
    distractors = [
        _clean_piece(value)
        for value in teacher.get("distractor_paragraphs", [])
        if _clean_piece(value)
    ]
    facts = list(skeleton.canonical_facts)
    # Ordered state transitions must remain chronologically readable.  Other
    # tasks are order-invariant and may distribute their evidence naturally.
    if skeleton.task_type != "relation_state_tracking":
        random.Random(seed).shuffle(facts)
    paragraphs: List[str] = []
    if title:
        paragraphs.append(title)
    if opening:
        paragraphs.append(opening)

    extras = transitions + distractors
    rng = random.Random(seed + 17)
    rng.shuffle(extras)
    fact_cursor = 0
    extra_cursor = 0
    while fact_cursor < len(facts) or extra_cursor < len(extras):
        if extra_cursor < len(extras):
            paragraphs.append(extras[extra_cursor])
            extra_cursor += 1
        if fact_cursor < len(facts):
            paragraphs.append(facts[fact_cursor])
            fact_cursor += 1
        if extra_cursor < len(extras) and rng.random() < 0.45:
            paragraphs.append(extras[extra_cursor])
            extra_cursor += 1
    return "\n\n".join(paragraphs).strip()


def task_distribution(rows: Iterable[Mapping[str, Any]]) -> Dict[str, int]:
    counts = {name: 0 for name in TASK_TYPES}
    for row in rows:
        counts[str(row["task_type"])] += 1
    return counts


def length_distribution(rows: Iterable[Mapping[str, Any]]) -> Dict[str, int]:
    counts = {name: 0 for name, _, _, _ in LENGTH_BINS}
    for row in rows:
        count = int(row["token_count"])
        matched = False
        for name, low, high, _ in LENGTH_BINS:
            if low <= count < high:
                counts[name] += 1
                matched = True
                break
        if not matched:
            raise ValueError(f"token count outside registered bins: {count}")
    return counts
