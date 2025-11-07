"""Query intent detection utilities.

This module provides lightweight heuristics (with optional LLM fallback)
to infer the entity type, relations, and filters implied by a natural
language question. The goal is to recognise structured aggregation
queries (e.g., "companies located in Finland") so we can fetch
pre-indexed facts instead of sending large raw context to the LLM.
"""

from __future__ import annotations

import json
import re
from typing import Dict, List, Optional


ENTITY_KEYWORDS: Dict[str, List[str]] = {
    "organization": [
        "company",
        "companies",
        "organization",
        "organisations",
        "startup",
        "business",
        "firm",
        "enterprise",
        "ngo",
    ],
    "person": ["person", "people", "employee", "founder", "ceo", "team"],
    "institution": ["university", "college", "school", "institute"],
    "project": ["project", "initiative", "program", "programme"],
}

RELATION_KEYWORDS: Dict[str, List[str]] = {
    "located_in": ["located", "based", "headquartered", "situated"],
    "offers": ["offers", "provides", "deliver", "selling"],
    "works_on": ["working on", "develop", "build", "focus"],
    "has_contact": ["contact", "phone", "email"],
    "founded_in": ["founded", "established", "started"],
}

ATTRIBUTE_KEYWORDS: Dict[str, List[str]] = {
    "name": ["name", "called"],
    "location": ["location", "city", "country", "based"],
    "phone": ["phone", "telephone", "contact number"],
    "email": ["email", "mail"],
    "website": ["website", "web site", "url"],
    "description": ["describe", "about"],
}

LOCATION_PATTERN = re.compile(
    r"(?:in|within|based in|located in|headquartered in|from)\s+([\w\s,\-]+)",
    re.IGNORECASE,
)


def _extract_location(question: str) -> Optional[str]:
    match = LOCATION_PATTERN.search(question)
    if match:
        location = match.group(1).strip()
        location = re.sub(r"[?.!]+$", "", location)
        if 1 <= len(location) <= 80:
            return location
    return None


def _heuristic_intent(question: str) -> Dict:
    lower_question = question.lower()
    entity_type = None
    relations: List[str] = []
    attributes: List[str] = []
    filters: Dict[str, str] = {}
    confidence = 0.0

    for candidate, keywords in ENTITY_KEYWORDS.items():
        if any(word in lower_question for word in keywords):
            entity_type = candidate
            confidence += 0.3
            break

    for canonical_relation, keywords in RELATION_KEYWORDS.items():
        if any(word in lower_question for word in keywords):
            relations.append(canonical_relation)

    if relations:
        confidence += 0.3

    for attribute, keywords in ATTRIBUTE_KEYWORDS.items():
        if any(word in lower_question for word in keywords):
            attributes.append(attribute)

    location = _extract_location(question)
    if location:
        filters["location"] = location
        confidence += 0.2
        if "located_in" not in relations:
            relations.append("located_in")

    if "list" in lower_question or lower_question.startswith("what"):
        confidence += 0.1

    return {
        "entity_type": entity_type,
        "relations": relations,
        "attributes": attributes,
        "filters": filters,
        "confidence": min(confidence, 0.95),
        "source": "heuristic",
    }


# Mapping of synonymous relation names to canonical normalized relations
RELATION_SYNONYMS = {
    "situated_in": "located_in",
    "based_in": "located_in",
    "located_in": "located_in",
    "headquartered_in": "located_in",
    "operates_in": "located_in",
    "operating_in": "located_in",
    "presence_in": "located_in",
    "has_office_in": "located_in",
    "contact": "has_contact",
    "phone": "has_contact",
    "contact_info": "has_contact",
    "contact_information": "has_contact",
    "has_contact": "has_contact",
    "has_email": "has_email",
    "email": "has_email",
    "emails": "has_email",
    "has_website": "has_website",
    "website": "has_website",
    "url": "has_website",
    "offers": "offers",
    "provides": "offers",
    "sells": "offers",
    "services": "offers",
    "develops": "works_on",
    "builds": "works_on",
    "works_on": "works_on",
    "focuses_on": "works_on",
    "focused_on": "works_on",
    "creates": "works_on",
    "founded_in": "founded_in",
    "established_in": "founded_in",
    "founded": "founded_in",
}


def normalize_relations(relations: List[str]) -> List[str]:
    normalized = []
    for relation in relations or []:
        rel = relation or ""
        rel_key = rel.lower().strip()
        rel_key = rel_key.replace(" ", "_")
        normalized.append(RELATION_SYNONYMS.get(rel_key, rel_key))
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for rel in normalized:
        if rel not in seen and rel:
            unique.append(rel)
            seen.add(rel)
    return unique


def _llm_fallback(question: str, openai_client) -> Optional[Dict]:
    if openai_client is None:
        return None

    try:
        prompt = (
            "You are a classifier that extracts structured query intent. "
            "Given a user question, respond with a JSON object containing:\n"
            "- entity_type: short noun such as organization, person, institution, project, dataset, etc.\n"
            "- relations: list of canonical verbs (snake_case) like located_in, offers, works_on.\n"
            "- attributes: list of requested fields (e.g., name, location, phone).\n"
            "- filters: key-value object, e.g., {\"location\": \"Finland\"}.\n"
            "If unsure, return null for that field. Keep lists short (<=3 items)."
        )

        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": question},
            ],
            temperature=0.0,
            max_tokens=300,
        )

        content = response.choices[0].message.content or ""
        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if not json_match:
            return None

        raw = json.loads(json_match.group(0))
        return {
            "entity_type": raw.get("entity_type"),
            "relations": raw.get("relations", []) or [],
            "attributes": raw.get("attributes", []) or [],
            "filters": raw.get("filters", {}) or {},
            "confidence": 0.6,
            "source": "llm",
        }
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"Query intent LLM fallback failed: {exc}")
        return None


def infer_query_intent(question: str, openai_client=None, min_confidence: float = 0.5) -> Dict:
    """Infer structured intent for a natural-language question."""

    intent = _heuristic_intent(question)
    intent["relations"] = normalize_relations(intent.get("relations"))

    if intent["confidence"] < min_confidence:
        llm_intent = _llm_fallback(question, openai_client)
        if llm_intent:
            llm_intent["relations"] = normalize_relations(llm_intent.get("relations"))
            return llm_intent

    return intent
