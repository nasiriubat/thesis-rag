"""Utility script for managing structured knowledge facts.

Usage examples:

    python manage_facts.py rebuild-facts

This command rebuilds the `KnowledgeFact` table from existing
`KnowledgeGraph` triples without re-running OpenAI extraction. Use it
after deploying the structured aggregation feature so that historical
documents gain structured coverage.
"""

import argparse

from app import create_app
from models import db, KnowledgeGraph, KnowledgeFact
from knowledge_graph import normalize_relation, infer_entity_type


def rebuild_facts() -> None:
    """Rebuild KnowledgeFact rows from existing KnowledgeGraph triples."""

    app = create_app()

    with app.app_context():
        print("Clearing existing KnowledgeFact entries...")
        deleted = KnowledgeFact.query.delete()
        db.session.commit()
        print(f"Deleted {deleted} existing facts")

        triples = KnowledgeGraph.query.all()
        print(f"Rebuilding facts from {len(triples)} triples")

        stored = 0
        for triple in triples:
            normalized = normalize_relation(triple.relation)
            entity_type = infer_entity_type(triple.subject, normalized)

            fact = KnowledgeFact(
                file_id=triple.source_id,
                entity_type=entity_type,
                subject=triple.subject,
                relation=triple.relation,
                normalized_relation=normalized,
                object=triple.object,
                attributes={"raw_relation": triple.relation},
                confidence=0.6,
            )
            db.session.add(fact)
            stored += 1

        db.session.commit()
        print(f"Stored {stored} knowledge facts")


def main():
    parser = argparse.ArgumentParser(description="Knowledge fact management")
    parser.add_argument(
        "command",
        choices=["rebuild-facts"],
        help="Operation to perform (currently only 'rebuild-facts').",
    )

    args = parser.parse_args()

    if args.command == "rebuild-facts":
        rebuild_facts()


if __name__ == "__main__":
    main()
