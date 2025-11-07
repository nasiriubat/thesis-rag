"""
Knowledge Graph module for extracting, storing, and searching triples (subject, relation, object)
"""
from models import db, KnowledgeGraph, File, KnowledgeFact
from sqlalchemy import or_, text
from typing import List, Dict, Optional
import re


def extract_triples_from_text(text: str, openai_client, language: str = 'en') -> List[Dict]:
    """
    Extract knowledge graph triples from text using OpenAI.
    
    Args:
        text: Input text to extract triples from
        openai_client: OpenAI client instance
        language: Language code ('en' for English, 'fi' for Finnish)
    
    Returns:
        List of dicts with 'subject', 'relation', 'object' keys
    """
    try:
        # Language-specific instructions
        language_name = "Finnish" if language == 'fi' else "English"
        language_instruction = (
            f"The text is in {language_name}. Extract triples preserving the original {language_name} language. "
            f"Keep all subjects, relations, and objects in {language_name} as they appear in the text."
        )
        
        prompt = f"""Extract knowledge graph triples from the following text. 
{language_instruction}

For each triple, provide the subject, relation, and object in the format:
SUBJECT | RELATION | OBJECT

Only extract factual, meaningful relationships. Ignore generic or trivial statements.
Limit to the most important triples (max 10).

Text:
{text[:4000]}  # Limit text to avoid token limits

Format your response as:
SUBJECT | RELATION | OBJECT
"""

        system_message = (
            f"You are a knowledge extraction expert. Extract clear, factual triples from text. "
            f"The text is in {language_name} - preserve the original language in all extracted triples."
        )

        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=500,
            timeout=30.0
        )
        
        content = response.choices[0].message.content
        triples = []
        
        # Parse the response
        for line in content.split('\n'):
            line = line.strip()
            if not line or '|' not in line:
                continue
            
            parts = [p.strip() for p in line.split('|')]
            if len(parts) >= 3:
                subject = parts[0]
                relation = parts[1]
                obj = '|'.join(parts[2:])  # In case object contains |
                
                if subject and relation and obj:
                    triples.append({
                        'subject': subject,
                        'relation': relation,
                        'object': obj
                    })
        
        return triples
    except Exception as e:
        print(f"Error extracting triples: {e}")
        return []


def normalize_relation(relation: str) -> str:
    rel = relation.lower().strip()
    rel = rel.replace('-', ' ')

    if any(keyword in rel for keyword in ["located", "based", "headquartered", "situated"]):
        return "located_in"
    if any(keyword in rel for keyword in ["phone", "contact", "telephone"]):
        return "has_contact"
    if "email" in rel:
        return "has_email"
    if any(keyword in rel for keyword in ["website", "url", "web site"]):
        return "has_website"
    if any(keyword in rel for keyword in ["offers", "provides", "delivers"]):
        return "offers"
    if any(keyword in rel for keyword in ["works", "develop", "build", "focus"]):
        return "works_on"
    if any(keyword in rel for keyword in ["founded", "established", "started"]):
        return "founded_in"

    # Fallback: normalize generic relation text to snake_case
    rel = re.sub(r"[^a-z0-9]+", "_", rel)
    return rel.strip("_") or "related_to"


def looks_like_organization(subject: str) -> bool:
    lower_subject = subject.lower()
    org_hints = ["inc", "ltd", "oy", "oyj", "corp", "company", "llc", "gmbh", "enterprise", "solutions"]
    return any(hint in lower_subject for hint in org_hints)


def infer_entity_type(subject: str, normalized_relation: str) -> Optional[str]:
    if normalized_relation in {"located_in", "has_contact", "has_email", "has_website"}:
        if looks_like_organization(subject):
            return "organization"
        return "entity"
    if normalized_relation == "founded_in":
        return "organization"
    if normalized_relation == "offers":
        return "organization"
    if normalized_relation == "works_on":
        return "organization"
    return None


def store_triples(triples: List[Dict], source_id: Optional[int] = None) -> int:
    """
    Store triples in the knowledge graph database.
    
    Args:
        triples: List of dicts with 'subject', 'relation', 'object'
        source_id: Optional file ID that these triples come from
    
    Returns:
        Number of triples stored
    """
    stored_count = 0
    try:
        for triple in triples:
            # Check if triple already exists (avoid duplicates)
            existing = KnowledgeGraph.query.filter_by(
                subject=triple['subject'],
                relation=triple['relation'],
                object=triple['object'],
                source_id=source_id
            ).first()
            
            if not existing:
                kg_entry = KnowledgeGraph(
                    subject=triple['subject'],
                    relation=triple['relation'],
                    object=triple['object'],
                    source_id=source_id
                )
                db.session.add(kg_entry)
                stored_count += 1
        
        db.session.commit()
        return stored_count
    except Exception as e:
        db.session.rollback()
        print(f"Error storing triples: {e}")
        return stored_count


def store_facts_from_triples(triples: List[Dict], file_id: int) -> int:
    stored_count = 0
    try:
        for triple in triples:
            subject = triple.get('subject')
            relation = triple.get('relation')
            obj = triple.get('object')

            if not (subject and relation and obj):
                continue

            normalized = normalize_relation(relation)
            entity_type = infer_entity_type(subject, normalized)

            fact = KnowledgeFact(
                file_id=file_id,
                entity_type=entity_type,
                subject=subject,
                relation=relation,
                normalized_relation=normalized,
                object=obj,
                attributes={"raw_relation": relation},
                confidence=0.6,
            )
            db.session.add(fact)
            stored_count += 1

        db.session.commit()
        return stored_count
    except Exception as e:
        db.session.rollback()
        print(f"Error storing knowledge facts: {e}")
        return stored_count


def process_file_to_graph(file_id: int, file_text: str, openai_client, language: str = 'en') -> bool:
    """
    Process a file to extract and store knowledge graph triples.
    
    Args:
        file_id: ID of the file in the database
        file_text: Text content of the file
        openai_client: OpenAI client instance
        language: Language code ('en' for English, 'fi' for Finnish)
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Delete existing triples for this file (if reprocessing)
        KnowledgeGraph.query.filter_by(source_id=file_id).delete()
        KnowledgeFact.query.filter_by(file_id=file_id).delete()
        db.session.commit()
        
        # Extract triples from text with language awareness
        triples = extract_triples_from_text(file_text, openai_client, language)
        
        if not triples:
            return False
        
        # Store triples
        stored_count = store_triples(triples, source_id=file_id)
        facts_count = store_facts_from_triples(triples, file_id=file_id)
        print(f"Stored {stored_count} triples and {facts_count} facts for file {file_id}")
        return (stored_count + facts_count) > 0
    except Exception as e:
        db.session.rollback()
        print(f"Error processing file to graph: {e}")
        return False


def search_knowledge_graph(query: str, top_k: Optional[int] = 5) -> List[Dict]:
    """
    Search the knowledge graph for matching triples.
    
    Uses a multi-strategy approach:
    1. Full query match
    2. Keyword-based matching
    3. Related entity search
    
    Args:
        query: Search query string
        top_k: Number of results to return. If None, returns all matching triples (unlimited)
    
    Returns:
        List of matching triples with 'subject', 'relation', 'object', 'source_id'
    """
    try:
        # Extract keywords from query (remove common words)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                     'of', 'with', 'by', 'from', 'tell', 'me', 'about', 'what', 'who', 
                     'where', 'when', 'why', 'how', 'is', 'are', 'was', 'were', 'be', 'been',
                     'have', 'has', 'had', 'do', 'does', 'did', 'can', 'could', 'should',
                     'would', 'will', 'that', 'this', 'these', 'those', 'some', 'any', 'all'}
        
        # Extract meaningful words (nouns, verbs, adjectives)
        words = re.findall(r'\b[a-zA-Z]{3,}\b', query.lower())
        keywords = [w for w in words if w not in stop_words]
        
        all_results = []
        seen_triples = set()  # Avoid duplicates
        
        # Strategy 1: Full query match (highest priority)
        query_filter = KnowledgeGraph.query.filter(
            or_(
                KnowledgeGraph.subject.ilike(f'%{query}%'),
                KnowledgeGraph.relation.ilike(f'%{query}%'),
                KnowledgeGraph.object.ilike(f'%{query}%')
            )
        )
        full_query_results = query_filter.limit(top_k).all() if top_k else query_filter.all()
        
        for r in full_query_results:
            triple_key = (r.subject, r.relation, r.object, r.source_id)
            if triple_key not in seen_triples:
                seen_triples.add(triple_key)
                all_results.append({
                    'subject': r.subject,
                    'relation': r.relation,
                    'object': r.object,
                    'source_id': r.source_id
                })
        
        # Strategy 2: Individual keyword matches (if we have keywords and need more results)
        if keywords and (top_k is None or len(all_results) < top_k):
            remaining = top_k - len(all_results) if top_k else None
            keyword_results = []
            
            for keyword in keywords[:5]:  # Limit to top 5 keywords
                keyword_filter = KnowledgeGraph.query.filter(
                    or_(
                        KnowledgeGraph.subject.ilike(f'%{keyword}%'),
                        KnowledgeGraph.relation.ilike(f'%{keyword}%'),
                        KnowledgeGraph.object.ilike(f'%{keyword}%')
                    )
                )
                keyword_matches = keyword_filter.limit(remaining).all() if remaining else keyword_filter.all()
                
                for r in keyword_matches:
                    triple_key = (r.subject, r.relation, r.object, r.source_id)
                    if triple_key not in seen_triples:
                        seen_triples.add(triple_key)
                        keyword_results.append({
                            'subject': r.subject,
                            'relation': r.relation,
                            'object': r.object,
                            'source_id': r.source_id
                        })
            
            if remaining:
                all_results.extend(keyword_results[:remaining])
            else:
                all_results.extend(keyword_results)
        
        # Strategy 3: Related entity search (if still need more results)
        if (top_k is None or len(all_results) < top_k) and keywords:
            remaining = top_k - len(all_results) if top_k else None
            
            # Find entities mentioned in the query and get their relationships
            for keyword in keywords[:3]:  # Limit to top 3 keywords
                if remaining is not None and remaining <= 0:
                    break
                
                # Search for entities that contain the keyword
                entity_filter = KnowledgeGraph.query.filter(
                    or_(
                        KnowledgeGraph.subject.ilike(f'%{keyword}%'),
                        KnowledgeGraph.object.ilike(f'%{keyword}%')
                    )
                )
                entity_matches = entity_filter.limit(remaining * 2 if remaining else None).all() if remaining else entity_filter.all()
                
                # Get related entities (second hop)
                related_entities = set()
                for match in entity_matches:
                    if match.subject:
                        related_entities.add(match.subject)
                    if match.object:
                        related_entities.add(match.object)
                
                # Find triples involving related entities
                for entity in list(related_entities)[:5]:  # Limit entities
                    if remaining is not None and remaining <= 0:
                        break
                    related_filter = KnowledgeGraph.query.filter(
                        or_(
                            KnowledgeGraph.subject.ilike(f'%{entity}%'),
                            KnowledgeGraph.object.ilike(f'%{entity}%')
                        )
                    )
                    related_results = related_filter.limit(remaining).all() if remaining else related_filter.all()
                    
                    for r in related_results:
                        triple_key = (r.subject, r.relation, r.object, r.source_id)
                        if triple_key not in seen_triples:
                            seen_triples.add(triple_key)
                            all_results.append({
                                'subject': r.subject,
                                'relation': r.relation,
                                'object': r.object,
                                'source_id': r.source_id
                            })
                            if remaining is not None:
                                remaining -= 1
        
        return all_results[:top_k] if top_k else all_results
        
    except Exception as e:
        print(f"Error searching knowledge graph: {e}")
        return []


def get_graph_stats() -> Dict:
    """
    Get statistics about the knowledge graph.
    
    Returns:
        Dict with 'total_triples', 'unique_subjects', 'unique_relations', 'documents_with_kg'
    """
    try:
        total_triples = KnowledgeGraph.query.count()
        unique_subjects = db.session.query(KnowledgeGraph.subject).distinct().count()
        unique_relations = db.session.query(KnowledgeGraph.relation).distinct().count()
        documents_with_kg = db.session.query(KnowledgeGraph.source_id).distinct().count()
        
        return {
            'total_triples': total_triples,
            'unique_subjects': unique_subjects,
            'unique_relations': unique_relations,
            'documents_with_kg': documents_with_kg
        }
    except Exception as e:
        print(f"Error getting graph stats: {e}")
        return {
            'total_triples': 0,
            'unique_subjects': 0,
            'unique_relations': 0,
            'documents_with_kg': 0
        }


def clear_all_graphs() -> bool:
    """
    Delete all knowledge graph entries.
    
    Returns:
        True if successful, False otherwise
    """
    try:
        KnowledgeGraph.query.delete()
        db.session.commit()
        return True
    except Exception as e:
        db.session.rollback()
        print(f"Error clearing knowledge graph: {e}")
        return False


def delete_document_graph(file_id: int) -> bool:
    """
    Delete all knowledge graph entries for a specific file.
    
    Args:
        file_id: ID of the file
    
    Returns:
        True if successful, False otherwise
    """
    try:
        KnowledgeGraph.query.filter_by(source_id=file_id).delete()
        db.session.commit()
        return True
    except Exception as e:
        db.session.rollback()
        print(f"Error deleting document graph: {e}")
        return False

