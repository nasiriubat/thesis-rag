import os
# Set OpenMP environment variable to handle multiple runtime libraries
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import openai
import numpy as np
import uuid
from flask import current_app
from pathlib import Path
from typing import List, Dict, Optional, Any
from collections import Counter
from tqdm import tqdm
import tiktoken
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Initialize storage paths
EMBEDDINGS_FOLDER = Path("dataembedding")
EMBEDDINGS_FOLDER.mkdir(parents=True, exist_ok=True)

# Initialize the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize tiktoken
encoding = tiktoken.encoding_for_model("text-embedding-ada-002")
try:
    completion_encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
except KeyError:
    completion_encoding = tiktoken.get_encoding("cl100k_base")
MAX_TOKENS = 500  # Smaller chunks for more precise matching
OVERLAP = 100
MIN_SCORE_THRESHOLD = 0.3  # Base threshold for all content types


def estimate_text_tokens(text: str) -> int:
    if not text:
        return 0
    try:
        return len(completion_encoding.encode(text))
    except Exception:
        return len(text.split())


def truncate_text_to_tokens(text: str, token_budget: int) -> str:
    if not text or token_budget <= 0:
        return ""
    try:
        tokens = completion_encoding.encode(text)
        if len(tokens) <= token_budget:
            return text
        truncated_tokens = tokens[:token_budget]
        return completion_encoding.decode(truncated_tokens)
    except Exception:
        return text[: token_budget * 4]


def extract_chunk_metadata(chunk: str, index: int) -> Dict[str, Any]:
    words = re.findall(r"\w+", chunk)
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", chunk) if s.strip()]
    headings = [
        line.strip()
        for line in chunk.splitlines()
        if line.strip() and (line.strip().startswith(("#", "-", "•")) or line.isupper())
    ]
    return {
        "index": index,
        "token_count": len(encoding.encode(chunk)),
        "char_count": len(chunk),
        "word_count": len(words),
        "sentence_count": len(sentences),
        "headings": headings[:5],
        "preview": chunk[:200],
    }


def extract_document_metadata(text: str, chunks: List[str]) -> Dict[str, Any]:
    words = [w.lower() for w in re.findall(r"\w+", text) if len(w) > 3]
    keyword_counts = Counter(words)
    keywords = [w for w, _ in keyword_counts.most_common(12)]
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    summary_sentences = sentences[:3]
    summary = " ".join(summary_sentences)
    if not summary:
        summary = text[:600]
    headings = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
        and len(line.strip()) < 120
        and (line.strip().startswith(("#", "-", "•")) or line.isupper())
    ]
    return {
        "token_count": len(encoding.encode(text)),
        "char_count": len(text),
        "chunk_count": len(chunks),
        "keywords": keywords,
        "headings": headings[:10],
        "summary": summary,
    }

def get_openai_key():
    """Get OpenAI API key from current app context."""
    return current_app.config.get('OPENAI_API_KEY')

def split_into_chunks(text: str, max_tokens: int = MAX_TOKENS, overlap: int = OVERLAP) -> List[str]:
    """Split text into chunks, trying to preserve natural boundaries where possible."""
    # Try to split by paragraphs first
    paragraphs = re.split(r'\n\s*\n', text)
    chunks = []
    
    for paragraph in paragraphs:
        if len(paragraph.strip()) == 0:
            continue
            
        # If paragraph is too long, split it further
        tokens = encoding.encode(paragraph)
        if len(tokens) > max_tokens:
            start = 0
            while start < len(tokens):
                end = min(start + max_tokens, len(tokens))
                chunk = encoding.decode(tokens[start:end])
                chunks.append(chunk)
                start += max_tokens - overlap
        else:
            chunks.append(paragraph)
    
    return chunks

def create_embedding(text: str) -> Optional[np.ndarray]:
    try:
        # Use sentence-transformers instead of OpenAI for embeddings
        embedding = model.encode(text, convert_to_numpy=True)
        return embedding.astype(np.float32)
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

def calculate_relevance_score(chunk: str, query: str, base_score: float) -> float:
    """Calculate a sophisticated relevance score based on content matching."""
    # Convert to lowercase for case-insensitive matching
    chunk_lower = chunk.lower()
    query_lower = query.lower()
    
    # Split query into words and remove common words
    query_words = set(re.findall(r'\w+', query_lower))
    common_words = {'what', 'does', 'do', 'is', 'are', 'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'who'}
    query_words = query_words - common_words
    
    if not query_words:
        return base_score
    
    # Count matching words
    matches = sum(1 for word in query_words if word in chunk_lower)
    
    # Calculate word match ratio
    word_match_ratio = matches / len(query_words)
    
    # Check for exact phrase matches (more weight)
    exact_phrase_match = query_lower in chunk_lower
    phrase_bonus = 0.3 if exact_phrase_match else 0
    
    # Check for consecutive word matches (good for names)
    query_words_list = [w for w in re.findall(r'\w+', query_lower) if w not in common_words]
    if len(query_words_list) > 1:
        consecutive_matches = 0
        for i in range(len(query_words_list) - 1):
            if f"{query_words_list[i]} {query_words_list[i+1]}" in chunk_lower:
                consecutive_matches += 1
        consecutive_bonus = 0.2 * (consecutive_matches / (len(query_words_list) - 1)) if len(query_words_list) > 1 else 0
    else:
        consecutive_bonus = 0
    
    # Check for semantic similarity bonus
    semantic_bonus = 0.1 if base_score > 0.5 else 0
    
    # Combine scores with weights
    final_score = (0.3 * base_score) + (0.3 * word_match_ratio) + phrase_bonus + consecutive_bonus + semantic_bonus
    
    # Debug output
    # print(f"Debug - Chunk: {chunk[:100]}...")
    # print(f"Debug - Score components: base={base_score:.3f}, word_match={word_match_ratio:.3f}, phrase_bonus={phrase_bonus}, consecutive_bonus={consecutive_bonus}, semantic_bonus={semantic_bonus}")
    # print(f"Debug - Final score: {final_score:.3f}")
    
    return final_score

def generate_and_store_embeddings(text: str) -> Optional[str]:
    file_id = str(uuid.uuid4())
    # print(f"Debug - Generating embeddings for text (first 100 chars): {text[:100]}...")
    
    chunks = split_into_chunks(text)
    # print(f"Debug - Split into {len(chunks)} chunks")
    
    if not chunks:
        print("No valid chunks created from input text")
        return None
    
    # Generate embeddings for all chunks
    embeddings = []
    chunk_metadatas: List[Dict[str, Any]] = []
    for i, chunk in enumerate(tqdm(chunks, desc="Generating embeddings")):
        embedding = create_embedding(chunk)
        if embedding is not None:
            embeddings.append(embedding)
            chunk_metadatas.append(extract_chunk_metadata(chunk, i))
            # print(f"Debug - Generated embedding {i+1}/{len(chunks)}")
        else:
            print(f"Debug - Failed to generate embedding for chunk {i+1}")
    
    if not embeddings:
        print("No embeddings generated for the text")
        return None
    
    try:
        # Convert embeddings to numpy array
        embeddings_np = np.array(embeddings, dtype=np.float32)
        # print(f"Debug - Created numpy array of shape {embeddings_np.shape}")
        
        # Save embeddings and chunks
        embeddings_file = EMBEDDINGS_FOLDER / f"{file_id}_embeddings.npy"
        np.save(embeddings_file, embeddings_np)
        # print(f"Debug - Saved embeddings to {embeddings_file}")
        
        # Save chunks and metadata
        chunks_file = EMBEDDINGS_FOLDER / f"{file_id}_chunks.json"
        document_metadata = extract_document_metadata(text, chunks)
        with open(chunks_file, 'w', encoding='utf-8') as f:
            json.dump({
                'chunks': chunks,
                'metadata': chunk_metadatas,
                'document_metadata': document_metadata,
                'file_id': file_id
            }, f, ensure_ascii=False)
        # print(f"Debug - Saved chunks to {chunks_file}")
        
        return file_id
    except Exception as e:
        print(f"Error storing embeddings: {e}")
        return None

def load_embeddings_and_chunks(file_id: str) -> tuple[Optional[np.ndarray], Optional[List[str]], List[Dict[str, Any]]]:
    try:
        # Load embeddings
        embeddings_file = EMBEDDINGS_FOLDER / f"{file_id}_embeddings.npy"
        embeddings = np.load(embeddings_file)
        
        # Load chunks
        chunks_file = EMBEDDINGS_FOLDER / f"{file_id}_chunks.json"
        with open(chunks_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            chunks = data['chunks']
            metadata = data.get('metadata', [])
        
        # print(f"Debug - Loaded {len(chunks)} chunks for file_id: {file_id}")
        return embeddings, chunks, metadata
    except Exception as e:
        print(f"Error loading embeddings and chunks for {file_id}: {e}")
        return None, None, []


def load_document_metadata(file_id: str) -> Dict[str, Any]:
    try:
        chunks_file = EMBEDDINGS_FOLDER / f"{file_id}_chunks.json"
        with open(chunks_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get('document_metadata', {})
    except Exception as e:
        print(f"Error loading document metadata for {file_id}: {e}")
        return {}

def _clean_candidate(result: Dict[str, Any]) -> Dict[str, Any]:
    cleaned = result.copy()
    cleaned.pop("embedding_vector", None)
    cleaned.pop("base_similarity", None)
    return cleaned


def mmr_rerank(
    candidates: List[Dict[str, Any]],
    query_embedding: np.ndarray,
    top_k: int,
    lambda_mult: float = 0.65,
) -> List[Dict[str, Any]]:
    if not candidates:
        return []

    if len(candidates) <= top_k:
        return sorted([
            _clean_candidate(candidate)
            for candidate in candidates
        ], key=lambda x: x.get("score", 0), reverse=True)

    candidate_embeddings = np.array([c["embedding_vector"] for c in candidates])
    similarity_to_query = cosine_similarity([query_embedding], candidate_embeddings)[0]

    selected_indices: List[int] = []
    candidate_indices = list(range(len(candidates)))

    while candidate_indices and len(selected_indices) < top_k:
        if not selected_indices:
            best_idx = int(np.argmax(similarity_to_query))
            candidate_indices.remove(best_idx)
            selected_indices.append(best_idx)
            continue

        mmr_scores = []
        selected_embeddings = candidate_embeddings[selected_indices]
        for idx in candidate_indices:
            diversity = cosine_similarity(
                [candidate_embeddings[idx]],
                selected_embeddings
            )[0].max()
            mmr_score = lambda_mult * similarity_to_query[idx] - (1 - lambda_mult) * diversity
            mmr_scores.append((mmr_score, idx))

        mmr_scores.sort(key=lambda x: x[0], reverse=True)
        best_idx = mmr_scores[0][1]
        candidate_indices.remove(best_idx)
        selected_indices.append(best_idx)

    selected_results = [_clean_candidate(candidates[idx]) for idx in selected_indices]
    return sorted(selected_results, key=lambda x: x.get("score", 0), reverse=True)


def search_across_indices(
    query: str,
    file_ids: List[str],
    top_k: int = 5,
    fetch_multiplier: int = 3,
    lambda_mult: float = 0.65,
    max_candidates: int = 100,
) -> List[Dict[str, Any]]:
    try:
        # print(f"Debug - Searching for query: {query}")
        # print(f"Debug - Searching across {len(file_ids)} files")
        
        # Create query embedding
        query_embedding = create_embedding(query)
        if query_embedding is None:
            return []
        
        all_results = []
        per_file_limit = max(top_k * fetch_multiplier, top_k)
        
        # Search in each file's embeddings
        for file_id in file_ids:
            embeddings, chunks, metadata_list = load_embeddings_and_chunks(file_id)
            if embeddings is None or chunks is None:
                continue
            
            # Calculate cosine similarities
            similarities = cosine_similarity([query_embedding], embeddings)[0]
            
            # Get top k results
            top_indices = np.argsort(similarities)[-per_file_limit:][::-1]
            
            for idx in top_indices:
                base_score = float(similarities[idx])
                
                # Calculate more sophisticated relevance score
                relevance_score = calculate_relevance_score(chunks[idx], query, base_score)
                
                if relevance_score >= MIN_SCORE_THRESHOLD:
                    chunk_metadata = metadata_list[idx] if idx < len(metadata_list) else {}
                    all_results.append({
                        "file_id": file_id,
                        "chunk": chunks[idx],
                        "distance": 1 - base_score,  # Convert similarity to distance
                        "score": relevance_score,
                        "metadata": chunk_metadata,
                        "embedding_vector": embeddings[idx],
                        "base_similarity": base_score,
                    })
        
        # Sort all results by score and return top_k
        all_results.sort(key=lambda x: x["score"], reverse=True)
        if not all_results:
            return []

        capped_candidates = all_results[: min(len(all_results), max(top_k * fetch_multiplier, top_k, max_candidates))]
        reranked_results = mmr_rerank(capped_candidates, query_embedding, top_k, lambda_mult=lambda_mult)
        # print(f"Debug - Found {len(reranked_results)} results above threshold")
        return reranked_results
    except Exception as e:
        print(f"Error searching: {e}")
        return []

# # Example usage:
# if __name__ == "__main__":
#     # Process single text
#     text = "Your long document text goes here..."
#     file_id = generate_and_store_embeddings(text)
    
#     if file_id:
#         # Search across specific files (could include multiple file IDs)
#         query = "What is the main topic?"
#         results = search_across_indices(query, [file_id], top_k=3)
        
#         for result in results:
#             print(f"\nFile ID: {result['file_id']}")
#             print(f"Score: {result['score']:.3f}")
#             print(f"Content: {result['chunk'][:200]}...")