import os
# Set OpenMP environment variable to handle multiple runtime libraries
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import openai
import numpy as np
import uuid
from flask import current_app
from pathlib import Path
from typing import List, Dict, Optional
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
MAX_TOKENS = 500  # Smaller chunks for more precise matching
OVERLAP = 100
MIN_SCORE_THRESHOLD = 0.3  # Base threshold for all content types

# Model context window sizes (approximate, leaving room for response)
MODEL_CONTEXT_WINDOWS = {
    "gpt-4": 8000,  # 8192 tokens, reserve ~200 for response
    "gpt-4-turbo": 120000,  # 128k tokens, reserve ~2000 for response
    "gpt-4-turbo-preview": 120000,
    "gpt-4-0125-preview": 120000,
    "gpt-4-1106-preview": 120000,
    "gpt-3.5-turbo": 15000,  # 16k tokens, reserve ~1000 for response
    "gpt-3.5-turbo-16k": 15000,
    "o1-preview": 200000,  # 200k tokens, reserve ~2000 for response
    "o1-mini": 200000,
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
    for i, chunk in enumerate(tqdm(chunks, desc="Generating embeddings")):
        embedding = create_embedding(chunk)
        if embedding is not None:
            embeddings.append(embedding)
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
        with open(chunks_file, 'w', encoding='utf-8') as f:
            json.dump({
                'chunks': chunks,
                'file_id': file_id
            }, f, ensure_ascii=False)
        # print(f"Debug - Saved chunks to {chunks_file}")
        
        return file_id
    except Exception as e:
        print(f"Error storing embeddings: {e}")
        return None

def load_embeddings_and_chunks(file_id: str) -> tuple[Optional[np.ndarray], Optional[List[str]]]:
    try:
        # Load embeddings
        embeddings_file = EMBEDDINGS_FOLDER / f"{file_id}_embeddings.npy"
        embeddings = np.load(embeddings_file)
        
        # Load chunks
        chunks_file = EMBEDDINGS_FOLDER / f"{file_id}_chunks.json"
        with open(chunks_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            chunks = data['chunks']
        
        # print(f"Debug - Loaded {len(chunks)} chunks for file_id: {file_id}")
        return embeddings, chunks
    except Exception as e:
        print(f"Error loading embeddings and chunks for {file_id}: {e}")
        return None, None

def count_tokens(text: str, model: str = "gpt-4") -> int:
    """
    Count tokens in text using tiktoken for the specified model.
    
    Args:
        text: Text to count tokens for
        model: OpenAI model name (default: gpt-4)
        
    Returns:
        Number of tokens
    """
    try:
        # Get encoding for the model
        encoding = tiktoken.encoding_for_model(model)
        tokens = encoding.encode(text)
        return len(tokens)
    except Exception as e:
        # Fallback to default encoding if model not found
        print(f"Warning: Could not get encoding for model {model}, using default: {e}")
        tokens = tiktoken.get_encoding("cl100k_base").encode(text)
        return len(tokens)


def get_model_context_limit(model: str) -> int:
    """
    Get the context window limit for a model.
    
    Priority order:
    1. Check Settings database for manually configured limit
    2. Check hardcoded MODEL_CONTEXT_WINDOWS dictionary
    3. Use safe default (gpt-4 limit: 8000 tokens)
    
    Args:
        model: OpenAI model name
        
    Returns:
        Maximum tokens available for context (reserving space for response)
    """
    # Try to get from Settings database first (manual override)
    try:
        from flask import current_app
        with current_app.app_context():
            from models import Settings
            setting_key = f"model_context_limit_{model}"
            context_limit_setting = Settings.query.filter_by(key=setting_key).first()
            if context_limit_setting and context_limit_setting.value:
                try:
                    limit = int(context_limit_setting.value)
                    if limit <= 0:
                        raise ValueError("Context limit must be positive")

                    # Determine known limit for this model (exact match or prefix)
                    known_limit = None
                    if model in MODEL_CONTEXT_WINDOWS:
                        known_limit = MODEL_CONTEXT_WINDOWS[model]
                    else:
                        for key, value in MODEL_CONTEXT_WINDOWS.items():
                            if model.startswith(key):
                                known_limit = value
                                break

                    if known_limit is not None and limit > known_limit:
                        print(
                            f"Warning: Stored context limit {limit} for {model} exceeds known maximum {known_limit}. "
                            "Clamping to the safe limit."
                        )
                        return known_limit

                    print(f"Using stored context limit for {model}: {limit} tokens")
                    return limit
                except ValueError:
                    print(f"Warning: Invalid context limit value '{context_limit_setting.value}' for {model}, using defaults")
    except Exception as e:
        # If we can't access database, continue with defaults
        pass
    
    # Check for exact match in hardcoded dictionary
    if model in MODEL_CONTEXT_WINDOWS:
        return MODEL_CONTEXT_WINDOWS[model]
    
    # Check for partial matches (e.g., "gpt-4-turbo-2024-04-09")
    for key, value in MODEL_CONTEXT_WINDOWS.items():
        if model.startswith(key):
            return value
    
    # Safe default: gpt-4 limit (8000 tokens)
    print(f"Warning: Unknown model {model}, using safe default context limit (8000 tokens)")
    return MODEL_CONTEXT_WINDOWS["gpt-4"]


def truncate_context(results: List[Dict], system_prompt: str, user_query: str, 
                     history_text: str, model: str, reserve_tokens: int = 200) -> List[Dict]:
    """
    Truncate context results if they exceed the model's context window.
    Keeps highest-scoring chunks when truncating.
    
    Args:
        results: List of result dictionaries with 'chunk', 'score', 'file_id'
        system_prompt: System prompt text
        user_query: User query text
        history_text: Conversation history text
        model: OpenAI model name
        reserve_tokens: Tokens to reserve for response (default: 200)
        
    Returns:
        Truncated list of results that fit within context window
    """
    if not results:
        return results
    
    # Get context limit for model
    context_limit = get_model_context_limit(model)
    available_tokens = context_limit - reserve_tokens
    
    # Count tokens for fixed parts
    system_tokens = count_tokens(system_prompt, model)
    query_tokens = count_tokens(user_query, model)
    history_tokens = count_tokens(history_text, model) if history_text else 0
    
    # Calculate available tokens for context
    fixed_tokens = system_tokens + query_tokens + history_tokens
    context_tokens_available = available_tokens - fixed_tokens
    
    if context_tokens_available <= 0:
        print(f"Warning: Fixed content exceeds context limit, returning empty results")
        return []
    
    # Sort results by score (highest first) to prioritize best matches
    sorted_results = sorted(results, key=lambda x: x.get('score', 0), reverse=True)
    
    # Build context incrementally until we hit the limit
    selected_results = []
    total_context_tokens = 0
    
    for result in sorted_results:
        chunk = result.get('chunk', '')
        # Format as it will appear in context
        formatted_chunk = f"Relevant content from {result.get('file_id', 'unknown')}:\n{chunk}"
        chunk_tokens = count_tokens(formatted_chunk, model)
        
        if total_context_tokens + chunk_tokens <= context_tokens_available:
            selected_results.append(result)
            total_context_tokens += chunk_tokens
        else:
            # Try to fit a truncated version if there's some space left
            remaining_tokens = context_tokens_available - total_context_tokens
            if remaining_tokens > 100:  # Only if meaningful space remains
                # Truncate chunk to fit remaining space
                # Rough estimate: 1 token ≈ 4 characters
                max_chars = remaining_tokens * 4
                truncated_chunk = chunk[:max_chars] if len(chunk) > max_chars else chunk
                result_copy = result.copy()
                result_copy['chunk'] = truncated_chunk
                selected_results.append(result_copy)
            break
    
    if len(selected_results) < len(results):
        print(f"Context truncated: {len(selected_results)}/{len(results)} results fit within {context_tokens_available} tokens")
    
    return selected_results


def search_across_indices(query: str, file_ids: List[str], top_k: int = 5) -> List[Dict]:
    try:
        # print(f"Debug - Searching for query: {query}")
        # print(f"Debug - Searching across {len(file_ids)} files")
        
        # Create query embedding
        query_embedding = create_embedding(query)
        if query_embedding is None:
            return []
        
        all_results = []
        
        # Search in each file's embeddings
        for file_id in file_ids:
            embeddings, chunks = load_embeddings_and_chunks(file_id)
            if embeddings is None or chunks is None:
                continue
            
            # Calculate cosine similarities
            similarities = cosine_similarity([query_embedding], embeddings)[0]
            
            # Get top k results
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            for idx in top_indices:
                base_score = float(similarities[idx])
                
                # Calculate more sophisticated relevance score
                relevance_score = calculate_relevance_score(chunks[idx], query, base_score)
                
                if relevance_score >= MIN_SCORE_THRESHOLD:
                    all_results.append({
                        "file_id": file_id,
                        "chunk": chunks[idx],
                        "distance": 1 - base_score,  # Convert similarity to distance
                        "score": relevance_score
                    })
        
        # Sort all results by score and return top_k
        all_results.sort(key=lambda x: x["score"], reverse=True)
        # print(f"Debug - Found {len(all_results)} results above threshold")
        return all_results[:top_k]
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