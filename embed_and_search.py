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
    print(f"Debug - Chunk: {chunk[:100]}...")
    print(f"Debug - Score components: base={base_score:.3f}, word_match={word_match_ratio:.3f}, phrase_bonus={phrase_bonus}, consecutive_bonus={consecutive_bonus}, semantic_bonus={semantic_bonus}")
    print(f"Debug - Final score: {final_score:.3f}")
    
    return final_score

def generate_and_store_embeddings(text: str) -> Optional[str]:
    file_id = str(uuid.uuid4())
    print(f"Debug - Generating embeddings for text (first 100 chars): {text[:100]}...")
    
    chunks = split_into_chunks(text)
    print(f"Debug - Split into {len(chunks)} chunks")
    
    if not chunks:
        print("No valid chunks created from input text")
        return None
    
    # Generate embeddings for all chunks
    embeddings = []
    for i, chunk in enumerate(tqdm(chunks, desc="Generating embeddings")):
        embedding = create_embedding(chunk)
        if embedding is not None:
            embeddings.append(embedding)
            print(f"Debug - Generated embedding {i+1}/{len(chunks)}")
        else:
            print(f"Debug - Failed to generate embedding for chunk {i+1}")
    
    if not embeddings:
        print("No embeddings generated for the text")
        return None
    
    try:
        # Convert embeddings to numpy array
        embeddings_np = np.array(embeddings, dtype=np.float32)
        print(f"Debug - Created numpy array of shape {embeddings_np.shape}")
        
        # Save embeddings and chunks
        embeddings_file = EMBEDDINGS_FOLDER / f"{file_id}_embeddings.npy"
        np.save(embeddings_file, embeddings_np)
        print(f"Debug - Saved embeddings to {embeddings_file}")
        
        # Save chunks and metadata
        chunks_file = EMBEDDINGS_FOLDER / f"{file_id}_chunks.json"
        with open(chunks_file, 'w', encoding='utf-8') as f:
            json.dump({
                'chunks': chunks,
                'file_id': file_id
            }, f, ensure_ascii=False)
        print(f"Debug - Saved chunks to {chunks_file}")
        
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
        
        print(f"Debug - Loaded {len(chunks)} chunks for file_id: {file_id}")
        return embeddings, chunks
    except Exception as e:
        print(f"Error loading embeddings and chunks for {file_id}: {e}")
        return None, None

def search_across_indices(query: str, file_ids: List[str], top_k: int = 5) -> List[Dict]:
    try:
        print(f"Debug - Searching for query: {query}")
        print(f"Debug - Searching across {len(file_ids)} files")
        
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
        print(f"Debug - Found {len(all_results)} results above threshold")
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