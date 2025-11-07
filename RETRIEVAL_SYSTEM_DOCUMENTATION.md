# Dual Retrieval System Documentation

## Overview

This system implements a dual retrieval approach for RAG (Retrieval-Augmented Generation):
- **Vector Search**: Semantic similarity-based retrieval using embeddings
- **Knowledge Graph (KAG)**: Structured relationship-based retrieval using triples
- **SQL Aggregation**: Direct SQL queries over normalised `KnowledgeFact` records for large, structured lists

Both methods process documents during upload and can be selected via admin settings. The system automatically processes all uploaded documents for both methods, allowing administrators to choose which retrieval method to use for answering questions.

---

## Table of Contents

1. [File Upload and Processing](#file-upload-and-processing)
2. [Vector Search Flow](#vector-search-flow)
3. [Knowledge Graph (KAG) Flow](#knowledge-graph-kag-flow)
4. [SQL Aggregation Flow](#sql-aggregation-flow)
5. [Query Processing](#query-processing)
6. [Answer Generation](#answer-generation)
7. [Comparison and Use Cases](#comparison-and-use-cases)

---

## File Upload and Processing

### Step 1: File Upload
**Location**: `routes.py` - `/api/file` endpoint (line ~1009)

When a file is uploaded:
1. Text is extracted from the file (PDF, DOCX, TXT, etc.)
2. A unique `file_identifier` (UUID) is generated
3. A `File` record is created in the database with:
   - `text`: Full text content
   - `file_identifier`: UUID for embedding files
   - `original_filename`: Display name
   - `user_id`: Uploader's ID

### Step 2: Dual Processing

**Both methods process the file simultaneously:**

#### A. Vector Search Processing
**Location**: `embed_and_search.py` - `generate_and_store_embeddings()`

1. **Text Chunking**:
   - Text is split into chunks (max 500 tokens, 100 token overlap)
   - Uses tiktoken for token counting
   - Preserves paragraph boundaries when possible

2. **Embedding Generation**:
   - Uses SentenceTransformer model: `all-MiniLM-L6-v2`
   - Generates 384-dimensional embeddings for each chunk
   - Embeddings are stored as numpy arrays

3. **Storage**:
   - Embeddings saved to: `dataembedding/{file_id}_embeddings.npy`
   - Chunks saved to: `dataembedding/{file_id}_chunks.json`
   - Returns `file_identifier` (UUID)

#### B. Knowledge Graph Processing
**Location**: `knowledge_graph.py` - `process_file_to_graph()`

1. **Triple Extraction**:
   - Uses OpenAI GPT-4 to extract knowledge graph triples
   - Prompt: "Extract knowledge graph triples from the following text"
   - Format: `SUBJECT | RELATION | OBJECT`
   - Limits to most important triples (max 10 per extraction)
   - Processes first 4000 characters of text

2. **Triple Parsing**:
   - Parses GPT-4 response line by line
   - Extracts subject, relation, and object from each line
   - Validates that all three components exist

3. **Storage**:
   - Triples stored in `KnowledgeGraph` database table
   - Each triple contains:
     - `subject`: Entity or concept
     - `relation`: Relationship type
     - `object`: Related entity or value
     - `source_id`: Foreign key to `File.id`
   - Duplicate triples are avoided (same subject, relation, object, source_id)
4. **Structured Fact Normalisation**:
   - Each triple is also converted into a `KnowledgeFact` record
   - Relations are normalised to canonical verbs (e.g., `located_in`, `has_contact`)
   - Stores optional `entity_type`, the original relation text, and a confidence score
   - Facts can be queried directly without loading full document chunks

**Code Reference**:
```python
# routes.py line ~1064-1076
# Extract and store knowledge graph triples
try:
    openai_key = Settings.query.filter_by(key="openai_key").first()
    if openai_key and openai_key.value:
        client = OpenAI(api_key=openai_key.value, timeout=30.0, max_retries=2)
        process_file_to_graph(db_file.id, text, client)
except Exception as kg_error:
    print(f"Error processing knowledge graph for file {db_file.id}: {kg_error}")
    # Don't fail the upload if KG extraction fails
```

**Rebuilding Existing Data**:
- Run `python manage_facts.py rebuild-facts` to regenerate `KnowledgeFact` entries from already extracted triples.
- Use after deploying the structured aggregation feature so legacy documents participate in structured queries.

---

## Query Processing

### Step 1: Query Preprocessing
**Location**: `query_preprocessing.py` - `preprocess_query()`

When a user asks a question:

1. **Normalization**:
   - Converts to lowercase
   - Trims whitespace
   - Removes extra spaces

2. **Keyword Extraction**:
   - Removes stop words (the, a, an, and, or, etc.)
   - Extracts meaningful words (minimum 3 characters)
   - Returns list of keywords

3. **Query Type Detection**:
   - Detects query type: `aggregate`, `comparison`, or `factual`
   - Aggregate patterns: "which", "how many", "all that", "list", etc.
   - Comparison patterns: "greater than", "above", "below", ">", "<", etc.

### Step 2: Query Intent Detection
**Location**: `query_intent.py` - `infer_query_intent()`

1. **Heuristic Pass**:
   - Matches keywords (companies, universities, people, etc.) to derive `entity_type`
   - Maps verbs to canonical relations (e.g., `located` → `located_in`)
   - Extracts simple filters such as location phrases ("in Finland")
2. **Optional LLM Fallback**:
   - When heuristic confidence is low, a lightweight GPT classification prompt returns a JSON intent payload
3. **Output**:
   - `entity_type`, `relations`, `attributes`, `filters`, `confidence`, and `source`
   - Logged for observability and consumed by downstream aggregation

### Step 3: Structured Fact Aggregation
**Location**: `routes.py` - `fetch_structured_facts()`

1. **Fact Filtering**:
   - Queries `KnowledgeFact` based on inferred relations and filters (e.g., `normalized_relation IN ('located_in')` and `object ILIKE '%Finland%'`)
   - Optionally constrains by `entity_type`
2. **Aggregation**:
   - Groups matching facts by subject, deduplicates `(relation, value)` pairs, and collects source files
   - Limits to 50 entities / 300 facts to stay well below model context limits
3. **Context Synthesis**:
   - Builds a compact text block (`Structured facts:`) passed to the LLM when available
   - Returns a `structured_results` payload so the UI can render tables or downloads
4. **Fallback**:
   - If no structured facts are found (or intent is ambiguous), the pipeline reverts to the traditional Vector / KAG retrieval flow

**Code Reference**:
```python
# routes.py line ~102-107
preprocessed = preprocess_query(question)
processed_question = preprocessed.get('normalized', question)
query_type = preprocessed.get('query_type', 'factual')
```

### Step 2: Retrieval Method Selection
**Location**: `routes.py` - `chat()` endpoint (line ~144-158)

The system checks the admin setting `retrieval_method`:
- **"Vector"** (default): Uses vector search
- **"KAG"**: Uses knowledge graph search

---

## Vector Search Flow

### Step 1: Query Embedding
**Location**: `embed_and_search.py` - `search_across_indices()`

1. Creates embedding for the user's query using SentenceTransformer
2. Same model used for document embeddings ensures compatibility

### Step 2: Similarity Search
**Location**: `embed_and_search.py` - `search_across_indices()`

For each file in the system:

1. **Load Embeddings**:
   - Loads `{file_id}_embeddings.npy` (numpy array)
   - Loads `{file_id}_chunks.json` (text chunks)

2. **Calculate Similarity**:
   - Computes cosine similarity between query embedding and all chunk embeddings
   - Uses `sklearn.metrics.pairwise.cosine_similarity`

3. **Relevance Scoring**:
   - Base score: Cosine similarity (0-1)
   - Enhanced with:
     - Word match ratio (30% weight)
     - Exact phrase match bonus (+0.3)
     - Consecutive word matches bonus (+0.2)
     - Semantic similarity bonus (+0.1)
   - Final score: Weighted combination
   - Minimum threshold: 0.3

4. **Ranking**:
   - Results sorted by relevance score (highest first)
   - Returns top K chunks (K = `chunk_number` setting, default 3)

**Code Reference**:
```python
# embed_and_search.py line ~315-340
def search_across_indices(query: str, file_ids: List[str], top_k: int = 5):
    query_embedding = create_embedding(query)
    for file_id in file_ids:
        embeddings, chunks = load_embeddings_and_chunks(file_id)
        similarities = cosine_similarity([query_embedding], embeddings)[0]
        # Calculate relevance scores and filter by threshold
```

### Step 3: Result Formatting
**Location**: `routes.py` - `chat()` endpoint (line ~257-261)

Returns list of dictionaries:
```python
{
    'file_id': 'uuid-string',
    'chunk': 'text content',
    'score': 0.85,  # Relevance score
    'distance': 0.15  # 1 - similarity
}
```

---

## Knowledge Graph (KAG) Flow

### Step 1: Triple Search
**Location**: `knowledge_graph.py` - `search_knowledge_graph()`

Uses a **multi-strategy approach**:

#### Strategy 1: Full Query Match (Highest Priority)
- Searches for triples where query appears in:
  - `subject` field
  - `relation` field
  - `object` field
- Uses SQL `ILIKE` for case-insensitive matching
- Returns top K results

#### Strategy 2: Keyword-Based Matching
- Extracts keywords from query (removes stop words)
- For each keyword, searches triples containing the keyword
- Combines results from multiple keywords
- Limits to top 5 keywords

#### Strategy 3: Related Entity Search (Graph Traversal)
- Finds entities mentioned in query
- Gets related entities (second hop in graph)
- Searches for triples involving related entities
- Enables finding indirectly related information

**Code Reference**:
```python
# knowledge_graph.py line ~148-274
def search_knowledge_graph(query: str, top_k: int = 5):
    # Strategy 1: Full query match
    full_query_results = KnowledgeGraph.query.filter(
        or_(
            KnowledgeGraph.subject.ilike(f'%{query}%'),
            KnowledgeGraph.relation.ilike(f'%{query}%'),
            KnowledgeGraph.object.ilike(f'%{query}%')
        )
    ).limit(top_k).all()
    
    # Strategy 2: Keyword matches
    # Strategy 3: Related entity search
```

### Step 2: Triple to Context Conversion
**Location**: `routes.py` - `chat()` endpoint (line ~196-250)

1. **Group by Source File**:
   - Groups triples by `source_id` (file ID)
   - Accumulates multiple triples from same file

2. **Build Context**:
   - Formats triple as: `"Knowledge: {subject} {relation} {object}"`
   - Combines multiple triples: `"Knowledge: ... Knowledge: ..."`
   - Appends first 500 characters of original file text
   - Creates chunk: `"{triples_text}. {file_text[:500]}"`

3. **Scoring**:
   - Base score: 0.8
   - Bonus: +0.1 per matching triple (more triples = higher relevance)
   - Sorts by score (highest first)

4. **Result Formatting**:
   - Returns same format as vector search for compatibility
   - Maps `source_id` to `file_identifier` for consistency

**Code Reference**:
```python
# routes.py line ~204-250
for triple in kg_results:
    source_id = triple.get('source_id')
    file_obj = File.query.filter_by(id=source_id).first()
    triple_info = f"Knowledge: {subject} {relation} {obj}"
    # Accumulate triples per file and build context
```

### Step 3: Fallback Mechanism
**Location**: `routes.py` - `chat()` endpoint (line ~252-256)

If KAG returns no results:
- Automatically falls back to vector search
- Ensures user always gets an answer if content exists

---

## SQL Aggregation Flow

**Location**: `routes.py` - `fetch_structured_facts()` and SQL retrieval branch

1. **Prerequisites**:
   - `KnowledgeFact` table must be populated (happens during ingestion or via `manage_facts.py rebuild-facts`)
   - Admin selects **SQL Aggregation** in Retrieval Method settings
2. **Intent Mapping**:
   - Relations inferred by `query_intent.py` are normalised through `normalize_relations()` which applies synonym mapping (e.g., “situated_in” → `located_in`)
3. **SQL Query**:
   - Filters `KnowledgeFact.normalized_relation` for the canonical relation list
   - Applies optional filters such as `location` by running `object ILIKE '%Finland%'`
   - Results limited to 50 entities / 300 facts to prevent oversized responses
4. **Aggregation & Sources**:
   - Groups facts by subject, deduplicates relation/value pairs, and records contributing files for source display
   - Produces both a compact text block and a structured JSON payload for the frontend
5. **Direct Answer**:
   - When SQL mode is active and structured results are found, the system skips the OpenAI call and returns the formatted list directly to the user (plus sources)
6. **Fallback**:
   - If no structured facts match (or intent confidence is low), SQL mode falls back to Vector retrieval (and subsequently KAG if needed)

---

## Answer Generation

### Step 1: Token Counting and Context Truncation
**Location**: `embed_and_search.py` - `truncate_context()`

**IMPORTANT**: The system **always counts tokens before sending to the model** and truncates context if it exceeds the model's token limit. This ensures we never exceed the model's context window.

#### Token Counting Process:

1. **Token Counting**:
   - Counts tokens for all components:
     - System prompt
     - User query
     - Conversation history
     - Retrieved context chunks
   - Uses `tiktoken` with model-specific encoding
   - Each model has its own tokenizer (e.g., `cl100k_base` for GPT-4)

2. **Context Limit Management**:
   
   The system uses a **priority-based approach** to determine context limits:
   
   **Priority Order:**
   1. **Manual Override** (Highest Priority): If admin has set a custom context limit in Settings, use that
   2. **Hardcoded Dictionary**: Check `MODEL_CONTEXT_WINDOWS` dictionary for known models
   3. **Safe Default**: Use gpt-4 limit (8,000 tokens) if model is unknown
   
   **Default Context Limits** (with reserved space for response):
   - `gpt-4`: 8,000 tokens (reserve 200 for response)
   - `gpt-4-turbo`: 120,000 tokens (reserve 2,000)
   - `gpt-4-turbo-preview`: 120,000 tokens
   - `gpt-4-0125-preview`: 120,000 tokens
   - `gpt-4-1106-preview`: 120,000 tokens
   - `gpt-3.5-turbo`: 15,000 tokens (reserve 1,000)
   - `gpt-3.5-turbo-16k`: 15,000 tokens
   - `o1-preview`: 200,000 tokens (reserve 2,000)
   - `o1-mini`: 200,000 tokens
   
   **Manual Configuration:**
   - Admin can set custom context limits in **Model Settings**
   - Stored as: `model_context_limit_{model_name}` in Settings table
   - If left empty, system uses safe defaults
   - Field auto-populates with default value when model is selected

3. **Truncation Strategy**:
   - **Sorts results by score** (highest first) to prioritize best matches
   - **Calculates available tokens**:
     ```
     available_tokens = context_limit - reserve_tokens - system_tokens - query_tokens - history_tokens
     ```
   - **Adds chunks incrementally** until token limit reached
   - **Keeps highest-scoring chunks** when truncating
   - **Excludes less similar context** if total exceeds limit
   - If fixed content (system + query + history) exceeds limit, returns empty results

**Code Reference**:
```python
# embed_and_search.py line ~219-263
def get_model_context_limit(model: str) -> int:
    # Priority 1: Check Settings database (manual override)
    # Priority 2: Check hardcoded MODEL_CONTEXT_WINDOWS
    # Priority 3: Use safe default (gpt-4: 8000 tokens)

# embed_and_search.py line ~266-320
def truncate_context(results, system_prompt, user_query, history_text, model, reserve_tokens):
    # Count tokens for all components
    # Sort results by score (highest first)
    # Add chunks until token limit reached
    # Keep highest-scoring chunks

# routes.py line ~286-294
truncated_results = truncate_context(
    results=results,
    system_prompt=system_prompt_text,
    user_query=question,
    history_text=history_text,
    model=model_name,
    reserve_tokens=600  # Reserve for response (max_tokens=600)
)
```

**Example Flow:**
1. User asks question → System retrieves 5 chunks (total: 10,000 tokens)
2. Model: `gpt-4` → Context limit: 8,000 tokens
3. System counts:
   - System prompt: 50 tokens
   - Query: 20 tokens
   - History: 100 tokens
   - Available for context: 8,000 - 600 - 50 - 20 - 100 = **7,230 tokens**
4. System sorts chunks by score (highest first)
5. Adds chunks until 7,230 tokens reached
6. Excludes lower-scoring chunks that exceed limit
7. Sends truncated context to OpenAI

### Step 2: Context Building
**Location**: `routes.py` - `chat()` endpoint (line ~296-311)

1. **Structured Facts First**:
   - If `fetch_structured_facts()` returns matches, prepend a compact block:
     ```
     Structured facts:
     {subject}:
       - located_in: Finland
     ```
   - Sources are summarised as `Structured facts (N entities)` with a list of contributing files
2. **Append RAG Chunks (Fallback)**:
   - Vector / KAG results are still truncated via `truncate_context()` when no structured facts are available or when additional context is useful
   - Each chunk formatted as:
     ```
     Relevant content from {filename}:
     {chunk_text}
     ```
   - Multiple results joined with double newlines
3. **Source Attribution**:
   - Creates `used_files` list with:
     - `name`: File name or summary label
     - `chunk`: Text chunk used (structured summary or raw chunk)
   - Used for displaying sources to user

### Step 3: OpenAI API Call
**Location**: `routes.py` - `chat()` endpoint (line ~321-343)

1. **Prompt Construction**:
   ```
   Conversation so far: {history_text}
   User's latest question: {question}
   Retrieved context: {context}
   ```
2. **System Prompt**:
   - Ensures natural, paragraph-style answers
   - Explicitly discourages numbered lists
3. **Response Metadata**:
   - Stores `sources_used`, `used_file_names`, and now `structured_results` (list of `{subject, facts, sources}`) for the frontend/UI

---

## Comparison and Use Cases

### Vector Search

**Best For:**
- Semantic similarity queries
- Finding conceptually similar content
- General Q&A
- Questions about topics, themes, or concepts
- Example: "What is machine learning?", "Explain neural networks"

**How It Works:**
- Converts text to numerical vectors (embeddings)
- Measures semantic similarity using cosine distance
- Finds chunks with similar meaning to query
- Works well for paraphrased questions

**Strengths:**
- Understands semantic meaning
- Handles synonyms and related concepts
- Good for exploratory questions
- Fast similarity computation

**Limitations:**
- May miss specific facts or relationships
- Struggles with aggregate queries
- Can't traverse relationships between entities

### Knowledge Graph (KAG)

**Best For:**
- Aggregate queries ("which companies have income > X")
- Structured queries with relationships
- Questions about specific entities and their properties
- Comparison queries ("companies with revenue above threshold")
- Example: "Which companies have yearly income above 3000 USD?"

**How It Works:**
- Extracts structured relationships (subject-relation-object)
- Stores as triples in database
- Searches using pattern matching and graph traversal
- Can find related entities through relationships

**Strengths:**
- Excellent for aggregate and comparison queries
- Can traverse relationships (find related entities)
- Good for structured data queries
- Handles numeric comparisons well

**Limitations:**
- Depends on quality of triple extraction
- May miss information not captured as triples
- Requires OpenAI API for extraction (cost)
- Less effective for conceptual/semantic queries

### SQL Aggregation

**Best For:**
- Large, structured lists of data (e.g., "List all companies in Finland")
- Direct SQL queries over normalised facts
- Efficient for pre-defined, static queries
- No need for LLM processing overhead

**How It Works:**
- Pre-processes query to extract filters and relations
- Maps relations to canonical forms
- Constructs a SQL query based on the normalised facts
- Executes the query against the `KnowledgeFact` table
- Aggregates results and formats them for display

**Strengths:**
- Extremely fast (no LLM overhead)
- Can handle complex, multi-step queries
- Scales well with large datasets
- No dependency on LLM API costs

**Limitations:**
- Limited to pre-defined query patterns
- May miss complex, context-aware relationships
- Requires careful normalisation of relations

### When to Use Each Method

| Query Type | Recommended Method | Reason |
|------------|-------------------|--------|
| "What is X?" | Vector Search | Semantic understanding |
| "Explain Y" | Vector Search | Conceptual similarity |
| "Which companies have X > Y?" | Knowledge Graph | Aggregate query |
| "List all entities with property Z" | Knowledge Graph | Structured query |
| "How does X relate to Y?" | Knowledge Graph | Relationship traversal |
| General Q&A | Vector Search | Better semantic matching |
| "List all companies in Finland" | SQL Aggregation | Direct SQL execution |

---

## System Architecture

### Data Flow Diagram

```
File Upload
    │
    ├─→ Extract Text
    │
    ├─→ Vector Processing ──→ Embeddings ──→ Filesystem
    │   └─→ Chunks ──→ Filesystem
    │
    └─→ KAG Processing ──→ GPT-4 ──→ Triples ──→ Database
        └─→ KnowledgeGraph Table

User Question
    │
    ├─→ Preprocess Query
    │
    ├─→ Check Retrieval Method Setting
    │
    ├─→ Vector Search ──→ Similarity ──→ Top K Chunks
    │   OR
    └─→ KAG Search ──→ Triple Matching ──→ Top K Triples ──→ Convert to Chunks
        │
        └─→ (Fallback to Vector if no results)
    
    └─→ Truncate Context (Token Management)
    └─→ Build Context String
    └─→ Send to OpenAI
    └─→ Return Answer + Sources
```

### Database Schema

**File Table:**
- `id`: Primary key
- `text`: Full text content
- `file_identifier`: UUID for embedding files
- `original_filename`: Display name
- `user_id`: Foreign key to User

**KnowledgeGraph Table:**
- `id`: Primary key
- `subject`: Entity/concept
- `relation`: Relationship type
- `object`: Related entity/value
- `source_id`: Foreign key to File.id
- `created_at`: Timestamp

**Settings Table:**
- `key`: Setting name (e.g., "retrieval_method")
- `value`: Setting value ("Vector" or "KAG")

### File Storage

**Vector Embeddings:**
- Location: `dataembedding/`
- Files:
  - `{uuid}_embeddings.npy`: NumPy array of embeddings
  - `{uuid}_chunks.json`: JSON with chunk texts

**Knowledge Graph:**
- Stored in database (SQLAlchemy)
- No separate file storage needed

---

## Configuration

### Admin Settings

**Retrieval Method Selection:**
- Location: Admin Settings → RAG Settings
- Options:
  - **Vector Search**: Default, best for semantic queries
  - **Knowledge Graph**: Best for aggregate/structured queries
- Setting key: `retrieval_method`
- Default: "Vector"

**Other RAG Settings:**
- `chunk_number`: Number of chunks to retrieve (1-5, default: 3)
- `message_history`: Conversation history pairs (1-15, default: 10)
- `show_matched_text`: Show source text in response (yes/no)

**Model Settings:**
- `openai_key`: OpenAI API key (required)
- `openai_model`: Selected OpenAI model (required)
- `model_context_limit_{model_name}`: Custom context limit for specific model (optional)
  - If not set, system uses safe defaults
  - Auto-populates with default value when model is selected
  - Used to truncate context before sending to model
  - Ensures highest-scoring chunks are kept when truncating

---

## Error Handling

### Vector Search Errors
- If embeddings fail to load: Skips that file, continues with others
- If no results found: Returns "no content" message
- If similarity calculation fails: Returns empty results

### Knowledge Graph Errors
- If triple extraction fails: Upload succeeds, but no KG data stored
- If no triples found: Falls back to vector search
- If database query fails: Returns empty results, logs error

### Token Management
- **Always counts tokens before sending to model** - Never exceeds context window
- If context exceeds limit: Truncates, keeping highest-scoring chunks
- If truncation needed: Logs warning with count of chunks kept
- If fixed content (system + query + history) exceeds limit: Returns empty context
- **Context limit priority**:
  1. Manual override from Settings (if set)
  2. Hardcoded defaults for known models
  3. Safe default (gpt-4: 8,000 tokens) for unknown models

---

## Performance Considerations

### Vector Search
- **Embedding Generation**: ~100ms per chunk (local model)
- **Similarity Calculation**: ~10ms per file
- **Storage**: ~1.5KB per chunk (384-dim float32)

### Knowledge Graph
- **Triple Extraction**: ~2-5 seconds per file (GPT-4 API call)
- **Storage**: ~200 bytes per triple
- **Search**: ~50-200ms depending on graph size

### SQL Aggregation
- **Query Execution**: Extremely fast (no LLM overhead)
- **Storage**: No additional storage needed for facts
- **Scalability**: Excellent for large datasets

### Recommendations
- **Vector Search**: Better for real-time queries, lower cost
- **Knowledge Graph**: Better for complex queries, but slower extraction and higher cost
- **SQL Aggregation**: Best for pre-defined, static queries, no LLM costs

---

## Future Enhancements

Potential improvements:
1. Hybrid retrieval (combine both methods)
2. Automatic method selection based on query type
3. Caching of embeddings and triples
4. Batch processing for large document sets
5. Incremental updates (reprocess only changed documents)

---

## Code References

### Key Files
- `routes.py`: Main API endpoints, chat logic
- `embed_and_search.py`: Vector search implementation
- `knowledge_graph.py`: Knowledge graph implementation
- `query_preprocessing.py`: Query normalization and analysis
- `models.py`: Database models

### Key Functions
- `generate_and_store_embeddings()`: Vector processing
- `process_file_to_graph()`: KAG processing
- `search_across_indices()`: Vector search
- `search_knowledge_graph()`: KAG search
- `truncate_context()`: Token management
- `preprocess_query()`: Query preprocessing

