"""
Query preprocessing module for normalizing and enhancing user queries
"""
import re
from typing import Dict, List, Tuple


def normalize_query(query: str) -> str:
    """
    Normalize a query by:
    - Converting to lowercase
    - Trimming whitespace
    - Removing extra spaces
    - Removing leading/trailing punctuation
    
    Args:
        query: Raw user query
        
    Returns:
        Normalized query string
    """
    if not query:
        return ""
    
    # Convert to lowercase
    normalized = query.lower()
    
    # Remove extra whitespace
    normalized = re.sub(r'\s+', ' ', normalized)
    
    # Trim whitespace
    normalized = normalized.strip()
    
    return normalized


def extract_keywords(query: str) -> List[str]:
    """
    Extract meaningful keywords from a query by removing stop words.
    
    Args:
        query: User query string
        
    Returns:
        List of meaningful keywords
    """
    # Common stop words
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'tell', 'me', 'about', 'what', 'who',
        'where', 'when', 'why', 'how', 'is', 'are', 'was', 'were', 'be', 'been',
        'have', 'has', 'had', 'do', 'does', 'did', 'can', 'could', 'should',
        'would', 'will', 'that', 'this', 'these', 'those', 'some', 'any', 'all',
        'which', 'whose', 'whom', 'there', 'their', 'they', 'them', 'then',
        'than', 'more', 'most', 'much', 'many', 'very', 'too', 'so', 'such'
    }
    
    # Extract words (minimum 3 characters)
    words = re.findall(r'\b[a-zA-Z]{3,}\b', query.lower())
    
    # Filter out stop words
    keywords = [w for w in words if w not in stop_words]
    
    return keywords


def detect_query_type(query: str) -> str:
    """
    Detect the type of query to help with retrieval strategy.
    
    Args:
        query: User query string
        
    Returns:
        Query type: 'aggregate', 'comparison', 'factual', or 'unknown'
    """
    query_lower = query.lower()
    
    # Aggregate queries (which, how many, list, all, etc.)
    aggregate_patterns = [
        r'\bwhich\b',
        r'\bhow many\b',
        r'\ball\b.*\bthat\b',
        r'\blist\b',
        r'\bshow\b.*\ball\b',
        r'\bfind\b.*\ball\b',
        r'\bcompanies\b.*\bhave\b',
        r'\bentities\b.*\bwith\b',
    ]
    
    # Comparison queries (greater than, less than, above, below, etc.)
    comparison_patterns = [
        r'\b(greater|more|above|over|higher|exceeds?)\b.*\b(than|of)?',
        r'\b(less|below|under|lower|fewer)\b.*\b(than|of)?',
        r'\b(equal|same|exactly)\b',
        r'\b(between|range)\b',
        r'[><=]',  # Direct comparison operators
    ]
    
    # Check for aggregate patterns
    for pattern in aggregate_patterns:
        if re.search(pattern, query_lower):
            return 'aggregate'
    
    # Check for comparison patterns
    for pattern in comparison_patterns:
        if re.search(pattern, query_lower):
            return 'comparison'
    
    # Default to factual
    return 'factual'


def preprocess_query(query: str) -> Dict:
    """
    Main preprocessing function that normalizes and analyzes a query.
    
    Args:
        query: Raw user query
        
    Returns:
        Dictionary with:
            - 'original': Original query
            - 'normalized': Normalized query
            - 'keywords': List of keywords
            - 'query_type': Type of query (aggregate, comparison, factual)
    """
    if not query:
        return {
            'original': '',
            'normalized': '',
            'keywords': [],
            'query_type': 'unknown'
        }
    
    normalized = normalize_query(query)
    keywords = extract_keywords(normalized)
    query_type = detect_query_type(normalized)
    
    return {
        'original': query,
        'normalized': normalized,
        'keywords': keywords,
        'query_type': query_type
    }

