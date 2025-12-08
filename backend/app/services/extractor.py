"""
Entity and relationship extractor from documents.
"""
import os
import json
import logging
from typing import List, Dict, Tuple, Optional

try:
    import spacy
    from spacy import displacy
except ImportError:
    spacy = None

from openai import OpenAI

logger = logging.getLogger(__name__)

# Global spaCy model (loaded lazily)
_nlp_model = None


def _load_spacy_model(model_name: str = "en_core_web_sm"):
    """Load spaCy model, downloading if necessary."""
    global _nlp_model
    if _nlp_model is None:
        try:
            logger.info(f"Loading spaCy model: {model_name}")
            _nlp_model = spacy.load(model_name)
            logger.info(f"Successfully loaded spaCy model: {model_name}")
        except OSError:
            logger.error(
                f"spaCy model '{model_name}' not found. "
                f"Please install it with: python -m spacy download {model_name}"
            )
            raise
    return _nlp_model


def extract_entities_sections(text: str) -> List[Dict]:
    """
    Extract named entities from text using spaCy.
    
    Args:
        text: Input text to extract entities from
        
    Returns:
        List of entity dictionaries with keys:
        - text: Entity text
        - label: Entity type (PERSON, ORG, GPE, etc.)
        - start: Start character position
        - end: End character position
        
    Raises:
        ImportError: If spaCy is not installed
        OSError: If spaCy model is not downloaded
    """
    if spacy is None:
        raise ImportError(
            "spaCy is not installed. Install it with: pip install spacy "
            "and download a model with: python -m spacy download en_core_web_sm"
        )
    
    nlp = _load_spacy_model()
    
    if not text or not text.strip():
        logger.warning("Empty text provided for entity extraction")
        return []
    
    logger.debug(f"Extracting entities from text ({len(text)} characters)")
    
    doc = nlp(text)
    entities = []
    
    for ent in doc.ents:
        entity_dict = {
            "text": ent.text.strip(),
            "label": ent.label_,
            "start": ent.start_char,
            "end": ent.end_char,
            "description": spacy.explain(ent.label_) or ent.label_
        }
        entities.append(entity_dict)
        logger.debug(f"Extracted entity: {entity_dict['text']} ({entity_dict['label']})")
    
    logger.info(f"Extracted {len(entities)} entities from text")
    return entities


def extract_relations_with_llm(text: str) -> List[Dict]:
    """
    Extract relationship triples from text using OpenAI.
    
    Args:
        text: Input text to extract relationships from
        
    Returns:
        List of relationship dictionaries with keys:
        - subject: Subject entity
        - relation: Relationship type
        - object: Object entity
        - confidence: Optional confidence score
        
    Raises:
        ValueError: If OpenAI API key is not set
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    if not text or not text.strip():
        logger.warning("Empty text provided for relation extraction")
        return []
    
    client = OpenAI(api_key=api_key)
    
    logger.debug(f"Extracting relations from text ({len(text)} characters)")
    
    # Truncate text if too long (OpenAI has token limits)
    max_chars = 8000  # Rough estimate for ~2000 tokens
    text_truncated = text[:max_chars] if len(text) > max_chars else text
    
    prompt = f"""Extract relationship triples from the following text. 
Return a JSON array of triples in the format:
[
  {{"subject": "Entity1", "relation": "relationship_type", "object": "Entity2"}},
  ...
]

Focus on extracting:
- Person-to-Person relationships (works_with, married_to, parent_of, etc.)
- Person-to-Organization relationships (works_at, founded, member_of, etc.)
- Person-to-Location relationships (lives_in, born_in, located_in, etc.)
- Organization-to-Organization relationships (owns, partners_with, competes_with, etc.)
- Organization-to-Location relationships (located_in, operates_in, etc.)
- Concept relationships (related_to, part_of, instance_of, etc.)

Text:
{text_truncated}

Return a JSON object with a "triples" key containing the array of triples:"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Cost-effective model
            messages=[
                {
                    "role": "system",
                    "content": "You are a relationship extraction expert. Extract relationship triples from text and return them as JSON."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,  # Low temperature for more consistent extraction
            response_format={"type": "json_object"}  # Response must be a JSON object
        )
        
        # Parse response
        content = response.choices[0].message.content
        logger.debug(f"LLM response: {content}")
        
        # Try to parse as JSON
        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                # Look for common keys that might contain the array
                for key in ["triples", "relationships", "relations", "data"]:
                    if key in parsed and isinstance(parsed[key], list):
                        relations = parsed[key]
                        break
                else:
                    # If no key found, try to find first list value
                    relations = next((v for v in parsed.values() if isinstance(v, list)), [])
            elif isinstance(parsed, list):
                relations = parsed
            else:
                relations = []
        except json.JSONDecodeError:
            # Fallback: try to extract JSON array from text
            logger.warning("Failed to parse JSON, attempting text extraction")
            start_idx = content.find('[')
            end_idx = content.rfind(']') + 1
            if start_idx >= 0 and end_idx > start_idx:
                json_str = content[start_idx:end_idx]
                relations = json.loads(json_str)
            else:
                relations = []
                logger.error(f"Could not extract JSON from response: {content}")
        
        # Validate and clean relations
        valid_relations = []
        for rel in relations:
            if isinstance(rel, dict) and "subject" in rel and "relation" in rel and "object" in rel:
                valid_rel = {
                    "subject": str(rel["subject"]).strip(),
                    "relation": str(rel["relation"]).strip(),
                    "object": str(rel["object"]).strip()
                }
                if valid_rel["subject"] and valid_rel["relation"] and valid_rel["object"]:
                    valid_relations.append(valid_rel)
        
        logger.info(f"Extracted {len(valid_relations)} valid relations from text")
        return valid_relations
        
    except Exception as e:
        logger.error(f"Error extracting relations with LLM: {e}", exc_info=True)
        # Return empty list on error rather than failing completely
        return []


def extract_entities_relationships(text: str) -> Tuple[List[Dict], List[Dict]]:
    """
    Extract both entities and relationships from text.
    
    Args:
        text: Input text
        
    Returns:
        Tuple of (entities, relationships)
    """
    entities = extract_entities_sections(text)
    relationships = extract_relations_with_llm(text)
    return entities, relationships


def extract_structured_information(documents: List[str]) -> Dict:
    """
    Extract structured information from multiple documents.
    
    Args:
        documents: List of document texts
        
    Returns:
        Dictionary with aggregated entities and relationships
    """
    all_entities = []
    all_relationships = []
    
    for i, doc_text in enumerate(documents):
        logger.info(f"Processing document {i+1}/{len(documents)}")
        entities, relationships = extract_entities_relationships(doc_text)
        all_entities.extend(entities)
        all_relationships.extend(relationships)
    
    # Deduplicate entities by text and label
    seen_entities = set()
    unique_entities = []
    for entity in all_entities:
        key = (entity["text"].lower(), entity["label"])
        if key not in seen_entities:
            seen_entities.add(key)
            unique_entities.append(entity)
    
    # Deduplicate relationships
    seen_relations = set()
    unique_relationships = []
    for rel in all_relationships:
        key = (rel["subject"].lower(), rel["relation"].lower(), rel["object"].lower())
        if key not in seen_relations:
            seen_relations.add(key)
            unique_relationships.append(rel)
    
    return {
        "entities": unique_entities,
        "relationships": unique_relationships,
        "total_entities": len(unique_entities),
        "total_relationships": len(unique_relationships)
    }
