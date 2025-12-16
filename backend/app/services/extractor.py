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

import ollama

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


def _correct_entity_type(entity_text: str, original_label: str) -> tuple:
    """
    Apply domain-specific rules to correct entity types.
    
    Args:
        entity_text: The entity name/text
        original_label: spaCy's predicted label
        
    Returns:
        Tuple of (corrected_label, corrected_description)
    """
    text_lower = entity_text.lower()
    text_upper = entity_text.upper()
    
    # Technical systems/frameworks/tools (should be PRODUCT or SYSTEM)
    technical_keywords = ['graf', 'sneps', 'api', 'ui', 'system', 'network', 'package', 'inference', 
                         'framework', 'algorithm', 'model', 'processor', 'engine']
    if any(keyword in text_lower for keyword in technical_keywords):
        return "PRODUCT", "Technical systems, frameworks, or software"
    
    # Acronyms in all caps (likely technical terms)
    if len(entity_text) > 1 and entity_text == text_upper and entity_text.replace(" ", "").isalpha():
        return "PRODUCT", "Technical acronym or system name"
    
    # Section references (like "1.1 Mind GRAF", "2.5 Mind GRAF Brief Discussion")
    if entity_text[0].isdigit() or (len(entity_text) > 3 and entity_text[:3].replace('.', '').isdigit()):
        return "CONCEPT", "Section or chapter reference"
    
    # Pure numbers or dates should stay as is
    if original_label in ["CARDINAL", "DATE", "TIME", "ORDINAL", "QUANTITY", "MONEY", "PERCENT"]:
        return original_label, spacy.explain(original_label)
    
    # If it contains "control", "acting", "reasoning" - likely a technical concept
    concept_keywords = ['control', 'acting', 'reasoning', 'discussion', 'implementation', 
                       'representation', 'architecture']
    if any(keyword in text_lower for keyword in concept_keywords):
        return "CONCEPT", "Technical concept or methodology"
    
    # Keep organizations and locations as-is (usually accurate)
    if original_label in ["ORG", "GPE", "LOC", "FAC"]:
        return original_label, spacy.explain(original_label)
    
    # Default: keep original
    return original_label, spacy.explain(original_label) or original_label


def extract_entities_with_llm(text: str) -> List[Dict]:
    """
    Extract named entities from ML / research text using Ollama LLM
    with strict JSON and atomic-entity guarantees.
    """

    if not text or not text.strip():
        logger.warning("Empty text provided for entity extraction")
        return []

    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
    model = os.getenv("OLLAMA_MODEL", "phi3:mini")  # recommended for M1

    logger.info(f"Using LLM-based entity extraction with {model}")

    # CRITICAL: Smaller chunks = less likely to overflow token limit
    # phi3:mini has limited context - keep it SHORT
    max_chars = 2500  # Reduced from 5000 to prevent JSON truncation
    text_truncated = text[:max_chars]

    prompt = f"""
You are a strict information extraction system.

TASK:
Extract named entities from the text below.

OUTPUT FORMAT:
- Output ONLY a valid JSON array of entity objects.
- The array MUST start with [ and end with ].
- Each entity object MUST have: "name", "type", "description"
- Use double quotes only (")
- Do NOT add extra text before or after the array
- If no entities are found, output []

ENTITY OBJECT SCHEMA (EXACT):
Each entity MUST be an object with these keys:
- "name": The exact text of the entity as it appears in the document
- "type": One of the allowed types (see below)
- "description": A brief description (optional, but recommended)

ENTITY RULES (MANDATORY):
- Each entity MUST be ATOMIC (one real-world object only).
- NEVER combine multiple people into one entity.
- If multiple names appear together, extract them as SEPARATE entities.
- Do NOT include commas in "name" unless part of the official name.
- Extract ONLY entities that appear verbatim in the text.
- Do NOT invent entities.
- Skip numbers, dates, and meaningless fragments.
- Skip entities shorter than 3 characters.
- Extract AT MOST 5 entities (KEEP IT SHORT!).

ALLOWED ENTITY TYPES (choose ONE):
PERSON
ORGANIZATION
DATASET
MODEL
METHOD
TASK
DOMAIN
CONCEPT

TEXT:
<<<
{text_truncated}
>>>
""".strip()

    try:
        client = ollama.Client(host=ollama_base_url)
        response = client.chat(
            model=model,
            format="json",
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": 0,
                "num_predict": 256  # Smaller output = more likely to complete
            }
        )

        response_text = response["message"]["content"].strip()
        logger.info(f"LLM response length: {len(response_text)} chars")

        # Parse JSON - if it fails, skip this chunk (no complex repair)
        try:
            entities_raw = json.loads(response_text)
        except json.JSONDecodeError as e:
            logger.warning(f"LLM returned invalid JSON (error: {e}). Skipping this chunk.")
            logger.debug(f"Invalid response was: {response_text[:300]}")
            return []  # SKIP CHUNK - don't try to repair broken JSON

        # Handle both formats: [{"name": ...}] or {"entities": [{"name": ...}]}
        if isinstance(entities_raw, dict):
            entities_raw = entities_raw.get("entities", [])
        
        entities = []
        for ent in entities_raw:
            if not isinstance(ent, dict):
                continue

            name = ent.get("name", "").strip()
            entity_type = ent.get("type", "").upper()
            description = ent.get("description", "").strip()
            
            # If no description, generate a simple one from the type
            if not description:
                description = f"A {entity_type.lower()} mentioned in the document"

            if not name or len(name) < 3:
                continue

            if name.isdigit():
                continue

            if entity_type not in {
                "PERSON",
                "ORGANIZATION",
                "DATASET",
                "MODEL",
                "METHOD",
                "TASK",
                "DOMAIN",
                "CONCEPT",
            }:
                continue

            entities.append({
                "text": name,
                "label": entity_type,
                "description": description,
                "start": 0,
                "end": 0,
            })

        logger.info(f"Extracted {len(entities)} entities using LLM")
        return entities

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM response as JSON: {e}")
        logger.error(f"Response was: {response_text[:500]}")
        return []

    except Exception as e:
        logger.error(f"LLM entity extraction failed: {e}")
        return []



def extract_entities_sections(text: str) -> List[Dict]:
    """
    Extract named entities from text using spaCy with domain-specific corrections.
    (Legacy function - prefer extract_entities_with_llm for better results)
    
    Args:
        text: Input text to extract entities from
        
    Returns:
        List of entity dictionaries with keys:
        - text: Entity text
        - label: Entity type (PERSON, ORG, GPE, PRODUCT, CONCEPT, etc.)
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
        entity_text = ent.text.strip()
        original_label = ent.label_
        
        # Apply domain-specific corrections
        corrected_label, corrected_description = _correct_entity_type(entity_text, original_label)
        
        entity_dict = {
            "text": entity_text,
            "label": corrected_label,
            "start": ent.start_char,
            "end": ent.end_char,
            "description": corrected_description
        }
        entities.append(entity_dict)
        
        # Log if we corrected the type
        if corrected_label != original_label:
            logger.debug(f"Corrected entity: {entity_text} ({original_label} → {corrected_label})")
        else:
            logger.debug(f"Extracted entity: {entity_text} ({corrected_label})")
    
    logger.info(f"Extracted {len(entities)} entities from text")
    return entities


def extract_relations_with_llm(text: str, known_entities: List[Dict] = None) -> List[Dict]:
    """
    Extract relationship triples from text using Ollama.
    Only creates relationships between entities that were already extracted.
    
    Args:
        text: Input text to extract relationships from
        known_entities: List of entity dicts with 'text' and 'label' keys
        
    Returns:
        List of relationship dictionaries with keys:
        - subject: Subject entity (must be in known_entities)
        - relation: Relationship type
        - object: Object entity (must be in known_entities)
    """
    if not text or not text.strip():
        logger.warning("Empty text provided for relation extraction")
        return []
    
    if not known_entities:
        logger.info("No entities provided - skipping relationship extraction")
        return []
    
    # Get configuration from environment
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
    model = os.getenv("OLLAMA_MODEL", "phi3:mini")

    logger.info(f"⚙️  Extracting relationships between {len(known_entities)} entities using {model}")
    
    # Create Ollama client
    client = ollama.Client(host=ollama_base_url)
    
    # CRITICAL: Smaller text chunks prevent token overflow and truncated JSON
    max_chars = 3000  # Reduced from 8000
    text_truncated = text[:max_chars] if len(text) > max_chars else text
    
    # Build entity list for prompt (limit to first 10 for shorter prompt)
    entity_names = [e["text"] for e in known_entities[:10]]
    entity_list = "\n".join([f"- {name}" for name in entity_names])
    
    prompt = f"""Extract relationships between entities from this research paper text.

YOU MUST FOLLOW THIS EXACT FORMAT:
[
  {{"subject": "CODA-LM", "relation": "evaluated_on", "object": "StreetHazards"}},
  {{"subject": "GPT-4V", "relation": "compared_with", "object": "Claude"}}
]

CRITICAL RULES:
1. Output MUST be a JSON array starting with [ and ending with ]
2. Each object MUST have exactly 3 keys: "subject", "relation", "object"
3. ONLY use entities from this list:
{entity_list}
4. Extract maximum 5 relationships (keep it short!)
5. If no relationships found, return []

ALLOWED RELATIONS:
uses, evaluated_on, compared_with, proposes, part_of, works_on, affiliated_with

TEXT:
{text_truncated}

Output the JSON array now:"""

    try:
        response = client.chat(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a research paper relationship extraction expert. Extract relationships ONLY between known entities. Output valid JSON only."
                },
                {"role": "user", "content": prompt}
            ],
            format="json",
            options={
                "temperature": 0,
                "num_predict": 512
            }
        )
        
        # Parse response
        content = response['message']['content'].strip()
        logger.info(f"LLM relation response (first 300 chars): {content[:300]}")
        
        # Try to parse as JSON
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            logger.warning("Could not parse relation JSON, skipping")
            return []
        
        # Handle multiple possible formats
        relations = []
        
        if isinstance(parsed, list):
            # Format 1: Direct array [{"subject": "X", "relation": "Y", "object": "Z"}]
            relations = parsed
            
        elif isinstance(parsed, dict):
            # Format 2: Wrapper object {"relationships": [...]}
            if "relationships" in parsed or "relations" in parsed or "triples" in parsed:
                relations = parsed.get("relationships", parsed.get("relations", parsed.get("triples", [])))
            
            # Format 3: Single relation object {"subject": "X", "relation": "Y", "object": "Z"}
            elif "subject" in parsed and "relation" in parsed and "object" in parsed:
                logger.info("Detected single relation object, wrapping in array...")
                relations = [parsed]
            
            # Format 4: Nested format {"Entity": {"relation": ["Target"]}}
            elif any(isinstance(v, dict) for v in parsed.values()):
                logger.info("Detected nested relation format, converting to triples...")
                for subject, rel_dict in parsed.items():
                    if not isinstance(rel_dict, dict):
                        continue
                    for relation, objects in rel_dict.items():
                        if isinstance(objects, list):
                            for obj in objects:
                                relations.append({
                                    "subject": subject,
                                    "relation": relation,
                                    "object": obj
                                })
                        elif isinstance(objects, str):
                            relations.append({
                                "subject": subject,
                                "relation": relation,
                                "object": objects
                            })
                logger.info(f"Converted nested format to {len(relations)} triples")
        
        if not relations:
            logger.info("No relations found in LLM response")
            return []
        
        # Validate relations - ensure both subject and object are in known entities
        entity_names_lower = {e["text"].lower() for e in known_entities}
        valid_relations = []
        filtered = {"not_dict": 0, "missing_fields": 0, "unknown_entity": 0}
        
        for rel in relations:
            if not isinstance(rel, dict):
                filtered["not_dict"] += 1
                continue
                
            subject = str(rel.get("subject", "")).strip()
            relation = str(rel.get("relation", "")).strip()
            obj = str(rel.get("object", "")).strip()
            
            if not (subject and relation and obj):
                filtered["missing_fields"] += 1
                continue
            
            # Check if both entities exist in known entities (case-insensitive)
            if subject.lower() not in entity_names_lower or obj.lower() not in entity_names_lower:
                filtered["unknown_entity"] += 1
                logger.debug(f"Filtered relation with unknown entity: '{subject}' -> '{obj}'")
                continue
            
            valid_relations.append({
                "subject": subject,
                "relation": relation,
                "object": obj
            })
        
        logger.info(f"✅ Extracted {len(valid_relations)} valid relations (filtered: {filtered})")
        return valid_relations
        
    except Exception as e:
        logger.error(f"Error extracting relations with LLM: {e}")
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
    
    Uses TWO-PHASE extraction:
    Phase 1: Extract ALL entities from ALL chunks (builds global entity registry)
    Phase 2: Extract relations using GLOBAL entity context (prevents cross-chunk drops)
    
    Args:
        documents: List of document texts
        
    Returns:
        Dictionary with aggregated entities and relationships
    """
    # CRITICAL PERFORMANCE LIMIT: Only process first N chunks with LLM for relationships
    max_relation_chunks = int(os.getenv("MAX_RELATION_CHUNKS", "5"))
    use_llm_entities = os.getenv("USE_LLM_ENTITY_EXTRACTION", "true").lower() == "true"
    
    logger.info(f"=== PHASE 1: Entity Discovery (all {len(documents)} chunks) ===")
    
    # PHASE 1: Extract entities from ALL chunks to build global registry
    all_entities = []
    
    for i, doc_text in enumerate(documents):
        logger.info(f"Phase 1 - Extracting entities from chunk {i+1}/{len(documents)}")
        
        # Extract entities - choose between LLM (better quality) or spaCy (faster)
        if use_llm_entities:
            entities = extract_entities_with_llm(doc_text)
        else:
            entities = extract_entities_sections(doc_text)
        
        all_entities.extend(entities)
    
    # Deduplicate entities (case-insensitive)
    seen_entities = set()
    unique_entities = []
    for entity in all_entities:
        key = (entity["text"].lower(), entity["label"])
        if key not in seen_entities:
            seen_entities.add(key)
            unique_entities.append(entity)
    
    logger.info(f"✅ Phase 1 Complete: {len(all_entities)} total entities, {len(unique_entities)} unique")
    logger.info(f"=== PHASE 2: Relation Extraction (first {max_relation_chunks} chunks with GLOBAL context) ===")
    
    # PHASE 2: Extract relationships using GLOBAL entity registry
    all_relationships = []
    
    for i in range(min(max_relation_chunks, len(documents))):
        logger.info(f"Phase 2 - Extracting relations from chunk {i+1}/{min(max_relation_chunks, len(documents))}")
        
        # CRITICAL FIX: Pass ALL unique entities (global registry), not just current chunk
        if unique_entities:
            relationships = extract_relations_with_llm(documents[i], known_entities=unique_entities)
            all_relationships.extend(relationships)
        else:
            logger.info("⏭️  Skipping relationship extraction (no entities found in global registry)")
    
    # Deduplicate relationships (use case-insensitive comparison)
    seen_relations = set()
    unique_relationships = []
    for rel in all_relationships:
        key = (rel["subject"].lower(), rel["relation"].lower(), rel["object"].lower())
        if key not in seen_relations:
            seen_relations.add(key)
            unique_relationships.append(rel)
    
    logger.info(f"✅ Phase 2 Complete: {len(all_relationships)} total relations, {len(unique_relationships)} unique")
    logger.info(
        f"=== Extraction Complete: {len(unique_entities)} entities, {len(unique_relationships)} relationships ==="
    )
    
    return {
        "entities": unique_entities,
        "relationships": unique_relationships,
        "total_entities": len(unique_entities),
        "total_relationships": len(unique_relationships)
    }
