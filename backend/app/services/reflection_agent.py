"""
Self-Reflection Agent for Agentic HybridRAG

This module implements a lightweight reflection step that evaluates retrieval quality
and decides whether to refine the retrieval strategy.

The reflection agent:
1. Analyzes retrieved documents against user intent
2. Checks for completeness, relevance, and coverage
3. Decides if retrieval should be retried with refined parameters
4. Provides actionable feedback for strategy adjustment
"""

import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

import ollama

logger = logging.getLogger(__name__)


class ReflectionDecision(Enum):
    """Possible reflection decisions"""
    ACCEPT = "accept"  # Retrieval is sufficient
    EXPAND = "expand"  # Need more results (increase top_k)
    REFINE = "refine"  # Need different strategy (change filters/entities)
    RETRY_VECTOR = "retry_vector"  # Switch to vector search only
    RETRY_GRAPH = "retry_graph"  # Add graph search


@dataclass
class ReflectionResult:
    """Result of reflection evaluation"""
    retry: bool
    decision: ReflectionDecision
    reason: str
    confidence: float  # 0.0 to 1.0
    suggestions: Dict[str, Any]  # Suggested changes to retrieval plan
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "retry": self.retry,
            "decision": self.decision.value,
            "reason": self.reason,
            "confidence": self.confidence,
            "suggestions": self.suggestions
        }


class ReflectionAgent:
    """
    Lightweight agent that evaluates retrieval quality and suggests improvements.
    
    This agent acts as a "quality gate" between retrieval and answer generation,
    ensuring that retrieved context is sufficient to answer the user's question.
    """
    
    # System prompt for reflection agent
    REFLECTION_PROMPT = """You are a retrieval quality evaluator for a HybridRAG system.

Your job is to analyze whether the retrieved documents are sufficient to answer the user's question.

Evaluate based on:
1. **Relevance**: Do the documents discuss the question topic?
2. **Completeness**: Is there enough information to provide a full answer?
3. **Coverage**: Are all parts of the question addressed?
4. **Specificity**: Are the documents specific enough, or too generic?

Output a JSON object with this schema:
{
  "retry": true/false,
  "decision": "accept|expand|refine|retry_vector|retry_graph",
  "reason": "Brief explanation (1 sentence)",
  "confidence": 0.0-1.0,
  "suggestions": {
    "action": "increase_top_k|add_graph|change_filters|add_entities",
    "details": "Specific changes to make"
  }
}

Decision types:
- **accept**: Documents are sufficient (retry=false)
- **expand**: Need more results, increase top_k (retry=true)
- **refine**: Need different documents, change filters (retry=true)
- **retry_vector**: Current strategy missed key info, try vector only (retry=true)
- **retry_graph**: Need relationships/entities, add graph search (retry=true)

Guidelines:
- If documents directly answer the question → accept
- If documents are relevant but incomplete → expand
- If documents are off-topic → refine
- If question asks "what/who/where" but no entities → retry_graph
- If question asks for comparison but no relationships → retry_graph
- Set confidence based on clarity of evaluation (0.7-1.0 typical)

Be conservative: Only retry if clearly necessary. Most retrievals are acceptable.

Examples:

Q: "What is CODA-LM?"
Retrieved: ["CODA-LM is a vision-language model...", "...evaluated on autonomous driving..."]
Output: {"retry": false, "decision": "accept", "reason": "Documents provide clear definition and context", "confidence": 0.95, "suggestions": {}}

Q: "How does GPT-4V compare to Claude?"
Retrieved: ["GPT-4V is a multimodal model...", "Claude is an AI assistant..."]
Output: {"retry": true, "decision": "retry_graph", "reason": "Documents describe models but lack comparison relationships", "confidence": 0.85, "suggestions": {"action": "add_graph", "details": "Query for 'compared_with' relationships between GPT-4V and Claude"}}

Q: "What datasets are used in autonomous driving research?"
Retrieved: ["Autonomous driving is a field...", "Self-driving cars use sensors..."]
Output: {"retry": true, "decision": "refine", "reason": "Documents discuss autonomous driving but don't mention specific datasets", "confidence": 0.80, "suggestions": {"action": "change_filters", "details": "Search specifically for dataset names and evaluation sections"}}

Q: "Explain the methodology in section 3"
Retrieved: ["Section 3 discusses...", "The methodology involves...", "Experiments were conducted..."]
Output: {"retry": false, "decision": "accept", "reason": "Documents cover methodology comprehensively", "confidence": 0.90, "suggestions": {}}

Now evaluate this retrieval:"""

    def __init__(
        self,
        model: str = "phi3:mini",
        ollama_url: str = "http://localhost:11434",
        max_retries: int = 2
    ):
        """
        Initialize reflection agent.
        
        Args:
            model: Ollama model for reflection (phi3:mini recommended)
            ollama_url: Ollama server URL
            max_retries: Maximum number of retrieval retries allowed
        """
        self.model = model
        self.ollama_client = ollama.Client(host=ollama_url)
        self.max_retries = max_retries
        
        logger.info(f"🔍 Initialized ReflectionAgent with model={model}, max_retries={max_retries}")
    
    def evaluate_retrieval(
        self,
        question: str,
        retrieved_chunks: List[str],
        retrieval_plan: Optional[Dict[str, Any]] = None,
        retry_count: int = 0
    ) -> ReflectionResult:
        """
        Evaluate whether retrieved documents are sufficient to answer the question.
        
        Args:
            question: User's question
            retrieved_chunks: List of retrieved document texts
            retrieval_plan: Original retrieval plan (for context)
            retry_count: Current retry attempt (0 = first retrieval)
            
        Returns:
            ReflectionResult with decision and suggestions
        """
        logger.info(f"🔍 Evaluating retrieval quality for: {question[:80]}...")
        
        # Quick checks before calling LLM
        if retry_count >= self.max_retries:
            logger.warning(f"⚠️ Max retries ({self.max_retries}) reached, accepting current retrieval")
            return ReflectionResult(
                retry=False,
                decision=ReflectionDecision.ACCEPT,
                reason=f"Max retries reached ({self.max_retries}), using available context",
                confidence=0.5,
                suggestions={}
            )
        
        if not retrieved_chunks:
            logger.warning("⚠️ No chunks retrieved, suggesting expansion")
            return ReflectionResult(
                retry=True,
                decision=ReflectionDecision.EXPAND,
                reason="No documents retrieved, need to expand search",
                confidence=0.95,
                suggestions={
                    "action": "increase_top_k",
                    "details": "Increase top_k to 10 and remove restrictive filters"
                }
            )
        
        # Build evaluation prompt
        prompt = self._build_evaluation_prompt(
            question,
            retrieved_chunks,
            retrieval_plan,
            retry_count
        )
        
        try:
            # Call reflection agent
            response = self.ollama_client.generate(
                model=self.model,
                prompt=prompt,
                format="json",
                options={
                    "temperature": 0.2,  # Slightly higher for nuanced evaluation
                    "num_predict": 200,  # Short responses
                }
            )
            
            response_text = response.get("response", "").strip()
            logger.debug(f"Reflection response: {response_text}")
            
            # Parse JSON response
            try:
                reflection_dict = json.loads(response_text)
            except json.JSONDecodeError as e:
                logger.error(f"❌ Invalid JSON from reflection agent: {e}")
                return self._get_fallback_reflection(retrieved_chunks)
            
            # Convert to ReflectionResult
            result = self._parse_reflection(reflection_dict)
            
            logger.info(
                f"✅ Reflection: {result.decision.value} "
                f"(retry={result.retry}, confidence={result.confidence:.2f})"
            )
            logger.info(f"📝 Reason: {result.reason}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Reflection agent failed: {e}")
            return self._get_fallback_reflection(retrieved_chunks)
    
    def _build_evaluation_prompt(
        self,
        question: str,
        chunks: List[str],
        plan: Optional[Dict[str, Any]],
        retry_count: int
    ) -> str:
        """Build the evaluation prompt with question and retrieved chunks"""
        
        # Format chunks
        chunks_text = "\n\n".join([
            f"[Document {i+1}]\n{chunk[:500]}..."  # Truncate long chunks
            for i, chunk in enumerate(chunks[:5])  # Max 5 chunks
        ])
        
        # Add retrieval plan context if available
        plan_text = ""
        if plan:
            plan_text = f"\n\nCurrent retrieval strategy:\n- Used vector DB: {plan.get('use_vector_db')}\n- Used graph DB: {plan.get('use_graph_db')}\n- Filters: {plan.get('vector_filters', {})}"
        
        # Add retry context
        retry_text = f"\n\nRetry attempt: {retry_count + 1} of {self.max_retries + 1}" if retry_count > 0 else ""
        
        prompt = f"""{self.REFLECTION_PROMPT}

Question: {question}

Retrieved Documents:
{chunks_text}
{plan_text}
{retry_text}

Evaluate the retrieval and output ONLY the JSON (no other text):"""
        
        return prompt
    
    def _parse_reflection(self, reflection_dict: Dict[str, Any]) -> ReflectionResult:
        """Parse reflection dictionary into ReflectionResult"""
        
        decision_str = reflection_dict.get("decision", "accept")
        try:
            decision = ReflectionDecision(decision_str)
        except ValueError:
            logger.warning(f"Unknown decision '{decision_str}', defaulting to ACCEPT")
            decision = ReflectionDecision.ACCEPT
        
        return ReflectionResult(
            retry=reflection_dict.get("retry", False),
            decision=decision,
            reason=reflection_dict.get("reason", "No reason provided"),
            confidence=float(reflection_dict.get("confidence", 0.7)),
            suggestions=reflection_dict.get("suggestions", {})
        )
    
    def _get_fallback_reflection(self, chunks: List[str]) -> ReflectionResult:
        """Generate fallback reflection when agent fails"""
        
        if not chunks:
            return ReflectionResult(
                retry=True,
                decision=ReflectionDecision.EXPAND,
                reason="Fallback: No documents retrieved",
                confidence=0.5,
                suggestions={"action": "increase_top_k", "details": "Expand search"}
            )
        
        return ReflectionResult(
            retry=False,
            decision=ReflectionDecision.ACCEPT,
            reason="Fallback: Agent failed, accepting current retrieval",
            confidence=0.5,
            suggestions={}
        )
    
    def apply_suggestions(
        self,
        original_plan: Dict[str, Any],
        reflection: ReflectionResult
    ) -> Dict[str, Any]:
        """
        Apply reflection suggestions to modify the retrieval plan.
        
        Args:
            original_plan: Original retrieval plan
            reflection: Reflection result with suggestions
            
        Returns:
            Modified retrieval plan
        """
        if not reflection.retry or not reflection.suggestions:
            return original_plan
        
        modified_plan = original_plan.copy()
        action = reflection.suggestions.get("action", "")
        
        logger.info(f"🔧 Applying suggestion: {action}")
        
        if action == "increase_top_k":
            # Expand search
            modified_plan["top_k"] = original_plan.get("top_k", 5) + 5
            logger.info(f"  → Increased top_k to {modified_plan['top_k']}")
        
        elif action == "add_graph":
            # Add graph search
            modified_plan["use_graph_db"] = True
            details = reflection.suggestions.get("details", "")
            if "compared_with" in details:
                modified_plan["graph_relation_types"] = ["compared_with", "evaluated_on"]
            logger.info(f"  → Enabled graph DB with relations")
        
        elif action == "change_filters":
            # Remove restrictive filters
            if "vector_filters" in modified_plan:
                # Keep document scope but remove other filters
                doc_filter = modified_plan["vector_filters"].get("document_id")
                modified_plan["vector_filters"] = {}
                if doc_filter:
                    modified_plan["vector_filters"]["document_id"] = doc_filter
                logger.info(f"  → Relaxed filters")
        
        elif action == "add_entities":
            # Enable entity-based graph search
            modified_plan["use_graph_db"] = True
            details = reflection.suggestions.get("details", "")
            # Parse entity names from details (simple extraction)
            import re
            entities = re.findall(r"'([^']+)'", details)
            if entities:
                modified_plan["graph_entities"] = entities
                logger.info(f"  → Added entities: {entities}")
        
        return modified_plan


# Integration with AgenticRetriever
class ReflectiveAgenticRetriever:
    """
    Wrapper around AgenticRetriever that adds self-reflection.
    
    This class retrieves documents, evaluates quality, and optionally
    retries with refined strategy based on reflection feedback.
    """
    
    def __init__(self, base_retriever, reflection_agent: Optional[ReflectionAgent] = None):
        """
        Initialize reflective retriever.
        
        Args:
            base_retriever: AgenticRetriever instance
            reflection_agent: ReflectionAgent instance (optional)
        """
        self.base_retriever = base_retriever
        self.reflection_agent = reflection_agent or ReflectionAgent()
        
        logger.info("🔄 Initialized ReflectiveAgenticRetriever with reflection enabled")
    
    def retrieve_with_reflection(
        self,
        question: str,
        context_docs: Optional[List[str]] = None,
        top_k: int = 5,
        enable_reflection: bool = True
    ) -> Dict[str, Any]:
        """
        Retrieve with self-reflection loop.
        
        Args:
            question: User's question
            context_docs: Document scope
            top_k: Number of results
            enable_reflection: Whether to enable reflection (can disable for speed)
            
        Returns:
            Dict with context, reflection_history, and final_reflection
        """
        retry_count = 0
        reflection_history = []
        
        while retry_count <= self.reflection_agent.max_retries:
            logger.info(f"🔍 Retrieval attempt {retry_count + 1}")
            
            # Execute retrieval
            context = self.base_retriever.retrieve(
                question=question,
                context_docs=context_docs,
                top_k=top_k
            )
            
            # Extract text chunks for reflection
            chunks = [chunk.get("text", "") for chunk in context.vector_chunks]
            
            if not enable_reflection:
                logger.info("⚡ Reflection disabled, accepting retrieval")
                return {
                    "context": context,
                    "reflection_history": [],
                    "final_reflection": None
                }
            
            # Reflect on retrieval quality
            reflection = self.reflection_agent.evaluate_retrieval(
                question=question,
                retrieved_chunks=chunks,
                retrieval_plan=context.retrieval_plan.__dict__,
                retry_count=retry_count
            )
            
            reflection_history.append(reflection.to_dict())
            
            # If accepted or max retries reached, return
            if not reflection.retry or retry_count >= self.reflection_agent.max_retries:
                logger.info(f"✅ Retrieval {'accepted' if not reflection.retry else 'finalized'} after {retry_count + 1} attempt(s)")
                return {
                    "context": context,
                    "reflection_history": reflection_history,
                    "final_reflection": reflection
                }
            
            # Apply suggestions and retry
            logger.info(f"🔄 Retrying retrieval with refinements...")
            modified_plan = self.reflection_agent.apply_suggestions(
                original_plan=context.retrieval_plan.__dict__,
                reflection=reflection
            )
            
            # Update parameters for next iteration
            top_k = modified_plan.get("top_k", top_k)
            retry_count += 1
        
        # Should never reach here, but just in case
        return {
            "context": context,
            "reflection_history": reflection_history,
            "final_reflection": reflection
        }


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 80)
    print("Self-Reflection Agent - JSON Schema")
    print("=" * 80)
    
    schema = {
        "type": "object",
        "required": ["retry", "decision", "reason", "confidence", "suggestions"],
        "properties": {
            "retry": {
                "type": "boolean",
                "description": "Whether to retry retrieval with refined strategy"
            },
            "decision": {
                "type": "string",
                "enum": ["accept", "expand", "refine", "retry_vector", "retry_graph"],
                "description": "Type of action to take"
            },
            "reason": {
                "type": "string",
                "description": "Brief explanation (1 sentence)"
            },
            "confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Confidence in evaluation (0.0-1.0)"
            },
            "suggestions": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["increase_top_k", "add_graph", "change_filters", "add_entities"],
                        "description": "Specific action to take"
                    },
                    "details": {
                        "type": "string",
                        "description": "Specific changes to make"
                    }
                }
            }
        }
    }
    
    print(json.dumps(schema, indent=2))
    
    print("\n" + "=" * 80)
    print("Example Reflection Results")
    print("=" * 80)
    
    examples = [
        {
            "scenario": "Sufficient documents",
            "result": {
                "retry": False,
                "decision": "accept",
                "reason": "Documents provide comprehensive answer",
                "confidence": 0.95,
                "suggestions": {}
            }
        },
        {
            "scenario": "Need more results",
            "result": {
                "retry": True,
                "decision": "expand",
                "reason": "Documents are relevant but incomplete",
                "confidence": 0.85,
                "suggestions": {
                    "action": "increase_top_k",
                    "details": "Increase from 5 to 10 results"
                }
            }
        },
        {
            "scenario": "Need graph relationships",
            "result": {
                "retry": True,
                "decision": "retry_graph",
                "reason": "Question asks for comparison but no relationships found",
                "confidence": 0.90,
                "suggestions": {
                    "action": "add_graph",
                    "details": "Query for 'compared_with' relationships"
                }
            }
        }
    ]
    
    for ex in examples:
        print(f"\nScenario: {ex['scenario']}")
        print(json.dumps(ex['result'], indent=2))
    
    print("\n✅ Reflection agent ready!")
