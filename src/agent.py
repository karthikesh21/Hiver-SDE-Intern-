"""
Main AI Customer Support Agent:
Integrates intent classification, historical resolution retrieval, grounded reply generation,
and transparent escalation logic into a unified, production-grade support system.
"""

from typing import Dict, Any, List, Optional, Set
from src.intent_classifier import IntentClassifier
from src.retriever import HistoricalRetriever
from src.reply_generator import GroundedReplyGenerator
from src.escalation import EscalationEngine
from src.config import RETRIEVAL_TOP_K

class CustomerSupportAgent:
    """
    Production AI Customer Support Agent for AmazonHelp.
    """
    def __init__(
        self,
        retriever: Optional[HistoricalRetriever] = None,
        classifier: Optional[IntentClassifier] = None,
        generator: Optional[GroundedReplyGenerator] = None,
        escalation_engine: Optional[EscalationEngine] = None,
        top_k: int = RETRIEVAL_TOP_K
    ):
        self.retriever = retriever or HistoricalRetriever(top_k=top_k)
        self.classifier = classifier or IntentClassifier()
        self.generator = generator or GroundedReplyGenerator()
        self.escalation_engine = escalation_engine or EscalationEngine()
        self.top_k = top_k
        self._is_initialized = False
        
    def initialize(self):
        """Warm up retriever index and classifier prototypes."""
        if not self._is_initialized:
            self.retriever.build_or_load_index()
            # Share encoder to avoid redundant model instances
            if self.retriever.encoder is not None:
                self.classifier.encoder = self.retriever.encoder
            self.classifier.build_prototypes()
            self._is_initialized = True
        return self

    def process_message(
        self,
        customer_message: str,
        exclude_conversation_ids: Optional[Set[str]] = None
    ) -> Dict[str, Any]:
        """
        Process incoming customer inquiry through the complete agent pipeline.
        Returns clean, structured JSON output.
        """
        if not self._is_initialized:
            self.initialize()
            
        # 1. Intent Classification
        clf_result = self.classifier.predict(customer_message)
        intent = clf_result.get("intent", "delivery_delay_tracking")
        intent_confidence = float(clf_result.get("confidence", 0.50))
        
        # 2. Historical Resolution Retrieval (with strict leakage exclusion)
        retrieved = self.retriever.retrieve(
            query=customer_message,
            intent_filter=None, # Semantic retrieval across corpus
            exclude_conversation_ids=exclude_conversation_ids,
            top_k=self.top_k
        )
        
        # 3. Grounded Reply Generation
        reply = self.generator.generate(
            customer_message=customer_message,
            retrieved_examples=retrieved,
            intent=intent
        )
        
        # 4. Escalation Policy Decision
        escalation = self.escalation_engine.evaluate(
            customer_message=customer_message,
            intent=intent,
            intent_confidence=intent_confidence,
            retrieved_examples=retrieved
        )
        
        # 5. Format Structured Agent Response
        sanitized_retrieved = [
            {
                "conversation_id": ex.get("conversation_id", ""),
                "similarity": ex.get("similarity", 0.0),
                "intent": ex.get("intent", ""),
                "resolution": ex.get("resolution", "")[:120] + "..." if len(ex.get("resolution", "")) > 120 else ex.get("resolution", "")
            }
            for ex in retrieved
        ]
        
        return {
            "intent": intent,
            "intent_confidence": intent_confidence,
            "retrieved_examples": sanitized_retrieved,
            "reply": reply,
            "decision": escalation["decision"],
            "decision_reason": escalation["decision_reason"]
        }

if __name__ == "__main__":
    agent = CustomerSupportAgent().initialize()
    test_msg = "Where is my refund for the sneakers I returned last week?"
    output = agent.process_message(test_msg)
    import pprint
    pprint.pprint(output)
