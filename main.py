"""
Command Line Interface for the AI Customer Support Agent (AmazonHelp).
Usage:
    python main.py --message "Where is my package? It was supposed to arrive yesterday."
    python main.py --interactive
"""

import sys
import argparse
import json
from src.agent import CustomerSupportAgent

def format_agent_output(output: dict, as_json: bool = False):
    if as_json:
        print(json.dumps(output, indent=2))
        return

    print("\n" + "=" * 75)
    print("AI CUSTOMER SUPPORT AGENT RESPONSE")
    print("=" * 75)
    print(f"INTENT:             {output['intent']} (Confidence: {output['intent_confidence']:.2f})")
    print(f"DECISION:           {output['decision']}")
    print(f"DECISION REASON:    {output['decision_reason']}")
    print("-" * 75)
    print("RETRIEVED HISTORICAL PRECEDENTS:")
    for i, ex in enumerate(output["retrieved_examples"], 1):
        sim = ex.get("similarity", 0.0)
        cid = ex.get("conversation_id", "")
        res = ex.get("resolution", "")
        print(f"  [{i}] [Sim: {sim:.3f} | ID: {cid[:12]}] {res}")
    print("-" * 75)
    print("DRAFT CUSTOMER-FACING REPLY:")
    print(f"\"{output['reply']}\"")
    print("=" * 75 + "\n")

def main():
    parser = argparse.ArgumentParser(description="AI Customer Support Agent for AmazonHelp")
    parser.add_argument("--message", "-m", type=str, help="Customer message text to process")
    parser.add_argument("--interactive", "-i", action="store_true", help="Launch interactive chat session")
    parser.add_argument("--json", action="store_true", help="Emit raw structured JSON")
    
    args = parser.parse_args()
    
    if not args.message and not args.interactive:
        parser.print_help()
        print("\nExample:")
        print("  python main.py --message \"Where is my refund for my return?\"")
        sys.exit(0)
        
    print("Initializing AI Customer Support Agent...")
    agent = CustomerSupportAgent().initialize()
    print("Agent ready.\n")
    
    if args.message:
        output = agent.process_message(args.message)
        format_agent_output(output, as_json=args.json)
        return
        
    if args.interactive:
        print("Starting interactive session. Type 'exit' or 'quit' to end.\n")
        while True:
            try:
                msg = input("Customer > ").strip()
                if msg.lower() in ["exit", "quit", "q"]:
                    print("Exiting session.")
                    break
                if not msg:
                    continue
                output = agent.process_message(msg)
                format_agent_output(output, as_json=args.json)
            except (KeyboardInterrupt, EOFError):
                print("\nSession terminated.")
                break

if __name__ == "__main__":
    main()
