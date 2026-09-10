"""
Configuration and constants for the AI Customer Support Agent system.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
REPORT_DIR = PROJECT_ROOT / "report"

# Load environment variables
load_dotenv(PROJECT_ROOT / ".env")

# Execution & Modeling Constants
RANDOM_SEED = int(os.getenv("RANDOM_SEED", 42))
SELECTED_BRAND = os.getenv("SELECTED_BRAND", "AmazonHelp")

# Model & API Settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Retrieval & Escalation Thresholds (Calibrated & Frozen on Development Set)
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", 3))
INTENT_CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.50))
RETRIEVAL_SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.45))

# Golden Set Paths
GOLDEN_SET_PATH = DATA_DIR / "golden_set.csv"
HISTORICAL_RESOLUTION_PATH = PROCESSED_DATA_DIR / f"{SELECTED_BRAND}_historical_resolutions.jsonl"
INTENTS_PATH = DATA_DIR / "intents.json"
