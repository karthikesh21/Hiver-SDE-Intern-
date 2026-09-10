# Engineering Decision Log: AI Customer Support Agent

This log records 12 non-obvious engineering decisions, their underlying technical and operational rationale, and their associated trade-offs made during the development of this project.

---

### Decision 1: Quantitative Selection of `AmazonHelp` over `AppleSupport`
- **Decision**: Select `AmazonHelp` (81,092 conversations) as the target brand rather than `AppleSupport` (76,639) or `Uber_Support` (41,185).
- **Reason**: Quantitative analysis of the raw conversations showed `AmazonHelp` has an average conversation depth of 8.65 turns (more than double AppleSupport's 3.70 and Uber's 3.56) and 14,214 unique vocabulary terms. Amazon interactions capture the full customer lifecycle: pre-purchase specs, in-transit shipping delays, physical unboxing damage, return logistics, financial ledger disputes, and subscription renewals.
- **Tradeoff**: Amazon queries contain higher stylistic noise, heavy reliance on external carrier tracking, and extensive account-specific references compared to Apple's standardized software troubleshooting guides.

---

### Decision 2: Processing a Reproducible 4,000-Conversation Subsample instead of 3 Million Tweets
- **Decision**: Sample and clean 4,000 high-quality, multi-turn Amazon conversations rather than indexing the full 3-million-tweet corpus.
- **Reason**: Processing 3 million tweets requires distributed cloud infrastructure, introduces millions of uninformative one-turn deflections, and causes local vector search latencies that impede rapid experimentation. 4,000 cleaned multi-turn interactions provide dense coverage across all 8 domain intents while enabling sub-second local retrieval on standard CPU hardware.
- **Tradeoff**: Extremely rare long-tail niche issues (e.g., obscure AWS/Kindle hardware bugs) have lower retrieval density than they would in the full corpus.

---

### Decision 3: Stratified Golden Evaluation Set Size of Exactly 200 (25 per Intent)
- **Decision**: Fix the golden benchmark size at exactly 200 examples, uniformly partitioned across the 8 discovered intents (25 examples each).
- **Reason**: A 200-example benchmark provides sufficient statistical power to distinguish significant performance deltas between baselines (e.g. 12.5% vs 44.5% vs 76.5%) while remaining small enough for meticulous manual verification and zero-leakage enforcement.
- **Tradeoff**: Uniform class balance does not mirror real-world operational distributions where delivery queries dominate (~40%). It tests intent classification capabilities equally across classes rather than weighting by production volume.

---

### Decision 4: Bi-Encoder Embeddings (`all-MiniLM-L6-v2`) for Vector Retrieval
- **Decision**: Utilize `all-MiniLM-L6-v2` with pre-computed normalized cosine dot products rather than BM25 keyword search or external vector databases (Chroma/Pinecone).
- **Reason**: `all-MiniLM-L6-v2` runs locally on CPU with an inference latency of <2 ms per query, produces 384-dimensional embeddings that fit easily in RAM (~12 MB for 4,000 records), and captures semantic paraphrases that BM25 misses. It requires zero cloud credentials and eliminates network latency.
- **Tradeoff**: Bi-encoders compress full sentence semantics into a single dense vector, which can occasionally dilute specific serial numbers or 17-digit Amazon order codes.

---

### Decision 5: Shared SentenceTransformer Instance between Retriever and Classifier
- **Decision**: Pass the loaded SentenceTransformer model instance directly from the retriever to the intent classifier during agent initialization.
- **Reason**: In memory-constrained environments, loading duplicate instances of PyTorch neural models wastes 200+ MB of RAM and adds redundant cold-start initialization latency.
- **Tradeoff**: Couples the encoder architecture between the intent classifier and historical retriever; if one component requires a different model dimension in the future, decoupling will be necessary.

---

### Decision 6: Calibrated Softmax Temperature ($T = 0.12$) on Semantic Intent Prototypes
- **Decision**: Apply a low temperature scale ($T=0.12$) to raw cosine similarities between query embeddings and intent prototype centroids.
- **Reason**: Raw cosine similarities across semantic prototypes tend to concentrate in the narrow band of 0.45 to 0.85. Standard softmax produces diffuse, uncalibrated probabilities. A temperature of 0.12 sharpens the probability distribution around the top intent, providing an interpretable confidence score that correlates with classification correctness.
- **Tradeoff**: Highly ambiguous queries with two genuinely overlapping intents can produce overconfident winner-takes-all scores if not tempered by lexical guards.

---

### Decision 7: Multi-Tiered Safety Gating Prioritizing False Auto-Handle Minimization
- **Decision**: Structure escalation criteria to aggressively penalize false auto-handles at the expense of nominal escalation accuracy.
- **Reason**: In production customer support, false auto-handles (falsely telling a customer their stolen package or double charge is taken care of) cause catastrophic customer churn and financial liability. Unnecessary escalations are merely an operational cost. We chose a conservative similarity threshold (0.60) and explicit hazard triggers (legal, fire, theft, 2FA lockout) to ensure safety.
- **Tradeoff**: Results in 103 unnecessary escalations and a nominal escalation accuracy of 45.5%, sacrificing automation rate for high escalation recall (92.6%).

---

### Decision 8: Grounded Template Synthesis as an In-Engine Local Fallback
- **Decision**: Implement a local deterministic grounded response synthesizer alongside the OpenAI API integration.
- **Reason**: Relying exclusively on external LLM APIs makes a project non-reproducible if API credits expire, network outages occur, or credentials are not supplied. The local synthesizer strips Twitter noise, formats navigational paths, and grounds replies in retrieved historical precedents with zero external dependencies.
- **Tradeoff**: Template-synthesized responses exhibit less conversational fluidity and stylistic variation than unconstrained LLM completions.

---

### Decision 9: Strict Exclusion Guard against Evaluation Leakage
- **Decision**: Dynamically pass all golden evaluation set conversation IDs and message signatures into the retriever's query method to explicitly forbid them from being retrieved as their own evidence.
- **Reason**: Without explicit leakage guards, nearest-neighbor vector search over historical data can retrieve the exact evaluation query if near-duplicates exist in the dataset, artificially inflating retrieval and generation metrics to 100%.
- **Tradeoff**: Requires maintaining a hash set of evaluation IDs during benchmark runs and slight filtering overhead during retrieval ranking.

---

### Decision 10: Intentional Omission of Autonomous Action Execution (Refund/Database Mutation)
- **Decision**: Deliberately omit automated database-mutating tools (e.g. issuing automated refunds or cancelling orders).
- **Reason**: In an enterprise social media integration, customer identities on Twitter are unauthenticated handles (`@user123`). Permitting an LLM to trigger financial transactions without secondary cryptographic or session authentication represents a critical security vulnerability.
- **Tradeoff**: The agent acts solely as a triage, informational guidance, and escalation assistant rather than an autonomous robotic refund processor.

---

### Decision 11: Choosing Spearman Rank Correlation ($\rho$) for Human-Judge Agreement
- **Decision**: Prioritize Spearman rank correlation ($\rho$) and Mean Absolute Difference (MAD) over raw Pearson linear correlation ($r$) when evaluating human-judge agreement.
- **Reason**: Evaluation rubric scores (1–5 scale) represent ordinal rankings with non-normal, bounded distributions. Pearson correlation assumes continuous bivariate normality, whereas Spearman measures monotonic ranking consensus.
- **Tradeoff**: Spearman evaluates relative order rather than exact scale calibration, which is why we pair it with Mean Absolute Difference to track absolute point deviations.

---

### Decision 12: Zero Hardcoded API Keys or Cloud Secrets
- **Decision**: Keep all API configurations in `.env.example`, `.gitignore`, and dynamic environment readers, with automatic graceful degradation if keys are absent.
- **Reason**: Hardcoding tokens into git repositories violates enterprise security standards and prevents graders/reviewers from running the code cleanly on their own machines.
- **Tradeoff**: Code must include dual-mode execution branches (API mode vs. local offline mode) for both inference and evaluation.
