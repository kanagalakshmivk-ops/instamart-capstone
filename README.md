# Instamart Customer Intelligence

All amounts are in INR. All data is synthetic; no network or API key is used.

## Reproduce
1. `git clone <repo> && cd instamart` (branch `main`)
2. `python -m venv .venv && source .venv/bin/activate` (Windows: `.venv\Scripts\activate`)
3. `pip install -r requirements.txt`
4. Put `generate_data.py` and `mock_llm.py` in the project root, then `python generate_data.py`
5. `python part_a.py` (clean, store, SQL, pandas) then `python validate.py` (prints `ALL CHECKPOINTS PASSED`)
6. `python part_b.py` (segments + lapse model), `python part_c.py` (triage), `python part_d.py` (RAG + agent)
   Or run everything: `./run_all.sh`.

## Part A: cleaning rules (rows or values changed)
| # | Rule | Changed |
|---|------|---------|
| 1 | `order_ts` converted to datetime | all rows |
| 2 | Exact duplicate rows removed | 119 rows |
| 3 | Per `order_id` keep latest `order_ts` (drops earlier pending records) | 656 rows |
| 4 | `order_amount` to numeric; unparseable filled from `round(price*qty*(1-disc/100),2)` | 112 values |
| 5 | Negative `delivery_minutes` set missing | 105 values |
| 6 | Missing delivery of completed orders filled with category median (completed only) | 774 values |
| 7 | Cancelled orders keep missing delivery (never delivered; imputing would invent data) | 1,199 left blank |
| C1 | City stripped + title case (24 spellings to 6) | |
| C2 | Gender mapped to male/female/unknown | blank to unknown: 30 |
| C3 | Ages outside 10-100 set missing (6), then 67 missing ages filled with city median | 67 |

Result: 12,309 orders (11,110 completed, 1,199 cancelled); completed revenue INR 2,998,668.80.
SQL Q1-Q7 and the pandas cross-check reproduce every reference value (see `part_a.py` output).
Q6 note: RANK() differs from ROW_NUMBER() on 8 rows because tied spends share one rank while ROW_NUMBER() still numbers them uniquely.

## Part B
K=4 chosen (elbow after K=4, best silhouette 0.526, four actionable segments). PCA ratios 0.634 / 0.218. Cohort 1,156 customers, 443 lapsed (38.32%), split 924/232. Random forest (depth 8, leaf 10) chosen: test F1 0.655 (logistic regression 0.648). Over-sampling lifts recall 0.65 to 0.85 but precision falls 0.64 to 0.59.
No encoder, scaler, imputer or resampler saw test rows; no data from 2025-09-01 onwards entered the features; segment labels were not used as features.
**Limitations and Next Steps:** the Premium flag is a post-hoc whole-window label (it includes September), so the retention list is illustrative and not a leakage-free evaluation.

## Part C
Redactions: 55 emails, 91 phones, 132 order ids (278 total, 0 left). Statuses per full run: 135 classified_json, 12 classified_keyword, 9 flagged_invalid_output, 4 skipped_empty, 4 flagged_long; 19 transient errors, all recovered.
Zero-shot accuracy 85.7% at about 75.6 tokens per classified ticket; few-shot 100% at about 165.8. Few-shot costs about 2.2x the tokens but removes every fraud/technical-to-billing error, worth it when a missed fraud ticket is costly.
Budget demo: zero-shot with TOKEN_BUDGET=6000 left 72 tickets budget_exhausted. Rule-validated tickets are routed automatically; empty, long, invalid and budget-exhausted tickets go to human review.

## Part D
Chunks: 15 (refund 4, delivery 3, escalation 3, payment 3, privacy 2). Retrieval/grounding 12/12, 12/12, fallback 2/2; with overlap 0 two questions break (refund-eligibility and payment-failure answers are split across a chunk boundary).
TF measures how often a word appears in a chunk, IDF how rare it is across chunks; TF-IDF cannot match synonyms (e.g. "money back" vs "refund").

| Scenario | Status | Refund (INR) | Model calls | Tokens |
|---|---|---|---|---|
| S1 | REFUND | 128.25 | 5 | 1102 |
| S2 | REFUND | 135.00 | 5 | 1102 |
| S3 | ESCALATE | 1999.00 | 5 | 1089 |
| S4 | DENIED | 0.00 | 4 | 769 |
| S5 | DENIED | 0.00 | 3 | 460 |
| S6 | NOT_FOUND | 0.00 | 2 | 154 |
| S7 | FALLBACK_MAX_ITERATIONS | 0.00 | 6 | 3246 |

Demos: S1 with token_limit=200 ends `FALLBACK_TOKEN_LIMIT`; S1 with time_limit_seconds=0 ends `FALLBACK_TIMEOUT`.
Crash and resume (S1): tools executed across crash and resume, in order: get_order, search_policy, calculate_refund, issue_refund (4, none repeated); final REFUND 128.25.

**ReAct vs plan-and-execute.** ReAct used 5 model calls for S1, 3 for S5 and 2 for S6, because it asks the model after every observation, so history (and tokens) grows each call. Plan-and-execute asks once for the whole step list, so it needs fewer calls and tokens on the happy path, but its plan is made before seeing data: a NOT_FOUND order or a cancelled order makes later steps pointless and forces a replan call. ReAct stops at once (S6 after 2 calls, S5 after 3). For this short, branching refund workflow I would choose ReAct: branches depend on observations (status, minutes, amount above INR 1000), the cost is only a few hundred extra tokens, and each step is checkpointable.
