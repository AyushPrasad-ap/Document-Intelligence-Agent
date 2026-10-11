# Document Intelligence Agent — Evaluation Report

## 1. Summary

The Document Intelligence Agent was evaluated using a small, manually curated set of retrieval and answer-generation questions. The regular automated test suite also completed successfully: **37 tests passed** (as reported after the latest test run).

| Evaluation | Result |
|---|---:|
| Automated test suite | 37/37 passed |
| Retrieval questions evaluated | 5 |
| Retrieval Hit@5 | 100% (5/5) |
| Mean Reciprocal Rank (MRR) | 0.80 |
| Answer questions evaluated | 10 |
| Answer checks passed automatically | 10/10 (100%) |
| Answer cases flagged for manual review | 10/10 |

## 2. Retrieval evaluation

Retrieval was evaluated with `top_k = 5` against five queries with expected source pages.

- **Hit@5 = 1.00:** at least one expected relevant page appeared in the top five results for all five queries.
- **MRR = 0.80:** the average reciprocal rank of the first relevant result across those queries was 0.80.
- For the FY25 net-profit, NIM, and capital-adequacy queries, a relevant page appeared at rank 1.
- For the FY24 net-profit query and the multi-step profit-growth query, the relevant page appeared at rank 2.

These results indicate that the current retriever found a relevant page within the top five for all tested questions. They do **not** establish that retrieval will perform equally well across the full document collection or on unseen query types; the evaluation contains only five retrieval questions.

## 3. Answer evaluation

The answer-evaluation set contained ten cases across these categories:

- Financial document questions (E01–E04)
- Arithmetic calculations (E05–E06)
- A multi-step financial question (E07)
- A general capability question (E08)
- Unsupported or unavailable facts requiring abstention (E09–E10)

All ten cases passed the evaluator's automated checks. Depending on the case, checks covered non-empty answers, expected tool routing, expected source retrieval, citation mapping, source grounding, numeric values, and appropriate handling of unsupported questions. Checks marked not applicable were not required for that case.

The unsupported-question cases did not invent an answer or a document citation when the requested Tesla FY2025 net profit or HDFC Bank FY2020 net profit could not be established from the available documents.

## 4. Observations

- The FY24 net-profit query retrieved a less relevant page at rank 1, but the expected page appeared at rank 2. This contributes to the difference between Hit@5 and MRR.
- The multi-step profit-growth query also retrieved its expected page at rank 2.
- In calculation case E06, the calculator rejected an initial `abs(...)` expression under its restricted expression rules; the agent retried with a supported subtraction expression and returned the expected result. The final answer passed, though the trace shows this recovery path.
- Every answer-evaluation result has `manual_review_required: true`. Therefore, the **100% figure represents automated checks only**, not a claim that a human reviewed every answer or that the evaluation proves complete factual correctness.

## 5. Limitations

1. The evaluation set is small and hand-curated: five retrieval questions and ten answer questions.
2. Automated numeric and citation checks are screening signals. They cannot fully judge reasoning quality, whether every claim is supported, whether all useful evidence was retrieved, or whether an answer is well-calibrated.
3. The expected retrieval pages were selected for these specific test questions. Results should not be generalized to all documents, question phrasing, or future uploads.
4. The reports do not constitute an independent benchmark comparison against other embedding models, chunking strategies, vector databases, or LLMs.

## 6. Conclusion

The current build passes its 37-test automated suite and meets the initial evaluation targets on the curated test set: **100% retrieval Hit@5, 0.80 MRR, and 10/10 automated answer checks passed**. Before presenting these results as a quality guarantee, manually review the recorded answers and their cited evidence, and expand the evaluation set with more documents, query phrasings, multi-hop questions, and deliberately unanswerable questions.

## 7. Supporting result files

- `data/processed/retrieval_evaluation_results.json`
- `data/processed/answer_evaluation_results.json`
