"""Part D driver. Run: python part_d.py (after part_a.py)"""
import json, os
import pandas as pd
import rag, agent
from agent import run_agent, resume_agent, SimulatedCrash

# ---------- 1. chunks
chunks = rag.build_chunks()
print("chunks:", len(chunks), pd.Series([c["source"] for c in chunks]).value_counts().sort_index().to_dict())
# ---------- 2. retrieval
print("\nquery 'refund policy late delivery' ->", [h["source"] for h in rag.retrieve("refund policy late delivery")])
# ---------- 3. one example answer
print("\nexample:", rag.answer("Who must approve a refund above INR 1000?"))
# ---------- 4. evaluation
res10, fail10 = rag.evaluate()
print("\noverlap 10:", res10, "| failures:", fail10)
res0, fail0 = rag.evaluate(rag.Retriever(40, 0))
print("overlap  0:", res0); [print("  breaks:", f) for f in fail0]
# ---------- 5-6. run the 7 scenarios
for p in ("outputs/refund_ledger.jsonl", "outputs/agent_runs.jsonl", "outputs/agent_demos.jsonl"):
    if os.path.exists(p): os.remove(p)
sc = pd.read_csv("data/raw/agent_scenarios.csv")
rows = []
for _, s in sc.iterrows():
    r = run_agent(s.message, s.scenario_id)
    refund = next((st["observation"]["refund_inr"] for st in r["steps"] if st["tool"] in ("issue_refund", "escalate_to_human")), 0.0)
    ok = (r["status"] == s.expected_outcome) and abs(refund - s.expected_refund_inr) <= 0.01 if s.expected_outcome != "FALLBACK" \
        else r["status"] == "FALLBACK_MAX_ITERATIONS"
    rows.append(dict(scenario_id=s.scenario_id, status=r["status"], refund_inr=refund, model_calls=r["llm_calls"],
                     tokens_used=r["tokens_used"], expected=s.expected_outcome, ok=ok))
    with open("outputs/agent_runs.jsonl", "a") as f:
        f.write(json.dumps({"scenario_id": s.scenario_id, "status": r["status"], "final_answer": r["final_answer"],
                            "llm_calls": r["llm_calls"], "tokens_used": r["tokens_used"],
                            "steps": [{"tool": st["tool"], "arguments": st["arguments"], "observation": st["observation"]} for st in r["steps"]]}) + "\n")
tab = pd.DataFrame(rows); print("\n", tab.to_string(index=False)); assert tab.ok.all()
s1 = sc.iloc[0]
# ---------- 7. safe stopping demos
d1 = run_agent(s1.message, "S1", token_limit=200)
d2 = run_agent(s1.message, "S1", time_limit_seconds=0)
print("\ntoken_limit=200 ->", d1["status"], "| tokens", d1["tokens_used"], "| calls", d1["llm_calls"], "| steps", len(d1["steps"]))
print("time_limit=0    ->", d2["status"], "| calls", d2["llm_calls"])
print("handoff text:", d1["final_answer"])
# ---------- 8. crash and resume
try:
    run_agent(s1.message, "S1", crash_after_tool=2, run_id="S1-crash-demo")
except SimulatedCrash as e:
    print("\ncrash:", e)
ck = json.load(open("state/S1-crash-demo.json")); print("checkpoint tools so far:", ck["tools_executed"])
rr = resume_agent("S1-crash-demo")
final_ck = json.load(open("state/S1-crash-demo.json"))
print("resumed ->", rr["status"], rr["final_answer"], "| tools across crash+resume:", final_ck["tools_executed"])
assert final_ck["tools_executed"] == ["get_order", "search_policy", "calculate_refund", "issue_refund"]
with open("outputs/agent_demos.jsonl", "w") as f:
    for name, r in [("token_limit_200", d1), ("time_limit_0", d2), ("resume", rr)]:
        f.write(json.dumps({"demo": name, "status": r["status"], "final_answer": r["final_answer"],
                            "llm_calls": r["llm_calls"], "tokens_used": r["tokens_used"]}) + "\n")
