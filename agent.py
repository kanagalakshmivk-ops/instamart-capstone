"""Part D tasks 5-8: tools, ReAct loop, safe stopping, checkpoint/resume."""
import json, math, os, sqlite3, time
from datetime import datetime
import mock_llm
from rag import retrieve

LEDGER, STATE_DIR = "outputs/refund_ledger.jsonl", "state"
os.makedirs("outputs", exist_ok=True); os.makedirs(STATE_DIR, exist_ok=True)

# ------------------------------------------------------------------ tools
def _clean(v):
    if v is None or (isinstance(v, float) and math.isnan(v)): return None
    return v

def get_order(order_id):
    with sqlite3.connect("data/instamart.db") as con:
        row = con.execute("SELECT order_id, status, order_amount, delivery_minutes FROM orders WHERE order_id = ?",
                          (order_id,)).fetchone()
    if row is None: return {"error": "order not found"}
    return {"order_id": row[0], "status": row[1], "order_amount": _clean(row[2]), "delivery_minutes": _clean(row[3])}

def search_policy(query):
    return {"results": [{"source": h["source"], "score": round(h["score"], 4), "text": h["text"]} for h in retrieve(query)]}

def calculate_refund(order_amount, delivery_minutes):
    """Rule: delivered in more than 45 minutes -> 100% of the amount, otherwise 0."""
    if delivery_minutes is None or delivery_minutes <= 45: return {"refund_inr": 0.0}
    return {"refund_inr": round(float(order_amount), 2)}

def _ledger(tool, order_id, refund_inr):
    with open(LEDGER, "a", encoding="utf-8") as f:
        f.write(json.dumps({"timestamp": datetime.now().isoformat(timespec="seconds"), "tool": tool,
                            "order_id": order_id, "refund_inr": refund_inr}) + "\n")
    return {"refund_inr": refund_inr}

def issue_refund(order_id, refund_inr):
    if refund_inr > 1000: return {"error": "refunds above INR 1000 must be escalated, not issued"}
    return _ledger("issue_refund", order_id, refund_inr)

def escalate_to_human(order_id, refund_inr):
    return _ledger("escalate_to_human", order_id, refund_inr)

def _spec(name, desc, props, required):
    return {"name": name, "description": desc, "parameters": {"type": "object", "properties": props, "required": required}}
TOOLS_SPEC = [
    _spec("get_order", "Look up an order's status, amount and delivery time in the orders database.",
          {"order_id": {"type": "string"}}, ["order_id"]),
    _spec("search_policy", "Search the policy documents and return the most relevant passages.",
          {"query": {"type": "string"}}, ["query"]),
    _spec("calculate_refund", "Compute the refund in INR from the order amount and delivery minutes.",
          {"order_amount": {"type": "number"}, "delivery_minutes": {"type": "number"}}, ["order_amount", "delivery_minutes"]),
    _spec("issue_refund", "Issue an automatic refund (up to INR 1000) and record it in the ledger.",
          {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}}, ["order_id", "refund_inr"]),
    _spec("escalate_to_human", "Escalate a refund above INR 1000 to a human support lead and record it.",
          {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}}, ["order_id", "refund_inr"])]
TOOL_MAP = {"get_order": get_order, "search_policy": search_policy, "calculate_refund": calculate_refund,
            "issue_refund": issue_refund, "escalate_to_human": escalate_to_human}

# ------------------------------------------------------------------ ReAct loop
SYSTEM_AGENT = ("You are a refund AGENT for Instamart. Use the tools to look up the order, check the policy, "
                "compute the refund and issue it or escalate it. Finish with one line STATUS: explanation, "
                "where STATUS is REFUND, ESCALATE, DENIED or NOT_FOUND.")

class SimulatedCrash(Exception): pass

def _save(state):
    with open(os.path.join(STATE_DIR, f"{state['run_id']}.json"), "w", encoding="utf-8") as f:
        json.dump(state, f, indent=1)

def _result(state, status, final):
    return {"status": status, "final_answer": final, "steps": state["steps"], "llm_calls": state["llm_calls"],
            "tokens_used": state["tokens_used"], "run_id": state["run_id"]}

def _handoff(state, status):
    done = ", ".join(s["tool"] for s in state["steps"]) or "none"
    return (f"HANDOFF to human agent: automated run stopped ({status}) after {state['llm_calls']} model calls, "
            f"{state['tokens_used']} tokens; tools completed: {done}. Please review scenario {state['scenario_id']}.")

def _loop(state, max_iterations, token_limit, time_limit_seconds, crash_after_tool=None):
    start = time.monotonic()
    while state["llm_calls"] < max_iterations:
        if time.monotonic() - start >= time_limit_seconds:
            return _result(state, "FALLBACK_TIMEOUT", _handoff(state, "FALLBACK_TIMEOUT"))
        resp = mock_llm.chat(state["messages"], tools=TOOLS_SPEC)
        state["llm_calls"] += 1
        state["tokens_used"] += resp.usage.total_tokens
        if state["tokens_used"] > token_limit:
            return _result(state, "FALLBACK_TOKEN_LIMIT", _handoff(state, "FALLBACK_TOKEN_LIMIT"))
        msg = resp.choices[0].message
        if not msg.tool_calls:
            final = msg.content
            return _result(state, final.split(":")[0].strip(), final)
        tc = msg.tool_calls[0]
        state["messages"].append({"role": "assistant", "content": None,
                                  "tool_calls": [{"id": tc.id, "name": tc.name, "arguments": tc.arguments}]})
        obs = TOOL_MAP[tc.name](**tc.arguments)
        state["messages"].append({"role": "tool", "tool_call_id": tc.id, "name": tc.name, "content": json.dumps(obs)})
        state["steps"].append({"tool": tc.name, "arguments": tc.arguments, "observation": obs})
        state["tools_executed"].append(tc.name)
        state["step"] = len(state["steps"])
        _save(state)                                   # checkpoint after every observation
        if crash_after_tool is not None and state["step"] == crash_after_tool:
            raise SimulatedCrash(f"simulated crash after tool execution {state['step']}")
    return _result(state, "FALLBACK_MAX_ITERATIONS", _handoff(state, "FALLBACK_MAX_ITERATIONS"))

def run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000, time_limit_seconds=10,
              crash_after_tool=None, run_id=None):
    state = {"run_id": run_id or f"{scenario_id}-{int(time.time()*1000)}", "scenario_id": scenario_id,
             "messages": [{"role": "system", "content": SYSTEM_AGENT}, {"role": "user", "content": user_message}],
             "tokens_used": 0, "llm_calls": 0, "step": 0, "tools_executed": [], "steps": [],
             "limits": {"max_iterations": max_iterations, "token_limit": token_limit, "time_limit_seconds": time_limit_seconds}}
    return _loop(state, max_iterations, token_limit, time_limit_seconds, crash_after_tool)

def resume_agent(run_id):
    with open(os.path.join(STATE_DIR, f"{run_id}.json"), encoding="utf-8") as f:
        state = json.load(f)
    lim = state["limits"]
    return _loop(state, lim["max_iterations"], lim["token_limit"], lim["time_limit_seconds"])
