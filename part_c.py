"""Part C: guarded support-ticket triage. Run: python part_c.py (after part_a.py and part_b.py)"""
import json, random, re, time
import pandas as pd
import mock_llm
from mock_llm import est_tokens, TransientLLMError

MAX_INPUT_TOKENS, BASE_DELAY, MAX_RETRIES = 400, 0.05, 3
CATEGORIES = ["billing", "delivery", "technical", "fraud"]
PATTERNS = [("EMAIL", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"),
            ("PHONE", re.compile(r"\b[6-9]\d{9}\b"), "[PHONE]"),
            ("ORDER_ID", re.compile(r"ORD-\d{6}"), "[ORDER_ID]")]
tickets = pd.read_csv("data/raw/tickets.csv", keep_default_na=False)

# ---------------- redaction ----------------
def redact(text, counter=None):
    for name, pat, ph in PATTERNS:
        text, n = pat.subn(ph, text)
        if counter is not None: counter[name] += n
    return text
from collections import Counter
sub = Counter()
redacted_all = [redact(t, sub) for t in tickets.text]
print("substitutions:", dict(sub), "total", sum(sub.values()))
left = sum(len(p.findall(t)) for t in redacted_all for _, p, _ in PATTERNS)
print("PII patterns left after redaction:", left); assert left == 0
print("empty:", int((tickets.text.str.strip() == "").sum()),
      "| long:", sorted({est_tokens(t) for t in tickets.text if est_tokens(t) > MAX_INPUT_TOKENS}))

# ---------------- prompts ----------------
FORMAT = ('Reply with JSON only, in exactly this form: {"category": "...", "reason": "..."} . '
          "Do not add any other text.")
ZERO_SHOT = ("You CLASSIFY customer support tickets into exactly one category: billing, delivery, technical or fraud. "
             + FORMAT)
FEW_SHOT = ("You CLASSIFY customer support tickets into exactly one category: billing, delivery, technical or fraud. "
            + FORMAT + "\nEXAMPLES:\n"
            'Ticket: Someone used my card without permission, this is an unauthorized transaction. -> {"category": "fraud", "reason": "unauthorized use"}\n'
            'Ticket: The app freezes when I open the cart. -> {"category": "technical", "reason": "app problem"}\n'
            'Ticket: I was charged twice and the invoice is wrong. -> {"category": "billing", "reason": "double charge"}\n'
            'Ticket: My rider never arrived with the order. -> {"category": "delivery", "reason": "order not delivered"}')
assert "CLASSIFY" in ZERO_SHOT and "EXAMPLES:" not in ZERO_SHOT and "EXAMPLES:" in FEW_SHOT

# ---------------- resilient call ----------------
retry_log, prompts_sent = [], []
def call_with_retry(messages, ticket_id):
    for n in range(MAX_RETRIES + 1):
        try:
            return mock_llm.chat(messages)
        except TransientLLMError:
            if n == MAX_RETRIES: raise
            wait = BASE_DELAY * 2 ** n * random.uniform(0.9, 1.15)
            retry_log.append((ticket_id, n, wait))
            print(f"  [retry] ticket={ticket_id} retry={n} wait={wait:.4f}s")
            time.sleep(wait)

# ---------------- validation ----------------
def validate(content):
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        words = set(re.findall(r"[a-z]+", str(content).lower()))
        found = [c for c in CATEGORIES if c in words]
        return ("classified_keyword", found[0]) if len(found) == 1 else ("flagged_invalid_output", "")
    if isinstance(data, dict) and data.get("category") in CATEGORIES:
        return "classified_json", data["category"]
    return "flagged_invalid_output", ""

# ---------------- pipeline ----------------
def run_pipeline(system_prompt, token_budget=None, verbose=False):
    mock_llm.reset(); retry_log.clear(); prompts_sent.clear()
    total, results = 0, []
    for (_, t), red in zip(tickets.iterrows(), redacted_all):
        r = dict(ticket_id=t.ticket_id, customer_id=t.customer_id, status="", category="", tokens_used=0, true=t.true_category)
        if t.text.strip() == "":
            r["status"] = "skipped_empty"
        elif est_tokens(t.text) > MAX_INPUT_TOKENS:
            r["status"] = "flagged_long"
        elif token_budget is not None and total >= token_budget:
            r["status"] = "budget_exhausted"
        else:
            msgs = [{"role": "system", "content": system_prompt}, {"role": "user", "content": red}]
            prompts_sent.append(msgs)
            resp = call_with_retry(msgs, t.ticket_id)
            used = resp.usage.total_tokens; total += used; r["tokens_used"] = used
            r["status"], r["category"] = validate(resp.choices[0].message.content)
        results.append(r)
    return pd.DataFrame(results), total

def report(name, df, total):
    cl = df[df.status.isin(["classified_json", "classified_keyword"])]
    acc = (cl.category == cl.true).mean()
    print(f"\n--- {name}: statuses {df.status.value_counts().to_dict()}")
    print(f"classified {len(cl)} | accuracy {acc*100:.1f}% ({int((cl.category==cl.true).sum())}/{len(cl)}) | "
          f"total tokens {total} | tokens/classified ticket {total/len(cl):.1f} | retries {len(retry_log)}")
    return cl

df_z, tok_z = run_pipeline(ZERO_SHOT)
cl_z = report("zero-shot (no budget)", df_z, tok_z)
print("retry waits all within range:", all(BASE_DELAY*2**n*0.9 <= w <= BASE_DELAY*2**n*1.15 for _, n, w in retry_log))
print("zero-shot confusion (rows = true, cols = predicted):"); print(pd.crosstab(cl_z.true, cl_z.category))
for msgs in prompts_sent:   # privacy assertion on every prompt sent
    u = msgs[1]["content"]
    assert not any(p.search(u) for _, p, _ in PATTERNS)
print("assertion passed: no email/phone/order id in any of", len(prompts_sent), "prompts")
df_f, tok_f = run_pipeline(FEW_SHOT)
report("few-shot (no budget)", df_f, tok_f)

# ---------------- routing ----------------
seg = pd.read_csv("outputs/customer_segments.csv")
seg_name, premium = seg.set_index("customer_id")["segment_name"], set(seg.loc[seg.is_premium, "customer_id"])
def route_df(df):
    out = []
    for _, r in df.iterrows():
        if r.status in ("skipped_empty", "flagged_long", "flagged_invalid_output", "budget_exhausted"):
            route, pr = "human_review", "NORMAL"
        elif r.category == "fraud": route, pr = "security_team", "HIGH"
        elif r.customer_id in premium: route, pr = "premium_desk", "HIGH"
        else: route, pr = "support_queue", "NORMAL"
        out.append(dict(ticket_id=r.ticket_id, customer_id=r.customer_id, status=r.status, category=r.category,
                        segment_name=seg_name.get(r.customer_id, ""), priority=pr, route=route, tokens_used=r.tokens_used))
    return pd.DataFrame(out)

BUDGET_DEMO = 6000
df_b, _ = run_pipeline(ZERO_SHOT, token_budget=BUDGET_DEMO)
route_df(df_b).to_csv("outputs/triage_budget_demo.csv", index=False)
print(f"\nbudget demo (zero-shot, TOKEN_BUDGET={BUDGET_DEMO}): budget_exhausted = {(df_b.status=='budget_exhausted').sum()}")
df_final, _ = run_pipeline(FEW_SHOT, token_budget=100000)
final = route_df(df_final); final.to_csv("outputs/triage_results.csv", index=False)
print("final results rows:", len(final)); print(final.route.value_counts().to_dict())
print("zero-shot est. tokens of system prompt:", est_tokens(ZERO_SHOT), "| few-shot:", est_tokens(FEW_SHOT))
