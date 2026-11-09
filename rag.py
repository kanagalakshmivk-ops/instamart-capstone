"""Part D tasks 1-4: chunking, TF-IDF retrieval, grounded answer, evaluation."""
import glob, os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import mock_llm

POLICY_DIR = os.path.join("data", "policies")

def chunk_words(words, size=40, overlap=10):
    step, chunks, start = size - overlap, [], 0
    while True:
        end = start + size
        chunks.append(words[start:end])
        if end >= len(words): break
        start += step
    return chunks

def build_chunks(size=40, overlap=10):
    out = []
    for path in sorted(glob.glob(os.path.join(POLICY_DIR, "*.txt"))):
        words = open(path, encoding="utf-8").read().split()
        for c in chunk_words(words, size, overlap):
            out.append({"source": os.path.basename(path), "text": " ".join(c)})
    return out

class Retriever:
    def __init__(self, size=40, overlap=10):
        self.chunks = build_chunks(size, overlap)
        self.vec = TfidfVectorizer()
        self.matrix = self.vec.fit_transform([c["text"] for c in self.chunks])
    def retrieve(self, query, k=3):
        sims = cosine_similarity(self.vec.transform([query]), self.matrix)[0]
        top = sims.argsort()[::-1][:k]
        return [{"source": self.chunks[i]["source"], "score": float(sims[i]), "text": self.chunks[i]["text"]} for i in top]

DEFAULT = Retriever()
def retrieve(query, k=3): return DEFAULT.retrieve(query, k)

SYSTEM_RAG = ("You are a POLICY ASSISTANT. Answer only from the supplied CONTEXT. "
              "If the context does not contain the answer, say that you could not find the information.")

def answer(question, retriever=None):
    hits = (retriever or DEFAULT).retrieve(question, 3)
    ctx = "\n".join(f"[{h['source']}]\n{h['text']}" for h in hits)
    user = f"CONTEXT:\n{ctx}\nQUESTION:\n{question}"
    resp = mock_llm.chat([{"role": "system", "content": SYSTEM_RAG}, {"role": "user", "content": user}])
    return resp.choices[0].message.content, [h["source"] for h in hits]

def evaluate(retriever=None, verbose=True):
    ev = pd.read_csv("data/raw/policy_eval.csv", keep_default_na=False)
    ans = ev[ev.expected_doc != "NONE"]; unans = ev[ev.expected_doc == "NONE"]
    doc_hits = fact_hits = fb = 0; failures = []
    for _, r in ans.iterrows():
        text, sources = answer(r.question, retriever)
        d, f = r.expected_doc in sources, r.expected_fact in text
        doc_hits += d; fact_hits += f
        if not (d and f): failures.append((r.question, d, f))
    for _, r in unans.iterrows():
        text, _ = answer(r.question, retriever); fb += text.startswith("I could not find")
    return dict(doc=f"{doc_hits}/{len(ans)}", fact=f"{fact_hits}/{len(ans)}", fallback=f"{fb}/{len(unans)}"), failures
