"""Optional: zero-shot large-language-model (LLM) intent classifier.

Not used for any result reported in the paper. Labels Banking77 queries with an
LLM's zero-shot intent prediction and self-reported confidence. Because the model
returns one intent and one confidence rather than a probability for every intent,
only P0 and P1 routing can be applied; P2 cannot.

Requirements:
    pip install anthropic
    export ANTHROPIC_API_KEY=...
    export LLM_MODEL=<model id>      # e.g. a current Claude model id

Outputs:
    results_llm_zeroshot.csv  (split, text, true_category, llm_intent, llm_confidence)
"""
import json
import os
import re
import time

import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
MODEL = os.environ.get("LLM_MODEL")
SAMPLE_N = None  # None = all queries; set an integer to limit cost

tr = pd.read_csv('data/train.csv')
te = pd.read_csv('data/test.csv')
_, val = train_test_split(tr, test_size=0.20, random_state=SEED, stratify=tr.category)
INTENTS = sorted(tr.category.unique().tolist())

PROMPT = """Classify this customer-support message from a banking app into exactly one
of these {n} intents: {intents}

Message: "{text}"

Respond with only a JSON object:
{{"intent": "<one intent from the list, verbatim>", "confidence": <number from 0 to 1>}}"""


def classify(client, text, retries=3):
    for attempt in range(retries):
        try:
            msg = client.messages.create(
                model=MODEL, max_tokens=100,
                messages=[{"role": "user", "content": PROMPT.format(
                    n=len(INTENTS), intents=", ".join(INTENTS), text=text)}])
            raw = msg.content[0].text
            out = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
            intent = out["intent"] if out["intent"] in INTENTS else None
            return intent, float(out["confidence"])
        except Exception:
            time.sleep(2 ** attempt)
    return None, 0.0  # unparseable answers are treated as low-confidence (sent to a human)


def main():
    if not os.environ.get("ANTHROPIC_API_KEY") or not MODEL:
        raise SystemExit("Set ANTHROPIC_API_KEY and LLM_MODEL first.")
    import anthropic
    client = anthropic.Anthropic()

    rows = []
    for split, df in [("validation", val), ("test", te)]:
        if SAMPLE_N:
            df = df.sample(min(SAMPLE_N, len(df)), random_state=SEED)
        for r in df.itertuples():
            intent, conf = classify(client, r.text)
            rows.append({"split": split, "text": r.text, "true_category": r.category,
                         "llm_intent": intent, "llm_confidence": conf})
    pd.DataFrame(rows).to_csv("results_llm_zeroshot.csv", index=False)
    print("Done. Select tau on the validation rows and apply P0/P1 routing "
          "to the test rows using the functions in main_results.py.")


if __name__ == "__main__":
    main()
