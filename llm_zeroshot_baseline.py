"""Zero-shot LLM baseline — NOT YET RUN.
This sandbox has no API credentials and no outbound access to model-hosting APIs,
so this script could not be executed as part of the automated pipeline. It is provided
complete and ready to run with the user's own API key (Anthropic or OpenAI).
Usage: set ANTHROPIC_API_KEY, then `pip install anthropic` and run this file.
It reproduces the exact P0/P1/P2 routing logic from analysis.py, substituting an LLM's
zero-shot intent prediction + self-reported confidence for the TF-IDF+LR classifier."""
import pandas as pd, json, time, os

INTENTS = sorted(pd.read_csv('data/train.csv').category.unique().tolist())
te = pd.read_csv('data/test.csv')
SAMPLE_N = 500  # adjust as budget allows; full 3,080 for an exact match to Table 3

PROMPT = """You are classifying a customer-support message from a banking app into exactly
one of these {n} intents: {intents}

Message: "{text}"

Respond with ONLY a JSON object: {{"intent": "<one of the intents above, verbatim>", "confidence": <float 0-1, your genuine estimate of P(this is the correct intent)>}}"""

def classify_with_anthropic(text):
    import anthropic
    client = anthropic.Anthropic()
    msg = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=100,
        messages=[{"role": "user", "content": PROMPT.format(n=len(INTENTS), intents=", ".join(INTENTS), text=text)}])
    out = json.loads(msg.content[0].text)
    return out["intent"], float(out["confidence"])

if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("Set ANTHROPIC_API_KEY first. This script was not run in the sandbox that produced this paper.")
    sample = te.sample(SAMPLE_N, random_state=42)
    rows = []
    for _, r in sample.iterrows():
        intent, conf = classify_with_anthropic(r.text)
        rows.append({"text": r.text, "true_category": r.category, "llm_intent": intent, "llm_confidence": conf})
        time.sleep(0.2)
    pd.DataFrame(rows).to_csv("results_llm_zeroshot.csv", index=False)
    print("Done. Feed results_llm_zeroshot.csv through the same route()/metrics() functions in analysis2.py.")
