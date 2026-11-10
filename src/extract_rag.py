"""
Step 1 - RAG extraction over paper PDFs in data/papers/.

Supports:
1. Google Gemini API (Free tier from https://aistudio.google.com/apikey) via GEMINI_API_KEY / GOOGLE_API_KEY
2. Anthropic Claude via ANTHROPIC_API_KEY

Per paper: parse PDF -> chunk -> embed (local) -> retrieve passages about the five
inputs + removal -> LLM returns rows as JSON -> clean -> data/raw/psh_rag.csv

Results are cached per paper in data/papers_cache/, so re-runs only process new PDFs.
"""
import os, re, json, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
PAPERS = ROOT / "data" / "papers"
CACHE = ROOT / "data" / "papers_cache"
OUT = ROOT / "data" / "raw" / "psh_rag.csv"

COLS = ["pH", "dosage_g_L", "conc_mg_L", "time_min", "temp_C",
        "removal_pct", "dye", "adsorbent", "source_doi", "origin"]
NUM = ["pH", "dosage_g_L", "conc_mg_L", "time_min", "temp_C", "removal_pct"]
QUERIES = [
    "table of experimental conditions pH dosage contact time temperature initial concentration removal %",
    "effect of adsorbent dose g/L on dye removal efficiency",
    "effect of pH on dye removal percentage",
    "effect of contact time and initial dye concentration",
    "experimental design Box-Behnken central composite runs table",
    "adsorption capacity qe mg/g adsorbent dosage temperature",
]
SYSTEM = """You extract experimental data from dye adsorption papers.
Return ONLY a JSON list (no prose). Each item has keys:
pH, dosage_g_L, conc_mg_L, time_min, temp_C, removal_pct, dye, adsorbent, origin.
Rules:
- Only rows for pumpkin seed husk / hull / shell adsorbents (Cucurbita pepo).
- Convert dosage to g/L (g per 100 mL = x10; mg per 50 mL = x0.02).
- If only qe (mg/g) is given: removal_pct = qe * dosage_g_L / conc_mg_L * 100.
- Use null for any value not stated in the text. Do NOT guess or fill in defaults.
  If a condition is stated once as fixed for a whole experiment (e.g. 'all runs at 30 C'), you may reuse it.
- origin = "table" if the number is in a table or the running text,
  "graph_estimate" if you had to infer it from a figure description.
- If nothing is extractable, return []."""


def load_env():
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"'))


def chunks_of(pdf, size=1500, overlap=250):
    import fitz
    text = "\n".join(p.get_text() for p in fitz.open(pdf))
    if len(text.strip()) < 500:
        return []
    step = size - overlap
    return [text[i:i + size] for i in range(0, len(text), step)]


def retrieve(emb, chunks, k=4):
    import numpy as np
    c = emb.encode(chunks, normalize_embeddings=True)
    q = emb.encode(QUERIES, normalize_embeddings=True)
    top = set()
    for s in q @ c.T:
        top.update(np.argsort(-s)[:k].tolist())
    return [chunks[i] for i in sorted(top)]


def extract_gemini(client, model_name, context, doi):
    prompt = f"{SYSTEM}\n\nPaper DOI: {doi}\n\nExcerpts:\n" + "\n---\n".join(context)
    try:
        response = client.generate_content(prompt)
    except Exception as e:
        if "429" in str(e) or "quota" in str(e).lower():
            import google.generativeai as genai
            alt = genai.GenerativeModel("gemini-3.5-flash")
            response = alt.generate_content(prompt)
        else:
            return None
    txt = response.text
    txt = re.sub(r"```json|```", "", txt).strip()
    try:
        data = json.loads(txt)
    except json.JSONDecodeError:
        m = re.search(r"\[\s*\{.*\}\s*\]", txt, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(0))
            except Exception:
                return None
        else:
            return None
    return data if isinstance(data, list) else None


def extract_anthropic(client, context, doi):
    msg = client.messages.create(
        model="claude-sonnet-5-5", max_tokens=4000, system=SYSTEM,
        messages=[{"role": "user",
                   "content": f"Paper DOI: {doi}\n\nExcerpts:\n" + "\n---\n".join(context)}],
    )
    txt = re.sub(r"```json|```", "", msg.content[0].text).strip()
    try:
        data = json.loads(txt)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, list) else None


def main():
    load_env()
    gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")

    backend = None
    if gemini_key:
        import google.generativeai as genai
        genai.configure(api_key=gemini_key)
        gemini_model = genai.GenerativeModel("gemini-3.5-flash-lite")
        backend = "gemini"
        print("Using Google Gemini API (gemini-3.5-flash-lite) for extraction.")
    elif anthropic_key:
        import anthropic
        anthropic_client = anthropic.Anthropic(api_key=anthropic_key)
        backend = "anthropic"
        print("Using Anthropic Claude API for extraction.")
    else:
        print("NOTE: No API key found in .env (GEMINI_API_KEY or ANTHROPIC_API_KEY).")
        print("To run AI RAG extraction on new PDFs:")
        print("1. Get a free Gemini key from https://aistudio.google.com/apikey")
        print("2. Put GEMINI_API_KEY=your_key in .env")
        print("Existing compiled data in data/raw/ is already ready for model training.")
        sys.exit(0)

    pdfs = sorted(PAPERS.glob("*.pdf"))
    if not pdfs:
        sys.exit(f"No PDFs found in {PAPERS}. Drop your papers there first.")

    import pandas as pd
    from sentence_transformers import SentenceTransformer
    emb = SentenceTransformer("all-MiniLM-L6-v2")
    CACHE.mkdir(exist_ok=True)

    rows, skipped = [], []
    for i, pdf in enumerate(pdfs, 1):
        doi = pdf.stem.replace("_", "/", 1)
        cache = CACHE / (pdf.stem + ".json")
        if cache.exists():
            got = json.loads(cache.read_text())
        else:
            chunks = chunks_of(pdf)
            if not chunks:
                skipped.append((pdf.name, "no text layer (scanned?) - needs OCR"))
                continue
            ctx = retrieve(emb, chunks)
            if backend == "gemini":
                got = extract_gemini(gemini_model, "gemini-1.5-flash", ctx, doi)
            else:
                got = extract_anthropic(anthropic_client, ctx, doi)
            if got is None:
                skipped.append((pdf.name, "model returned unparsable output"))
                continue
            cache.write_text(json.dumps(got))
        for r in got:
            r["source_doi"] = doi
        rows.extend(got)
        print(f"[{i}/{len(pdfs)}] {pdf.name}: {len(got)} rows")

    df = pd.DataFrame(rows).reindex(columns=COLS)
    for c in NUM:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["dye", "adsorbent"]:
        df[c] = df[c].astype("string").str.strip().str.lower()
    df["origin"] = df["origin"].fillna("graph_estimate")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    five = ["pH", "dosage_g_L", "conc_mg_L", "time_min", "temp_C"]
    complete = df[five + ["removal_pct"]].notna().all(axis=1)
    print(f"\nSaved {len(df)} rows -> {OUT}")
    print(f"Complete rows (all 5 inputs + removal): {int(complete.sum())}")
    if len(df):
        print(df[complete].groupby("dye").size().sort_values(ascending=False).to_string())
    for name, why in skipped:
        print(f"SKIPPED {name}: {why}")
    print("\nNext: check rows against the papers, then run 2_build_dataset_and_train.bat")


if __name__ == "__main__":
    main()
