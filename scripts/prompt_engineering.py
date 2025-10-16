#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LLM-driven syntactic parsing over CoNLL-U splits (train/dev/test) + EVALUATION.

- Reads:  <input_dir>/{train,dev,test}.conllu
- For dev/test, constructs prompt-engineered requests to produce UD-style analyses.
- Optional few-shot: sample K sentences from train as in-context exemplars.
- Optional keep-tokenization: force the LLM to keep exactly the same token forms as input.
- Validates structure, ID/HEAD ranges, optional token match, single root preference.

NEW: Evaluation metrics (written per split to JSON/TXT):
    - UPOS accuracy
    - Lemma accuracy
    - UAS
    - LAS
    - FEATS micro P/R/F1 (features parsed as key=val pairs)
    - Coverage stats (#sents, #tokens evaluated; skipped due to invalid/mismatch)

Outputs (append-safe parsing logs + evaluated summaries):
    <out_dir>/<split>.jsonl         # per-sentence logs: prompt, response, validity
    <out_dir>/<split>.llm.conllu    # only VALID blocks
    <out_dir>/<split>.metrics.json  # metrics
    <out_dir>/<split>.metrics.txt   # pretty summary

Environment:
  pip install openai>=1.0.0
  export OPENAI_API_KEY=sk-...

Example:
  python conllu_llm_parse.py \
    --input_dir data/wolof \
    --out_dir outputs/wolof \
    --model gpt-4o-mini \
    --few_shot_k 6 \
    --lang_hint "Wolof" \
    --keep_tokenization true \
    --ignore_punct true \
    --temperature 0.0 \
    --sleep 0.2 \
    --verbose
"""

import os
import re
import sys
import json
import time
import random
import argparse
from typing import List, Dict, Tuple, Optional

# -------------------- OpenAI client --------------------
def _get_openai():
    try:
        from openai import OpenAI
        return OpenAI()
    except Exception as e:
        raise RuntimeError("Install openai>=1.0.0 and set OPENAI_API_KEY") from e

# -------------------- CoNLL-U utilities --------------------
SENT_SPLIT_RE = re.compile(r"\n\s*\n", re.MULTILINE)
TAB = "\t"

def read_conllu_sentences(path: str) -> List[Dict]:
    """
    Parse a CoNLL-U file into a list of sentence dicts:
      {
        "sent_id": str | None,
        "text": str | None,
        "text_en": str | None (if present),
        "raw_block": str,
        "tokens": [ (id, form) ... ]  # id:int, form:str (skipping ranges/decimals)
      }
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()

    if not raw:
        return []

    chunks = re.split(SENT_SPLIT_RE, raw + "\n\n")
    sents: List[Dict] = []
    for ch in chunks:
        ch = ch.strip()
        if not ch:
            continue
        lines = ch.splitlines()
        sent_id = None
        text = None
        text_en = None
        tokens = []
        for ln in lines:
            if ln.startswith("# sent_id"):
                sent_id = ln.split("=", 1)[1].strip() if "=" in ln else None
            elif ln.startswith("# text_en"):
                text_en = ln.split("=", 1)[1].strip() if "=" in ln else None
            elif ln.startswith("# text"):
                if not ln.startswith("# text_en"):
                    text = ln.split("=", 1)[1].strip() if "=" in ln else None
            elif ln and not ln.startswith("#"):
                # skip ranges (1-2) and decimal IDs (3.1 for empty nodes/mwt)
                first = ln.split("\t", 1)[0]
                if "." in first or "-" in first:
                    continue
                try:
                    tid = int(first)
                except ValueError:
                    continue
                cols = ln.split("\t")
                if len(cols) >= 2:
                    tokens.append((tid, cols[1]))
        sents.append({
            "sent_id": sent_id,
            "text": text,
            "text_en": text_en,
            "raw_block": ch,
            "tokens": sorted(tokens, key=lambda x: x[0])
        })
    return sents

def sample_few_shot(train_sents: List[Dict], k: int) -> List[str]:
    if k <= 0 or not train_sents:
        return []
    chosen = random.sample(train_sents, min(k, len(train_sents)))
    return [s["raw_block"].strip() for s in chosen]

# -------------------- Prompt engineering --------------------
SYSTEM_PROMPT = """You are a syntactic parser and formatter.
Given a sentence (and optional English translation), output a Universal Dependencies-style analysis.

Output STRICTLY as:
1) Three header lines:
   # sent_id = <ID>
   # text = <original sentence>
   # text_en = <English translation or leave the original if not given>
2) Then a token table with EXACTLY 10 TAB-separated columns per row:
   ID  FORM  LEMMA  UPOS  FEATS  HEAD  DEPREL  DEPS  MISC  Gloss=<english gloss>

Rules:
- IDs are 1..N with no gaps. Use 0 only as HEAD for the root token.
- If unknown, write underscore `_`.
- UPOS must be universal POS (NOUN, VERB, ADJ, ADV, PRON, DET, ADP, AUX, PROPN, NUM, PART, SCONJ, CCONJ, INTJ, PUNCT, SYM, X).
- DEPREL should be UD-like (root, nsubj/subj, obj/comp:obj, obl/comp:obl, amod/mod, case/udep, cc, conj, advmod, aux, cop, punct, etc.).
- Keep punctuation as separate tokens with UPOS=PUNCT.
- Put any flags (e.g., NoSpaceAfter=Yes) in MISC; place the English gloss in the final column as `Gloss=...`.
- Do NOT include explanations or code fences. Only output the formatted block.
"""

def build_user_prompt(sent: Dict, lang: str, keep_tokenization: bool, exemplars: List[str]) -> str:
    headers = [
        f"# sent_id = {sent.get('sent_id') or 'N/A'}",
        f"# text = {sent.get('text') or ''}",
        f"# text_en = {sent.get('text_en') or ''}",
        "",
        "Instructions:",
        f"- Language: {lang or 'Unknown'}",
        "- Tokenize reasonably for the language.",
        "- Ensure exactly 10 TAB-separated columns per token row.",
        "- Use 0 ONLY as HEAD for the single root token.",
        "- Keep punctuation tokens.",
    ]
    if keep_tokenization and sent["tokens"]:
        forms = [tok for _, tok in sent["tokens"]]
        headers.append("- KEEP THE EXACT TOKENIZATION below (same number/order of tokens and surface forms).")
        headers.append("Tokens:")
        for i, form in enumerate(forms, 1):
            headers.append(f"{i}\t{form}")
    headers.append("- Output only the final formatted block (no commentary).")

    sections = []
    if exemplars:
        sections.append("### Examples")
        for ex in exemplars:
            sections.append(ex.strip())
        sections.append("### Target")
    sections.extend(headers)
    return "\n".join(sections).strip()

# -------------------- Validation --------------------
def validate_block(
    block: str,
    expect_n_tokens: Optional[int],
    expected_forms: Optional[List[str]],
    require_single_root: bool = False,
) -> Tuple[bool, Optional[str]]:
    if not block.strip().startswith("# sent_id ="):
        return False, "Missing `# sent_id` header."
    lines = [ln for ln in block.strip().splitlines() if ln.strip()]
    if len(lines) < 4:
        return False, "Too few lines."
    tok_lines = lines[3:]
    ids, heads, forms = [], [], []
    root_count = 0
    for ln in tok_lines:
        cols = ln.split(TAB)
        if len(cols) != 10:
            return False, f"Row must have 10 columns: `{ln}`"
        try:
            tid = int(cols[0]); head = int(cols[5])
        except ValueError:
            return False, f"ID/HEAD not int in row: `{ln}`"
        ids.append(tid); heads.append(head); forms.append(cols[1])
        if cols[6].strip().lower() == "root":
            root_count += 1
    if not ids:
        return False, "No token rows."
    n = len(ids)
    if ids != list(range(1, n + 1)):
        return False, f"IDs not contiguous 1..N: {ids}"
    if any(h < 0 or h > n for h in heads):
        return False, f"HEAD out of range 0..{n}"
    if expect_n_tokens is not None and n != expect_n_tokens:
        return False, f"Token count mismatch. Expected {expect_n_tokens}, got {n}"
    if expected_forms is not None and forms != expected_forms:
        return False, "Surface forms mismatch with required tokenization."
    if require_single_root and root_count != 1:
        return False, f"Require exactly 1 root, found {root_count}"
    return True, None

# -------------------- Parsing helpers for evaluation --------------------
def parse_conllu_table(block: str) -> List[Dict]:
    """
    Parse our 10-col table from a model block:
    0:ID 1:FORM 2:LEMMA 3:UPOS 4:FEATS 5:HEAD 6:DEPREL 7:DEPS 8:MISC 9:Gloss=...
    Returns list of dict per token.
    """
    lines = [ln for ln in block.strip().splitlines() if ln.strip()]
    tok_lines = lines[3:]  # after 3 headers
    rows = []
    for ln in tok_lines:
        cols = ln.split(TAB)
        if len(cols) != 10:
            raise ValueError(f"Expected 10 columns, got {len(cols)} in `{ln}`")
        rows.append({
            "id": int(cols[0]),
            "form": cols[1],
            "lemma": cols[2],
            "upos": cols[3],
            "feats": cols[4],
            "head": int(cols[5]),
            "deprel": cols[6],
            # deps, misc, gloss are not needed for core metrics
        })
    return rows

def parse_gold_conllu_rows(raw_block: str) -> List[Dict]:
    """
    Parse a standard UD 10-column CoNLL-U block (gold).
    Returns only integer-ID rows (skips ranges and decimal IDs).
    Gold columns (UD): ID FORM LEMMA UPOS XPOS FEATS HEAD DEPREL DEPS MISC
    """
    rows = []
    for ln in raw_block.splitlines():
        if not ln or ln.startswith("#"):
            continue
        first = ln.split("\t", 1)[0]
        if "." in first or "-" in first:
            continue
        cols = ln.split("\t")
        if len(cols) < 10:
            continue
        try:
            tid = int(cols[0]); head = int(cols[6])
        except Exception:
            continue
        rows.append({
            "id": tid,
            "form": cols[1],
            "lemma": cols[2],
            "upos": cols[3],
            "feats": cols[5],
            "head": head,
            "deprel": cols[7],
        })
    rows.sort(key=lambda r: r["id"])
    return rows

def parse_feats(feats_str: str) -> List[Tuple[str, str]]:
    """
    Convert FEATS string like 'Case=Nom|Number=Sing' to sorted list of (key,val).
    '_' -> empty list.
    """
    if not feats_str or feats_str == "_":
        return []
    pairs = []
    for item in feats_str.split("|"):
        if not item:
            continue
        if "=" in item:
            k, v = item.split("=", 1)
            pairs.append((k.strip(), v.strip()))
    pairs.sort()
    return pairs

# -------------------- Metrics --------------------
def compute_metrics_for_pair(
    gold_rows: List[Dict],
    pred_rows: List[Dict],
    ignore_punct: bool
) -> Dict:
    """
    Compute per-sentence counts (to be aggregated) for:
      - tokens considered
      - UPOS correct
      - lemma correct
      - UAS (head match)
      - LAS (head+label match)
      - FEATS micro TP/FP/FN (on key=val units)
    """
    n = min(len(gold_rows), len(pred_rows))
    counts = {
        "tokens_considered": 0,
        "upos_correct": 0,
        "lemma_correct": 0,
        "uas_correct": 0,
        "las_correct": 0,
        "feats_tp": 0,
        "feats_fp": 0,
        "feats_fn": 0,
    }
    for i in range(n):
        g, p = gold_rows[i], pred_rows[i]
        # optional punctuation filter by UPOS on gold (common practice)
        if ignore_punct and g["upos"] == "PUNCT":
            continue

        counts["tokens_considered"] += 1

        # UPOS
        if g["upos"] == p["upos"]:
            counts["upos_correct"] += 1

        # Lemma (normalize lower-case)
        if (g["lemma"] or "_").lower() == (p["lemma"] or "_").lower():
            counts["lemma_correct"] += 1

        # UAS / LAS
        if g["head"] == p["head"]:
            counts["uas_correct"] += 1
            if g["deprel"] == p["deprel"]:
                counts["las_correct"] += 1

        # FEATS micro P/R/F1
        g_feats = set(parse_feats(g["feats"]))
        p_feats = set(parse_feats(p["feats"]))
        tp = len(g_feats & p_feats)
        fp = len(p_feats - g_feats)
        fn = len(g_feats - p_feats)
        counts["feats_tp"] += tp
        counts["feats_fp"] += fp
        counts["feats_fn"] += fn

    return counts

"""
def aggregate_counts(counts_list: List[Dict]) -> Dict:
    agg = {
        "tokens_considered": 0,
        "upos_correct": 0,
        "lemma_correct": 0,
        "uas_correct": 0,
        "las_correct": 0,
        "feats_tp": 0,
        "feats_fp": 0,
        "feats_fn": 0,
    }
    for c in counts_list:
        for k in agg:
            agg[k] += c.get(k, 0)
    # derive metrics
    denom = max(agg["tokens_considered"], 1)
    tp, fp, fn = agg["feats_tp"], agg["feats_fp"], agg["feats_fn"]
    upos_acc = agg["upos_correct"] / denom
    lemma_acc = agg["lemma_correct"] / denom
    uas = agg["uas_correct"] / denom
    las = agg["las_correct"] / denom
    tp, fp, fn = agg["feats_tp"], agg["feats_fp"], agg["feats_fn"]
    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    f1   = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
    agg.update({
        "upos_accuracy": upos_acc,
        "lemma_accuracy": lemma_acc,
        "uas"
           "las": las,
        "feats_precision": prec,
        "feats_recall": rec,
        "feats_f1": f1,
    })
    return agg
"""

def aggregate_counts(counts_list: List[Dict]) -> Dict:
    agg = {
        "tokens_considered": 0,
        "upos_correct": 0,
        "lemma_correct": 0,
        "uas_correct": 0,
        "las_correct": 0,
        "feats_tp": 0,
        "feats_fp": 0,
        "feats_fn": 0,
    }
    for c in counts_list:
        for k in agg:
            agg[k] += c.get(k, 0)

    denom = max(agg["tokens_considered"], 1)  # avoid divide-by-zero
    tp, fp, fn = agg["feats_tp"], agg["feats_fp"], agg["feats_fn"]

    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    f1   = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)

    # always add derived metrics
    agg["upos_accuracy"] = agg["upos_correct"] / denom
    agg["lemma_accuracy"] = agg["lemma_correct"] / denom
    agg["uas"] = agg["uas_correct"] / denom
    agg["las"] = agg["las_correct"] / denom
    agg["feats_precision"] = prec
    agg["feats_recall"] = rec
    agg["feats_f1"] = f1

    return agg


def write_metrics(out_dir: str, split: str, metrics: Dict):
    json_path = os.path.join(out_dir, f"{split}.metrics.json")
    txt_path  = os.path.join(out_dir, f"{split}.metrics.txt")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    # Pretty text
    lines = []
    lines.append(f"=== {split.upper()} METRICS ===")
    lines.append(f"Sentences evaluated: {metrics['sentences_evaluated']} / {metrics['sentences_total']}")
    lines.append(f"Tokens considered:   {metrics['tokens_considered']}")
    lines.append("")
    lines.append(f"UPOS accuracy:       {metrics['upos_accuracy']:.4f}")
    lines.append(f"Lemma accuracy:      {metrics['lemma_accuracy']:.4f}")
    lines.append(f"UAS:                 {metrics['uas']:.4f}")
    lines.append(f"LAS:                 {metrics['las']:.4f}")
    lines.append(f"FEATS precision:     {metrics['feats_precision']:.4f}")
    lines.append(f"FEATS recall:        {metrics['feats_recall']:.4f}")
    lines.append(f"FEATS F1:            {metrics['feats_f1']:.4f}")
    lines.append("")
    lines.append(f"Skipped sentences (invalid/mismatch): {metrics['sentences_skipped']}")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

# -------------------- I/O helpers --------------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def write_jsonl(path: str, obj: Dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def write_block(path: str, block: str):
    with open(path, "a", encoding="utf-8") as f:
        f.write(block.rstrip() + "\n\n")

# -------------------- Main processing --------------------
def process_split(
    split_name: str,
    sents: List[Dict],
    out_dir: str,
    model: str,
    temperature: float,
    max_tokens: int,
    few_shot_examples: List[str],
    lang_hint: str,
    keep_tokenization: bool,
    ignore_punct: bool,
    sleep_s: float,
    dry_run: bool,
    verbose: bool,
):
    jsonl_path = os.path.join(out_dir, f"{split_name}.jsonl")
    conllu_path = os.path.join(out_dir, f"{split_name}.llm.conllu")

    client = None if dry_run else _get_openai()

    # For evaluation aggregation
    per_sent_counts: List[Dict] = []
    sentences_evaluated = 0
    sentences_skipped = 0

    for idx, sent in enumerate(sents, 1):
        expect_n = len(sent["tokens"]) if (keep_tokenization and sent["tokens"]) else None
        expected_forms = [f for _, f in sent["tokens"]] if (keep_tokenization and sent["tokens"]) else None

        user_msg = build_user_prompt(
            sent=sent,
            lang=lang_hint,
            keep_tokenization=keep_tokenization,
            exemplars=few_shot_examples
        )

        if dry_run:
            raw = "## DRY-RUN ##\n" + user_msg + "\n## END DRY-RUN ##"
        else:
            resp = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=max_tokens,
            )
            raw = (resp.choices[0].message.content or "").strip()

        start = raw.find("# sent_id")
        block = raw[start:].strip() if start >= 0 else raw.strip()

        ok, msg = validate_block(
            block=block,
            expect_n_tokens=expect_n,
            expected_forms=expected_forms,
            require_single_root=False
        )

        record = {
            "split": split_name,
            "idx": idx,
            "sent_id": sent.get("sent_id"),
            "text": sent.get("text"),
            "text_en": sent.get("text_en"),
            "prompt": user_msg,
            "response": block,
            "valid": ok,
            "validation_msg": msg,
        }
        write_jsonl(jsonl_path, record)
        if ok:
            write_block(conllu_path, block)

        if verbose:
            tag = "OK" if ok else f"INVALID ({msg})"
            print(f"[{split_name} {idx}/{len(sents)}] {sent.get('sent_id') or ''} -> {tag}")

        # ---- Evaluation (only if valid and tokenization aligned if requested) ----
        try:
            if ok:
                gold_rows = parse_gold_conllu_rows(sent["raw_block"])
                pred_rows = parse_conllu_table(block)
                # Evaluate only if lengths align (safety)
                if len(gold_rows) == len(pred_rows):
                    counts = compute_metrics_for_pair(
                        gold_rows=gold_rows,
                        pred_rows=pred_rows,
                        ignore_punct=ignore_punct
                    )
                    per_sent_counts.append(counts)
                    sentences_evaluated += 1
                else:
                    sentences_skipped += 1
            else:
                sentences_skipped += 1
        except Exception as e:
            sentences_skipped += 1
            if verbose:
                print(f"[{split_name} {idx}] Eval error: {e}", file=sys.stderr)

        if sleep_s > 0:
            time.sleep(sleep_s)

    # Aggregate metrics
    agg = aggregate_counts(per_sent_counts)
    agg.update({
        "sentences_total": len(sents),
        "sentences_evaluated": sentences_evaluated,
        "sentences_skipped": sentences_skipped,
        "ignore_punct": ignore_punct,
    })
    write_metrics(out_dir, split_name, agg)

def main():
    ap = argparse.ArgumentParser(description="LLM UD parsing over CoNLL-U train/dev/test + evaluation.")
    ap.add_argument("--input_dir", required=True, help="Directory with wol.Wolof.train.conllu, wol.Wolof.dev.conllu, wol.Wolof.test.conllu")
    ap.add_argument("--out_dir", required=True, help="Directory to write outputs")
    ap.add_argument("--model", default="gpt-4o-mini", help="OpenAI chat model")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_tokens", type=int, default=1400)
    ap.add_argument("--few_shot_k", type=int, default=0, help="Number of train exemplars for in-context learning")
    ap.add_argument("--lang_hint", default="", help="Language name to include in the prompt (e.g., Swahili)")
    ap.add_argument("--keep_tokenization", type=lambda x: x.lower() in {"1","true","yes"}, default=False)
    ap.add_argument("--ignore_punct", type=lambda x: x.lower() in {"1","true","yes"}, default=True,
                    help="If true, punctuation tokens (gold UPOS=PUNCT) are excluded from metrics.")
    ap.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between requests")
    ap.add_argument("--dry_run", type=lambda x: x.lower() in {"1","true","yes"}, default=False)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed)
    ensure_dir(args.out_dir)

    paths = {
        "train": os.path.join(args.input_dir, "wol.Wolof.train.conllu"),
        "dev":   os.path.join(args.input_dir, "wol.Wolof.dev.conllu"),
        "test":  os.path.join(args.input_dir, "wol.Wolof.test.conllu"),
    }
    for k, p in paths.items():
        if k in {"dev", "test"} and not os.path.exists(p):
            print(f"Warning: {k}.conllu not found at {p}. Skipping.")
    train_sents = read_conllu_sentences(paths["train"]) if os.path.exists(paths["train"]) else []
    dev_sents   = read_conllu_sentences(paths["dev"]) if os.path.exists(paths["dev"]) else []
    test_sents  = read_conllu_sentences(paths["test"]) if os.path.exists(paths["test"]) else []

    few_shot_examples = sample_few_shot(train_sents, args.few_shot_k)

    if dev_sents:
        process_split(
            split_name="wol.Wolof.dev",
            sents=dev_sents,
            out_dir=args.out_dir,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            few_shot_examples=few_shot_examples,
            lang_hint=args.lang_hint,
            keep_tokenization=args.keep_tokenization,
            ignore_punct=args.ignore_punct,
            sleep_s=args.sleep,
            dry_run=args.dry_run,
            verbose=args.verbose,
        )
    if test_sents:
        process_split(
            split_name="wol.Wolof.test",
            sents=test_sents,
            out_dir=args.out_dir,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            few_shot_examples=few_shot_examples,
            lang_hint=args.lang_hint,
            keep_tokenization=args.keep_tokenization,
            ignore_punct=args.ignore_punct,
            sleep_s=args.sleep,
            dry_run=args.dry_run,
            verbose=args.verbose,
        )

    if args.verbose:
        print("Done.")

if __name__ == "__main__":
    main()
