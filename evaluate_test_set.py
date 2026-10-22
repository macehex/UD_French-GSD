"""
Evaluate the French agreement checker on the UD French-GSD test set.

Three tests, run in this order:
  1. Stanza accuracy (UPOS / Lemma / UAS / LAS) on gold tokens
  2. False positives: run the checker on clean test sentences (ideally flags nothing)
  3. Synthetic errors: corrupt test sentences with a number error, then measure
     how many the checker detects and how many it fixes exactly

Needs, in the same folder:
  fr_gsd-ud-train.conllu   (to build the inflection lexicon)
  fr_gsd-ud-test.conllu    (the test set)

Install: pip install stanza conllu
"""

import gc
import random
from collections import defaultdict, Counter

import stanza
import torch
from conllu import parse_incr

# ----------------------------- CONFIG -----------------------------
TRAIN_PATH = "fr_gsd-ud-train.conllu"
TEST_PATH = "fr_gsd-ud-test.conllu"
N_SENTENCES = 300       # set to None to use the whole test set
USE_GPU = True          # set False if you still run out of GPU memory (3.6 GB is tight)
CHUNK = 50              # sentences per Stanza batch when scoring
SEED = 0
# ------------------------------------------------------------------

random.seed(SEED)


def free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =================== 1. DATA + INFLECTION LEXICON ===================

def load_sentences(path, limit=None):
    sents = []
    with open(path, encoding="utf-8") as f:
        for sent in parse_incr(f):
            if "text" in sent.metadata:
                sents.append(sent)
            if limit and len(sents) >= limit:
                break
    return sents


def build_lexicon(path):
    """(lemma, upos) -> Counter{(feats, form): count}"""
    lex = defaultdict(Counter)
    with open(path, encoding="utf-8") as f:
        for sent in parse_incr(f):
            for t in sent:
                if not isinstance(t["id"], int):      # skip multiword ranges (du, au...)
                    continue
                feats = tuple(sorted((t["feats"] or {}).items()))
                lex[(t["lemma"], t["upos"])][(feats, t["form"])] += 1
    return lex


# =================== 2. THE CHECKER (rules A and B) ===================

def inflect(lex, lemma, upos, target):
    """Pick the known form whose features best match `target` (Number must match)."""
    best, best_key = None, (-1, -1)
    for (items, form), count in lex.get((lemma, upos), {}).items():
        d = dict(items)
        if d.get("Number") != target.get("Number"):
            continue
        score = sum(d.get(k) == v for k, v in target.items())
        if (score, count) > best_key:
            best, best_key = form, (score, count)
    return best


def feats(w):
    return dict(p.split("=") for p in w.feats.split("|")) if w.feats else {}


def match_case(src, new):
    return new.capitalize() if src[:1].isupper() else new


def span(w):
    """Character offsets of a word in the original text (None for du/au-style contractions)."""
    tok = w.parent
    if len(tok.words) > 1:
        return None
    return tok.start_char, tok.end_char


def check(nlp, lex, text):
    """Returns list of (start, end, fix, message, alt)."""
    doc, issues = nlp(text), []
    for sent in doc.sentences:
        words = sent.words
        for w in words:
            if w.head == 0:
                continue
            head, wf, hf = words[w.head - 1], feats(w), feats(words[w.head - 1])

            # Rule A: determiner / adjective vs noun
            if w.deprel in ("det", "amod") and head.upos in ("NOUN", "PROPN"):
                target = dict(wf)
                for k in ("Number", "Gender"):
                    if k in hf and k in wf and hf[k] != wf[k]:
                        target[k] = hf[k]
                if target != wf:
                    fix = inflect(lex, w.lemma, w.upos, target)
                    sp = span(w)
                    if fix and sp and fix.lower() != w.text.lower():
                        issues.append((sp[0], sp[1], match_case(w.text, fix),
                                       f"'{w.text}' should agree with '{head.text}'", None))

            # Rule B: verb vs subject
            if w.deprel == "nsubj" and head.upos in ("VERB", "AUX"):
                if any(x.head == w.id and x.deprel == "conj" for x in words):
                    continue
                vf = feats(head)
                target = dict(vf)
                if "Number" in wf and wf["Number"] != vf.get("Number"):
                    target["Number"] = wf["Number"]
                    if w.upos == "PRON" and "Person" in wf:
                        target["Person"] = wf["Person"]
                if target != vf and vf.get("VerbForm") == "Fin":
                    verb_fix = inflect(lex, head.lemma, head.upos, target)
                    vsp = span(head)

                    subj_fix, ssp = None, span(w)
                    if w.upos == "PRON" and "Number" in vf:
                        subj_fix = inflect(lex, w.lemma, w.upos, {**wf, "Number": vf["Number"]})

                    if verb_fix and vsp and verb_fix.lower() != head.text.lower():
                        alt = None
                        if subj_fix and ssp and subj_fix.lower() != w.text.lower():
                            alt = (ssp[0], ssp[1], match_case(w.text, subj_fix))
                        issues.append((vsp[0], vsp[1], verb_fix,
                                       f"'{head.text}' should agree with subject '{w.text}'", alt))
    return issues


def apply_edits(text, issues):
    # right-to-left so earlier offsets stay valid
    for start, end, fix, *_ in sorted(issues, key=lambda x: x[0], reverse=True):
        text = text[:start] + fix + text[end:]
    return text


# =================== 3. TEST A: STANZA ACCURACY ===================

def eval_stanza(test_sents):
    print("\n=== Test 1: Stanza accuracy on gold tokens ===")
    nlp_gold = stanza.Pipeline(
        "fr",
        processors="tokenize,pos,lemma,depparse",
        tokenize_pretokenized=True,
        use_gpu=USE_GPU,
        pos_batch_size=500,
        lemma_batch_size=200,
        depparse_batch_size=200,
        depparse_min_length_to_batch_separately=100,
        verbose=False,
    )

    # syntactic words only (skip multiword ranges like du/au)
    gold_sents = [[t for t in s if isinstance(t["id"], int)] for s in test_sents]

    pred_sents = []
    for i in range(0, len(gold_sents), CHUNK):
        chunk = [[t["form"] for t in s] for s in gold_sents[i:i + CHUNK]]
        pred_sents.extend(nlp_gold(chunk).sentences)
        free_gpu()
    assert len(pred_sents) == len(gold_sents)

    def base(rel):                      # "nsubj:pass" -> "nsubj"
        return (rel or "").split(":")[0]

    total = upos_ok = lemma_ok = uas_ok = las_ok = 0
    for gold, pred in zip(gold_sents, pred_sents):
        for g, p in zip(gold, pred.words):
            total += 1
            upos_ok += g["upos"] == p.upos
            lemma_ok += g["lemma"] == p.lemma
            head_ok = g["head"] == p.head
            uas_ok += head_ok
            las_ok += head_ok and base(g["deprel"]) == base(p.deprel)

    print(f"Words scored: {total}")
    print(f"UPOS : {upos_ok / total:.3f}")
    print(f"Lemma: {lemma_ok / total:.3f}")
    print(f"UAS  : {uas_ok / total:.3f}   (correct head)")
    print(f"LAS  : {las_ok / total:.3f}   (correct head + relation)")

    # release this pipeline before loading the next one
    del nlp_gold, pred_sents
    free_gpu()


# =================== 4. TEST B: FALSE POSITIVES ===================

def eval_clean(nlp, lex, test_sents):
    print("\n=== Test 2: false positives on clean sentences ===")
    flagged, examples = 0, []
    for s in test_sents:
        text = s.metadata["text"]
        issues = check(nlp, lex, text)
        if issues:
            flagged += 1
            examples.append((text, [i[3] for i in issues]))
    print(f"Clean sentences flagged: {flagged}/{len(test_sents)} ({flagged / len(test_sents):.1%})")
    return examples


# =================== 5. TEST C: SYNTHETIC ERRORS ===================

def flip_number(lex, lemma, upos, f):
    """Find a known form with identical features except opposite Number."""
    want = dict(f)
    if want.get("Number") == "Sing":
        want["Number"] = "Plur"
    elif want.get("Number") == "Plur":
        want["Number"] = "Sing"
    else:
        return None
    cands = [(fm, c) for (items, fm), c in lex.get((lemma, upos), {}).items()
             if dict(items) == want]
    if not cands:
        return None
    return max(cands, key=lambda x: x[1])[0]


def corrupt(lex, sent):
    """Return (corrupted_text, original_text) with ONE number error, or None."""
    text = sent.metadata["text"]

    covered, units, cursor = set(), [], 0
    for t in sent:
        if isinstance(t["id"], tuple):                      # multiword range, e.g. du = de + le
            covered.update(range(t["id"][0], t["id"][2] + 1))
    for t in sent:
        if isinstance(t["id"], int) and t["id"] in covered:
            continue
        pos = text.find(t["form"], cursor)
        if pos == -1:
            return None
        cursor = pos + len(t["form"])
        units.append((t, pos, cursor))

    options = []
    for t, s, e in units:
        if isinstance(t["id"], int) and t["upos"] in ("DET", "ADJ", "VERB", "AUX"):
            new = flip_number(lex, t["lemma"], t["upos"], t["feats"] or {})
            if new and new.lower() != t["form"].lower():
                options.append((s, e, match_case(t["form"], new)))
    if not options:
        return None
    s, e, new = random.choice(options)
    return text[:s] + new + text[e:], text


def eval_synthetic(nlp, lex, test_sents):
    print("\n=== Test 3: synthetic number errors ===")
    n = detected = fixed = 0
    misses = []
    for sent in test_sents:
        c = corrupt(lex, sent)
        if not c:
            continue
        bad, orig = c
        n += 1
        issues = check(nlp, lex, bad)
        if issues:
            detected += 1
        if apply_edits(bad, issues) == orig:
            fixed += 1
        else:
            misses.append((bad, orig))
    print(f"Corrupted sentences: {n}")
    print(f"Error detected:      {detected}/{n} ({detected / n:.1%})")
    print(f"Exactly fixed:       {fixed}/{n} ({fixed / n:.1%})")
    return misses


# =================== MAIN ===================

def main():
    try:
        stanza.download("fr", verbose=False)
    except Exception as e:
        print("stanza.download skipped:", e)

    test_sents = load_sentences(TEST_PATH, N_SENTENCES)
    print(f"Loaded {len(test_sents)} test sentences")

    # Test 1 uses its own pipeline, freed before the checker pipeline is loaded
    eval_stanza(test_sents)

    lex = build_lexicon(TRAIN_PATH)
    print(f"\nLexicon built: {len(lex)} (lemma, upos) entries")

    nlp = stanza.Pipeline(
        "fr",
        processors="tokenize,mwt,pos,lemma,depparse",
        use_gpu=USE_GPU,
        verbose=False,
    )

    fp_examples = eval_clean(nlp, lex, test_sents)
    misses = eval_synthetic(nlp, lex, test_sents)

    print("\nSample false positives (clean sentence -> what was flagged):")
    for text, msgs in fp_examples[:5]:
        print(" ", text)
        print("   ->", msgs)

    print("\nSample misses (corrupted vs original):")
    for bad, orig in misses[:5]:
        print("  BAD :", bad)
        print("  GOLD:", orig)


if __name__ == "__main__":
    main()