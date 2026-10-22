"""
Type your own French sentences and see what the checker flags.

Put this file in the same folder as evaluate_test_set.py and
fr_gsd-ud-train.conllu, then run:  python try_sentence.py
"""

import stanza
from evaluate_test_set import build_lexicon, check, apply_edits, TRAIN_PATH, USE_GPU

lex = build_lexicon(TRAIN_PATH)
nlp = stanza.Pipeline(
    "fr",
    processors="tokenize,mwt,pos,lemma,depparse",
    use_gpu=USE_GPU,
    verbose=False,
)

SHOW_PARSE = False   # set True to print how Stanza parsed the sentence


def run(text):
    if SHOW_PARSE:
        for w in nlp(text).sentences[0].words:
            print(f"  {w.id:2} {w.text:12} {w.upos:6} {w.feats or '-':45} head={w.head} {w.deprel}")

    issues = check(nlp, lex, text)
    if not issues:
        print("No agreement errors found.")
        return

    for start, end, fix, msg, alt in issues:
        print(f"- {msg}")
        print(f"  replace '{text[start:end]}' with '{fix}'")
        if alt:
            a_start, a_end, a_fix = alt
            print(f"  or change the subject: '{text[a_start:a_end]}' -> '{a_fix}'")
    print("Corrected:", apply_edits(text, issues))


if __name__ == "__main__":
    print("Type a French sentence (empty line to quit).")
    while True:
        text = input("\n> ").strip()
        if not text:
            break
        run(text)