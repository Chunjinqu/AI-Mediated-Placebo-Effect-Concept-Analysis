# -*- coding: utf-8 -*-
"""
step02: cleaning + sentence segmentation
Input:  data/texts/*.md
Output: data/output/sentences.csv (columns: sent_id, article, n_words, text)
Run:    python step02_clean_segment.py

Cleaning rules follow the reference paper (Lee et al., 2026):
1. Repair hyphenated word breaks across lines (manage-\\nment -> management)
2. Merge sentences interrupted by line breaks
3. Remove URLs, [bracketed] content, and non-text symbols
4. Sentence segmentation with spaCy
5. Drop sentences shorter than MIN_WORDS or longer than MAX_WORDS
Note: no stopword removal or lemmatization (SBERT consumes whole-sentence
semantics, so token-level normalization is unnecessary).
"""
import re
import pandas as pd
import spacy
from config import TEXT_DIR, OUT_DIR, MIN_WORDS, MAX_WORDS


def fix_hyphen_breaks(text: str) -> str:
    # "manage-\nment" -> "management" (hyphen + newline + lowercase letter)
    return re.sub(r"-\n(?=[a-z])", "", text)


def merge_broken_lines(text: str) -> str:
    # Lines not ending in sentence-final punctuation are treated as
    # interrupted by a line break and merged with the next line
    lines = [ln.strip() for ln in text.splitlines()]
    merged, buf = [], ""
    for ln in lines:
        if not ln:
            if buf:
                merged.append(buf)
                buf = ""
            continue
        buf = (buf + " " + ln).strip()
        if buf.endswith((".", "!", "?", ":", ";")) or len(buf) > 800:
            merged.append(buf)
            buf = ""
    if buf:
        merged.append(buf)
    return "\n".join(merged)


def remove_artifacts(text: str) -> str:
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)   # URLs
    text = re.sub(r"\[[^\]]{0,80}\]", " ", text)          # [12] [doi:...] etc.
    text = re.sub(r"[^\x00-\x7F]", " ", text)             # non-ASCII symbols
    text = re.sub(r"[_{}<>|~*#`\\^+=]", " ", text)        # markdown/symbol residue
    text = re.sub(r"\s{2,}", " ", text)                   # extra whitespace
    return text


# -- Two explicit extensions to the Lee et al. cleaning pipeline (to be
#    reported in the Methods section) --
# 1) Remove the entire References section: bibliography entries contain no
#    instances of concept usage and would occupy whole clusters in a small
#    corpus. Structural truncation (located by a standalone heading line);
#    in-text citations are unaffected.
# 2) Remove journal header/copyright metadata lines (ISSN etc.), which
#    likewise carry no conceptual content.
_REF_HEADING = re.compile(
    r"^\s{0,3}#{0,6}\s*\*{0,2}\s*(references|bibliography)\s*\*{0,2}\s*#*\s*$",
    re.I)
# Some journals (e.g., Mary Ann Liebert) have no "References" heading; the
# reference list directly follows the Disclosure Statement section, which is
# then used as the truncation point
_FALLBACK_HEADING = re.compile(
    r"^\s{0,3}#{0,6}\s*\*{0,2}\s*disclosure statement\s*\*{0,2}\s*#*\s*$", re.I)
_META_LINE = re.compile(r"issn|journal homepage|creative commons|"
                        r"©\s*\d|copyright\s*©|licensee", re.I)


def drop_backmatter(text: str) -> tuple:
    lines = text.splitlines()
    cut = None
    for i, ln in enumerate(lines):
        if _REF_HEADING.match(ln):
            cut = i                       # last match = true start of reference list
    if cut is None:                       # journals without a References heading:
        for i, ln in enumerate(lines):    # fall back to Disclosure Statement
            if _FALLBACK_HEADING.match(ln):
                cut = i
    found = cut is not None
    if found:
        lines = lines[:cut]
    lines = [ln for ln in lines if not _META_LINE.search(ln)]
    return "\n".join(lines), found


def main():
    nlp = spacy.load("en_core_web_sm")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    md_files = sorted(TEXT_DIR.glob("*.md"))
    if not md_files:
        print("!! data/texts is empty — run step01 first.")
        return

    for md in md_files:
        article = md.stem
        text = md.read_text(encoding="utf-8", errors="ignore")
        text, ref_found = drop_backmatter(text)
        if not ref_found:
            print(f"    !! Warning: no References heading located in {article}; "
                  f"please check manually whether bibliography entries leaked in.")
        text = fix_hyphen_breaks(text)
        text = merge_broken_lines(text)
        text = remove_artifacts(text)

        for para in text.split("\n"):
            para = para.strip()
            if not para:
                continue
            doc = nlp(para)
            for sent in doc.sents:
                words = [w for w in sent.text.split() if any(c.isalnum() for c in w)]
                n = len(words)
                if n < MIN_WORDS or n > MAX_WORDS:
                    continue
                rows.append({"article": article, "n_words": n,
                             "text": sent.text.strip()})

    df = pd.DataFrame(rows)
    df.insert(0, "sent_id", range(len(df)))
    df = df.drop_duplicates(subset=["article", "text"]).reset_index(drop=True)
    df["sent_id"] = range(len(df))

    out = OUT_DIR / "sentences.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"Done: {len(md_files)} articles -> {len(df)} sentences")
    print(f"Average {len(df)/len(md_files):.0f} sentences per article "
          f"(reference paper: ~16218 sentences / 73 articles = ~222 per article)")
    print(f"Saved to {out}")
    print("Next: run step03_embed.py (first run downloads the ~90 MB SBERT model)")


if __name__ == "__main__":
    main()
