# Evidence ↔ Markdown matching methodology (record)

How evidence sentences and claim-keywords from `CR_Main_LLM_Extraction.csv` were
verified against the source paper bodies (`Main_process/{DOI@}/{DOI@}.md`).
All matching code is in `evidence_audit_lib.py`, `audit.py`, `keyword_audit_v2.py`,
`keyword_audit_v3.py`. Inputs are never modified; every artifact is a new file.

---

## 0. Why plain substring fails

A contiguous substring match of an evidence sentence against the md scored only
**33–68%** (`substring%` column in `evidence_grep_summary.csv`). Paper markdown is
full of noise that breaks any contiguous match:

- inline tags `<sub>`/`<sup>` (`H<sub>2</sub>O`), HTML
- LaTeX `$...$` + commands `\cdot`, sub/superscripts `_2`, `^2`
- unicode: en-dash `–` (most common, 15k+), middot `·`, `Å`, `°`, `±`, `×`, `′`,
  curly quotes, accented latin, replacement char `�`, Greek `α`
- PDF-extraction glue: missing spaces after commas, hyphenation line-breaks
  (`single-crystal` → `singlecrystal`), citation digits glued to words
- non-English papers (CJK)

So matching is **token-based**, not contiguous.

## 1. Normalization — `norm(s)` (evidence_audit_lib.py)

```
NFKC normalize  (folds Å, sub/superscripts, ligatures, smart quotes)
strip <[^>]+>            -> ' '     (inline tags)
strip \[a-zA-Z]+         -> ' '     (latex commands)
dashes ‐‑‒–—―−          -> '-'
middot/bullet/× · • ∙ × -> ' '
strip ${}^_~|*#          -> ' '
lowercase
```
`toks(s)` = `[a-z0-9]+` tokens with len>2.

## 2. Sentence grounding — token coverage (audit.py)

For each evidence sentence: `coverage = |evidence content-tokens present in md| / |evidence content-tokens|`.
Three signals make a token "present":
1. plain — token in the md token set
2. **digit-glue** — formula tokens (`h2o`,`c6h6`,`cm3`) checked against the md
   with all non-alnum removed (`h 2 o` → `h2o`); only for tokens containing a digit
3. **synonym** — abbrev/canonical families (`dmf`↔`dimethylformamide`,
   `h2o`↔`water`/`dehydration`/`dried`, …) via `SYN_INDEX`

Status thresholds: `grounded ≥0.85`, `weak 0.5–0.85`, `absent <0.5`.
`md_line` = the md line maximizing evidence-token overlap.

**Result (3 axes, `evidence_grep_summary.csv`):** solvent 98.2% / temperature 96.7%
/ gas 98.4% grounded; absent ≈0%. Manual grep of the absent+weak rows: **~0 real
hallucinations** — they are (a) extractor meta/absence notes ("no … reported"),
(b) real quotes failing on notation (°C, 2θ, P/P0, curly quotes), or (c) non-English
papers. See `evidence_absent_weak_classified.csv`.

## 3. Claim-keyword grounding — anchor window (keyword_audit_v2.py)

A sentence can be 98% grounded yet have its single decisive word ("irreversible")
injected. So critical words are checked separately, and crucially **near the cited
location**, not anywhere in the paper (the whole paper almost always contains the
word, which hides hallucination).

- **anchor window** = `md_line ± 2` lines. (Tested: md uses long paragraph-lines,
  so evidence-token coverage by the window is median 0.96; ±2 is sufficient.)
- 47 keywords in 4 groups: positive (flexible, breathing, gate-opening, SC-SC,
  phase transition, swelling/shrinkage/expansion/contraction, np/lp, guest/adsorption-
  induced, NTE, …), negative (rigid, robust, unchanged, retained, stable, …),
  reversibility (reversible, irreversible, recover, restore, regenerate, revert),
  decomposition (collapse, amorphous, decomposition, hysteresis, distortion).
- **left-boundary matching** `(?<![a-z])root`: distinguishes `reversible` from
  `irreversible` (the latter contains the former). Roots (`revers`, `transformat`)
  are prefix-matched to catch inflections; phrases use substring; short acronyms
  (NTE, np, lp) use both-side word boundary.
- **flip detection**: evidence has X but the window has only the antonym
  (reversible↔irreversible, rigid↔flexible) ⇒ `flip`.

`status` = grounded (in window) / absent (not) / flip.

## 4. Absent/flip taxonomy (keyword_absent_flip_v2_classified.csv)

Each non-grounded keyword classified by grep:
- **b_meta_absence** — keyword used to state ABSENCE ("no breathing", "without gate-
  opening"). Not a body claim.
- **d_keyword_elsewhere_in_paper** — keyword is in the paper but not within ±2 of the
  cited line (the extractor quoted a neighbouring sentence). Location imprecision, not
  fabrication.
- **a_window_boundary** — keyword just outside ±2 but within ±5.
- **c_keyword_NOWHERE_in_paper** — keyword absent from the whole paper (the only true
  hallucination candidate).
- **flip**.

**v2 numbers:** 418 absent+flip = 209 meta (50%) + 163 elsewhere (39%) + 29 window +
15 nowhere + 2 flip. Grep of the 15 nowhere + 2 flip ⇒ **0 real hallucinations**:
all were matcher notation gaps (SC-SC written `single-crystal to single-crystal`),
standard terminology (`reversible type-I` for a hysteresis-free isotherm), AI framing
labels over a phenomenon present in the body, or md_line localization false-positives.
The 2 flips: one paper actually says "structural flexibility" (false positive), one is
a malformed evidence cell ("Was THIS EXACT MOF (…)", the only such junk row).

## 5. Notation-gap augmentation (keyword_audit_v3.py)

Matcher-only fix (no label comparison, no semantic loosening) for surface variants
that were wrongly `absent`. Data-driven from the ±2-window diagnostic:

| keyword | added pattern |
|---|---|
| single_crystal_to_single_crystal | `single[\s-]?crystal[\s-]?to[\s-]?single[\s-]?crystal` (hyphen/space mix) |
| sc_sc | `sc[\s-]sc`, `sc[\s-]to[\s-]sc` (abbreviation only) |
| crystal_to_crystal | `crystal[\s-]?to[\s-]?crystal` |
| narrow_pore / large_pore | `np` / `lp` abbrev (word-boundary) + full name |
| gate_opening | `gating` (dropped in v2) + `gate[\s-]?open` |
| phase_transition | `phase transformat` (not generic `phase change`) |

**Before → after grounded% (`keyword_grounding_v3_summary.csv`):**
overall **94.8% → 95.0%**; single_crystal_to_single_crystal 90.4→93.4,
crystal_to_crystal 93.3→100, narrow_pore 97.1→100, gate_opening 91.1→92.2,
phase_transition 88.5→89.3, sc_sc stable (its absents were meta, not notation).

It does not reach ~97% because most non-grounded rows are **not** notation gaps —
they are meta-absence (50%) and keyword-located-elsewhere (39%), which are out of
scope for a notation-only fix.

## 5b. Anchoring fix + reliability gate (keyword_audit_v4.py)

v2/v3 set the anchor line = single md line with max token-SET overlap. Concern raised:
because md paragraphs are one ~2000-char line, frequency matching could grab an
unrelated long paragraph, faking `grounded`. Measured on `keyword_grounding_v3.csv`,
evidence↔`anchor_window_text` overlap was median 0.17 (90% <0.5).

**Root cause was a truncation artifact, not bad anchoring:** the `anchor_window_text`
*column* is truncated to 300 chars while the real ±2 window is ~2000+. Recomputing
overlap against the FULL window gives median ≈1.0; a difflib longest-contiguous-token
match lands on the SAME line with the evidence present verbatim. (IGUTUB's solvent
evidence is genuinely inside L45, not a temperature paragraph.)

The robust anchor was still implemented:
- **anchor_line** = line of the longest contiguous evidence-token block (difflib
  `find_longest_match`) — a long contiguous run is not coincidental, unlike set overlap.
- **anchor_match** = evidence content-word overlap with the FULL ±2 window (computed
  in-code, not from the truncated column). **anchor_run_frac** = longest run / |evidence|.
- **anchor_reliable** = anchor_match ≥ 0.6. Unreliable rows get status
  `anchor_unreliable` and are excluded from grounded%.

**Result:** 8,298 claims → **anchor_reliable 8,000 (96.4%)**; grounded% over reliable
rows = **97.5%** (vs 95.0% over all in v3). The 298 unreliable (3.6%) are 70%
extractor meta/negative sentences ("… was not reported", a malformed cell) plus
heavy paraphrase / non-English — correctly not verbatim in the body.

## 5c. Full 1:1 evidence coverage (evidence_keyword_grounding_full.py)

The keyword-grounding outputs (v2–v4) emit **one row per (evidence, keyword)** via
`for kw in kws_present(...)`, so an evidence sentence with **no** claim keyword produces
**zero rows** and is silently dropped, while a multi-keyword sentence produces several.
Example NIFRUS (gas), dict of 4 guests: CH4 (2 keywords→2 rows), O2 (1→1 row),
N2 (0→dropped), Ar (0→dropped) — only 3 rows, no guest_key column to tell which.
Root cause = the keyword-emit filter, NOT dedup/cache/index.

`evidence_keyword_grounding_full.py` produces **exactly one row per (cif, axis, guest_key)
evidence** (strict 1:1 with CR_Main's evidence units, dict guests split):
- sentence check always: `sentence_coverage` (whole-md token coverage, synonyms+digit-glue)
  and `sentence_status` (grounded ≥0.85 / weak 0.5–0.85 / absent <0.5).
- keyword check when the evidence has ≥1 claim keyword: `keyword` / `keyword_group` /
  `in_anchor_window` / `flip_flag` are `;`-joined per keyword; `keyword_status` aggregates
  (grounded / partial / absent / flip / anchor_unreliable). No keyword → all keyword
  columns = `none`, `keyword_status=no_keyword`. The row is kept either way.
- anchor columns (`anchor_line`, `anchor_match`, `anchor_reliable`) are evidence-level
  (difflib contiguous anchor, §5b).
- Output `evidence_keyword_grounding_full.csv`, UTF-8 BOM + ASCII group labels; text
  fields stripped of CR/LF/TAB so Excel never mis-splits a row. Standalone (no imports).

## 6. Files

| file | content |
|---|---|
| `evidence_audit_lib.py` | norm / tokenizers / coverage / digit-glue / synonyms |
| `audit.py` | 3-axis sentence-coverage audit |
| `keyword_audit_v2.py` | claim-keyword anchor-window audit + flip |
| `keyword_audit_v3.py` | + notation-gap augmentation, before/after |
| `evidence_grep_audit.csv` / `_summary.csv` | per-sentence grounding |
| `evidence_absent_weak[_classified].csv` | absent/weak sentences + hallucination class |
| `keyword_grounding_v2.csv` / `_v2_summary.csv` | per-keyword grounding (anchor) |
| `keyword_absent_flip_v2[_classified].csv` | absent/flip keywords + taxonomy |
| `keyword_grounding_v3.csv` / `_v3_summary.csv` | augmented, with v2→v3 columns |
| `keyword_absent_flip_v3.csv` | remaining absent/flip after augmentation |
| `keyword_audit_v4.py` | difflib contiguous anchor + reliability gate |
| `keyword_grounding_v4.csv` / `_v4_summary.csv` | + anchor_match / anchor_run_frac / anchor_reliable; grounded% over reliable rows |
| `keyword_audit_v4_standalone.py` | self-contained merge of the whole chain |
| `evidence_keyword_grounding_full.py` | 1:1 evidence coverage (all guests, keyword-optional) |
| `evidence_keyword_grounding_full.csv` | one row per (cif,axis,guest_key) evidence |

## 7. Known limits

- Author-defined abbreviations (`p-DVB`=p-divinylbenzene) need the definition sentence
  parsed; not handled.
- Notation still costing a little coverage: `°C`, `2θ`, inequalities `P/P0<…`, curly
  quotes; CJK papers unscorable by the latin tokenizer.
- Anchor window ±2 is line-based; one md_line localization landing on a figure caption
  can cause a false `flip` (seen once).
