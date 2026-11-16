# Data

Everything under `data/` except this file is git-ignored. To rebuild it, put the two
raw files in `data/raw/` and run `python scripts/prepare_data.py` from the repo root.

```
data/
  raw/
    Sarcasm_Headlines_Dataset_v2.json   # News Headlines v2 (JSON lines)
    task_A_En_test.csv                  # iSarcasmEval Task A English test
  processed/
    train.jsonl  val.jsonl  test.jsonl  # in-domain headline splits
    isarcasm_test.jsonl                 # out-of-domain test set
    stats.json                          # counts, balances, baseline, SHA-256
```

Each processed row is `{"id": str, "text": str, "label": 0|1, "domain": str}`, where
`label = 1` means sarcastic. The output never includes `article_link`, because the
link's domain gives away the label (see [Known artifacts](#known-artifacts)).

## Sources

### News Headlines Dataset for Sarcasm Detection, v2 (Misra & Arora)
- URL: https://github.com/rishabhmisra/News-Headlines-Dataset-For-Sarcasm-Detection
  (mirror: https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection)
- File: `Sarcasm_Headlines_Dataset_v2.json`, with fields `is_sarcastic`, `headline` and `article_link`
- Downloaded: 2026-11-10
- Raw records: **28,619** (13,634 sarcastic from The Onion; 14,985 non-sarcastic,
  14,984 of them from HuffPost)
- Citation: Misra, R. & Arora, P. (2023). *Sarcasm Detection using News Headlines Dataset.* AI Open, 4, 13–18.

### iSarcasmEval, SemEval-2022 Task 6, Subtask A English test (Abu Farha et al., 2022)
- URL: https://github.com/iabufarha/iSarcasmEval, file `test/task_A_En_test.csv`
- Commit: [`dfc708b`](https://github.com/iabufarha/iSarcasmEval/tree/dfc708b53bde1bb571abfb5692f63231c2232195)
  (`main` at download time; the file was last changed in `5d21d47`, 2022-02-07)
  &nbsp;·&nbsp; Downloaded: 2026-11-10
- Columns: `text`, `sarcastic`
- Rows: **1,400** (200 sarcastic). The file has 1,616 physical lines because some
  tweets span several lines.
- Citation: Abu Farha, I., Oprea, S., Wilson, S. & Magdy, W. (2022). *SemEval-2022
  Task 6: iSarcasmEval, Intended Sarcasm Detection in English and Arabic.* SemEval-2022.

## Processing

`scripts/prepare_data.py` runs these steps in order. Each count is checked by an
`assert`, so if the raw data changes the script stops before writing anything.

### Headlines
1. **Domain.** `domain` comes from `article_link` by substring match: `theonion` gives
   `onion`, `huffingtonpost` gives `huffpost`. The script uses substring matching
   because about 580 HuffPost links are malformed (`huffingtonpost.comhttp:…`). No
   rows fall into `other`.
2. **Dedupe key.** `" ".join(text.split())`, which removes leading and trailing
   whitespace and collapses runs of internal whitespace. Case and punctuation are kept;
   the headlines are already all lowercase. The key is used only to find duplicates.
   Models see the original `text`.
3. **Label-conflict groups dropped: 0 groups / 0 rows.** If copies of the same key had
   different labels, every copy would be dropped. None did.
4. **Exact duplicates removed: 116 rows** (copies of 80 distinct headlines). The first
   occurrence in file order is kept. Result: 28,503 rows.
5. **Length filter: 1 row dropped** (more than 60 whitespace tokens). This is raw line
   7302, a 151-word Onion "headline" that begins *"hot wheels ranked number one toy for
   rolling down ramp…"*. Result: **28,502 rows**.
6. **IDs:** `h` followed by the row's position in raw-file order after steps 3–5,
   zero-padded to 5 digits (`h00000` … `h28501`). IDs are assigned before shuffling,
   so a given ID always refers to the same headline whatever the seed.
7. **Splits:** stratified on `label`, 80/10/10, with two calls to
   `sklearn.model_selection.train_test_split` (`random_state=13`). The first call
   splits 80 / 20; the second splits the 20 into 10 / 10. The script asserts that the
   splits share no IDs.

**Near-duplicates (estimated, not removed): about 23 pairs.** Pairs were counted using
the nearest neighbour under char 3–5-gram TF-IDF cosine similarity ≥ 0.90, measured
over all 28,502 kept headlines. That gives 18 pairs at ≥ 0.95 and 29 at ≥ 0.85. Of the
23 pairs, 5 are split across train/val/test, and none disagree on label. They are
mostly HuffPost newsletter templates (*"huffpost rise: what you need to know on
december 3"* vs *"… december 9"*) and small rewrites (*works* vs *worked*). Only 1
pair is still a duplicate after lowercasing and removing punctuation. This is too
small to affect test scores, so the pairs stay in.

### iSarcasmEval
- Read with `dtype=str, keep_default_na=False`, so tweets such as `"NA"` stay as
  literal text. Text is otherwise kept as-is, with no dedupe or length filter.
- Labels are stripped and checked to be in `{0, 1}`.
- **IDs:** `t` followed by the CSV row index, zero-padded to 4 digits (`t0000` … `t1399`).
- `domain = "twitter"`.

## Splits (seed = 13)

| split | rows | sarcastic % | onion | huffpost |
|---|---:|---:|---:|---:|
| train | 22,801 | 47.55% (10,841) | 10,841 | 11,960 |
| val   |  2,850 | 47.54% (1,355)  |  1,356 |  1,494 |
| test  |  2,851 | 47.53% (1,355)  |  1,355 |  1,496 |
| isarcasm_test | 1,400 | 14.29% (200) | n/a | n/a |

The val split has one more Onion row than it has sarcastic rows. That row is
`h00563` (*"do you know why i'm pulling you over, being wildly aggressive, and
charging you with assault today, sir?"*). It comes from The Onion but is labelled
`0` in the raw data. It is the only row where label and domain disagree, and it is
kept as-is.

**Reference baseline on `isarcasm_test`:** predicting every tweet as sarcastic gives
F1 (sarcastic) = 2p / (1 + p) = **0.25**, where p = 0.142857.

## Integrity

SHA-256 (also written to `processed/stats.json`):

| file | sha256 |
|---|---|
| `raw/Sarcasm_Headlines_Dataset_v2.json` | `ab0faf86c2dbbef47a95080bc7601981f3ea9dc48ccca73382f90b9a56e02b49` |
| `raw/task_A_En_test.csv` | `0ea25619ca3358ecc8440daac1b9bba4d18455bf5b76d874d8dd6a959e72fc49` |
| `processed/train.jsonl` | `1044b41aafbff31a2be39191e86d77edb98544837ac17ed71841cb2fec8911b9` |
| `processed/val.jsonl` | `8a5cc33c83e9c105ae04555cbcc1fc126bc9efc1120c7f0fa8a5ae92dc4f0cfb` |
| `processed/test.jsonl` | `1cad157b19637b02900ad539385b8b424266abbf4960d59614442cd716530b2b` |
| `processed/isarcasm_test.jsonl` | `07036154c9f06ae1d74275b1e02ac5225bdb351ebcdb096336b312b22f0e3d5a` |

To regenerate the files, run this from the repo root:

```bash
python scripts/prepare_data.py   # seed is the SEED constant (13) in the script; no CLI flag
shasum -a 256 data/processed/*.jsonl   # compare with the table above
```

Output is byte-identical for the same raw files, seed, and pandas/scikit-learn
versions. The files are written as UTF-8 with `\n` line endings and
`ensure_ascii=False`.

## Known artifacts

- **Label ≈ source.** In the headline data, every sarcastic headline comes from The
  Onion and every non-sarcastic one except `h00563` comes from HuffPost. A model
  can score well in-domain by recognizing each outlet's style (for example HuffPost
  newsletter templates or Onion phrasing such as *"area man"* and *"nation's"*)
  without detecting sarcasm. In-domain val/test scores therefore measure skill at
  classifying the source as much as at detecting sarcasm. **`isarcasm_test` measures
  this:** it is tweets labelled by their authors for intended sarcasm, with no
  Onion/HuffPost signal.
- **Domain and class-balance shift.** The headlines are short, lowercase, edited news
  text and about 48% sarcastic. iSarcasmEval is informal, mixed-case tweets and 14%
  sarcastic. Report F1 on the sarcastic class rather than accuracy on the OOD set,
  and compare it with the 0.25 baseline above.
- **Residual near-duplicates.** About 23 pairs remain, 5 of them across splits (see
  [Processing](#processing)).
- **One label/domain mismatch** (`h00563`), which may be a labelling error in the
  source data.
