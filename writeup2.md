# Part 2: Reopening the Project

This is the continuation of [`writeup.md`](./writeup.md). It is written step by step, as the work happens.

---

## Entry 1 (17 Oct 2026): The audit and the data leakage

### Why I reopened the project

- The DODa project released DODa-500K. It adds about 452,000 synthetic sentences to the human data.
- The AbjadNLP 2027 workshop accepts regular papers until 5 Jan 2027.
- Before I build on the old results, I audited them.

### What the audit found

I checked the English sentences of my splits. I lowercased them and removed extra spaces. Then I counted the sentences that appear in two sets. I ran the check on `Train.csv`, `Val.csv` and `Test.csv`.

| Check | Result |
| --- | --- |
| English sentences of Test also in Train | 459 sentences (9.5%) |
| English sentences of Val also in Train | 381 sentences (8.3%) |
| English sentences of Test also in Val | 81 sentences |

Script: `scripts/check_overlap.py` (commit `cb7e306`). Data measured: the CSV files of commit `f322f53`.

The BLEU score of 18.30 comes from a split with this leakage. I no longer consider it reliable.

### The cause

The cause is in `data/prepare_data.py`:

```python
df_merged.drop_duplicates(subset=["eng", "darija_ar"], inplace=True)
df_new_train, df_new_val = train_test_split(
    df_merged, test_size=0.10, random_state=42
)
```

There are two weaknesses:

1. `drop_duplicates` removes only identical pairs. An English sentence with two different Darija translations keeps both rows.
2. `train_test_split` works row by row. It does not know that two rows share the same English sentence. The two rows can go to different sets.

Example: "thank you" has two Darija translations. One row goes to Train and one row goes to Test. The model sees the English sentence during training. The test is then easier than it should be.

### Other findings, not yet verified by me

A code audit made with Claude Code reported these points. I did not check them by hand yet.

- BLEU was computed differently for DODa and for TerjamaBench. The "44% drop" in `writeup.md` is therefore not proven.
- `fine_tune_big.py` may have a bug with `text_target`. The 32.71% UNK rate may be wrong.
- 59 English sentences of TerjamaBench are already in Train. (Checked in Entry 2: I found 67 with my own normalization.)

### Limit of these numbers

- I normalized only case and spaces. Other differences (punctuation, for example) are not covered. The real overlap can be higher.

### Decisions

- Build a clean split (split B). An English sentence goes into only one set. I use `GroupShuffleSplit` on the normalized English sentence.
- Keep the old random split (split A). I train the same small model on A and on B. The score gap measures the effect of the leakage.
- Test the synthetic data of DODa-500K only on split B.
- Write the paper and this document during the work, not at the end.

### What I learned

- A high score on a test set is not a proof. I must check that the test set is independent of the training set.
- I did not run this check in the first version of the project.

### Next step

Create split B. Add an automatic test that checks that Train, Val and Test have no English sentence in common.

---

## Entry 2 (20 Oct 2026): DODa-500K, TerjamaBench overlap and split decisions

### What DODa-500K contains

I downloaded `doda-500k/darija_parallel.jsonl` from the DODa repository (229 MB) and explored it in Google Colab. Notebook: `notebooks/explore_doda500k.ipynb`.

| Check | Result |
| --- | --- |
| Human rows (`doda_human`) | 48,350 |
| Synthetic rows (`doda_grounded`) | 226,000 |
| Synthetic rows (`exploratory`) | 226,000 |
| Generator of all synthetic rows (`source` field) | `gemini-3.1-flash-lite` |

- Fields: `id`, `darija_arabic`, `darija_latin`, `english`, `code_switch`, `register`, `domain`, `generation_mode`, `source`.
- No field links a synthetic row to a human row. The documentation does not explain how the `doda_grounded` rows were generated.
- I will use only the human rows for the splits.

### Cleaning the human part

English normalization: lowercase, punctuation replaced by a space, multiple spaces replaced by one space.

| Check | Result |
| --- | --- |
| Rows with empty English or empty Darija | 0 |
| Identical pairs (normalized English + Darija) | 372 |
| Darija rows with less than 70% Arabic letters | 0 |
| Rows after removing identical pairs | 47,978 |
| Unique English sentences | 44,652 |
| English sentences with more than one Darija translation | 2,175 |

The 2,175 sentences are the source of the leakage. Together they add 3,326 extra rows. Example: "leave me alone" has 23 different translations. In a random 80/10/10 split, the chance that all 23 rows go to the same set is about 0.6% (0.8^23). This sentence leaks almost every time.

### A bug in my first Arabic-script filter

My first filter counted all characters: letters, digits, punctuation and spaces. It rejected 31 rows. I looked at them: all were valid Darija in Arabic script, for example "لا!" (No!) or "هادي 10:30" (it's 10:30). Digits and punctuation lowered the ratio below 70%. I changed the filter to count letters only. Result: 0 rows rejected.

This matters: these rows contain numbers and times, and TerjamaBench has a `numeric_and_date` category.

The filter in my old `data/prepare_data.py` has the same weakness (it counts all characters, threshold 50%). It applied only to the atlasia data, not to the DODa sentences.

### TerjamaBench is also in my training data

TerjamaBench is my independent benchmark. The model must not see its sentences during training. I checked this.

Normalization: lowercase, punctuation replaced by a space, multiple spaces replaced by one space.

| Check | Result |
| --- | --- |
| TerjamaBench rows | 850 |
| Unique English sentences in TerjamaBench | 773 |
| TerjamaBench sentences also in Train | 67 (8.7%) |
| TerjamaBench sentences also in Val | 28 |
| TerjamaBench sentences also in Test | 21 |

Script: `scripts/check_terjama_overlap.py` (commit `e43c456`). Data measured: the CSV files of commit `f322f53`.

The common sentences are short and frequent: "thanks", "hello", "turn right", "how are you", "leave me alone". DODa did not copy TerjamaBench. These sentences exist naturally in both datasets. They are still a leak: the model saw these English inputs during training.

### Decisions for the new splits

- Data: the human rows of DODa-500K only.
- Proportions: 80% Train, 10% Val, 10% Test.
- Normalization of English sentences: the same as above. Without punctuation removal, "Thank you!" and "thank you" stay two different groups, and the leak continues.
- Split A: random, row by row (the old method). Split B: grouped by normalized English sentence.
- Remove every TerjamaBench English sentence from Train and Val, in split A and in split B. Val counts too, because it selects the best checkpoint.
- I will measure the final number of removed sentences on the new splits.

### What I learned

- A benchmark is independent only if I check it. "Independent benchmark" is a claim that needs a test.
- The normalization choice changes the numbers (59 in the audit, 67 in my check). I must write it down with every number.
- Before I remove data with a filter, I look at the rejected rows. Here, all 31 were false positives.

### Next step

Build splits A and B. Check that split B gives 0 with `scripts/check_overlap.py`.


---

## Entry 3 (21 Oct 2026): Splits A and B

### What I built

Script: `scripts/make_splits.py`. Check: `scripts/verify_splits.py`. Commit `30e89cc`.

1. I extracted the human rows of DODa-500K in Colab and saved them as `data/doda_human.csv` (48,350 rows).
2. I normalized the English sentences and removed identical pairs: 47,978 rows, 44,652 unique English sentences.
3. I removed every TerjamaBench English sentence from the data, before the split: 70 sentences, 257 rows. Result: 47,721 rows, 44,582 unique English sentences.
4. Split A: random, row by row (`train_test_split`), 80/10/10.
5. Split B: grouped by normalized English sentence (`GroupShuffleSplit`), 80/10/10 of the groups.
6. Both splits use the same seed (42).

The 70 TerjamaBench sentences have 257 rows: almost 4 Darija translations per sentence. They are short and frequent phrases.

### Results

| Split | Train | Val | Test |
| --- | --- | --- | --- |
| A (rows) | 38,176 | 4,772 | 4,773 |
| B (rows) | 38,097 | 4,821 | 4,803 |

Split B uses proportions of groups, not rows. Its row proportions are 79.8 / 10.1 / 10.1%.

| Shared English sentences | A | B |
| --- | --- | --- |
| Test ∩ Train | 433 (9.2% of 4,724 unique Test sentences) | 0 |
| Val ∩ Train | 450 | 0 |
| Test ∩ Val | 91 | 0 |

The script stops with an error if split B has a shared sentence (`assert`). I read the saved files again with `verify_splits.py`: same numbers.

Split A reproduces the old situation: 9.2% here, 9.5% in my old files.

### Limit

Split B has no shared English sentence after normalization. Paraphrases ("thank you" / "thanks") and spelling variants ("thank u") can still be in two sets. I call it a grouped split, not a leakage-free split.

### Next step

Train the small model (MarianMT `opus-mt-en-ar`) on A and on B, with the same settings. Evaluate both on their own Test set and on TerjamaBench.

---

## Entry 4 (24 Oct 2026): Training on splits A and B, evaluation and leakage analysis

### What I changed in the code

**`src/translation/fine_tune.py`** (commit `4786421`)

- New arguments: `--split` (`A` or `B`) and `--run_name`.
- Fixed seed: `torch.manual_seed(42)` and `torch.cuda.manual_seed_all(42)`, before the model and the `DataLoader`.
- Data: `data/splits/<split>/train.csv` and `val.csv`.
- Output folder: `models/<run_name>_<date>` and `runs/<run_name>_<date>`. A run cannot overwrite another run.

**`src/translation/evaluate_mt.py`** (commit `66ef4d1`): one script for all evaluations.

- Arguments: `--model` (local folder or Hub ID), `--dataset` (`test` or `terjama`), `--split` (`A` or `B`, for `test` only).
- Decoding, the same for all models: prefix `>>ary<<`, `num_beams=4`, `length_penalty=1.2`, batch size 32.
- Metrics: BLEU (order 4) and chrF++, with a 95% bootstrap confidence interval (1,000 resamples).
- Output: `results/<model>_<dataset>[_<split>].csv` (English, reference, hypothesis, + `topic` for TerjamaBench) and a `.json` file (scores, intervals, signatures, decoding settings).

Signatures (sacreBLEU 2.6.0):

- BLEU: `nrefs:1|bs:1000|seed:12345|case:mixed|eff:no|tok:13a|smooth:exp|version:2.6.0`
- chrF++: `nrefs:1|bs:1000|seed:12345|case:mixed|eff:yes|nc:6|nw:2|space:no|version:2.6.0`

Versions: `transformers` 5.20.2, `torch` 2.11.0+cu130, `sacrebleu` 2.6.0. Hardware: Google Colab T4.

### Problems found in the old evaluation scripts

I read the git history of `src/translation/evaluate.py` and `evaluate_terjama.py`.

| Finding | Evidence |
| --- | --- |
| BLEU 18.30 used `max_ngram_order=3` and no decoding options | commit `1690022` (25 Apr 2026) |
| Beam search (`num_beams=4`, `length_penalty=1.2`) was added later | commit `8684484` (1 May 2026) |
| TerjamaBench evaluation used model `v2`, beam search 4, order 3 | commit `59bc545` (4 May 2026) |
| `evaluate.py` changed to order 4; `evaluate_terjama.py` stayed at order 3 | commit `36d3a11` (17 Jul 2026) |

The 18.30 and the 10.23 do not use the same decoding. The "44% drop" in `writeup.md` compares two scores that are not comparable.

### Training

Settings, the same for A and B: `opus-mt-en-ar`, AdamW, lr 2e-5, 8 epochs, batch size 16, cosine schedule with 6% warmup, mixed precision, seed 42. The script keeps the checkpoint with the lowest validation loss.

| Run | Best epoch | Validation loss |
| --- | --- | --- |
| `small_A_20261023_221426` | 5 | 2.5182 |
| `small_B_20261023_224611` | 6 | 2.4892 |

For both runs, the validation loss stops decreasing after epoch 5 or 6.

### Results

The ± value is the half-width of the 95% confidence interval.

| Model | Dataset | BLEU | chrF++ | Length ratio |
| --- | --- | --- | --- | --- |
| base (`opus-mt-en-ar`) | Test A | 0.22 ± 0.10 | 11.20 ± 0.38 | 1.821 |
| base | Test B | 0.25 ± 0.11 | 11.04 ± 0.37 | 1.895 |
| base | TerjamaBench | 0.31 ± 0.18 | 16.24 ± 1.49 | 2.490 |
| small_A | Test A | 14.06 ± 0.84 | 36.99 ± 0.68 | 0.895 |
| small_A | TerjamaBench | 4.90 ± 1.26 | 34.28 ± 1.29 | 1.075 |
| small_B | Test B | 13.08 ± 0.90 | 35.60 ± 0.69 | 0.894 |
| small_B | TerjamaBench | 5.39 ± 1.38 | 34.44 ± 1.28 | 1.074 |

TerjamaBench by category (about 70 sentences per category, so the intervals are large):

| Category | BLEU A | BLEU B | chrF++ A | chrF++ B |
| --- | --- | --- | --- | --- |
| common_phrases | 6.57 | 5.44 | 40.19 | 41.56 |
| dialect_variation | 1.52 | 1.43 | 15.04 | 15.58 |
| educational | 4.00 | 3.93 | 31.38 | 32.29 |
| humor | 2.79 | 2.88 | 22.08 | 22.66 |
| idioms | 1.10 | 0.84 | 21.69 | 21.21 |
| incorrect_spellings | 3.63 | 5.95 | 32.60 | 34.50 |
| long_sentences | 6.28 | 5.77 | 38.09 | 37.41 |
| mixed_language | 4.49 | 4.59 | 29.89 | 29.09 |
| named_entities | 6.40 | 10.00 | 43.40 | 42.71 |
| numeric_and_date | 5.44 | 6.56 | 34.67 | 35.83 |
| religion | 4.05 | 6.58 | 34.29 | 34.54 |
| single_words | 6.54 | 0.00 | 53.94 | 52.45 |

The global TerjamaBench score is a corpus-level score. It is not the mean of the categories (mean of the 12 BLEU scores of small_A: 4.40; global: 4.90).

### The base model

- The tokenizer reads `>>ary<<` as one token.
- In 9 of 10 random examples, the base model translates into Modern Standard Arabic, not Darija. Example: "take the pillow" gives خذ الوسادة; the reference is هازي لوسادا.
- The base model also produces repetition loops. Example: "Teach me" gives `# علمني # # # # # #`.

I found the loops with the length ratio of the BLEU output (1.821), then with examples.

| Evaluation | Ratio > 3 | Contains `#` |
| --- | --- | --- |
| base, Test A | 149 / 4,773 | 76 |
| base, TerjamaBench | 38 / 850 | 5 |
| small_A, Test A | 2 / 4,773 | 0 |
| small_A, TerjamaBench | 4 / 850 | 0 |
| small_B, Test B | 3 / 4,803 | 0 |
| small_B, TerjamaBench | 0 / 850 | 0 |

On TerjamaBench, many references have 1 or 2 words. A correct translation of 4 words already gives a ratio above 3. These counts are a maximum, not an exact count of loops.

### Decision: decoding

I keep the current decoding (no rule against repetitions).

- The loops occur almost only with the base model.
- An option such as `no_repeat_ngram_size=3` blocks any repeated sequence of 3 tokens. Only 8 of 4,773 references of Test A contain such a sequence, so the cost is small. But it adds a setting that the paper must justify.
- The result on the base model is a negative result: without fine-tuning, the model does not produce Darija.

### BLEU-4 and short sentences

BLEU-4 needs sequences of 4 words. When a group of translations contains no such sequence, sacreBLEU stops the calculation and returns 0, even if all translations are correct.

| `single_words` | 4-word sequences in the translations | BLEU | chrF++ |
| --- | --- | --- | --- |
| small_A | 4 | 6.54 | 53.94 |
| small_B | 0 | 0.00 | 52.45 |

The BLEU difference comes from the length of the translations, not from their quality. For a single word, a long translation is more likely an error.

### Analysis: seen and unseen sentences in Test A

Script: `scripts/analyze_leak.py` (commit `69596b1`). I split Test A into two groups: the English sentence is in Train A ("seen") or not ("unseen"). Same model (small_A), same file, same decoding.

| Group | Rows | Unique sentences | BLEU | chrF++ |
| --- | --- | --- | --- | --- |
| unseen | 4,307 | 4,291 | 13.84 ± 0.83 | 37.23 ± 0.67 |
| seen | 466 | 433 | 18.45 ± 4.60 | 33.74 ± 2.60 |

The two metrics do not agree. BLEU is not reliable for the "seen" group:

| Group | 1-gram | 2-gram | 3-gram | 4-gram | Translations with 4 words or more |
| --- | --- | --- | --- | --- | --- |
| unseen | 39.8% | 20.2% | 11.1% | 6.4% | 2,447 / 4,307 (57%) |
| seen | 40.3% | 20.7% | 18.8% | 18.6% | 103 / 466 (22%) |

- The 1-gram and 2-gram precisions are almost equal. The difference comes only from the 3-gram and 4-gram precisions.
- In the "seen" group, only 103 translations contain 4-word sequences. A small number of sentences decide the 4-gram precision. This explains the large interval (± 4.60).
- The "seen" sentences are shorter: 3.5 tokens per reference (1,637 / 466) against 4.7 (20,431 / 4,307).

With chrF++, small_A translates the "seen" sentences less well. A probable cause: `make_splits.py` removes identical pairs. A "seen" sentence in Test A always has a Darija translation that is different from the one in Train A. The model learned the English sentence with another translation.

### What this changes for Part 1

- On its own test set, split A gives a higher score than split B: +0.98 BLEU, +1.39 chrF++.
- On TerjamaBench, A and B have the same level (chrF++ 34.28 and 34.44).
- The overlap of English sentences (9.2% of Test A) does not explain the gap: on the "unseen" sentences only, small_A still has chrF++ 37.23, against 35.60 for small_B on Test B.

The gap comes from something else. Two hypotheses, not verified:

1. The composition of the test sets. In split B, a sentence with many translations goes completely into one set. In split A, these sentences end up mostly in the "seen" group.
2. The variation between two training runs. I have only one run per split.

### Limits

- One training run per split.
- Test A and Test B do not contain the same sentences.
- The confidence intervals are not a formal statistical test.

### What I learned

- Read the length ratio and the BP in the BLEU output, not only the score. The ratio showed the loops of the base model.
- BLEU-4 is not stable on short sentences. The score can go to 0 or become too high.
- Save the translations. Without them, I could not have done the seen/unseen analysis or explained the BLEU results.
- A removed duplicate changes the meaning of "leakage". Here, a "seen" English sentence never comes with the same Darija translation.

### Decisions

- Metrics: chrF++ is the main metric for all datasets and categories. I always report BLEU too, with the number of 4-word sequences.
- TerjamaBench: I report the global score on all 850 sentences and the score of each category. I do not remove or reweight any category.
- `results/` stays in Google Drive, not in Git. The TerjamaBench files contain sentences of a gated dataset.

### Next step

- Test on the sentences that are in Test A and in Test B, and not in Train A or Train B. Both models never saw them. If the scores are close, hypothesis 1 is correct.
- Look at the 2 translations of small_A with a ratio above 3.