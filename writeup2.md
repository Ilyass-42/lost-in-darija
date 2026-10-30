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

TerjamaBench by category (46 to 163 entries per category, so the intervals are large):

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

---

## Entry 5 (29 Oct 2026): Common-sentence test

### Why

In Entry 4, small_A scored higher than small_B on its own test set (chrF++ 36.99 against 35.60). The English overlap did not explain the gap. Test A and Test B contain different sentences, so their scores are not comparable. I need both models to translate the same sentences with the same references.

### Method

Script: `scripts/analyze_common.py` (commit `79c2715`). Input: the result CSV files of small_A on Test A and small_B on Test B (Entry 4, same decoding).

1. Normalize the English sentences (same function as `make_splits.py`).
2. Keep the sentences that are in Test A and in Test B, and in neither Train A nor Train B.
3. Keep the rows that have the same Darija reference in both files.
4. Compute chrF++ and BLEU-4 for both models on these rows.
5. Compute the difference A − B (chrF++) with a paired bootstrap: 1,000 resamples, seed 12345.

### Results

| Check | Result |
| --- | --- |
| Common English sentences | 415 |
| Rows in Test A / rows in Test B | 419 / 420 |
| Rows with the same reference in A and B | 419 |

One row of Test B has no matching reference in Test A. The script excludes it.

| Model | chrF++ | BLEU | 4-grams in hypotheses |
| --- | --- | --- | --- |
| small_A | 35.56 | 12.54 | 635 |
| small_B | 35.85 | 13.75 | 628 |

- Difference chrF++ (A − B): −0.27. 95% confidence interval: [−1.13; 0.64]. The interval contains 0.
- The two models have a similar number of 4-grams in their hypotheses (635 and 628).

### What this shows

- On the same sentences and the same references, I do not see a difference between the two models.
- This agrees with TerjamaBench (chrF++ 34.28 and 34.44): the two models have the same level on shared data.
- The gap of 1.39 chrF++ on the own test sets is therefore not a general quality gap. It is compatible with hypothesis 1 (different test composition).
- This result makes hypothesis 2 (training variation) less likely for a general gap. It does not exclude it.

### Limits

- The common set is small: 419 rows, about 9% of each test set.
- 415 sentences for 419 rows: almost all sentences have one translation. This test does not cover the sentences with many translations. Hypothesis 1 is not tested directly.
- One training run per split.
- I computed the confidence interval for the difference only for chrF++.

### What I learned

- Two scores on two different test sets are not a comparison. A comparison needs the same sentences and the same references.
- A paired bootstrap gives one interval for the difference between two models. Two separate intervals do not.

### Decisions

- The result is written in `paper_draft.md` (Sections 4.3, 5.1, 5.2, 6.2 and Limitations).
- Hypothesis 1 stays open. The test by number of translations (one or many translations per English sentence, in Test A and Test B) is optional.
- Next: Part 2 (DODa-500K synthetic data). I did not find a dataset card for DODa-500K. The generation method of `doda_grounded` is unknown. I will ask Aissam and Hamza.
- The repository license is CC BY-NC 4.0 (checked in the `LICENSE` file). It forbids commercial use of the data.


---

## Entry 6 (29 Oct 2026): Synthetic data of DODa-500K: overlap, cleaning and training setup

### Overlap with the human data

Notebook: `notebooks/explore_doda500k.ipynb` (one new cell). I normalized the English sentences (same function as before). I compared each synthetic type with the 44,652 unique English sentences of the human part. Data: the raw file, before cleaning.

| Synthetic type | Rows | Rows identical to a human sentence | Unique sentences identical to a human sentence |
| --- | --- | --- | --- |
| `doda_grounded` | 226,000 | 57 | 43 |
| `exploratory` | 226,000 | 482 | 296 |

- The exact overlap is small for both types.
- `doda_grounded` has less overlap than `exploratory`. I had thought that "grounded" could mean "based on human sentences". The numbers do not support this idea. I did not verify what "grounded" means. I found no dataset card, so the generation method is still unknown.
- The identical sentences that I looked at are short and frequent ("You're welcome.", "It is hot."). I read this as natural coincidence. I did not verify it.
- This test sees only identical sentences. It does not see paraphrases.

### Cleaning

Script: `scripts/decontaminate_synthetic.py` (commit `48e2d53`). I ran it in Colab on CPU. It loads TerjamaBench from Hugging Face (`atlasia/TerjamaBench`, split `test`, same code as `evaluate_mt.py`), because the CSV file is not in Git.

For each synthetic type, in this order:

1. Remove rows with an empty English or Darija side, and rows with less than 70% Arabic letters (letters only).
2. Remove identical pairs inside the type (same normalized English, same Darija).
3. Remove rows whose normalized English sentence is in Test B (4,459 unique sentences), Val B (4,458) or TerjamaBench (773).
4. Check that no such sentence remains (`assert`), then write `data/synthetic/<type>.csv` (columns `eng`, `darija_ar`).

| Step | `doda_grounded` | `exploratory` |
| --- | --- | --- |
| Rows at start | 226,000 | 226,000 |
| Empty side, or less than 70% Arabic letters | 0 | 0 |
| Identical pairs | 744 | 342 |
| Rows in Test B | 9 | 40 |
| Rows in Val B | 6 | 37 |
| Rows in TerjamaBench | 2 | 40 |
| Removed in step 3 (union of the three sets) | 17 | 117 |
| Rows after cleaning | 225,239 | 225,541 |
| Unique English sentences | 223,830 | 224,451 |

The counts of step 3 are made after step 2. A row can be in more than one set. The cleaned files are in Google Drive (`MyDrive/synthetic/`), not in Git.

### Changes in `fine_tune.py`

Commit `c362880` (hash read in the output of `git pull` in Colab).

- New argument `--synthetic`: one or more CSV files added to Train.
- The script refuses `--synthetic` with `--split A`. The files are cleaned against Test B and Val B only.
- `DarijaDataset` reads a list of CSV files, concatenates them, and stops if a value is empty.
- The script prints the number of rows of Train and Validation.
- Each epoch prints the elapsed time.
- Validation: `val.csv` only (human, Val B).

### Decisions

- I keep both cleaned files in full. The two types differ by 302 rows (0.13%), so I do not subsample.
- Training rows: C1 263,336; C2 263,638; C3 488,877. In C1 and C2, about 86% of the rows are synthetic.
- Validation: Val B (human only, 4,821 rows) for all conditions. The rule is the same for C0 to C3. Val B has no sentence in common with the synthetic data (the script checks it). I chose this after a discussion. A source that I read only in excerpts (Sennrich et al., 2016, WMT 16) is an example of this practice. I must verify it in the full text before I cite it.
- Settings for C1 to C3: the same as C0 (8 epochs, lr 2e-5, batch 16, seed 42).
- I will measure the sentences that the two synthetic types share when I start C3. I did not measure them yet.

### Limit

The cleaning removes identical sentences only. A paraphrase of a Test B or TerjamaBench sentence can stay in the synthetic data.

### What I learned

- A count has a meaning next to a reference point. 57 and 482 mean something only next to each other, and next to the 226,000 rows.
- A script that needs a protected file must load it from the same source as the other scripts. The TerjamaBench CSV is not in Git, so the first version of the script could not run in a clean clone.
- Colab: copy every output to Drive before I close the session.

---

## Entry 7 (29–30 Oct 2026): Training and evaluation of C1 and C2

### Training

```
python src/translation/fine_tune.py --split B --run_name small_C1 --synthetic data/synthetic/doda_grounded.csv
python src/translation/fine_tune.py --split B --run_name small_C2 --synthetic data/synthetic/exploratory.csv
```

Same settings as C0. GPU: NVIDIA A100-SXM4-40GB (C0 used a T4). The Colab panel showed 19.3 GB of 40 GB of GPU memory in use during C1.

| Run | Train rows | Epoch 0 time | Epochs completed | Best epoch (from 0) | Best validation loss | Training time |
| --- | --- | --- | --- | --- | --- | --- |
| C0 `small_B_20261023_224611` | 38,097 | not measured | 8 | 6 | 2.4892 | not measured |
| C1 `small_C1_20261029_142927` | 263,336 | 15.9 min | 7 | 5 | 2.4449 | 112.0 min |
| C2 `small_C2_20261030_124415` | 263,638 | 16.3 min | 8 | 6 | 2.4449 | 129.2 min |

- Epochs are counted from 0, as the script prints them. I checked the Colab logs of small_A and small_B (Part 1). The best epochs are 5 (validation loss 2.5182) and 6 (2.4892). They use the same convention, and they agree with Entry 4. Both logs show eight complete epochs (0 to 7). These two runs did not print the elapsed time, so the training time of C0 is not measured.
- C1: I stopped the run by hand during the eighth epoch. Seven epochs were complete. The best checkpoint (epoch 5, validation loss 2.4449) was already written. The validation loss of epoch 6 was 2.4533. The cell time was 123.5 min, which includes the loading and the part of the last epoch. I copied the model to Drive and checked `model.safetensors` (292 MB).
- C2: eight complete epochs. Best validation loss 2.4449 at epoch 6. Epoch 7 gave 2.4465. The cell time was 130.6 min.
- I estimated the duration of C2 from the time of epoch 0: 127 min. The measure was 129.2 min.
- C1 and C2 have almost the same lowest validation loss. Both are lower than C0 (2.4892) on the same Val B. A loss does not tell the quality of the translation. The training loss is not comparable between conditions, because about 86% of the rows are synthetic.
- Compute: about 11 Colab units for each run (estimate, from the rate of 5.3 units per hour shown by Colab).
- Colab Pro does not keep a notebook running after I close the tab. According to the third-party sources that I read, this feature is for Pro+. I kept the tab open.

### Evaluation

Script: `evaluate_mt.py`, same decoding as before. C1 and C2 on an A100. The C0 results are from Entry 4 (T4). The ± value is the half-width of the 95% bootstrap interval (1,000 resamples).

| Model | Test B BLEU | Test B chrF++ | TerjamaBench BLEU | TerjamaBench chrF++ |
| --- | --- | --- | --- | --- |
| C0 | 13.08 ± 0.90 | 35.60 ± 0.69 | 5.39 ± 1.38 | 34.44 ± 1.28 |
| C1 | 11.90 ± 0.80 | 35.26 ± 0.65 | 10.70 ± 1.71 | 40.57 ± 1.57 |
| C2 | 12.14 ± 0.81 | 35.47 ± 0.69 | 10.10 ± 1.78 | 41.13 ± 1.56 |

Differences with C0: TerjamaBench chrF++ +6.13 (C1) and +6.69 (C2); TerjamaBench BLEU +5.31 and +4.71; Test B chrF++ −0.34 and −0.13; Test B BLEU −1.18 and −0.94.

TerjamaBench by category (chrF++ and BLEU). C0 is the split B model of Entry 4.

| Category | chrF++ C0 | chrF++ C1 | chrF++ C2 | BLEU C0 | BLEU C1 | BLEU C2 |
| --- | --- | --- | --- | --- | --- | --- |
| common_phrases | 41.56 | 45.48 | 45.53 | 5.44 | 15.00 | 12.26 |
| dialect_variation | 15.58 | 21.66 | 21.11 | 1.43 | 4.44 | 3.95 |
| educational | 32.29 | 35.30 | 38.56 | 3.93 | 6.06 | 6.42 |
| humor | 22.66 | 25.65 | 27.51 | 2.88 | 4.11 | 4.39 |
| idioms | 21.21 | 24.74 | 24.95 | 0.84 | 1.31 | 0.93 |
| incorrect_spellings | 34.50 | 37.94 | 34.71 | 5.95 | 7.46 | 4.21 |
| long_sentences | 37.41 | 48.31 | 47.60 | 5.77 | 15.18 | 14.14 |
| mixed_language | 29.09 | 35.80 | 37.25 | 4.59 | 7.32 | 5.77 |
| named_entities | 42.71 | 45.08 | 45.89 | 10.00 | 11.87 | 13.20 |
| numeric_and_date | 35.83 | 40.92 | 41.46 | 6.56 | 8.92 | 8.82 |
| religion | 34.54 | 43.81 | 49.49 | 6.58 | 7.45 | 10.79 |
| single_words | 52.45 | 52.31 | 54.48 | 0.00 | 5.57 | 8.30 |

### What this shows

- On TerjamaBench, C1 and C2 are higher than C0. In chrF++, the intervals of C0 (33.16 to 35.72) and of C1 (39.00 to 42.14) or C2 (39.57 to 42.69) do not overlap. I computed these ranges as score ± half-width.
- On Test B, I do not see a difference in chrF++. BLEU is lower for C1 and C2. I did not analyze why.
- C1 and C2 do not differ on TerjamaBench: the chrF++ difference is 0.56, much smaller than the intervals (about ±1.6).
- By category, C1 is higher than C0 in chrF++ in 11 of 12 categories (`single_words`: −0.14). C2 is higher in 12 of 12. The largest gains are `long_sentences` and `religion`. Between C1 and C2, the two largest differences are `religion` (+5.68 for C2) and `incorrect_spellings` (−3.23 for C2). Each category has 46 to 163 entries, and I did not compute their intervals.
- BLEU for `single_words` (0.00, 5.57, 8.30) shows again that BLEU-4 is unstable on short sentences (Entry 4). chrF++ stays the main metric.
- The gain appears on TerjamaBench, which is a separate dataset. It does not appear on Test B, which comes from the same human source as the training data.

### Limits

- One run per condition.
- C0 ran on a T4. C1 and C2 ran on an A100.
- I stopped C1 after seven of eight epochs. I cannot know if the eighth epoch would have given a better checkpoint.
- The intervals of two models are separate intervals. A paired bootstrap of the difference is not done.
- The cleaning removes identical sentences only. I do not know how the synthetic rows were generated. A paraphrase of a TerjamaBench sentence can stay in the synthetic data, and it can explain a part of the gain. I cannot exclude it.
- C3 is not done. It mixes the effect of the type and the effect of the quantity.

### What I learned

- A lower validation loss (C1 and C2 against C0) did not give a higher chrF++ on Test B. I use the loss to choose a checkpoint, not to compare models.
- The time of epoch 0 gave a good estimate of the full run (127 min estimated, 129.2 min measured).
- A checkpoint is written only when the validation loss improves. After a "Writing model shards" line, I can stop the run safely.

### Decisions

- The results go in `paper_draft.md` (Sections 4.1, 5.4, 5.5, 6.3 and Limitations).
- Next: C3 on the A100 (estimate: about 4 hours and 21 Colab units; not run yet). When C3 starts, I measure the sentences that the two synthetic types share.
- Later, without GPU: paired bootstrap C1 − C0 and C2 − C0; near-duplicate test between TerjamaBench and the synthetic data.
- To do: ask Aissam and Hamza how `doda_grounded` was generated. Optional: evaluate C0 again on the A100.