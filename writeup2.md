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