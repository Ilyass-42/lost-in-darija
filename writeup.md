# From 18 to 10: Diagnosing a Distribution Shift in a Low-Resource NLP Project

## The Project

`lost-in-darija` is a speech-to-speech translation app that converts English audio into Moroccan Darija Arabic, targeting English-speaking tourists in Morocco the same feeling as Coppola's *Lost in Translation*, just in Marrakech instead of Tokyo. The pipeline: Whisper (STT) → fine-tuned MarianMT (translation) → edge-tts (TTS).

I built this as a 3rd-year CS engineering student to learn PyTorch and the HuggingFace ecosystem hands-on. The constraint was real: no research budget, a CPU-only laptop for development, and a Google Colab T4 GPU for training.

## Building the Baseline

The first challenge was the data. No ready-made EN→Darija translation model existed on HuggingFace. I used `Helsinki-NLP/opus-mt-en-ar` as a base a model trained on Modern Standard Arabic (MSA) and fine-tuned it on DODa (Darija Open Dataset), an open-source community dataset by Outchakoucht & Es-Samaali with ~150k entries and over 86k translated sentence pairs, covering both Arabic and Latin-script Darija. The model required a `>>ary<<` language token prepended to each English input to target Moroccan Darija (ISO 639-3: `ary`).

First run: 4 epochs, AdamW optimizer, lr=2e-5. BLEU score: **17.27**. Decent for a first attempt, but I was only tracking training loss no validation.

I then added validation loss tracking to `fine_tune.py`: `model.eval()`, `torch.no_grad()`, per-epoch logging, best checkpoint saving. Re-trained for 8 epochs. Best checkpoint at epoch 7 (before overfitting set in). New BLEU: **18.30**. A methodological improvement that directly translated into a score gain.

## Chasing a Better Model and a Dead End

With a working baseline, I tested `Helsinki-NLP/opus-mt-tc-big-en-ar` (~600M parameters vs ~77M for the base model). Full fine-tuning was out of the question on a T4 (15GB VRAM) so I used **LoRA** (Low-Rank Adaptation via `peft`), freezing the base weights and injecting trainable rank-decomposition matrices into the attention layers. Configuration: r=16, lora_alpha=32, target_modules on q/k/v/out projections, ~1% trainable parameters.

It failed before training even started. The tokenizer produced a **UNK rate of 32.71%** on Darija Arabic text nearly one in three tokens was unknown. The model generated incoherent sequences. The larger model had been trained on a different Arabic data distribution that covered Darija even less than the base model, despite its higher reported BLEU.

Back to `opus-mt-en-ar`: 0% UNK on the same data. Its SentencePiece vocabulary, despite being MSA-oriented, covers Darija script well enough for fine-tuning to work.

**Lesson:** always check tokenizer UNK rate on your target data before committing to a base model. It's a better selection criterion than benchmark BLEU on a different test set.

## The Number That Didn't Add Up

I then evaluated the model on **TerjamaBench** an independent benchmark of 850 EN↔Darija sentence pairs across 12 semantic categories, published by AtlasIA.

Global BLEU: **10.23**.

That's a 44% drop from 18.30. The model hadn't changed. The task hadn't changed. What changed was the data distribution.

## Diagnosing the Problem

I broke the evaluation down by category using `.filter()` on the `topic` column:

| Category | BLEU |
|---|---|
| common_phrases | 17.40 ✅ |
| long_sentences | 11.84 |
| named_entities | 11.94 |
| numeric_and_date | 11.63 |
| incorrect_spellings | 9.51 |
| religion | 9.83 |
| educational | 8.72 |
| single_words | 8.60 |
| mixed_language | 7.69 |
| humor | 4.39 |
| dialect_variation | 2.02 ❌ |
| idioms | 1.14 ❌ |

The pattern was clear. Categories well-represented in DODa scored well. Categories absent from DODa scored near zero. The model had learned DODa's distribution not "Darija" as a language.

This is a classic distribution shift: training and test sets drawn from different distributions. The bottleneck was the data, not the model architecture.

## Attempting the Fix: Data Augmentation

The fix I attempted was targeted data augmentation. Rather than generating random Darija pairs, I used TerjamaBench's own English column as seed sentences directly targeting the benchmark's distribution. Four categories were prioritized: Common Phrases, Named Entities, Numeric/Date, and Mixed Language. Idioms were excluded: LLM hallucination risk is high and they're outside the tourist use case anyway.

The pipeline: NVIDIA NIM (Qwen) generated candidate pairs → an LLM judge evaluated fluency and authenticity → a native speaker spot-check before any pair would enter training data.

The broader lesson, independent of how the attempt ended: a BLEU score on a held-out test set from the same distribution as training is not a generalization benchmark. Evaluating on an independent, domain-specific dataset is what surfaces the real gaps and tells you exactly where to look.

## Engineering Lessons

Building the generator-judge augmentation pipeline surfaced lessons that have less to do with NLP and more to do with engineering against LLM APIs at scale.

**LLM output isn't guaranteed to match the expected schema.** Qwen would occasionally switch into an internal reasoning mode and return an empty response field instead of the generated pairs. **Lesson:** treat the shape of an LLM response as untrusted input validate before parsing, don't assume.

**Token budgets need to be sized for the task, not the default.** Generating structured JSON in batches of 20 pairs exceeded the default token limit, which silently truncated the output mid-object. **Lesson:** for structured generation, size the token limit to the expected output, then verify the result actually parses.

**A single prompt can't enforce a precise ratio.** Asking one prompt to hit a target percentage of French/English borrowings produced inconsistent results across batches. **Lesson:** decompose a complex constraint into separate, simpler generation tasks rather than asking one prompt to balance several objectives at once.

**Reliability is its own quality metric.** Repeated availability issues with one provider were enough on their own to justify migrating to a different stack for generation and judging. **Lesson:** a model's accuracy doesn't matter if the API serving it isn't stable enough to run a pipeline against.

## Pausing Augmentation, Shipping What Works

Past the engineering issues above, a deeper problem persisted: the judge itself wasn't reliable for Darija-specific nuance. A smaller judge model avoided rate limits but missed semantic errors AM/PM inversions, wrong dates, mistranslated status words slipping through as "valid." A larger judge caught more, but hit rate limits hard enough to stall the pipeline. Few-shot examples meant to anchor the judge's criteria caused it to memorize the literal example rather than generalize the principle. One batch of 224 generated pairs was reviewed and deleted outright for quality.

At that point, continuing to tune the judge had a worse cost-to-signal ratio than I was willing to pay for a portfolio project on a deadline. I paused the augmentation work rather than ship synthetic data I didn't trust, and redirected the remaining time to closing the loop on the part of the project that was fully in my control: making the pipeline something a reviewer could actually run, not just a model card with a BLEU table.

## Containerizing the Pipeline

The full pipeline Whisper STT, the fine-tuned MarianMT model, and edge-tts now runs end-to-end inside a single Docker image, exposing the existing Gradio UI on port 7860. The model is pulled from the HuggingFace Hub at build time rather than copied from local disk, so the image builds identically on any machine. The image is CPU-only by design, matching the constraint the whole project was built under.

Details of the build (base image, layer ordering, model loading strategy) are documented in the repo's `Dockerfile` and `CLAUDE.md` rather than repeated here.


## Shipping It

With the augmentation approach paused, I containerized the pipeline and deployed it to HuggingFace Spaces so the model is actually reachable, not just a checkpoint on a hard drive. Docker SDK, CPU Basic hardware no GPU needed for inference at this scale (Whisper `tiny.en` + a 77M-parameter MarianMT model).

The main friction point was git-based: HF Spaces deploys are pushed to a separate Git remote, and it's easy to commit the surface files (`Dockerfile`, `app.py`, `requirements.txt`) while missing a source directory that isn't at the repo root `src/` didn't make it into the first commit, which surfaced immediately as a `ModuleNotFoundError` at container startup. A useful reminder that "it builds" and "it imports" are two different checkpoints.

The app is live at [huggingface.co/spaces/ILyass-42/lost-in-darija](https://huggingface.co/spaces/ILyass-42/lost-in-darija).

---

*Model published at [ILyass-42/lost-in-darija-marian](https://huggingface.co/ILyass-42/lost-in-darija-marian). Dataset: DODa. Evaluation: TerjamaBench (AtlasIA).*