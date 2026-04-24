# 🗣️ Lost in Darija

A speech-to-speech translation pipeline that converts spoken English into Moroccan Darija (الدارجة), designed for tourists navigating Morocco.

---

## ✨ Features

- 🎙️ **Speech-to-Text** — transcribes English audio using OpenAI Whisper
- 🌍 **Neural Machine Translation** — translates English to Moroccan Darija using a fine-tuned MarianMT model (Helsinki-NLP/opus-mt-en-ar)
- 🔊 **Text-to-Speech** — synthesizes Darija audio using Microsoft Edge TTS (ar-MA-JamalNeural voice)
- 📊 **BLEU score evaluation** — measures translation quality on a test set (BLEU = 17.27 after 4 epochs)

---

## 🗂️ Project Structure
```
lost-in-darija/
├── data/
│   ├── Train.csv          # Training set (eng → darija_ar)
│   ├── Val.csv            # Validation set
│   ├── Test.csv           # Test set
│   └── sentences.csv      # Sample sentences
├── notebooks/
│   └── notebook.ipynb     # Experimentation notebook
├── src/
│   ├── pipeline.py        # End-to-end pipeline (STT → Translation → TTS)
│   ├── stt/
│   │   └── transcribe.py  # Whisper transcription
│   ├── translation/
│   │   ├── fine_tune.py   # MarianMT fine-tuning script
│   │   ├── translate.py   # Inference with fine-tuned model
│   │   └── evaluate.py    # BLEU score evaluation
│   └── tts/
│       └── synthetise.py  # Edge TTS synthesis
├── tests/
│   └── test_pipeline.py
└── requirements.txt
```
---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Speech-to-Text | OpenAI Whisper (tiny.en) |
| Translation | MarianMT fine-tuned on DODa |
| Text-to-Speech | Microsoft Edge TTS (ar-MA-JamalNeural) |
| Training | PyTorch + HuggingFace Transformers |
| Evaluation | SacreBLEU |

---

## 📚 Dataset

The translation model was fine-tuned on **[DODa — Darija Open Dataset](https://github.com/darija-open-dataset/dataset)**, the largest open-source English ↔ Darija parallel corpus, containing ~86,000 translated sentences. DODa covers both Arabic-script and Latin-script Darija, and accounts for the dialect's characteristic code-switching between Arabic, French, and Spanish.

Only the `sentences.csv` file was used, filtered to the `eng` → `darija_ar` (Arabic script) column pair, yielding ~48,800 clean pairs split 81/9/10 into train/val/test sets.

> Outchakoucht, A. & Es-Samaali, H. (2024). *The Evolution of Darija Open Dataset: Introducing Version 2*. arXiv:2405.13016.

---

## 🧪 Fine-tuning

The base model [`Helsinki-NLP/opus-mt-en-ar`](https://huggingface.co/Helsinki-NLP/opus-mt-en-ar) was fine-tuned on the DODa corpus to adapt from Modern Standard Arabic (MSA) to Moroccan Darija. Source sentences were prepended with the `>>ary<<` language token (ISO 639-3 code for Moroccan Arabic) to direct the model's output toward Darija rather than MSA.

| Parameter | Value |
|-----------|-------|
| Base model | Helsinki-NLP/opus-mt-en-ar |
| Training pairs | ~39,000 |
| Epochs | 4 |
| Learning rate | 2e-5 |
| Optimizer | AdamW |
| Hardware | Google Colab T4 GPU |
| BLEU score (test) | 17.27 |

The fine-tuned model is available on the HuggingFace Hub: **[Ilyass-42/opus-mt-en-ary-darija](https://huggingface.co/Ilyass-42/opus-mt-en-ary-darija)**

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/Ilyass-42/lost-in-darija.git
cd lost-in-darija
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download or train the model

The fine-tuned model is not included in the repo (too large). You can either:

**Option A — Download from HuggingFace Hub** *(recommended)*:
```python
from transformers import MarianMTModel, MarianTokenizer
tokenizer = MarianTokenizer.from_pretrained("Ilyass-42/opus-mt-en-ary-darija")
model = MarianMTModel.from_pretrained("Ilyass-42/opus-mt-en-ary-darija")
```

**Option B — Fine-tune yourself:**
```bash
python src/translation/fine_tune.py
```

**Option C — Use the base model** (no Darija adaptation):
Change `model_path` in `src/translation/translate.py` to `"Helsinki-NLP/opus-mt-en-ar"`.

### 4. Run the pipeline

```bash
python src/pipeline.py path/to/your/audio.mp3
```

The translated Darija audio will be saved to `data/results/`.

---

## 📊 Results

| Metric | Value |
|--------|-------|
| BLEU Score | 17.27 |
| Training epochs | 4 |
| Base model | Helsinki-NLP/opus-mt-en-ar |
| TTS Voice | ar-MA-JamalNeural (Moroccan Arabic) |
| Training hardware | Google Colab T4 GPU |

---

## ⚠️ Known Limitations

- TTS output has a slight MSA (Modern Standard Arabic) accent rather than a pure Darija accent
- BLEU score is modest — Darija is a low-resource language with limited parallel data

---

## 📌 Roadmap

- [x] Whisper STT integration
- [x] MarianMT fine-tuning on DODa
- [x] Edge TTS synthesis
- [x] BLEU evaluation
- [ ] Gradio UI for live demo
- [ ] Validation loss tracking during training
- [ ] Improve BLEU score with larger dataset or `transformer-big` variant
- [ ] Native Darija TTS voice

---

## 👤 Author

**Ilyass** — 3rd-year Computer Engineering Student, INSA Rouen (ML/AI track)  
[GitHub](https://github.com/Ilyass-42) · [LinkedIn](https://www.linkedin.com/in/ilyass-el-assad-965598321/)
