"""Évaluation unique : Test A, Test B ou TerjamaBench.

Exemples (depuis la racine du dépôt) :
    python src/translation/evaluate_mt.py --model Helsinki-NLP/opus-mt-en-ar --dataset test --split A
    python src/translation/evaluate_mt.py --model models/small_A_20261023_1430 --dataset terjama

Sorties :
    results/<modèle>_<jeu>[_<split>].csv   traductions
    results/<modèle>_<jeu>[_<split>].json  scores et réglages
"""
import argparse
import json
import os
from datetime import datetime

import pandas as pd
import torch
from sacrebleu.metrics import BLEU, CHRF
from transformers import MarianMTModel, MarianTokenizer

# Réglages fixes : les mêmes pour tous les modèles.
PREFIX = ">>ary<< "
NUM_BEAMS = 4
LENGTH_PENALTY = 1.2
BATCH_SIZE = 32
N_BOOTSTRAP = 1000
RESULTS_DIR = "results"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True, choices=["test", "terjama"])
    parser.add_argument("--split", type=str, choices=["A", "B"])
    args = parser.parse_args()
    if args.dataset == "test" and args.split is None:
        parser.error("--split est obligatoire avec --dataset test")
    if args.dataset == "terjama" and args.split is not None:
        parser.error("--split ne s'utilise pas avec --dataset terjama")
    return args


def output_stem(model, dataset, split):
    """'models/small_A_x/' + 'test' + 'A' -> 'small_A_x_test_A'."""
    name = os.path.basename(model.rstrip("/"))
    parts = [name, dataset] + ([split] if split else [])
    return "_".join(parts)


def load_data(dataset, split):
    """Retourne un DataFrame avec les colonnes english, reference (+ topic)."""
    if dataset == "test":
        df = pd.read_csv(f"./data/splits/{split}/test.csv")
        data = pd.DataFrame({"english": df["eng"], "reference": df["darija_ar"]})
    else:
        from datasets import load_dataset  # jeu protégé : jeton Hugging Face requis
        ds = load_dataset("atlasia/TerjamaBench")["test"]
        data = pd.DataFrame({
            "english": ds["English"],
            "reference": ds["Darija"],
            "topic": ds["topic"],
        })
    missing = data[["english", "reference"]].isna().sum().sum()
    assert missing == 0, f"{missing} valeurs vides dans english/reference"
    return data


def translate(sentences, model, tokenizer, device):
    hypotheses = []
    for i in range(0, len(sentences), BATCH_SIZE):
        batch = [PREFIX + s for s in sentences[i:i + BATCH_SIZE]]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, num_beams=NUM_BEAMS, length_penalty=LENGTH_PENALTY)
        hypotheses.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))
    return hypotheses


def score(hypotheses, references):
    """BLEU (ordre 4) et chrF++, avec intervalle de confiance par bootstrap."""
    bleu = BLEU()
    chrf = CHRF(word_order=2)
    b = bleu.corpus_score(hypotheses, [references], n_bootstrap=N_BOOTSTRAP)
    c = chrf.corpus_score(hypotheses, [references], n_bootstrap=N_BOOTSTRAP)
    return {
        "n": len(hypotheses),
        "bleu": float(b.score),
        "bleu_mean": float(b._mean),
        "bleu_ci": float(b._ci),
        "bleu_str": str(b),
        "bleu_signature": str(bleu.get_signature()),
        "chrf": float(c.score),
        "chrf_mean": float(c._mean),
        "chrf_ci": float(c._ci),
        "chrf_str": str(c),
        "chrf_signature": str(chrf.get_signature()),
    }


def main():
    args = parse_args()
    stem = output_stem(args.model, args.dataset, args.split)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = MarianTokenizer.from_pretrained(args.model)
    model = MarianMTModel.from_pretrained(args.model).to(device)
    model.eval()

    data = load_data(args.dataset, args.split)
    data["hypothesis"] = translate(data["english"].tolist(), model, tokenizer, device)

    refs = data["reference"].tolist()
    hyps = data["hypothesis"].tolist()
    report = {
        "model": args.model,
        "dataset": args.dataset,
        "split": args.split,
        "date": datetime.now().isoformat(timespec="seconds"),
        "decoding": {"prefix": PREFIX, "num_beams": NUM_BEAMS,
                     "length_penalty": LENGTH_PENALTY, "batch_size": BATCH_SIZE},
        # Vérifie si >>ary<< est un seul token pour ce tokenizer.
        "prefix_tokens": tokenizer.tokenize(PREFIX.strip()),
        "global": score(hyps, refs),
    }
    if args.dataset == "terjama":
        report["per_topic"] = {}
        for topic in sorted(data["topic"].unique()):
            part = data[data["topic"] == topic]
            report["per_topic"][topic] = score(part["hypothesis"].tolist(),
                                               part["reference"].tolist())

    data.to_csv(f"{RESULTS_DIR}/{stem}.csv", index=False, encoding="utf-8")
    with open(f"{RESULTS_DIR}/{stem}.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"{stem}")
    print(f"  BLEU  : {report['global']['bleu_str']}")
    print(f"  chrF++: {report['global']['chrf_str']}")
    print(f"  prefix_tokens: {report['prefix_tokens']}")
    if args.dataset == "terjama":
        for topic, s in report["per_topic"].items():
            print(f"  {topic:22s} BLEU {s['bleu']:6.2f}  chrF++ {s['chrf']:6.2f}")


if __name__ == "__main__":
    main()
