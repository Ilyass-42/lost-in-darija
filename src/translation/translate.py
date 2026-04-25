""" Module : translate.py : sert a traduire grace a au modele Helsinki-NLP/opus-mt-en-ar"""

from pathlib import Path
from transformers import MarianTokenizer, MarianMTModel


model_path = Path(__file__).parent.parent.parent/"models"/"fine_tuned_marian_v2"

_model = None
_tokenizer = None


def translate(texte: str) :
    global _model, _tokenizer
    if _model is None:
        _tokenizer = MarianTokenizer.from_pretrained(model_path)
        _model = MarianMTModel.from_pretrained(model_path)
    texte = ">>ary<< " + texte
    inputs = _tokenizer(texte, return_tensors="pt")
    translated = _model.generate(**inputs)
    res = _tokenizer.decode(translated[0], skip_special_tokens=True)

    return res


if __name__ == "__main__":
    texte = "Hello, I would like to order a sandwich."
    res = translate(texte)
    print(res)
