""" Module : translate.py : sert a traduire grace a au modele Helsinki-NLP/opus-mt-en-ar"""

from pathlib import Path
from transformers import MarianTokenizer, MarianMTModel


model_path = Path(__file__).parent.parent.parent/"models"/"fine_tuned_marian_v2"

tokenizer = MarianTokenizer.from_pretrained(model_path)
model = MarianMTModel.from_pretrained(model_path)


def translate(texte: str) :  
    texte = ">>ary<< " + texte
    inputs = tokenizer(texte, return_tensors="pt")
    translated = model.generate(**inputs)
    res = tokenizer.decode(translated[0], skip_special_tokens=True)

    return res


if __name__ == "__main__":
    texte = "Hello, I would like to order a sandwich."
    res = translate(texte)
    print(res)
