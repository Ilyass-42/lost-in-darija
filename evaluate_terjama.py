from datasets import load_dataset
from sacrebleu.metrics import BLEU
from pathlib import Path
from transformers import MarianTokenizer, MarianMTModel


ds = load_dataset("atlasia/TerjamaBench")
# ds_common = ds["test"].filter(lambda x:x ["topic"]== "common_phrases")
print(set(ds["test"]["topic"]))


model_path = Path(__file__).parent / "models" / "fine_tuned_marian_v2"

tokenizer = MarianTokenizer.from_pretrained(model_path)
model = MarianMTModel.from_pretrained(model_path)
model = model.to("cpu")
model.eval()

def translate(phrases: list[str]) :
    phrases = [">>ary<< " + texte for texte in phrases]
    inputs = tokenizer(phrases, return_tensors="pt",padding=True,truncation= True)
    inputs = {k: v.to("cpu") for k, v in inputs.items()}
    translated = model.generate(**inputs,num_beams=4,length_penalty=1.2)
    res = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]

    return res

batch_size = 32

scores = {}
for categorie in set(ds["test"]["topic"]):
    hypotheses = []
    references = []
    ds_categorie = ds["test"].filter(lambda x:x ["topic"]== categorie)
    for index in range(0,len(ds_categorie["English"]),batch_size):
        predicted_phrase = translate(ds_categorie["English"][index:index+batch_size])
        hypotheses.extend(predicted_phrase)
    references = [list(ds_categorie["Darija"])]

    bleu = BLEU(max_ngram_order=3)
    bleu_score = bleu.corpus_score(hypotheses,references)
    scores[f"bleu_score_{categorie}"] = bleu_score.score
    
print(scores)
