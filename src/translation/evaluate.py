from sacrebleu.metrics import BLEU
import pandas as pd
from pathlib import Path
from transformers import MarianTokenizer, MarianMTModel

test_path = "/content/lost-in-darija/data/Test.csv"
tokenizer = MarianTokenizer.from_pretrained("/content/lost-in-darija/models/fine_tuned_marian")
model = MarianMTModel.from_pretrained("/content/lost-in-darija/models/fine_tuned_marian")
model = model.to("cuda")

def translate(phrases: list[str]) :
    phrases = [">>ary<< " + texte for texte in phrases]
    inputs = tokenizer(phrases, return_tensors="pt",padding=True,truncation= True)
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    translated = model.generate(**inputs)
    res = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]

    return res

batch_size = 32
test_df = pd.read_csv(test_path)
hypohteses = []
references = []
for index in range(0,len(test_df["eng"]),batch_size):
    predicted_phrase = translate(test_df["eng"][index:index+batch_size])
    hypohteses.extend(predicted_phrase)
references = [list(test_df["darija_ar"])]


bleu = BLEU(max_ngram_order=3)
bleu_score = bleu.corpus_score(hypohteses,references)
print(f"\n",{bleu_score})
