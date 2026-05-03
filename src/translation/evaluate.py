from sacrebleu.metrics import BLEU
import pandas as pd
from pathlib import Path
from transformers import MarianTokenizer, MarianMTModel
from peft import PeftModel


test_path = Path(__file__).parent.parent.parent / "data" / "Test.csv"
model_path = Path(__file__).parent.parent.parent / "models" / "fine_tuned_marian_big"

tokenizer = MarianTokenizer.from_pretrained(model_path)
base_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-tc-big-en-ar")
model = PeftModel.from_pretrained(base_model, model_path)
model = model.to("cuda")
model.eval()

def translate(phrases: list[str]) :
    phrases = [">>ary<< " + texte for texte in phrases]
    inputs = tokenizer(phrases, return_tensors="pt",padding=True,truncation= True)
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    translated = model.generate(**inputs,num_beams=4,length_penalty=1.2)
    res = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]

    return res

batch_size = 32
test_df = pd.read_csv(test_path)
hypotheses = []
references = []
for index in range(0,len(test_df["eng"]),batch_size):
    predicted_phrase = translate(test_df["eng"][index:index+batch_size])
    hypotheses.extend(predicted_phrase)
references = [list(test_df["darija_ar"])]


bleu = BLEU(max_ngram_order=3)
bleu_score = bleu.corpus_score(hypotheses,references)
print(f"\n",{bleu_score})
