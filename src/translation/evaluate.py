from sacrebleu.metrics import BLEU
from translate import translate
import pandas as pd
from pathlib import Path

test_path = Path(__file__).parent.parent.parent / "data" / "Test.csv"


if __name__=="__main__":
    test_df = pd.read_csv(test_path)
    hypohteses = []
    references = []
    for phrase_eng in test_df.head(10)["eng"]:
        predicted_phrase = translate(phrase_eng)
        hypohteses.append(predicted_phrase)
    for phrases_darija in test_df.head(10)["darija_ar"]:
        references.append([phrases_darija])


    bleu = BLEU()
    bleu_score = bleu.corpus_score(hypohteses,references)
    print(f"\n",{bleu_score})
