""" Module : pipeline.py : module qui regroupe les 3 modules """
print("pipeline.py démarre")
from stt.transcribe import transcribe
from translation.translate import translate
from tts.synthetise import synthetise
import sys
import os

if __name__=="__main__":
    
    if len(sys.argv)<2:
        print("There is one argument missing !!")
        sys.exit(0)


    param = sys.argv[1]
    print("Étape : Transcription")
    text_audio = transcribe(param)
    print(text_audio)
    print("Étape : Translation")
    text_traduit = translate(text_audio)
    print(f"Texte Traduit : {text_traduit}")
    print("Étape : Synthetise")
    nom_fichier = os.path.basename(param)
    nom_audio= os.path.splitext(nom_fichier)[0]
    output_path = f"data/results/{nom_audio}_darija.mp3"
    synthetise(text_traduit,output_path)
