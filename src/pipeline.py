""" Module : pipeline.py """
from stt.transcribe import transcribe
from translation.translate import translate
from tts.synthetise import synthetise
import sys
import os

def run_pipline(audio_path):
        print("Étape : Transcription")
        try:
            text_audio = transcribe(audio_path)
        except Exception as e:
            raise RuntimeError(f"Erreur STT : {e}")
        
        print("Étape : Translation")
        try:
            text_traduit = translate(text_audio)
        except Exception as e:
            raise RuntimeError(f"Erreur Translation : {e}")

        print("Étape : Synthetise")
        try:
            nom_fichier = os.path.basename(audio_path)
            nom_audio= os.path.splitext(nom_fichier)[0]
            output_path = f"data/results/{nom_audio}_darija.mp3"
            synthetise(text_traduit,output_path)
        except Exception as e:
            raise RuntimeError(f"Erreur TTS : {e}")
        
        return text_audio, text_traduit, output_path





if __name__=="__main__":
    print("pipeline.py démarre")
    
    if len(sys.argv)<2:
        print("There is one argument missing !!")
        sys.exit(0)


    param = sys.argv[1]
    print("Étape : Transcription")
    try:
        text_audio = transcribe(param)
    except Exception as e:
        print(f"Erreur STT : {e}")
        sys.exit(1)
    print(text_audio)
    print("Étape : Translation")
    try:
        text_traduit = translate(text_audio)
    except Exception as e:
        print(f"Erreur Translation : {e}")
        sys.exit(1)
    print(f"Texte Traduit : {text_traduit}")
    print("Étape : Synthetise")
    try:
        nom_fichier = os.path.basename(param)
        nom_audio= os.path.splitext(nom_fichier)[0]
        output_path = f"data/results/{nom_audio}_darija.mp3"
        synthetise(text_traduit,output_path)
    except Exception as e:
        print(f"Erreur TTS : {e}")
        sys.exit(1)
