""" Module : pipeline.py : module qui regroupe les 3 modules """
print("pipeline.py démarre")
from stt.transcribe import transcribe
from translation.translate import translate
from tts.synthetise import synthetise
import sys
import whisper

if __name__=="__main__":
    
    if len(sys.argv)<2:
        print("There is one argument missing !!")
        sys.exit(0)

    model = whisper.load_model("tiny.en")

    param = sys.argv[1]
    print("Étape : Transcription")
    text_audio = transcribe(param)
    print(text_audio)
    print("Étape : Translation")
    text_traduit = translate(text_audio)
    print("Étape : Synthetise")
    synthetise(text_traduit)
