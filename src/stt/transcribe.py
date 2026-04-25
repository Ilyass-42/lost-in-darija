""" Module : transcribe.py : sert a transformer la voix en texte (speech to text)"""

import whisper
import sys

_model = None

def transcribe(path : str) :
    global _model
    if _model is None:
        _model = whisper.load_model("tiny.en")

    result = _model.transcribe(path)

    return result["text"]


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("A paremeter is missing !!")
        sys.exit(0)

    param = sys.argv[1]
    result = transcribe(param)

    print( f"THE MESSAGE : {result}")
