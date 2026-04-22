""" Module : synthetise.py : sert synthetiser une voix pour un texte"""

from gtts import gTTS



def synthetise(texte: str):
    path ="data/results/" 
    tts = gTTS(texte,lang = "ar")
    index = 3
    tts.save(f"{path}audio{index}.mp3")


if __name__ == "__main__":

    texte = "مرحباً، أريد أن أطلب شطيرة"
    synthetise(texte)
    