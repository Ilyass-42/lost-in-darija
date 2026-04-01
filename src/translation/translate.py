""" Module : translate.py : sert a traduire grace a deep translator"""

from deep_translator import GoogleTranslator



def translate(texte: str) : 
    res = GoogleTranslator('en','fr').translate(text = texte)

    return res


if __name__ == "__main__":

    texte = "Hello, I would like to order a sandwich."
    res = translate(texte)
    print(res)