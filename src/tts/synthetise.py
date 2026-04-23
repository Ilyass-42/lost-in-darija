""" Module : synthetise.py : sert synthetiser une voix pour un texte"""

import asyncio
import edge_tts
VOICE ="ar-MA-JamalNeural"

def synthetise(texte,output_path):
    async def amain() -> None:
        """ Main Function"""
        communicate = edge_tts.Communicate(texte,VOICE)
        await communicate.save(output_path)
    asyncio.run(amain())


if __name__ == "__main__":
    synthetise("مرحباً، أريد أن أطلب شطيرة","data/results/sandwich_order2.mp3")
    
    