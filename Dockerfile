FROM python:3.14-slim

# ffmpeg est requis par Whisper pour décoder l'audio
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dépendances : étape stable -> en premier, profite du cache
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Télécharge le modèle fine-tuné et le sauvegarde exactement à l'endroit
# où translate.py va le chercher
RUN python -c "\
from transformers import MarianTokenizer, MarianMTModel; \
tok = MarianTokenizer.from_pretrained('ILyass-42/lost-in-darija-marian'); \
mdl = MarianMTModel.from_pretrained('ILyass-42/lost-in-darija-marian'); \
tok.save_pretrained('models/fine_tuned_marian_v2'); \
mdl.save_pretrained('models/fine_tuned_marian_v2')"

# Code de l'appli -> en dernier, change le plus souvent
COPY . .

EXPOSE 7860

CMD ["python", "app.py"]
