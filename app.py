import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from src.pipeline import run_pipline
import gradio as gr

with gr.Blocks() as interface:
    audio_input = gr.Audio(type="filepath", label="audio")
    transcription = gr.Textbox(label="Transcription")
    translation = gr.Textbox(label="Traslation")
    darija_audio = gr.Audio(label="Darija Audio")

    outputs = [transcription, translation, darija_audio]
    audio_input.stop_recording(fn=run_pipline, inputs=audio_input, outputs=outputs)
    audio_input.upload(fn=run_pipline, inputs=audio_input, outputs=outputs)

interface.launch(server_name="0.0.0.0", server_port=7860)
