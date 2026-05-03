import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from src.pipeline import run_pipline
import gradio as gr

interface = gr.Interface(
    fn=run_pipline,
    inputs=gr.Audio(type="filepath",label="audio"),
    outputs=[gr.Textbox(label="Transcription"),gr.Textbox(label="Traslation"),gr.Audio(label="Darija Audio")]
)

interface.launch()
