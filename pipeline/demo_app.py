#!/usr/bin/env python3
"""Web demo: microphone input → English speech output"""
import gradio as gr
from pipeline.orchestrator import SpeechToSpeechPipeline

pipe = SpeechToSpeechPipeline()

def demo(audio, ref_voice=None):
    out_wav, eng, rel = pipe.translate(audio, ref_voice)
    return out_wav, f"Rendille: {rel}\n\nEnglish: {eng}"

with gr.Blocks() as demo:
    gr.Markdown("# 🎤 Rendille → English Speech Translator")
    gr.Markdown("Speak Rendille, get English audio response")
    with gr.Row():
        mic = gr.Audio(source="microphone", type="filepath")
        ref = gr.Audio(label="Optional voice reference (3-6 sec)", type="filepath")
    btn = gr.Button("Translate")
    with gr.Row():
        out_audio = gr.Audio(label="English Translation")
        out_text = gr.Textbox(label="Transcription", lines=5)
    btn.click(fn=demo, inputs=[mic, ref], outputs=[out_audio, out_text])

if __name__ == "__main__":
    demo.launch(share=True)  # creates public link
