# -*- coding: utf-8 -*-
"""Multimodal-AI-Assistant-Llava7B-Whisper"""

# Install necessary libraries for the project
!pip install -q -U transformers==4.37.2
!pip install -q bitsandbytes==0.41.3 accelerate==0.25.0
!pip install -q git+https://github.com/openai/whisper.git  # Installing Whisper from OpenAI
!pip install -q gradio
!pip install -q gTTS

import torch
from transformers import BitsAndBytesConfig, pipeline

# Configure model quantization to reduce memory usage and improve performance
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

# Specify the model ID for the image-to-text pipeline
model_id = "llava-hf/llava-1.5-7b-hf"

# Initialize the image-to-text pipeline with the specified model and quantization configuration
pipe = pipeline("image-to-text",
                model=model_id,
                model_kwargs={"quantization_config": quantization_config})

import whisper
import gradio as gr
import time
import warnings
import os
from gtts import gTTS
from PIL import Image

# Load and display an image
image_path = "img.jpg"
image = Image.open((image_path))
image

import nltk
nltk.download('punkt')
from nltk import sent_tokenize

# Set the maximum number of tokens to generate in the response
max_new_tokens = 200

# Define the prompt instructions for the image-to-text model
prompt_instructions = """
Describe the image using as much detail as possible,
is it a painting, a photograph, what colors are predominant, what's happening in the image
what is the image about?
"""

# Combine the user input and prompt instructions
prompt = "USER: <image>\n" + prompt_instructions + "\nASSISTANT:"

# Generate text description of the image using the pipeline
outputs = pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": 200})

# Display the generated text outputs
outputs

# Print each sentence of the generated text
for sent in sent_tokenize(outputs[0]["generated_text"]):
    print(sent)

# Suppress warnings
warnings.filterwarnings("ignore")

import warnings
from gtts import gTTS
import numpy as np

# Check if CUDA is available and set the device accordingly
torch.cuda.is_available()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using torch {torch.__version__} ({DEVICE})")

import whisper
# Load the Whisper model for speech recognition
model = whisper.load_model("medium", device=DEVICE)  # Model size can be tiny, small, base, medium, or large
print(
    f"Model is {'multilingual' if model.is_multilingual else 'English-only'} "
    f"and has {sum(np.prod(p.shape) for p in model.parameters()):,} parameters."
)

import re
import datetime
import os

# Create a log file with a timestamp
tstamp = datetime.datetime.now()
tstamp = str(tstamp).replace(' ','_')
logfile = f'{tstamp}_log.txt'

# Function to write history to the log file
def writehistory(text):
    with open(logfile, 'a', encoding='utf-8') as f:
        f.write(text)
        f.write('\n')
    f.close()

# Function to convert image to text description
def img2txt(input_text, input_image):
    # Load the image
    image = Image.open(input_image)

    # Log the input text and its properties
    writehistory(f"Input text: {input_text} - Type: {type(input_text)} - Dir: {dir(input_text)}")
    
    # Define prompt instructions based on input text type
    if type(input_text) == tuple:
        prompt_instructions = """
        Describe the image using as much detail as possible, is it a painting, a photograph, what colors are predominant, what's happening in the image, what is the image about?
        """
    else:
        prompt_instructions = """
        Act as an expert in imagery descriptive analysis, using as much detail as possible from the image, respond to the following prompt:
        """ + input_text

    # Log the prompt instructions
    writehistory(f"prompt_instructions: {prompt_instructions}")
    prompt = "USER: <image>\n" + prompt_instructions + "\nASSISTANT:"

    # Generate text description of the image
    outputs = pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": 200})

    # Extract the response text from the outputs
    if outputs is not None and len(outputs[0]["generated_text"]) > 0:
        match = re.search(r'ASSISTANT:\s*(.*)', outputs[0]["generated_text"])
        if match:
            # Extract the text after "ASSISTANT:"
            reply = match.group(1)
        else:
            reply = "No response found."
    else:
        reply = "No response generated."

    return reply

# Function to transcribe audio to text
def transcribe(audio):
    # Check if the audio input is None or empty
    if audio is None or audio == '':
        return ('','',None)  # Return empty strings and None audio file

    # Load and preprocess the audio
    audio = whisper.load_audio(audio)
    audio = whisper.pad_or_trim(audio)

    # Convert audio to mel spectrogram
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # Detect the language of the audio
    _, probs = model.detect_language(mel)

    # Decode the audio to text
    options = whisper.DecodingOptions()
    result = whisper.decode(model, mel, options)
    result_text = result.text

    return result_text

# Function to convert text to speech and save it as an audio file
def text_to_speech(text, file_path):
    language = 'en'

    # Convert text to speech using gTTS
    audioobj = gTTS(text = text,
                    lang = language,
                    slow = False)

    # Save the audio file
    audioobj.save(file_path)

    return file_path

import locale
print(locale.getlocale())  # Before running the pipeline
# Run the pipeline
print(locale.getlocale())  # After running the pipeline

# Set the preferred encoding to UTF-8
locale.getpreferredencoding = lambda: "UTF-8"  # Required for compatibility

# Generate a silent audio file using ffmpeg
!ffmpeg -f lavfi -i anullsrc=r=44100:cl=mono -t 10 -q:a 9 -acodec libmp3lame Temp.mp3

import gradio as gr
import base64
import os

# Function to process audio and image inputs
def process_inputs(audio_path, image_path):
    # Process the audio file using the transcribe function
    speech_to_text_output = transcribe(audio_path)

    # Process the image input using img2txt function
    if image_path:
        chatgpt_output = img2txt(speech_to_text_output, image_path)
    else:
        chatgpt_output = "No image provided."

    # Convert the chatgpt_output text to speech
    processed_audio_path = text_to_speech(chatgpt_output, "Temp3.mp3")  # Save the audio file

    return speech_to_text_output, chatgpt_output, processed_audio_path

# Create the Gradio interface for the multimodal AI assistant
iface = gr.Interface(
    fn=process_inputs,
    inputs=[
        gr.Audio(sources=["microphone"], type="filepath"),
        gr.Image(type="filepath")
    ],
    outputs=[
        gr.Textbox(label="Speech to Text"),
        gr.Textbox(label="AI Output"),
        gr.Audio("Temp.mp3")
    ],
    title="Multi Modal AI Assistant Using Whisper and Llava",
    description="Upload an image and interact via voice input and audio response."
)

# Launch the Gradio interface
iface.launch(debug=True)
