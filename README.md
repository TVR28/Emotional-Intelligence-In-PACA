# Emotional-Intelligence-In-PACA

As a result of my independent research, this repository focuses on incorporate emotional intelligence into PACA, which aims to briefly explain multiple methods to build multiodal-PACA and finetuning of `Whisper` and `Llava` models on custom data. Refer to the individual folders in the repo for specific notebooks.

## Table of Contents

- [Overview](#overview)
- [Modules](#modules)
- [Contributing](#contributing)
- [License](#license)

## Overview

Emotional intelligence in PACA is crucial to assess and understand the user personality and emotions from diverse human emotion ques through multimodal interactions(audio and text). One approach, unlike the standard method of performing ASR (Speech-to-text) and providing it as an input to a finetuned text LLM, is building an audio/multimodal language model(MPACA), which is capable of understanding even the slightest human emotion ques based on their tone and voice and assessing the human emotion, mood and personality to provide more relavant empathetic reponses. 

Unlike [LTU](https://github.com/YuanGongND/ltu) and [GAMA](https://sreyan88.github.io/gamaaudio/) models, which possess audio understanding capabilities but lack speech emotion understanding, the focus on MPACA is mainly on providing the model with both audio & speech understanding abilities where it can understand human speech including every utterance and sound (like "hmmm", "ughh", ""huh? etc). There will be a strong emphasis on the explainable aspects of the model as well, where it can explain why it provides certain responses based on it's understanding. This aids in better model evaluation and AI safety which is crucial in Mental Health AI Assistants.

Another approach can be speech emotion recognition using text-based LLMs post ASR. Read more about it [here](https://github.com/YuanGongND/llm_speech_emotion_challenge/tree/main)

## Modules

1. **Multimodal PACA**

Multimodal-PACA stands for Multimodal Personality Adaptive Conversational AI. It is an attempt to build a POC to implement audio feature in carebot using Whisper to interact with users through both text and speech, adapting its responses based on the user's personality traits and displaying their live personality scores. The application utilizes `Llama-2-counsel-finetuned`, `Llama3-8B`, OpenAI's `GPT-3.5-Turbo` and `GPT-4` models models and `Whisper` for ASR and TTS.

Checkout the full code [here](https://github.com/TVR28/Emotional-Intelligence-In-PACA/tree/main/Multimodal-PACA) and the model demo [here](https://huggingface.co/spaces/TVRRaviteja/Multimodal-PACA)

2. **Llava Finetuning**

A guide for fine-tuning LLaVA, a multimodal model integrating language and vision, on custom data using LoRA(Low-Rank Adaptation). It uses tools like DeepSpeed to enhance model performance for domain-specific tasks involving text and images. The folder includes a notebook demonstrating how to fine-tune LLaVA using DeepSpeed, which facilitates efficient distributed training. Key steps involve environment setup, model loading, and integrating LoRA weights for improved contextual understanding.

For full finetuning steps and guide, access the notebook [here](https://github.com/TVR28/Emotional-Intelligence-In-PACA/tree/main/Llava-Finetuning)

3. **Whisper Finetuning**

A step-by-step guide on how to fine-tune Whisper for any multilingual Automatic Speech Recognition dataset using Hugging Face 🤗 Transformers. I have finetuned the model on `Hindi` language using the Common Voice Dataset. The notebook is designed to be run on Google Colab with a T4 GPU and includes steps for data preparation, model training, evaluation, and deployment using Gradio.

For full finetuning steps and guide, access the notebook [here](https://github.com/TVR28/Emotional-Intelligence-In-PACA/tree/main/Whisper-Finetuning)

4. **Multimodal AI Assistant**

`Note`: This is a project I worked on, which can be helpful if you want to know more about Llava and whisper models implementations.

The Multimodal AI Assistant is an advanced tool designed to bridge the gap between human and computer interaction for VQA(Visual-Question Answering) tasks. Utilizing the power of OpenAI's `Whisper` and `LlaVa-7B` models, this assistant provides a seamless experience, processing live audio and visual inputs to deliver insightful and context-aware responses. This guide will help you set up and explore the full capabilities of the Multimodal AI Assistant.

Access the Multimodal AI Assistant [here](https://colab.research.google.com/drive/1EObkOG0Cpzm_6i0v1ryctEEfNlpX60cN?usp=sharing)


## Contributing
I'm happy to receive feedbacks or any questions on any implementations, updations or suggestions on the methods and techniques. Do give it a ⭐️ if you feel the work is helpful.

If you wish to contribute to the project, please follow these guidelines:
- Fork the repository.
- Create a new branch for your feature or bug fix.
- Submit a pull request with a detailed description of your changes.

## License
This project is licensed under the [MIT license] license. See the [LICENSE](LICENSE) file for more details.
