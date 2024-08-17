# Emotional-Intelligence-In-PACA

As a result of my independent research, this repository focuses on incorporate emotional intelligence into PACA, which aims to briefly explain multiple methods to build multiodal-PACA and finetuning of `Whisper` and `Llava` models on custom data. Refer to the individual folders in the repo for specific notebooks.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Modules](#modules)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Emotional intelligence in PACA is crucial to assess and understand the user personality and emotions from diverse human emotion ques through multimodal interactions(audio and text). One approach, unlike the standard method of performing ASR (Speech-to-text) and providing it as an input to a finetuned text LLM, is building an audio/multimodal language model(MPACA), which is capable of understanding even the slightest human emotion ques based on their tone and voice and assessing the human emotion, mood and personality to provide more relavant empathetic reponses. Unlike LTU and GAMA models, which possess audio understanding capabilities but lack speech emotion understanding, the focus on MPACA is mainly on providing the model with both audio & speech understanding abilities where it can understand human speech including every utterance and sound (like "hmmm", "ughh", ""huh? etc). There will be a strong emphasis on the explainable aspects of the model as well, where it can explain why it provides certain responses based on it's understanding. This aids in better model evaluation and AI safety which is crucial in Mental Health AI Assistants.

Another approach can be speech emotion recognition using text-based LLMs post ASR. Read more about it [here](https://github.com/YuanGongND/llm_speech_emotion_challenge/tree/main)
