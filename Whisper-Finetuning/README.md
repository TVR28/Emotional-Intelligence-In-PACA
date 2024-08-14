# Fine-Tuning Whisper for Multilingual ASR

This repository contains a Jupyter Notebook for fine-tuning the Whisper model on the Common Voice dataset for multilingual automatic speech recognition (ASR). The notebook is designed to be run on Google Colab with a T4 GPU and includes steps for data preparation, model training, evaluation, and deployment using Gradio.

Access the finetuned whisper model on Hindi language on my HugingFace space: [TVRRaviteja/whisper-finetuned-hi](https://huggingface.co/TVRRaviteja/whisper-finetuned-hi/tree/main)

![image](https://github.com/user-attachments/assets/c8610193-b75c-4360-b23a-75b741480a2a)


## Table of Contents

- [Introduction](#introduction)
- [Setup and Installation](#setup-and-installation)
- [Data Preparation](#data-preparation)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Deployment](#deployment)
- [Acknowledgments](#acknowledgments)
- [References](#References)
- [License](#license)

## Introduction

Whisper is a state-of-the-art model for automatic speech recognition developed by OpenAI. This project demonstrates how to fine-tune the Whisper model using the Hugging Face Transformers library on the Common Voice dataset, which includes multilingual audio samples.

![image](https://github.com/user-attachments/assets/8bc48470-2546-443c-8c62-4cb102c1ae8e)

## Setup and Installation

To run the notebook, you need to have access to Google Colab. The following packages will be installed as part of the setup:

- `datasets[audio]`
- `transformers`
- `accelerate`
- `evaluate`
- `jiwer`
- `tensorboard`
- `gradio`

You will also need to authenticate with the Hugging Face Hub to access the datasets and push the fine-tuned model.

## Data Preparation

The Common Voice dataset is used for training and evaluation. The dataset is preprocessed by removing unnecessary columns and resampling audio to a consistent sampling rate. The Hindi subset of the dataset is used in this example, but the notebook can be adapted for other languages.

## Model Training

The Whisper model is fine-tuned using the `Seq2SeqTrainer` from the Hugging Face Transformers library. Key components of the training setup include:

- **Feature Extractor**: Converts audio inputs into feature vectors.
- **Tokenizer**: Transforms text into token IDs.
- **Processor**: Combines the feature extractor and tokenizer.
- **Data Collator**: Prepares batches of data for training.
- **Training Arguments**: Configure the training process, including batch size, learning rate, and evaluation strategy.

## Evaluation

The model's performance is evaluated using the Word Error Rate (WER) metric. The evaluation script computes WER by comparing the model's predictions with the ground truth transcriptions.

## Deployment

After fine-tuning, the model is pushed to the Hugging Face Hub and deployed using Gradio for real-time ASR. The Gradio interface allows users to input audio via a microphone and receive transcriptions in real-time.

## Acknowledgments

This project is based on the [Hugging Face blog post](https://huggingface.co/blog/fine-tune-whisper) on fine-tuning Whisper for multilingual ASR. Special thanks to the authors and contributors for providing detailed guidance and resources.

## References
[LLM-Based Post ASR Speech Emotion Recognition Challenge](https://github.com/YuanGongND/llm_speech_emotion_challenge/tree/main)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
