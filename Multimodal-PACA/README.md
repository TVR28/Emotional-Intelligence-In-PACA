# Multimodal-PACA: Personality Adaptive Conversational AI

Multimodal-PACA is a sophisticated conversational AI application designed to adapt to user personalities through multimodal interactions. It leverages advanced natural language processing (NLP) and speech synthesis technologies to provide personalized and engaging conversations.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Modules](#modules)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Multimodal-PACA stands for Multimodal Personality Adaptive Conversational AI. It is designed to interact with users through both text and speech, adapting its responses based on the user's personality traits. The application utilizes OpenAI's GPT models and Whisper for speech recognition, alongside personality prediction algorithms to enhance user interaction.

## Features

- **Personality Prediction**: Analyzes user input to predict personality traits based on the Big Five model usig GPT-4.
- **Speech Synthesis**: Converts text responses into natural-sounding speech using OpenAI's Whisper API.
- **Multimodal Interaction**: Supports both text and audio inputs for a seamless user experience.

## Installation

To set up Multimodal-PACA, follow these steps:

1. **Clone the Repository**:
```bash
git clone https://github.com/TVR28/Emotional-Intelligence-In-PACA.git
cd multimodal-paca
```
2. Install Dependencies:
Ensure you have Python installed. Then, install the required packages:
  ```bash
  pip install -r requirements.txt
  ```
3. Environment Variables:
Set up your environment variables for OpenAI and Hugging Face API keys. Create a `.env` file in the root directory:
  ``` text
  OPENAI_API_KEY=your_openai_api_key
  HF_TOKEN=your_huggingface_token
  ```
## Usage
To run the application, execute the following command:
  ``` python
  python app.py
  ```
This will launch a Gradio interface where you can interact with the AI using text and audio inputs.

## Modules

`utils.py`
- Personality Prediction: Uses a few-shot prompt to predict personality scores from text.
- Speech Synthesis: Generates speech from text using OpenAI's API.
- Chat Completion: Handles both OpenAI and Hugging Face chat models for generating responses.

`sample_audio.py`
- Text-to-Speech Conversion: Converts input text into speech and plays the audio using Gradio.

`app.py`
- Main Application: Integrates all functionalities into a Gradio interface, handling user inputs and updating chat history.

## Contributing
Contributions are welcome! Please follow these steps to contribute:
1. Fork the repository.
2. Create a new branch (git checkout -b feature/YourFeature).
3. Commit your changes (git commit -m 'Add your feature').
4. Push to the branch (git push origin feature/YourFeature).
5. Open a pull request.

