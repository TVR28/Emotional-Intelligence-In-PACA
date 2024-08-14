import openai # type: ignore
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv # type: ignore
from huggingface_hub import InferenceClient # type: ignore
import pandas as pd # type: ignore
import os, time

load_dotenv(find_dotenv())
# Setup API keys
openai.api_key = os.getenv("OPENAI_API_KEY")
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

client = OpenAI()

# Define a few-shot prompt for personality prediction
few_shot_prompt = """
You are an expert in personality psychology. Based on the text provided, predict the personality scores for the Big Five personality traits: Openness, Conscientiousness, Extraversion, Agreeableness, and Neuroticism. Each score should be a floating-point number between 0 and 1.

Example 1:
Text: "I love exploring new ideas and trying new things."
Scores: Openness: 0.9, Conscientiousness: 0.4, Extraversion: 0.7, Agreeableness: 0.5, Neuroticism: 0.3

Example 2:
Text: "I prefer to plan everything in advance and stick to the plan."
Scores: Openness: 0.3, Conscientiousness: 0.8, Extraversion: 0.4, Agreeableness: 0.6, Neuroticism: 0.4

Now, predict the scores for the following text:
"""

def predict_personality(text):
    # Prepare the prompt with the user's text
    prompt = few_shot_prompt + f"Text: \"{text}\"\nScores:"
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    
    # Call the OpenAI API to get the prediction
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=messages,
        max_tokens=50,
        temperature=0.5
    )
    
    # Extract the predicted scores from the response
    scores_text = response.choices[0].message.content.strip()
    scores = [float(score.split(":")[1].strip()) for score in scores_text.split(",")]
    return scores

def create_line_plot(scores):
    labels = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']
    data = {'Personality': labels, 'Score': scores}
    return pd.DataFrame(data)

# Gradio interface
def personality_app(text):
    scores = predict_personality(text)
    df = create_line_plot(scores)
    return df


def transcribe_audio(audio_path):
    with open(audio_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(model="whisper-1", file=audio_file)
        return transcript.text

def openai_chat_completion(messages: list, selected_model: str) -> list[str]:
    try:
        response = openai.chat.completions.create(
            # model='gpt-3.5-turbo',
            model=selected_model,
            messages=messages,
            # temperature=0.5,
        )
        collected_messages = response.choices[0].message.content.strip().split('\n')
        return collected_messages   # return all the collected chunks of messages
    
    except Exception as e:
        return [str(e)]

def llama2_chat_completion(messages: list, hf_model_id: str, selected_model: str) -> list[str]:
    try:
        hf_token = os.getenv("HF_TOKEN")
        client = InferenceClient(model=hf_model_id, token=hf_token)
        # Start the chat completion process with streaming enabled
        response_stream = client.chat_completion(messages, max_tokens=400, stream=True)

        # Collect the generated message chunks
        collected_messages = []
        for completion in response_stream:
            # Assuming the response structure is similar to OpenAI's
            delta = completion['choices'][0]['delta']
            if 'content' in delta.keys():
                collected_messages.append(delta['content'])
        # Return the collected messages
        return collected_messages
    
    except Exception as e:
        return [str(e)]

def generate_messages(messages: list) -> list:
    formatted_messages = [   # first format of messages for chat completion
        {
            'role': 'system',
            'content': 'You are a helpful assistant.'
        }
        
    ]
    for m in messages:   # Loop over the existing chat history and create user, assistant responses.
        formatted_messages.append({
            'role': 'user',
            'content': m[0]
        })
        if m[1] != None:
            formatted_messages.append({
                'role': 'assistant',
                'content': m[1]
            })
    return formatted_messages

def generate_audio_response(chat_history: list, selected_model: str) -> list:  # type: ignore
    messages = generate_messages(chat_history)  # generates messages based on chat history
    if selected_model == "gpt-4" or "gpt-3.5-turbo":
        bot_message = openai_chat_completion(messages, selected_model) # Get all the collected chunks of messages for streaming
    if selected_model == "Llama-3-8B":
        hf_model_id = "meta-llama/Meta-Llama-3-8B"
        bot_message = llama2_chat_completion(messages, hf_model_id, selected_model)
    if selected_model == "Llama-2-7b-chat-Counsel-finetuned":
        hf_model_id = "TVRRaviteja/Llama-2-7b-chat-Counsel-finetuned"
        bot_message = llama2_chat_completion(messages, hf_model_id, selected_model)
    else:
        selected_model='gpt-3.5-turbo'
        bot_message = openai_chat_completion(messages, selected_model)
    
    chat_history[-1][1] = ''        # [-1] -> last conversation, [1] -> current carebot message
    for bm in bot_message:     # Loop over the collected messages
        chat_history[-1][1] += bm
        time.sleep(0.05)
        yield chat_history   # For streamed carebot responses

def generate_text_response(chat_history: list, selected_model: str) -> list:  # type: ignore
    messages = generate_messages(chat_history) # generates messages based on chat history
    if selected_model == "gpt-4" or "gpt-3.5-turbo":
        bot_message = openai_chat_completion(messages, selected_model) # Get all the collected chunks of messages for streaming
    if selected_model == "Llama-3-8B":
        hf_model_id = "meta-llama/Meta-Llama-3-8B"
        bot_message = llama2_chat_completion(messages, hf_model_id, selected_model)
    if selected_model == "Llama-2-7b-chat-Counsel-finetuned":
        hf_model_id = "TVRRaviteja/Llama-2-7b-chat-Counsel-finetuned"
        bot_message = llama2_chat_completion(messages, hf_model_id, selected_model)
    else:
        selected_model='gpt-3.5-turbo'
        bot_message = openai_chat_completion(messages, selected_model)
        
    chat_history[-1][1] = ''        # [-1] -> last conversation, [1] -> current carebot message
    for bm in bot_message:     # Loop over the collected messages
        chat_history[-1][1] += bm
        time.sleep(0.05)
        yield chat_history   # For streamed carebot responses

def set_user_response(user_message: str, chat_history: list) -> tuple:
    chat_history += [[user_message, None]]  #Append the recent user message into the chat history
    return '', chat_history
