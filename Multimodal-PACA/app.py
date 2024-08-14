import gradio as gr # type: ignore
from utils import generate_audio_response, generate_text_response, set_user_response, transcribe_audio, personality_app, create_line_plot, predict_personality
from huggingface_hub import login # type: ignore
import os

# Function to handle audio input and update chatbot
def handle_audio_input(audio_file_path, chat_history):
    if audio_file_path is not None:
        # Transcribe the audio
        output = transcribe_audio(audio_file_path)
        personality_scores=personality_app(output)
        # Update the chat history with the transcription
        _, chat_history = set_user_response(output, chat_history)
        return output, chat_history, personality_scores
    return None, chat_history, None

def clear_audio():
    return None

def hide_textbox():
    return gr.Textbox(visible=False)

def open_textbox():
    return gr.Textbox(visible=True)

# Function to handle the model selection
def update_selected_model(selected_model):
    print(f"Selected model: {selected_model}")
    return selected_model

with gr.Blocks() as demo:
    gr.Markdown("<center><h1>Multimodal Personality Adaptive Conversational AI</h1></center>")
    gr.Markdown("<center><h5>Personality Adaptive AI This application uses LLMs to create a personality adaptive conversational AI that interacts with users and displays personality scores. (Description with links goes here)</h5></center>")
    with gr.Row():
        with gr.Column(scale=6): 
            # Audio recording component
            audio_input = gr.Microphone(sources=["microphone"], type="filepath", label="Tell Me How You're Feeling", container=True, interactive=True)
            output_text = gr.Textbox(label="Transcription", placeholder="What you said appears here..")
            chatbot = gr.Chatbot(label="Carebot", height=450) #Chatbot interface
            msg = gr.Textbox(label="Type your message here:") # Textbox for user input
            
            # with gr.Group():
            with gr.Row():
                Run = gr.Button("Run",variant="primary", size="sm")
                clear = gr.ClearButton(size="sm") #To clear the chat
                # generate = gr.Button("Generate", size="sm")
                # save_chat = gr.Button("Save", size="sm")
            
            # Display some query examples
            examples = gr.Examples(examples=["I'm feeling Sad all the time", "Tell me a joke.", "Cheer Me Up!", "Tell me about Seattle"], inputs=msg)
            #Clear the message
            clear.click(lambda: None, None, chatbot, queue=False)
            
        # Right side - Information, Visualization, and Dropdown
        with gr.Column(scale=4):
            # 1st component - Dropdown to choose models
            model_selection = gr.Dropdown(
            ["Llama-2-7b-chat-Counsel-finetuned", "Llama-3-8B", "gpt-4", "gpt-3.5-turbo"], label="Models", info="Choose your LLM model", value="Llama-2-7b-chat-Counsel-finetuned")
            
            # Textbox to display the selected model
            selected_model = gr.Textbox(label="Selected Model", interactive=False, visible=False) # not displayed in the app
            
            model_selection.change(fn=update_selected_model, inputs=model_selection, outputs=selected_model)
            
            # 2nd component - Live Personality Score Visualization
            personality_score = gr.LinePlot(x="Personality", y="Score",label="Personality Scores", height=300)

        #Generate responses to the user's audio query
        if audio_input is not None and output_text != None:
            
            gr.on(audio_input.change, fn=handle_audio_input, inputs=[audio_input, chatbot], outputs=[output_text, chatbot, personality_score], queue=False).then(fn=generate_audio_response, inputs=[chatbot,selected_model], outputs=chatbot)
            audio_input.change(clear_audio, inputs=None, outputs=audio_input)
            
            pass
        
        if msg is not None:
            # Submit the response to LLM
            gr.on(triggers=[msg.submit, Run.click],fn=personality_app, inputs=msg, outputs=personality_score).then(fn=set_user_response, inputs=[msg, chatbot], outputs=[msg, chatbot], queue=False).then(fn=generate_text_response, inputs=[chatbot, selected_model], outputs=chatbot)

# Launch the Gradio app
demo.queue()

if __name__ == '__main__':
    login(token = os.getenv("HF_TOKEN")) # HF Login
    demo.launch()
