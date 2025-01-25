import re
import streamlit as st
from groq import Groq
from PIL import ImageGrab, Image
import cv2
import pyperclip
import google.generativeai as genai
from openai import OpenAI
import pyaudio
from faster_whisper import WhisperModel
import os
import speech_recognition as sr
import pyttsx3  # For text-to-speech
import time
import threading  # For threading support
from dotenv import load_dotenv
import os

load_dotenv()
# Initialization
wake_word = 'spidey'
groq_client = Groq(api_key="GROQ_API_KEY")
genai.configure(api_key='genaiapikey')
openai_client = OpenAI(api_key='openaikey')
web_cam = cv2.VideoCapture(0)

sys_msg = (
    'You are a multi-modal AI voice assistant. Your user may or may not have attached a photo for context '
    '(either a screenshot or a webcam capture). Any photo has already been processed into a highly detailed '
    'text prompt that will be attached to their transcribed voice prompt. Generate the most useful and '
    'factual response possible, carefully considering all previous generated text in your response before '
    'adding new tokens to the response. Do not expect or request images, just use the context if added. '
    'Use all of the context of this conversation so your response is relevant to the conversation. Make '
    'your responses clear and concise, avoiding any verbosity.'
)

convo = [{'role': 'system', 'content': sys_msg}]

generation_config = {
    'temperature': 0.7,
    'top_p': 1,
    'top_k': 1,
    'max_output_tokens': 2048
}

safety_settings = [
    {'category': 'HARM_CATEGORY_HARASSMENT', 'threshold': 'BLOCK_NONE'},
    {'category': 'HARM_CATEGORY_HATE_SPEECH', 'threshold': 'BLOCK_NONE'},
    {'category': 'HARM_CATEGORY_SEXUALLY_EXPLICIT', 'threshold': 'BLOCK_NONE'},
    {'category': 'HARM_CATEGORY_DANGEROUS_CONTENT', 'threshold': 'BLOCK_NONE'},
]

model = genai.GenerativeModel('gemini-1.5-flash-latest',
                              generation_config=generation_config,
                              safety_settings=safety_settings)

num_cores = os.cpu_count()
whisper_size = 'base'
whisper_model = WhisperModel(
    whisper_size,
    device='cpu',
    compute_type='int8',
    cpu_threads=num_cores // 2,
    num_workers=num_cores // 2
)

# Initialize Text to Speech engine
engine = pyttsx3.init()

# Functions
def groq_prompt(prompt, img_context):
    if img_context:
        prompt = f'USER PROMPT: {prompt}\n\n  IMAGE CONTEXT: {img_context}'
    convo.append({'role': 'user', 'content': prompt})
    chat_completion = groq_client.chat.completions.create(messages=convo, model='llama3-70b-8192')
    response = chat_completion.choices[0].message
    convo.append(response)

    return response.content

def take_screenshot():
    path = 'screenshot.jpg'
    screenshot = ImageGrab.grab()
    rgb_screenshot = screenshot.convert('RGB')
    rgb_screenshot.save(path, quality=15)
    return path

def web_cam_capture():
    if not web_cam.isOpened():
        return 'Error: Camera did not open successfully'
    path = 'webcam.jpg'
    ret, frame = web_cam.read()
    cv2.imwrite(path, frame)
    return path

def get_clipboard_text():
    clipboard_content = pyperclip.paste()
    if isinstance(clipboard_content, str):
        return clipboard_content
    else:
        return 'No Clipboard text to copy'

def vision_prompt(prompt, photo_path):
    img = Image.open(photo_path)
    prompt = (
        'You are the vision analysis AI that provides semantic meaning from images to provide context '
        'to send to another AI that will create a response to the user. Do not respond as the AI assistant '
        'to the user. Instead, take the user prompt input and try to extract all meaning from the photo '
        'relevant to the user prompt. Then generate as much objective data about the image for the AI '
        f'assistant who will respond to the user. \nUSER PROMPT: {prompt}'
    )
    response = model.generate_content([prompt, img])
    return response.text

def listen_to_voice_command():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        print("Listening for voice command...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        command = recognizer.recognize_google(audio)
        print(f"Voice Command: {command}")
        return command.lower()  # Convert command to lowercase for easy comparison
    except sr.UnknownValueError:
        print("Sorry, I did not understand that.")
        return None
    except sr.RequestError:
        print("Sorry, there was an issue with the speech recognition service.")
        return None

def speak_response(response):
    def speak():
        engine.say(response)
        engine.runAndWait()

    # Run speech synthesis in a separate thread to avoid blocking the main loop
    threading.Thread(target=speak).start()

def handle_voice_commands(command):
    if 'screen' in command or 'screenshot' in command:
        screenshot_path = take_screenshot()
        st.image(screenshot_path, caption="Screenshot Captured")

        # Automatically process the screenshot and generate a response
        img_context = vision_prompt(command, screenshot_path)
        response = groq_prompt(command, img_context)
        st.text_area("Assistant Response", response)
        speak_response(response)
    elif 'webcam' in command or 'camera' in command:
        webcam_path = web_cam_capture()
        if webcam_path.startswith('Error'):
            st.error(webcam_path)
        else:
            st.image(webcam_path, caption="Webcam Capture")

            # Automatically process the webcam image and generate a response
            img_context = vision_prompt(command, webcam_path)
            response = groq_prompt(command, img_context)
            st.text_area("Assistant Response", response)
            speak_response(response)
    else:
        # Default response for general queries or conversations
        response = groq_prompt(command, None)  # Use 'None' as there's no image context
        st.text_area("Assistant Response", response)
        speak_response(response)

# Streamlit Interface
st.title("Spidey AI Assistant")

# Placeholder to dynamically update the status
status_placeholder = st.empty()

# Function to continuously listen for commands
def continuous_listening():
    while True:
        status_placeholder.text("Listening for voice command...")
        voice_command = listen_to_voice_command()
        if voice_command:
            status_placeholder.text(f"Command received: {voice_command}")
            handle_voice_commands(voice_command)
        time.sleep(1)  # Add a small delay to avoid excessive resource usage

# Automatically start continuous listening in the background
if __name__ == "__main__":
    continuous_listening()
