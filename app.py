import os
import openai
import streamlit as st
from streamlit_chat import message
import tempfile
import speech_recognition as sr
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
import numpy as np
from transformers import pipeline
import torch

# Initialize the inappropriate content detector
classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY") or st.secrets["OPENAI_API_KEY"]
def detect_inappropriate(text):
    results = classifier(text)
    for result in results:
        if result['label'] == 'NEGATIVE' and result['score'] > 0.85:
            return True
    return False

def record_audio():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Speak now...")
        audio = r.listen(source)
        st.info("Processing your voice...")
    return audio

def audio_to_text(audio):
    r = sr.Recognizer()
    try:
        text = r.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return "Could not understand audio"
    except sr.RequestError:
        return "Could not request results"

def text_to_audio(text):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts = gTTS(text=text, lang='en')
        tts.save(fp.name)
        audio = AudioSegment.from_mp3(fp.name)
        play(audio)
        os.unlink(fp.name)

def generate_response(prompt):
    if detect_inappropriate(prompt):
        return "I'm sorry, but I can't respond to that content."
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content

def main():
    st.title("Voice-Based Chatbot 🤖")
    st.write("Speak to the chatbot and get voice responses!")

    if 'generated' not in st.session_state:
        st.session_state['generated'] = []

    if 'past' not in st.session_state:
        st.session_state['past'] = []

    if 'audio_data' not in st.session_state:
        st.session_state['audio_data'] = None

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Record Voice"):
            audio = record_audio()
            st.session_state['audio_data'] = audio
            user_input = audio_to_text(audio)
            if user_input and user_input not in ["Could not understand audio", "Could not request results"]:
                st.session_state.past.append(user_input)
                output = generate_response(user_input)
                st.session_state.generated.append(output)
                text_to_audio(output)

    with col2:
        if st.button("Type Instead"):
            user_input = st.text_input("You:", key="text_input")
            if user_input:
                st.session_state.past.append(user_input)
                output = generate_response(user_input)
                st.session_state.generated.append(output)
                text_to_audio(output)

    if st.session_state['generated']:
        for i in range(len(st.session_state['generated'])-1, -1, -1):
            message(st.session_state["generated"][i], key=str(i))
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')

if __name__ == "__main__":
    main()
