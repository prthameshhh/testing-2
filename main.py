import streamlit as st
import json
import re
from crewai import Agent, Task, Process,Crew
from langchain_groq import ChatGroq
import tempfile
import os
from deepgram import DeepgramClient, PrerecordedOptions, FileSource

 
DG_KEY = "88b968f3e3cfc8eaf5a596e15c579ffca9a59aed"
deepgram = DeepgramClient(DG_KEY)

def transcribe_audio_file(audio_file_path):
    # Read the audio file from the local path
    with open(audio_file_path, "rb") as audio_file:
        buffer_data = audio_file.read()

    # Define the transcription options
    options = {
        "model": "nova-2",
        "smart_format": True,
        "language": "en",
        "diarize": True,
        "profanity_filter": False
    }
    payload = {
        "buffer": buffer_data,
    }
    # Call the transcribe_file method with the audio buffer and options
    response = deepgram.listen.prerecorded.v("1").transcribe_file(payload, options)
    return response

def process_diarized_transcript(res):
    transcript = res['results']['channels'][0]['alternatives'][0]
    words = res['results']['channels'][0]['alternatives'][0]['words']
    current_speaker = None
    current_sentence = []
    output = []
    for word in words:
        # This checks if the speaker has changed from the previous word.
        if current_speaker != word['speaker']:
            if current_sentence:
                output.append((current_speaker, ' '.join(current_sentence)))
                current_sentence = []
            current_speaker = word['speaker'] # This updates the current speaker.

        current_sentence.append(word['punctuated_word']) # adds current word to the sentence being built.

        # This checks if the current word ends a sentence (by punctuation).
        if word['punctuated_word'].endswith(('.', '?', '!')):
            output.append((current_speaker, ' '.join(current_sentence)))
            current_sentence = []

    # adds any remaining words as a final sentence.
    if current_sentence:
        output.append((current_speaker, ' '.join(current_sentence)))
    return output

def format_speaker(speaker_num):
    return f"speaker {speaker_num}"


def transcribe_and_process_audio(audio_file_path):
    # Transcribe the audio file
    res = transcribe_audio_file(audio_file_path)

    # Process the diarized transcript
    diarized_result = process_diarized_transcript(res)

    # Check if the result is available
    if not diarized_result:
        return "No transcription available. The audio might still be too low quality or silent."

    # Initialize an empty string variable to store the transcription
    transcription = ""

    # Open a text file to write the result
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
        file_path = temp_file.name
        # Iterate over the diarized result
        for speaker, sentence in diarized_result:
            # Format the speaker and sentence
            line = f"{format_speaker(speaker)}: {sentence}\n"

            # Append the line to the transcription variable
            transcription += line

            # Write the line to the text file
            temp_file.write(line.encode('utf-8'))

    return transcription


#Streamlit interface
st.title("Audio Transcription and Diarization")

uploaded_file = st.file_uploader("Choose an audio file", type=["mp3", "wav", "m4a"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as temp_audio_file:
        temp_audio_file.write(uploaded_file.read())
        temp_audio_file_path = temp_audio_file.name

    st.write("Transcribing audio...")
    transcription = transcribe_and_process_audio(temp_audio_file_path)

    st.write("Transcription:")
    st.text(transcription)
