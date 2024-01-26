import tkinter as tk
from tkinter import scrolledtext
from tkinter.font import Font
from keras.models import load_model
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
import random
from transformers import pipeline

# Load model and data
lemmatizer = WordNetLemmatizer()
nltk.download('punkt')
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.keras')

# Initialize Hugging Face pipeline for emotion detection globally
emotion_classifier = pipeline('sentiment-analysis', model='j-hartmann/emotion-english-distilroberta-base')

# Initialize the Tkinter root window
window = tk.Tk()
window.title("Chatbot")
window.geometry("600x550")
window.configure(bg="#333333")

# Create the custom font
custom_font = Font(family="Helvetica", size=12)

# Function to preprocess sentence
def preprocess_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words if word.isalpha()]
    return sentence_words

# Function to create a bag of words
def bag_of_words(sentence):
    sentence_words = preprocess_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

# Function to predict class
def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.75  # Adjusted threshold for more accuracy
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    # If no class meets the threshold, return an empty list
    if not results:
        return []

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

# Enhanced Function to get response with better error handling
def get_response(intents_list, intents_json, emotion):
    if not intents_list:
        # Customized message for handling unknown queries
        return "I'm sorry, I didn't quite catch that. Could you please ask about our classes, training programs, or other services?"

    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            if 'emotion_responses' in i and emotion in i['emotion_responses']:
                result = random.choice(i['emotion_responses'][emotion])
            else:
                result = random.choice(i['responses'])
            break
    return result

# Typing animation function
def display_typing_effect(text, index=0):
    if index < len(text):
        ChatLog.config(state=tk.NORMAL)
        ChatLog.insert(tk.END, text[index], 'bot')
        ChatLog.config(state=tk.DISABLED)
        window.after(20, lambda: display_typing_effect(text, index + 1))
    else:
        ChatLog.config(state=tk.NORMAL)
        ChatLog.insert(tk.END, '\n\n')
        ChatLog.config(state=tk.DISABLED)
        ChatLog.yview(tk.END)

# Function to send messages
def send(event=None):
    msg = EntryBox.get("1.0", 'end-1c').strip()
    EntryBox.delete("0.0", tk.END)

    if msg != '':
        ChatLog.config(state=tk.NORMAL)
        ChatLog.insert(tk.END, "You: " + msg + '\n\n', 'user')
        ChatLog.config(foreground="#442265", font=custom_font)

        emotion = emotion_classifier(msg)[0]['label']
        ints = predict_class(msg)
        res = get_response(ints, intents, emotion)

        ChatLog.insert(tk.END, "Bot: ", 'bot')
        display_typing_effect(res)
        ChatLog.config(state=tk.DISABLED)
        ChatLog.yview(tk.END)

# Chat log frame
chat_frame = tk.Frame(window, bd=2, bg="#333333", relief=tk.SUNKEN)
chat_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

ChatLog = scrolledtext.ScrolledText(chat_frame, bd=0, bg="#333333", fg="#FFFFFF", font=custom_font, wrap=tk.WORD)
ChatLog.config(state=tk.DISABLED)
ChatLog.pack(fill=tk.BOTH, expand=True)

# User input box frame
input_frame = tk.Frame(window, bd=2, bg="#333333")
input_frame.pack(padx=10, pady=5, fill=tk.X, expand=False)

EntryBox = tk.Text(input_frame, bd=0, bg="#444444", fg="#FFFFFF", font=custom_font, width="29", height="5")
EntryBox.bind("<Return>", send)
EntryBox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
EntryBox.focus()

# Send button
SendButton = tk.Button(input_frame, font=custom_font, text="Send", width="12", height=5, bd=0, bg="#007BFF", activebackground="#007BFF", fg='#FFFFFF', command=send)
SendButton.pack(side=tk.RIGHT, fill=tk.X, expand=False)

# Configure text tags for different colors
ChatLog.tag_config('user', foreground="#FF0000")  # Red color for user text
ChatLog.tag_config('bot', foreground="#FFFFFF")  # White color for bot text

window.mainloop()
