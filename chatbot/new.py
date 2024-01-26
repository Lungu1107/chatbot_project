import random
import json
import pickle
import numpy as np
import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer
from transformers import pipeline

# Download necessary NLTK datasets
nltk.download('punkt')
nltk.download('wordnet')

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Load the intents file
try:
    with open('intents.json') as file:
        intents = json.load(file)
except FileNotFoundError:
    print("Error: The 'intents.json' file was not found.")
    exit()
except json.JSONDecodeError:
    print("Error: JSON decoding has failed.")
    exit()

words, classes, documents = [], [], []

# Process each sentence in our intents patterns
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize each word in the sentence
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        # Add to documents in our corpus
        documents.append((word_list, intent['tag']))
        # Add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize and lower each word and remove duplicates
words = [lemmatizer.lemmatize(word.lower()) for word in words if word.isalpha()]
words = sorted(set(words))

# Sort classes
classes = sorted(set(classes))

# Create training data
training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    # List of tokenized words for the pattern
    pattern_words = doc[0]
    # Lemmatize each word - create base word, in attempt to represent related words
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # Create our bag of words array with 1, if word match found in current pattern
    for word in words:
        bag.append(1) if word in pattern_words else bag.append(0)
    
    # Output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    
    training.append([bag, output_row])

# Shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training, dtype=object)

# Create train and test lists. X - patterns, Y - intents
train_X = np.array(list(training[:, 0]), dtype=float)
train_Y = np.array(list(training[:, 1]), dtype=float)

# Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
# equal to number of intents to predict output intent with softmax
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(128, input_shape=(len(train_X[0]),), activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(len(train_Y[0]), activation='softmax'))

# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fitting and saving the model
model.fit(train_X, train_Y, epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.keras', save_format='h5')

# Save words and classes
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Initialize Hugging Face pipeline for emotion detection
emotion_classifier = pipeline('sentiment-analysis', model='j-hartmann/emotion-english-distilroberta-base')

# Define a function to get a response from the chatbot, including emotion detection
def chatbot_response(text):
    ints = predict_class(text)
    emotion = emotion_classifier(text)[0]['label']
    res = get_response(intents_list=ints, intents_json=intents, emotion=emotion)  # Pass all three arguments
    return res

# Add the bag_of_words function definition here
def bag_of_words(sentence):
    # Tokenize the sentence
    sentence_words = nltk.word_tokenize(sentence)
    # Lemmatize and lowercase each word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    # Create an array of 0s for each word in the vocabulary
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    # Generate bag of words from the sentence
    bow = bag_of_words(sentence)
    # Predict the class using the model
    model_output = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(model_output) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json, emotion):
    print(f"Detected Emotion: {emotion}")  # Debugging print
    if not intents_list:
        return "I'm sorry, I didn't understand that. Can you rephrase it?"

    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            if 'emotion_responses' in i and emotion in i['emotion_responses']:
                result = random.choice(i['emotion_responses'][emotion])
                # Removed the debugging print statement
            else:
                result = random.choice(i['responses'])
            break
    return result

# Example usage
response = chatbot_response("I am feeling sad but want to learn karate")
response1 = chatbot_response("The thought of joining a martial arts class makes me feel nervous.")
print(response)
print(response1)
