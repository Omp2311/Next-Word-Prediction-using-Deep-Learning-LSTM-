import streamlit as st
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

### Load the All models

model = load_model('next_wrod_model.h5')

with open('tokenizer.pkl','rb') as file:
    tokenizer = pickle.load(file)

reverse_index = {idx:word for word,idx in tokenizer.word_index.items()}
max_len = 78


## num_word  = 10 means next 10 word predcit

def generate_text(seed_text,num_words=10):
    text = seed_text
    for _ in range(num_words):
        seq = tokenizer.texts_to_sequences([text])[0]
        padded = pad_sequences([seq],maxlen = max_len,padding = 'pre')
        preds = model.predict(padded,verbose=0)
        pos = np.argmax(preds)

        next_word = reverse_index.get(pos, " ")
        text += " " + next_word
    return text

st.title("Next Word Prediction With Deep Learning")
seed = st.text_input("Enter a Starting Text:", 'Hello')

num_word = st.slider('Number of words to generate', min_value=1, max_value=20, value=10, step=1)
if st.button('Generate'):
    result = generate_text(seed,num_word)
    st.write(result)