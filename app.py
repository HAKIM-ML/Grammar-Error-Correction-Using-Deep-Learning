import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

# Set Streamlit page configuration as the first command
st.set_page_config(page_title="Grammar Correction App", layout="wide")

model_path = 'D:\\Deep Learning\\End to End Projects\\Grammer Corretion\\saved_model'
tokenizer_path = 'D:\\Deep Learning\\End to End Projects\\Grammer Corretion\\tokenizer'

@st.experimental_singleton
def load_model():
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)
    return model, tokenizer

model, tokenizer = load_model()

def correct_grammar(input_text):
    inputs = tokenizer.encode("grammar: " + input_text, return_tensors="pt", add_special_tokens=True)
    outputs = model.generate(inputs, max_length=128, num_beams=4, early_stopping=True)
    corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return corrected_text

def highlight_differences(original, corrected):
    from difflib import SequenceMatcher
    matcher = SequenceMatcher(None, original.split(), corrected.split())
    display_text = ""

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            display_text += original.split()[i1:i2][0] + " "
        elif tag == 'insert':
            display_text += f"<span style='background-color: #87CEEB;'>{' '.join(corrected.split()[j1:j2])}</span> "
        elif tag == 'delete':
            display_text += f"<span style='background-color: #FF6347;text-decoration: line-through;'>{' '.join(original.split()[i1:i2])}</span> "
        elif tag == 'replace':
            display_text += f"<span style='background-color: #FF6347;text-decoration: line-through;'>{' '.join(original.split()[i1:i2])}</span> <span style='background-color: #87CEEB;'>{' '.join(corrected.split()[j1:j2])}</span> "

    return display_text.strip()

def add_custom_css():
    st.markdown("""
        <style>
        .stTextArea {
            height: 150px;
        }
        </style>
    """, unsafe_allow_html=True)

# Custom CSS can be called later in the code
add_custom_css()

st.title('Grammar Error Correction App')
input_text = st.text_area('Enter text with grammar mistakes:', value='', height=150)

if st.button('Correct Grammar'):
    if input_text:
        corrected_text = correct_grammar(input_text)
        st.write("### Corrected Text")
        st.write(corrected_text)
        st.markdown("### Detailed Correction")
        st.markdown(highlight_differences(input_text, corrected_text), unsafe_allow_html=True)
    else:
        st.write('Please enter some text to correct.')
