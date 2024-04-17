# Grammar Error Correction Using Deep Learning

## Overview
This project is focused on developing a deep learning-based application for correcting grammatical errors in English text. It leverages the capabilities of the T5 (Text-to-Text Transfer Transformer) model, which has been fine-tuned on the Lang-8 dataset to specifically address the task of grammar correction.

## Key Features
- **Grammar Correction Model**: Utilizes a fine-tuned T5 model to identify and correct grammatical errors.
- **Dataset**: Trained on the Lang-8 dataset, which is rich in various grammatical error types made by language learners.
- **Streamlit App**: A user-friendly web application built with Streamlit that allows users to input text and receive grammatically corrected output.

## Technologies Used
- **T5ForConditionalGeneration**: A pre-trained transformer model from Hugging Face's Transformers library, fine-tuned for grammar correction.
- **Hugging Face Transformers**: Provides the architecture for T5 and the utilities for working with pre-trained models.
- **Streamlit**: An open-source app framework for Machine Learning and Data Science projects.
- **PyTorch**: As the underlying framework supporting the Transformers library.
- **SentencePiece**: A library for subword tokenization necessary for T5 model inputs.

## Project Structure
- `app.py`: The main Streamlit application script.
- `model/`: Directory containing the fine-tuned model files.
- `tokenizer/`: Directory containing the tokenizer files.

## Setup and Installation
To set up this project locally, follow these steps:

1. **Clone the Repository**
   ```bash
   git clone https://github.com/HAKIM-ML/Grammar-Error-Correction-Using-Deep-Learning.git
   cd grammar-error-correction
