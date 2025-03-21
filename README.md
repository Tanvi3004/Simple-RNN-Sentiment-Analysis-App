# Simple-RNN-Sentiment-Analysis-App

This project implements a sentiment analysis model using a Simple Recurrent Neural Network (RNN) trained on the IMDB movie reviews dataset. The model is built using TensorFlow/Keras and trained on preprocessed text data. The trained model is then used in a Streamlit web application to provide an interactive interface for users to predict the sentiment of movie reviews.

## Project Structure
 - simple_rnn_imdb.h5 - Saved model file after training, containing the learned parameters of the Simple RNN.
 - Prediction.ipynb - Jupyter Notebook for loading the trained model and making sentiment predictions on user-provided reviews.
 - main.py - Streamlit web application that provides an interactive interface for sentiment prediction using the trained model.
 - README.md - Documentation file that explains the project setup, installation, and usage.

## Installation
### Prerequisites
Before running the project, ensure that you have the following installed:

 - Python (>=3.7)
 - TensorFlow
 - Streamlit
 - NumPy
 - Keras

## Steps to Install and Run
### 1. Clone the repository:
```bash
git clone https://github.com/Tanvi3004/Simple-RNN-Sentiment-Analysis-App.git
cd Simple-RNN-Sentiment-Analysis
```
### 2. Create a virtual environment (optional but recommended):
``` bash
python -m venv env
source env/bin/activate  # On macOS/Linux
env\Scripts\activate  # On Windows
```
### 3.Install required dependencies:
```bash
Install required dependencies:
```
## Running the Streamlit App
To launch the Streamlit web application, use the following command:
```bash
streamlit run main.py
```

## Understanding the Model

### Data Preprocessing

The IMDB dataset is loaded using tensorflow.keras.datasets.imdb.

Reviews are tokenized and indexed as integers.

Reviews are padded to a fixed length using pad_sequences to ensure uniform input size.

### Model Architecture

An Embedding Layer maps words to dense vector representations.

A Simple RNN Layer processes the sequential data to extract temporal patterns.

A Dense Layer with Sigmoid Activation outputs a probability score for sentiment classification.

### Training Process

The model is compiled using the Adam optimizer and binary cross-entropy loss function.

An EarlyStopping callback prevents overfitting by stopping training when validation loss stagnates.

## Streamlit Sentiment Analysis App

This is how the app looks when running:

<img width="880" alt="Image" src="https://github.com/user-attachments/assets/75edf108-7be2-4a2a-8f85-9005e00aa9e1" />
<img width="861" alt="Image" src="https://github.com/user-attachments/assets/646fbd42-360f-44bb-b5eb-3abcb72e717b" />

# Note: 
The initial model had a low accuracy due to limitations in Simple RNN. To improve accuracy, consider using Bidirectional LSTM, pretrained embeddings (GloVe), and balancing the dataset. 🚀

# 🚀 How to Improve Prediction Score?
### 1. Use a More Advanced Model (LSTM or Bidirectional LSTM)
Simple RNNs are weak at capturing long-term dependencies. Replace SimpleRNN with Bidirectional LSTM.
### 2.Train for More Epochs
Your model might not have trained enough to learn proper representations.
Try increasing epochs gradually (e.g., from 5 → 10 → 15) to see improvements.
- Why?
  - Training longer allows the model to generalize better.
  - Monitor val_loss: If it increases while train_loss decreases, stop early (use EarlyStopping).
### 3. Use Pretrained Word Embeddings (GloVe or Word2Vec)
Instead of random word embeddings, use pretrained embeddings like GloVe for better word understanding
- Why?
  - Pretrained embeddings understand word meanings better.
  - Helps model generalize better, improving accuracy.
 ### 4. Balance the Dataset for Better Predictions
Check if your dataset is biased towards one sentiment
- Why?
  - If the dataset is imbalanced, the model learns a bias.
  - Balancing it ensures fair learning.
### 5.Lower Learning Rate for Better Generalization
Modify the learning rate for more stable training:
- Why?
  - A high learning rate makes training unstable.
  - A lower learning rate helps model learn fine-grained patterns.
## Final Steps
### 1. Replace SimpleRNN with Bidirectional LSTM ✅
### 2. Train for more epochs (10-15) with EarlyStopping ✅
### 3. Use pretrained embeddings (GloVe) ✅
### 4. Balance dataset (if needed) ✅
### 5. Reduce learning rate (0.0001) for stable training ✅

### Next Actions
Try these improvements one by one.
Retrain model and check new prediction scores.
# This should boost accuracy and confidence! 🚀
