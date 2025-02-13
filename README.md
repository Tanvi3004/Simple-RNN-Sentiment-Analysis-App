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
https://github.com/Tanvi3004/Simple-RNN-Sentiment-Analysis-App/tree/main
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
