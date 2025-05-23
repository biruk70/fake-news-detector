# Fake News Detector 📰🤖

This is a simple AI/ML project that detects fake news using Natural Language Processing (NLP) and a machine learning model (Passive Aggressive Classifier). Built with Python, scikit-learn, and NLTK.

## 🔧 How It Works

- Loads real and fake news datasets
- Cleans and vectorizes text using TF-IDF
- Trains a model to classify news as real or fake
- Evaluates the model using accuracy and a confusion matrix

## 📁 Files

- `fake_news_detector.py` — main training script
- `README.md` — project overview

## 📦 Dataset

You can download the dataset from [this Kaggle link](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset).

Place `Fake.csv` and `True.csv` in the same folder as the Python script to run the project.

## ✅ Accuracy

Tested accuracy: **~99%** on unseen data.

## 🚀 Tech Stack

- Python
- scikit-learn
- pandas
- NLTK
- matplotlib
