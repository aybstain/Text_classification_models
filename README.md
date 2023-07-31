# Topic Classification with ML Models

This repository contains the code and data for a topic classification project using various Machine Learning (ML) models. The goal is to classify news articles into four categories: World, Sports, Business, and Sci/Tech.

## Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Project Structure](#project-structure)
4. [Data Preprocessing](#data-preprocessing)
5. [ML Models](#ml-models)
6. [Confusion Matrix](#confusion-matrix)
7. [Usage](#usage)
8. [License](#license)

## Overview

In this project, we explore different ML models, including Random Forest, Naive Bayes, SVM, and CNN, for topic classification on a dataset of news articles. We compare their performance using accuracy and F1-score metrics.

## Prerequisites

To run the code in this repository, you'll need the following Python libraries:

- pandas
- numpy
- scikit-learn
- nltk
- keras
- gradio

You can install these dependencies using `pip`:


- `data`: get your data from "https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset?select=test.csv".
- `trained_cnn_model.h5`: The saved trained CNN model.
- `main.py`: Python script to load the trained CNN model and make predictions on a gradio app.
- `all_in_one.ipynb`: Jupyter notebook with the code for data preprocessing, model training, and evaluation.
- `README.md`: This readme file providing an overview of the project.

## Data Preprocessing

In the Jupyter notebook `all_in_one.ipynb`, we load and preprocess the news articles data. The preprocessing steps include tokenization, removing stop words, stemming, and converting the text data into sequences for feeding into the models.

## ML Models

We explore four different ML models for topic classification:

1. Random Forest
2. Naive Bayes
3. SVM
4. CNN

Each model is trained on the preprocessed data and evaluated on the test set. We present the accuracy and the classification report that includes precision, recall, and F1-score for each class.

## Confusion Matrix

We visualize the performance of each ML model using confusion matrices. The confusion matrix helps us to understand the model's predictions and identify areas of improvement.

## Usage

Run the main.py script in your local environment or on your server. The script will start the Gradio app and provide a user interface for entering text. The Gradio app will display the predicted class on the screen

## Results
<img width="690" alt="Capture1" src="https://github.com/aybstain/Text_classification_models/assets/103702856/63275bfc-e695-4ec0-9cf5-816b140a4e6c">

<img width="704" alt="Capture" src="https://github.com/aybstain/Text_classification_models/assets/103702856/ad287adf-8b90-41f2-8504-ed83f79088c6">

<img width="692" alt="Capture2" src="https://github.com/aybstain/Text_classification_models/assets/103702856/441e2aff-c771-4024-9083-e30c1c59a03e">

<img width="748" alt="capture3" src="https://github.com/aybstain/Text_classification_models/assets/103702856/22003465-48ed-47a8-94dd-034793864900">


