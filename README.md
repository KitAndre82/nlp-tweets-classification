# nlp-tweets-classification

# Hate Speech Classification Project

### Overview

This project aims to develop machine learning models to classify hate speech, offensive language, and neutral tweets. The dataset used contains tweets labeled with one of three classes: hate speech, offensive language, or neither. The classification task is approached using both a one-model approach and a two-model approach. In the one-model approach, a single model is trained to classify tweets into all three classes. In the two-model approach, two separate models are trained: one to distinguish between neutral and bad language (offensive language and hate speech), and another to further classify bad language into offensive language and hate speech.

### Dataset

The dataset used for this project is a CSV file named labeled_data.csv, containing 24783 tweets with six columns:

tweet: The text content of the tweet.

class: The class label for the tweet, with three unique values: 0 (hate speech), 1 (offensive language), and 2 (neither).

hate_speech: The count of ratings classifying the tweet as hate speech.

offensive_language: The count of ratings classifying the tweet as offensive language.

neither: The count of ratings classifying the tweet as neither hate speech nor offensive language.

count: The total count of ratings for the tweet.

### Methodology

The project follows these main steps:

#### Data Preprocessing:

Compute tweet length and create a new feature.
Remove tweets longer than 280 characters.
Remove Twitter handles and preprocess text data.

#### Exploratory Data Analysis (EDA):

Visualize tweet length distribution by class.

Analyze class distribution.

#### Model Building:

Utilize various machine learning algorithms including Naive Bayes, Decision Tree, Random Forest, Support Vector Machine (SVM), and K Nearest Neighbors (KNN).
Evaluate models using metrics such as confusion matrix and classification report.

### Two-Model Approach:
Build separate models for classifying neutral language vs. bad language, and for further classifying bad language into offensive language vs. hate speech.

Files Included:

Code file: File containing the Python code for data preprocessing, EDA, model building, and evaluation.

labeled_data.csv: Dataset used for training and testing.

README.md: Readme file providing an overview of the project, instructions, and details about the dataset and methodology.

### Instructions:

Clone the repository to your local machine.
Ensure you have Python installed along with necessary libraries (Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, etc.).
Run the code file to reproduce the analysis, train the models, and evaluate the results.
Follow along with the comments in the file for a detailed understanding of each step.
Explore different models, hyperparameters, and feature engineering techniques for further experimentation and improvement.

### Results:

The one-model approach achieved moderate performance, with Decision Tree Classifier being the best-performing algorithm.
The two-model approach yielded better results, particularly with Support Vector Machines performing well for both Model 1 (neutral vs. bad language) and Model 2 (offensive language vs. hate speech).

### Conclusion:

Hate speech classification is a challenging task due to the nuanced nature of language and context.
Model performance can be improved by experimenting with different algorithms, feature engineering, and data augmentation techniques.
Continued research and development in this area are crucial for building effective tools to combat hate speech and promote a safer online environment.
