### Sentiment Analysis using TF-IDF 
This project focuses on classifying IMDB movie reviews into positive or negative sentiments using classical machine learning algorithms. It demonstrates a complete NLP pipeline from text preprocessing to model evaluation.

 ### Project Overview
Data preprocessing: Data preprocessing is a crucial step in any NLP project. In this project, we used NLTK (Natural Language Toolkit) to clean and prepare the IMDB movie review texts. This involved several steps such as removing punctuation, converting text to lowercase, and eliminating stopwords — common words like “the,” “and,” and “is” that don’t carry meaningful sentiment. Tokenization was also applied, which splits the text into individual words or tokens to make analysis easier. Proper preprocessing helps reduce noise in the data and improves the quality of features extracted later.

# Feature extraction:
 Since machine learning algorithms require numerical input, the raw text data needs to be transformed into a format that models can understand. We used TF-IDF (Term Frequency-Inverse Document Frequency) vectorization to convert each review into a fixed-length numeric vector. TF-IDF highlights important words by giving higher weights to terms that appear frequently in a review but are rare across the entire dataset. This helps emphasize distinctive words that can better indicate sentiment. We limited the number of features to the top 5,000 terms to keep the model efficient while capturing meaningful patterns.

# Modeling: 
Two classic machine learning models were trained and compared for sentiment classification: Logistic Regression and Naive Bayes. Logistic Regression is a popular linear model well-suited for binary classification problems like sentiment analysis, and it estimates the probability that a review is positive or negative. Naive Bayes, on the other hand, is a probabilistic classifier based on Bayes' theorem and assumes independence between features, making it computationally efficient and effective for text classification tasks. Both models were trained on the TF-IDF vectors and fine-tuned to optimize their predictive performance.

# Evaluation: 
Measuring performance with accuracy, confusion matrix, and classification reports
To assess how well the models performed, we used multiple evaluation metrics. Accuracy measures the overall percentage of correct predictions. The confusion matrix provides detailed insight by showing counts of true positives, true negatives, false positives, and false negatives, which help identify the types of errors the models make. We also generated classification reports that include precision, recall, and F1-score for both positive and negative classes, giving a balanced view of model effectiveness. These metrics collectively ensure that the sentiment classification is reliable and useful for real-world applications.

### Included Files

Sentiment_Analysis_LogisticRegression_NaiveBayes.ipynb — Jupyter notebook with code, results, and visualizations

requirements.txt — List of Python libraries needed to run the project

README.md — This documentation file

### Libraries Used
pandas

numpy

scikit-learn

nltk

matplotlib

seaborn

