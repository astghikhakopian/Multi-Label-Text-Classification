import pandas as pd
import nltk

# vectorization
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# metrics
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import hamming_loss

# BinaryRelevance
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.linear_model import LogisticRegression

import TextCleaningHelper


# Load Training data
df = pd.read_csv('data/clean/Trainning_reviews.csv')
initial_df = pd.read_csv('data/clean/Trainning_reviews.csv')

# cleaning up datasource 
df['Description'] = df['Description'].apply(TextCleaningHelper.clean_up)

df.rename(columns = {'Symptoms':'symptom_list'}, inplace = True) 

# extract symptoms
symptoms = [] 

for i in df['symptom_list']: 
    symptoms.append(i.split(', ')) 
# add to  dataframe  
df['Symptoms'] = symptoms

# get all symptom tags in a list
all_symptoms = sum(symptoms,[])

all_symptoms = nltk.FreqDist(all_symptoms) 
# create dataframe
all_symptoms_df = pd.DataFrame({'Symtom': list(all_symptoms.keys()), 
'Count': list(all_symptoms.values())})

# Vectorization
from sklearn.preprocessing import MultiLabelBinarizer
multilabel_binarizer = MultiLabelBinarizer()
multilabel_binarizer.fit(df['Symptoms'])
# transform target variable
y = multilabel_binarizer.transform(df['Symptoms'])

tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=10000)

# split dataset into training and validation set
xtrain, xval, ytrain, yval = train_test_split(df['Description'], y, test_size=0.2, random_state=9)

# create TF-IDF features
xtrain_tfidf = tfidf_vectorizer.fit_transform(xtrain)
xval_tfidf = tfidf_vectorizer.transform(xval)

# Binary Relevance
br_classifier = BinaryRelevance(LogisticRegression(C=40,class_weight='balanced'))
br_classifier.fit(xtrain_tfidf, ytrain)
br_predictions = br_classifier.predict(xval_tfidf)

print("Accuracy = ",accuracy_score(yval,br_predictions.toarray()))
print("F1 score = ",f1_score(yval,br_predictions, average="micro"))
print("Hamming loss = ",hamming_loss(yval,br_predictions))

# public methods
def infer_tags(q):
    q_vec = tfidf_vectorizer.transform([q])
    q_pred = br_classifier.predict(q_vec)
    return multilabel_binarizer.inverse_transform(q_pred)



















