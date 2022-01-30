
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import string
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('words')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')
from nltk.corpus import wordnet, stopwords

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree



df = pd.read_csv('Areviews.csv')

print(df.head())

df = df.drop(columns = ['Unnamed: 0','review_id','product_id'], axis =1)
print(df.head())


list_of_stopwords = stopwords.words('spanish')

def ReviewProcessing(df):
  # remove non alphanumeric
  df['reviewed1'] = df.review_body.str.replace('[^a-zA-Z0-9 ]', '')
  # lowercase
  df.reviewed1 = df.reviewed1.str.lower()
  # split into list
  df.reviewed1 = df.reviewed1.str.split(' ')
  # remove stopwords
  df.reviewed1 = df.reviewed1.apply(lambda x: [item for item in x if item not in list_of_stopwords])
  return df



def get_wordnet_pos(word):
  tag = nltk.pos_tag([word])[0][1][0].upper()
  tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

  return tag_dict.get(tag, wordnet.NOUN)

lemmatizer = nltk.stem.WordNetLemmatizer()
def get_lemmatize(sent):
  return " ".join([lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in nltk.word_tokenize(sent)])



clean_data = ReviewProcessing(df)
clean_data.reviewed1 = clean_data.reviewed1.apply(' '.join)
clean_data['reviewed1_lemmatized'] = clean_data.reviewed1.apply(get_lemmatize)

print(df.head())


nb = Pipeline([('vectorize', CountVectorizer(ngram_range=(1, 2))),
               ('tfidf', TfidfTransformer()),
               ('clf', MultinomialNB()),
              ])


sgd = Pipeline([('vect', CountVectorizer(ngram_range=(1, 2))),
                ('tfidf', TfidfTransformer()),
                ('clf', SGDClassifier()),
               ])


logreg = Pipeline([('vect', CountVectorizer(ngram_range=(1, 2))),
                               ('tfidf', TfidfTransformer()),
                               ('clf', LogisticRegression(max_iter=300)),
                              ])
tree2 = Pipeline([('vect', CountVectorizer(ngram_range=(1, 2))),
                               ('tfidf', TfidfTransformer()),
                               ('clf', DecisionTreeClassifier()),
                              ])

x = clean_data['reviewed1_lemmatized']
y = clean_data['Class']
X_train, X_test, y_train, y_test = train_test_split(x, y,test_size=0.2, random_state = 44)

print("MultinomialNB")
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)
print(accuracy_score(y_test, y_pred_nb))
print(confusion_matrix(y_test, y_pred_nb))
print(classification_report(y_test, y_pred_nb))


print("SVM")
sgd.fit(X_train, y_train)
y_pred_sgd = sgd.predict(X_test)
print(accuracy_score(y_test, y_pred_sgd))
print(confusion_matrix(y_test, y_pred_sgd))
print(classification_report(y_test, y_pred_sgd))



print("Logistic Regression")
logreg.fit(X_train, y_train)
y_pred_log = logreg.predict(X_test)
print(accuracy_score(y_test, y_pred_log))
print(confusion_matrix(y_test, y_pred_log))
print(classification_report(y_test, y_pred_log))


knn = Pipeline([('vect', CountVectorizer(ngram_range=(1, 2))),
                ('tfidf', TfidfTransformer()),
                ('clf', KNeighborsClassifier(n_neighbors=10)),
               ])

knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)


print("knn:10")
print(accuracy_score(y_test, y_pred_knn))
print(confusion_matrix(y_test, y_pred_knn))
print(classification_report(y_test, y_pred_knn))

x = 0
n = 0
for i in range(1,100):

    knn = Pipeline([('vect', CountVectorizer(ngram_range=(1, 2))),
                    ('tfidf', TfidfTransformer()),
                    ('clf', KNeighborsClassifier(n_neighbors=i)),
                   ])

    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)

    if(accuracy_score(y_test, y_pred_knn) > x):
        n = i
        x = accuracy_score(y_test, y_pred_knn)

print("knn ")
print(i)
print(accuracy_score(y_test, y_pred_knn))
print(confusion_matrix(y_test, y_pred_knn))
print(classification_report(y_test, y_pred_knn))


print("Tree")



tree2.fit(X_train, y_train)
y_pred_tree = tree2.predict(X_test)
print(accuracy_score(y_test, y_pred_tree))
print(confusion_matrix(y_test, y_pred_tree))
print(classification_report(y_test, y_pred_tree))
