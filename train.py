import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
import pandas as pd
from sklearn.metrics import classification_report
from utilities import clean_text


def train(file_url, save_model = False):
    df = pd.read_csv(file_url)
    df.columns = ['category', 'description']

    total_words = df['description'].apply(lambda x: len(x.split(' '))).sum()
    enc = LabelEncoder()
    df['numerical_category'] = enc.fit_transform(df['category'])
    labels = list(enc.classes_)

    df['description'] = df['description'].apply(clean_text)

    x = df.description
    y = df.numerical_category

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42, stratify=y)
    my_tags = list(df.category.unique())

    nb = Pipeline([('vect', CountVectorizer()),
                   ('tfidf', TfidfTransformer()),
                   ('clf', MultinomialNB()),
                   ])
    nb.fit(x_train, y_train)

    y_pred = nb.predict(x_test)

    print('accuracy %s' % accuracy_score(y_pred, y_test))
    print(classification_report(y_test, y_pred, target_names=my_tags))
    if save_model:
        # save the classifier
        with open('naive_bayes_classifier.pkl', 'wb') as fid:
            pickle.dump(nb, fid)

    return nb