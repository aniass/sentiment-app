import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as imbpipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from joblib import dump

stop_words = stopwords.words('english')
clothes = ['dress', 'color', 'wear', 'top', 'sweater', 'material', 'shirt',
           'jeans', 'pant', 'skirt', 'order', 'white', 'black', 'fabric',
           'blouse', 'sleeve', 'even', 'jacket']
lem = WordNetLemmatizer()


URL_DATA = 'data\review_final.csv'


def text_preprocess(text):
    """Function to clean text data: remove punctuation,
    stopwords and apply stemming"""
    words = re.sub("[^a-zA-Z]", " ", text)
    words = [word.lower() for word in words.split() if word.lower() not in
             stop_words and word.lower() not in clothes]
    words = [lem.lemmatize(word) for word in words]
    return " ".join(words)


def read_data(path):
    """Function to read and preprocess data"""
    data = pd.read_csv(path, header=0, index_col=0)
    data['Review'] = data['Review'].apply(text_preprocess)
    return data


def splitting_data(data):
    """Function to split data on train and test set"""
    X = data['Review']
    y = data['Recommended']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                        random_state=42)
    return X_train, X_test, y_train, y_test


def train_models(X_train, X_test, y_train, y_test):
    """Calculate models with score"""
    model = imbpipeline([('vect', CountVectorizer(min_df=5, ngram_range=(1, 2))),
                       ('tfidf', TfidfTransformer()),
                       ('smote', SMOTE()),
                       ('model', LogisticRegression()), ])

    model.fit(X_train, y_train)
    # save model
    dump(model, 'models/LR_model.pkl')
    # score
    pred = model.score(X_test, y_test)
    return pred


if __name__ == '__main__':
    df = read_data(URL_DATA)
    X_train, X_test, y_train, y_test = splitting_data(df)
    result_models = train_models(X_train, X_test, y_train, y_test)
    print(result_models)
