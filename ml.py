import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from seaborn import heatmap
import argparse
from utils import split_data
from matplotlib import pyplot as plt
import json


def select_model(k):
    """
    Select model via parameter value
    :param k: abbreviation of classifier algorithm like k which is KNeighbors
    :return: algorithm constructor
    """
    if k == 'd':
        return DecisionTreeClassifier()
    elif k == 'r':
        return RandomForestClassifier(n_estimators=5)
    elif k == 'l':
        return LogisticRegression(solver='liblinear')
    elif k == 'k':
        return KNeighborsClassifier(n_neighbors=5)
    else:
        return KMeans(n_clusters=5)


def plot_confusion_matrix(testY, y_pred):
    """
    Draw confusion matrix for all algorithms' last accuracy scores
    :param testY: real data for an input
    :param y_pred: predicted data for an input
    :return:
    """
    cm = confusion_matrix(testY, y_pred, normalize='true')
    print(cm)
    cm_df = pd.DataFrame(cm, columns=[0, 1, 2, 3, 4], index=[0, 1, 2, 3, 4])
    cm_df.index.name = 'Actual'
    cm_df.columns.name = 'Predicted'
    plt.figure(figsize=(10, 7))
    heatmap(cm_df, cmap='Blues', annot=True)
    plt.show()


def evaluate_metric(testY, y_pred):
    """
    Get evaluation metrics(precision, recall etc.) of different types of classification methods.
    :param testY: float; y_test value to evaluate accuracy score.
    :param y_pred: float; prediction value of each classifier using x_test data.
    :return:
    """
    from sklearn.metrics import classification_report, accuracy_score
    report = classification_report(testY, y_pred)
    print("Classification Report:", )
    print(report)
    score = accuracy_score(testY, y_pred)
    print("Accuracy:", score)
    return score


def convert(string):
    """
    Converting tokens column values from string type to list
    :param string: tokens column value '[token1, token2, ...]' to [token1, token2, ...]
    :return: list of tokens
    """
    li = list(string.split(" "))
    return li


def preprocess(data, original):
    """
    Preprocessing of dataframe
    :param data: to convert tokens column values from list to average of token tf-idf results
    :param original: get original value of dataframe to write tokens columns result
    :return: original dataframe with changed values
    """
    data['tokens'] = data['tokens'].str.replace("]", "")
    data['tokens'] = data['tokens'].str.replace("[", "")
    data['tokens'] = data['tokens'].str.replace("'", "")
    data['tokens'] = data['tokens'].str.replace(",", "")

    cv = CountVectorizer()
    word_count_vector = cv.fit_transform(data['tokens'])
    tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)  # tf idf transformer
    tfidf_transformer.fit(word_count_vector)  # tf idf with count vectorizer as number of words
    df_idf = pd.DataFrame(tfidf_transformer.idf_, index=cv.get_feature_names(), columns=["idf_weights"])

    index = 0
    for elem in original['tokens']:
        text_list = convert(elem)
        transformed = []
        for token in text_list:
            if token == '':
                break
            tfidf_result = float(df_idf._get_value(token, "idf_weights"))  # get all words' tf idf weight
            transformed.append(tfidf_result)
        if transformed:
            original.iloc[index, original.columns.get_loc('tokens')] = sum(transformed) / len(transformed)
        else:
            original.iloc[index, original.columns.get_loc('tokens')] = "[]"
        index += 1

    return original


if __name__ == '__main__':
    df = pd.read_csv('dataset/url_features.csv')
    df = df[df.astype(str)['tokens'] != "['']"]
    df = df[df.astype(str)['tokens'] != "[]"]
    df = split_data(df)
    # df[['tokens', 'main_category']].to_csv('dataset/URL_ML.csv', index=False)
    new_df = preprocess(df, df)
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--method", default="k", help='selected method for classification')

    args = vars(parser.parse_args())
    method = args['method']

    with open('config.json', 'r') as f:
        json_config = json.load(f)
        category_map = json_config['category_map']

    new_df = new_df[new_df.astype(str)['tokens'] != "[]"]
    X = np.vstack(new_df['tokens'])
    y = np.vstack(new_df['main_category'].map(category_map))

    k = 10
    classifier = select_model(method)
    kf = KFold(n_splits=k, random_state=None)
    acc_score = []
    values = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index, :], X[test_index, :]
        y_train, y_test = y[train_index], y[test_index]

        classifier.fit(X_train, y_train)
        pred_values = classifier.predict(X_test)

        acc = evaluate_metric(y_test, pred_values)
        acc_score.append(acc)
        values.append([y_test, pred_values])

    index = np.argmax(acc_score)
    print(f'Best fold: {index}\n')
    y_test = values[index][0]
    pred_values = values[index][1]

    plot_confusion_matrix(y_test, y_pred=pred_values)

