import re
import requests
import json
from seaborn import heatmap
from matplotlib import pyplot as plt
import pandas as pd
from nltk.tokenize import word_tokenize  # split sentence to words
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer  # go to word's root (was -> to be)
from nltk.corpus import stopwords
import nltk
import pickle
from sklearn.utils import shuffle
import warnings
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


wnl = WordNetLemmatizer()
warnings.filterwarnings('ignore')
# df = pd.read_csv('dataset/test_category.csv', usecols=[0, 1])
# df_test = pd.read_csv('dataset/test_predict.csv', header=None, usecols=[0, 1])

with open('config.json', 'r') as f:
    json_config = json.load(f)
    headers = json_config['headers']
    frequency = json_config['frequency']
    frequency_path = json_config['frequency_path']
    domains = json_config['domains']
    category_map = json_config['category_map']
    

stop_words = set(stopwords.words('english'))
with open("dataset/stopwords.txt") as f:
    """
    add desired stop words into global stopwords list.
    """
    for word in f:
        stop_words.add(word.replace('\n', ''))

for tld in domains:
    stop_words.add(tld)


def scrape_url(url):
    """
    Scrapping the url with request's response type.
    :param url: given url for scrapping.
    :param words_frequency: dictionary that has categories and its tokens with frequency order.
    :return:
    """
    try:
        res = requests.get(url, headers=headers, timeout=15)
        if res.status_code == 200:
            soup = BeautifulSoup(res.text, "html.parser")
            [tag.decompose() for tag in soup("script")]
            [tag.decompose() for tag in soup("style")]
            text = soup.get_text()
            cleaned_text = re.sub('[^a-zA-Z]+', ' ', text).strip()
            tokens = word_tokenize(cleaned_text)
            tokens_lemmatize = remove_stopwords(tokens)
            return tokens_lemmatize
            # return predict_category(words_frequency, tokens_lemmatize)
        else:
            print(
                f'Error occurred : ({res.status_code}).')
    except Exception as e:
        print(f'Error :\n {e}')
        return False


def predict_category(words_frequency, tokens):
    """
    Get intersection of given tokens and words frequency.
    Find weights of words for each category and add this result into category weights.
    Pull the category index which has maximum category weight.
    :param words_frequency:
    :param tokens:
    :return:
    """
    category_weights = []
    for category in words_frequency:
        weight = 0
        intersect_words = set(words_frequency[category]).intersection(set(tokens))
        for word in intersect_words:
            if word in tokens:
                index = words_frequency[category].index(word)
                weight += frequency - index
        category_weights.append(weight)

    category_index = category_weights.index(max(category_weights))
    main_category = list(words_frequency.keys())[category_index]
    category_weight = max(category_weights)
    category_weights[category_index] = 0
    category_index = category_weights.index(max(category_weights))
    main_category_2 = list(words_frequency.keys())[category_index]
    category_weight_2 = max(category_weights)
    return main_category, category_weight, main_category_2, category_weight_2


def remove_stopwords(tokens):
    tokens_list = []
    for word in tokens:
        word = wnl.lemmatize(word.lower())
        if word not in stop_words:
            tokens_list.append(word)
    return list(filter(lambda x: len(x) > 1, tokens_list))


def scrape(props):
    """
    Scrapping the url. Usage in preprocess.
    :param props:
    :return: response of get request.
    """
    i = props[0]
    url = props[1]
    print(i, url)
    try:
        return requests.get(url, headers=headers, timeout=15)
    except:
        return ''


def parse_request(props):
    """
    Parse request result gathered from process executor.
    This function runs when response of request works properly.
    :param props:
    :return: tokens passed from lemmatization, cleaning text and removing stop words operations.
    """
    i = props[0]
    response = props[1]
    if response != '' and response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        [tag.decompose() for tag in soup("script")]
        [tag.decompose() for tag in soup("style")]
        text = soup.get_text()
        # regex: Clean non letter characters from the HTML response (syntax).
        cleaned_text = re.sub('[^a-zA-Z]+', ' ', text).strip()
        # token: split text into tokens.
        tokens = word_tokenize(cleaned_text)
        # lemma: Clean stop words like common words(and, you etc.) from the tokens list.
        tokens_lemmatize = remove_stopwords(tokens)
        return i, tokens_lemmatize
    else:
        return i, ['']


def create_test_data(map_dict):
    """
    returns: dataframe for testing
    """
    df = pd.read_csv('dataset/URL_Classification.csv', header=None, usecols=[1,2])
    df = df[df[2].isin(['Health', 'Sports', 'Shopping', 'News', 'Computers'])]
    df[2] = df[2].replace(map_dict)
    df.to_csv('dataset/test_category.csv', index=False)
    return df


def one_url_predict(url):
    pickle_in = open(frequency_path, "rb")
    words_frequency = pickle.load(pickle_in)
    tokens = scrape_url(url)
    results = predict_category(words_frequency, tokens)
    return results[0]


def get_statistics(real_data, predicted_data):
    """
    returns statistics of model
    :param real_data: real prediction class data exp: Y_test
    :param predicted_data: predicted  prediction class data
    :return:
    """
    report = classification_report(real_data, predicted_data)
    print("Classification Report:", )
    print(report)

    score = accuracy_score(real_data, predicted_data)
    print("Accuracy:", score)

    cm = confusion_matrix(real_data, predicted_data, normalize='true')
    print(cm)
    cm_df = pd.DataFrame(cm, columns=[0, 1, 2, 3, 4], index=[0, 1, 2, 3, 4])
    cm_df.index.name = 'Actual'
    cm_df.columns.name = 'Predicted'
    plt.figure(figsize=(10, 7))
    heatmap(cm_df, cmap='Blues', annot=True)
    plt.show()


def split_data(df, save=False):
    counts = [0, 0, 0, 0, 0]
    size = 100
    df = shuffle(df)
    new_df = pd.DataFrame()

    for i in range(1, len(df) - 1):
        id = category_map.get(str(df.iloc[i][1]))
        if counts[id] < size:
            new_df = new_df.append(df.iloc[i], ignore_index=True)
            counts[id] += 1
        sum_counts = counts[0] + counts[1] + counts[2] + counts[3] + counts[4]
        if sum_counts >= size * len(counts):
            break
    if save:
        new_df.to_csv(path_or_buf='dataset/test_predict.csv', index=False)
        print('Out csv written !')
    return new_df


def get_results(df):
    results = []
    real_classes = []

    for i in range(1, len(df) - 1):
        try:
            predicted_class = one_url_predict(str(df.iloc[i][0]))
            predicted_class_id = category_map.get(predicted_class)
            results.append(predicted_class_id)

            id = category_map.get(str(df.iloc[i][1]))
            real_classes.append(id)
        except Exception as err:
            print(err)
    get_statistics(real_classes, results)
