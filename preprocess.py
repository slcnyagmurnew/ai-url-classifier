import os
import sys
import nltk
from utils import scrape, parse_request
import json
import pickle
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

file_name = 'dataset/URL-categorization-DFE.csv'
file_path = os.path.join(sys.path[0], file_name)


with open('config.json', 'r') as f:
    json_config = json.load(f)
    domains = json_config['domains']
    thread = json_config['thread']
    multiprocessing = json_config['multiprocessing']
    token = json_config['token']
    frequency = json_config['frequency']
    frequency_path = json_config['frequency_path']

df = pd.read_csv(file_path)
df = df[df['main_category'].isin(['Health', 'Sports', 'Shopping', 'News_and_Media', 'Computer_and_Electronics'])]
df = df[['url', 'main_category', 'main_category:confidence']]

print('Selected columns: ', df['main_category'].value_counts())

df['url'] = df['url'].apply(lambda x: 'http://' + x)
df['tld'] = df['url'].apply(lambda x: x.split('.')[-1])
# drops the current index of the DataFrame and replaces it with an index of increasing integers.
df = df[df.tld.isin(domains)].reset_index(drop=True)
df['tokens'] = ''


def get_features(dataframe, file=token):
    with ThreadPoolExecutor(thread) as thread_executor:
        """
        Urls in dataframe are sent to scrape function with enumerated values.
        This operation split into 16 threads to increase retrieving url speed.
        It takes ~50 minute to finish scrapping.
        Example: 0 - http://url.com
        """
        results = thread_executor.map(scrape, [(i, elem) for i, elem in enumerate(dataframe['url'])])

    with ProcessPoolExecutor(multiprocessing) as process_executor:
        """
        Single process runs very slowly with this operation.
        To increase analysis of obtained urls' speed, this operation split into 2 processes.
        Urls are mapped to parse request function to get data from urls.
        """
        response = process_executor.map(parse_request, [(i, elem) for i, elem in enumerate(results)])

    for props in response:
        """
        Add result of scrapped url content with tokens into related dataframe location.
        """
        i = props[0]
        tokens = props[1]
        dataframe.at[i, 'tokens'] = tokens

    dataframe.to_csv(file, index=False)
    return dataframe


if __name__ == '__main__':
    df = get_features(dataframe=df)
    words_frequency = {}
    for category in df.main_category.unique():
        """
        nltk.FreqDist(all_words): create frequency distribution.
        nltk.FreqDist(all_words).most_common(frequency): returns a list with given frequency number.
        In this list, elements are in ascending order by frequency of each element.
        For each category, tokens are copied into a list named all words.
        All words are put in order by frequency of each word(in nltk.FreqDist).
        Finally, category and its list of words are put into dictionary.
        """
        # print(category)
        all_words = []
        df_temp = df[df.main_category == category]
        for word in df_temp.tokens:
            all_words.extend(word)
        most_common = [word[0] for word in nltk.FreqDist(all_words).most_common(frequency)]
        words_frequency[category] = most_common

    # Save word frequency model with pickle
    pickle_out = open(frequency_path, "wb")
    pickle.dump(words_frequency, pickle_out)
    print('Model saved.')
    pickle_out.close()
