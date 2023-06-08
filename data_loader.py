import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import re

class DataLoader:

    def __init__(self, config_dict: dict):
        self.config_dict = dict
        self.data_text = None
        self.text_list = None
        self.embeddings = None
        self.labels = None

        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.tokenizer = AutoTokenizer.from_pretrained(config_dict["tokenizer"])

        self.get_data(dataset_name=config_dict["dataset"])
        self.lemmatize()
        self.get_embeddings()

    def get_data(self, dataset_name: str):
        folder_name = "datasets/"
        if dataset_name == "imdb":
            folder_name += "aclImdb/imdb_reviews.csv"
        elif dataset_name == "yelp-review":
            folder_name += "yelp-review/yelp_review.csv"
        elif dataset_name == "tweet-eval":
            folder_name += "tweet-eval-emoji/tweet_eval.csv"
        elif dataset_name == "tweet-topic":
            folder_name += "tweet-topic/tweet_topic_multi.csv"
        else:
            print("READ ERROR")
            exit(1)

        self.data_text = pd.read_csv(folder_name, index_col=0).sample(50000)  # .sample(frac=1)
        self.labels = self.data_text.pop('label').to_numpy().astype(int)  # .reshape(-1, 1)

    def lemmatize(self):
        lemmatized_text = []
        for sample_old in self.data_text["text"].tolist():
            sample = re.sub(r'([^\w\s]|<.*\/>)', '', sample_old)
            sample_list = sample.split()
            lemmatized_sample = [self.lemmatizer.lemmatize(word) for word in sample_list]
            lemmatized_sample_str = ' '.join(lemmatized_sample)
            lemmatized_text.append(lemmatized_sample_str)
        self.text_list = lemmatized_text

    def get_embeddings(self):
        # text_list = self.data_text["text"].tolist()
        embeddings_raw = self.tokenizer(self.text_list, padding=True, truncation=True, max_length=256, return_tensors="pt")

        embeddings = []
        for idx in range(len(self.text_list)):
            emb_raw = embeddings_raw[idx]
            embedding = emb_raw.ids
            embedding = (np.array(embedding) / 50000).tolist()
            # embedding.extend(emb_raw.attention_mask)
            embeddings.append(embedding)

        self.embeddings = np.array(embeddings)
