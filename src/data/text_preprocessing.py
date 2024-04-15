from sklearn.feature_extraction.text import TfidfVectorizer
import string
import numpy as np
import re
from tqdm.auto import tqdm

import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)


def get_top_tf_idf_words(feature_names, response, top_n=2):

    sorted_nzs = np.argsort(response.data)[:-(top_n + 1):-1]
    return feature_names[response.indices[sorted_nzs]]


class TextProcessor:
    @classmethod
    def normalize_text(self, text: str):
        stop_words = set()
        for language in stopwords.fileids():
            stop_words.update(stopwords.words(language))
        punct_words = set(string.punctuation)

        normalized = [w for w in word_tokenize(text.lower())
                      if (w not in punct_words)
                      and (w not in stop_words)]

        return ' '.join(normalized)
    
    @classmethod
    def tf_idf_ordering(self, df):
        if self.tf_idf:
            vectorizer = TfidfVectorizer()
            train_tf = vectorizer.fit(df['text'].fillna('').values)
            feature_array = np.array(train_tf.get_feature_names())

            def apply_tfidf(desc):
                top_tf_idf_words = get_top_tf_idf_words(feature_array, train_tf.transform(
                    [desc]), int(len(desc.split(' ')) * .5))
                low_tf_idf_words = set(feature_array[train_tf.transform([desc]).indices]) - set(top_tf_idf_words)
                for word in low_tf_idf_words:
                    desc = re.sub(r'(^|\s+)' + word + r'($|\s+)', ' ', desc)
                return desc

            df.loc[df['text'].apply(type) == str, 'text'] = df.loc[df['text'].apply(type) == str, 'text'] \
                .swifter \
                .allow_dask_on_strings(enable=True) \
                .progress_bar(desc='[Textprocessor] Applying TF-IDF ordering...') \
                .apply(apply_tfidf)

        return df
