import pandas as pd
import nltk
from nltk.corpus import stopwords
import spacy
from nltk.tokenize import ToktokTokenizer
from nltk.tokenize import word_tokenize,sent_tokenize
import matplotlib.pyplot as plt

# Defining stopwords necessary for remove_stopwords function
nltk.download('stopwords', quiet=True, raise_on_error=True)
stopword_list = set(nltk.corpus.stopwords.words('english'))
tokenized_stop_words = nltk.word_tokenize(' '.join(nltk.corpus.stopwords.words('english')))


# Function to remove stopwords and return text intact rather than tokenized

def remove_stopwords(text):
    tokenizer = nltk.tokenize.toktok.ToktokTokenizer()
    # convert sentence into token of words
    all_addresses = []
    for addresses in text:
        tokens = tokenizer.tokenize(addresses)
        tokens = [token.strip() for token in tokens]
        # check in lowercase
        t = [token for token in tokens if token.lower() not in stopword_list]
        text=' '.join(t)
        all_addresses.append(text)
    return pd.Series(all_addresses)

# Function to lemmatize

def lemma(text,allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    # Initialize spacy 'en' model, keeping only tagger component needed for lemmatization
    nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
    # Create list to store all addresses in
    all_addresses = []
    for address in text:
        doc = nlp(address)
        t = " ".join([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        all_addresses.append(t)
    return pd.Series(all_addresses)

# Function to plot top stopwords
def plot_top_words(model, feature_names, n_top_words, title,dim_1,dim_2):
    fig, axes = plt.subplots(dim_1, dim_2, figsize=(30, 15), sharex=True)
    axes = axes.flatten()
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]

        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7)
        ax.set_title(f'Topic {topic_idx +1}',
                     fontdict={'fontsize': 30})
        ax.invert_yaxis()
        ax.tick_params(axis='both', which='major', labelsize=20)
        for i in 'top right left'.split():
            ax.spines[i].set_visible(False)
        fig.suptitle(title, fontsize=40)

    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    plt.show()
    fig.tight_layout()
    plt.savefig("{}".format(title))
