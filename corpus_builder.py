from gensim.corpora import Dictionary
from gensim.models import Phrases
from gensim.models.phrases import Phraser
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import os
import string
import tqdm
from unidecode import unidecode

LETTERS = set(string.ascii_letters)
SYMBOLS = set('()[]\/|={}><@✦#')

class CorpusBuilder(object):
    def __init__(
        self, 
        ndocs,
        phrase_min_count=5, 
        vocabulary_size=10000,
        bigram_min_count=5,
        bigram_threshold=10,
        trigram_min_count=5,
        trigram_threshold=10,
        substitutions=dict(),
        data_directory=None,
        model_directory=None
    ): 
        self.ndocs = ndocs
        self.phrase_min_count = phrase_min_count
        self.vocabulary_size = vocabulary_size
        self.bigram_min_count = bigram_min_count
        self.bigram_threshold = bigram_threshold
        self.trigram_min_count = trigram_min_count
        self.trigram_threshold = trigram_threshold
        self.substitutions = substitutions
        self.data_directory = data_directory
        self.model_directory = model_directory
        
        self.load_bad_phrases()

    def tokenize(self, text):
        return [token.lower() for token in word_tokenize(text)]

    def stream_sentences(self, texts):
        tqdm.tqdm.write("Streaming sentences.")
        for text in tqdm.tqdm(texts):
            for sentence in sent_tokenize(text):
                yield self.tokenize(sentence)       

    def load_bad_phrases(self):
        with open("%s/bad-phrases.txt" % self.data_directory, mode='r', encoding='UTF-8') as fp: 
            self.bad_phrases = set([phrase.strip() for phrase in fp.readlines()])

    def add_bad_phrase(self, phrase): self.bad_phrases.add(phrase)
        
    def save_bad_phrases(self):
        bad_phrases = list(self.bad_phrases)
        bad_phrases.sort()
        with open("%s/bad-phrases.txt" % self.data_directory, mode='w', encoding='UTF-8') as fp:
            for phrase in bad_phrases: fp.write("%s\n" % phrase)
                
    def train_phrasers(self, texts):
        bigrams = Phrases(
            self.stream_sentences(texts), 
            min_count=self.bigram_min_count, 
            threshold=self.bigram_threshold
        ) 
        #print("Training bigram phraser ...")
        self.bigram_phraser = Phraser(bigrams)

        #print("Collecting trigrams ...")
        trigrams = Phrases(
            self.bigram_phraser[self.stream_sentences(texts)], 
            min_count=self.trigram_min_count, 
            threshold=self.trigram_threshold
        )
        #print("Training trigram phraser ...")
        self.trigram_phraser = Phraser(trigrams)
            
    def save_phrasers(self):
        path = os.path.join(self.model_directory, "bigram-phraser.pkl")
        self.bigram_phraser.save(path)

        path = os.path.join(self.model_directory, "trigram-phraser.pkl")
        self.trigram_phraser.save(path)

    def load_phrasers(self):
        path = os.path.join(self.model_directory, "bigram-phraser.pkl")
        self.bigram_phraser = Phraser.load(path)

        path = os.path.join(self.model_directory, "trigram-phraser.pkl")
        self.trigram_phraser = Phraser.load(path)

    def prepare_text(self, text):
        for key, value in self.substitutions.items(): text = text.replace(key, value) 
        tokens = self.tokenize(text)
        tokens = self.bigram_phraser[tokens]
        tokens = self.trigram_phraser[tokens]
        return [token for token in tokens if not token in self.bad_phrases]

    def prepare_texts(self, texts):
        return [self.prepare_text(text) for text in tqdm.tqdm(texts)]

    def keep_phrase(self, phrase, cnt):
        if "'" in phrase: return False
        if phrase in self.bad_phrases: return False
        phrase_set = set(phrase)
        if SYMBOLS & phrase_set: return False
        if (LETTERS & set(phrase)) and cnt > self.phrase_min_count: return True
        return False

    def build_vocabulary(self, texts, save=False):
        self.ndocs = len(texts)
        tqdm.tqdm.write("Building vocabulary over %d documents." % self.ndocs)
        phrase_map = {}
        for document in tqdm.tqdm(texts):
            for phrase in document:
                if not phrase in phrase_map: phrase_map[phrase] = 0
                phrase_map[phrase] += 1
        phrases = list(phrase_map.keys())
        phrases = sorted(phrases, key=lambda phrase: -phrase_map[phrase])        

        vocabulary = [
            phrase for phrase in phrases if self.keep_phrase(phrase, phrase_map[phrase])
        ]

        hyphenated = {phrase.replace('-', '_') for phrase in vocabulary if "-" in phrase}
        vocabulary = [phrase for phrase in vocabulary if not phrase in hyphenated][:self.vocabulary_size]
        if save: 
            path = os.path.join(
                self.data_directory, 
                "vocabulary-%d-%d-%d.tsv" % (len(texts), self.phrase_min_count, self.vocabulary_size)
            )
            fp = open(path, mode='w', encoding='UTF-8')
            for phrase in vocabulary: fp.write("%s\t%d\n" % (phrase, phrase_map[phrase]))
            fp.close()
        self.vocabulary = set(vocabulary)

    def load_vocabulary(self):
        path = os.path.join(
            self.data_directory, 
            "vocabulary-%d-%d-%d.tsv" % (self.ndocs, self.phrase_min_count, self.vocabulary_size)
        )
        fp = open(path, mode='r', encoding='UTF-8')
        self.vocabulary = set([])
        for line in fp:
            line = line.strip()
            if line:
                phrase, cnt = line.split('\t')
                self.vocabulary.add(phrase)
        fp.close()

    def build_document(self, text):
        return [phrase for phrase in text if phrase in self.vocabulary]

    def build_corpus(self, texts): 
        corpus = [self.build_document(text) for text in tqdm.tqdm(texts)]
        return corpus

    def build_dictionary(self, corpus, save=False): 
        self.dictionary = Dictionary(corpus)
        self.dictionary.filter_extremes(no_below=self.phrase_min_count, no_above=0.6, keep_n=self.vocabulary_size)
        if save: self.save_dictionary()

    def save_dictionary(self):
        path = os.path.join(
            self.model_directory, 
            "dictionary-%d-%d-%d.pkl" % (self.ndocs, self.phrase_min_count, self.vocabulary_size)
        )
        self.dictionary.save(path)

    def load_dictionary(self):
        path = os.path.join(
            self.model_directory, 
            "dictionary-%d-%d-%d.pkl" % (self.ndocs, self.phrase_min_count, self.vocabulary_size)
        )
        self.dictionary = Dictionary.load(path)

    def encode_corpus(self, corpus):
        return [self.dictionary.doc2bow(document) for document in corpus]