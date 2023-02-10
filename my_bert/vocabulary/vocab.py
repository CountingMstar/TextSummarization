import pickle
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import Counter

class Token(object):
    """
    text를 문장, 단어단위로 token화

    load_text_data: text data를 불러옴
    save_pickle_data: data을 pickle로 저장
    load_pickle_data: pickle data를 불러옴
    sent_token: text를 문장단위로 나눔
    word_token: text를 단어단위로 나눔
    """
    def __init__(self):
        super(Token, self).__init__()
    
    def load_text_data(self, path):
        f = open(path)
        data = f.read()
        return data

    def save_pickle_data(self, data, path):
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load_pickle_data(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data

    def sent_token(self, text):
        tokens = sent_tokenize(text)
        return tokens

    def word_token(self, text):
        tokens = word_tokenize(text)
        return tokens

class Vocab(Token):
    """
    text의 word들을 토대로 vocabulary 사전 구축

    count: text에 사용된 각 word의 빈도수 딕셔너리
    dictionary: 각 word의 index를 부여한 딕셔너리
    """
    def __init__(self):
        self.special_token = ["<pad>", "<unk>", "<eos>", "<sos>", "<mask>"]
        self.pad_index = 0
        self.unk_index = 1
        self.eos_index = 2
        self.sos_index = 3
        self.mask_index = 4
        super(Vocab, self).__init__()

    def count(self, text):
        count_text = Counter(text)
        return count_text  

    def dictionary(self, text):
        count_text = self.count(text)
        words = list(count_text.keys())
        words = self.special_token + words
        words = enumerate(words)

        word2index = {}
        index2word = {}
        for i, word in words:
            word2index[word] = i
            index2word[i] = word

        return word2index, index2word

    def vocab_main(self):
        corpus = self.load_text_data("data/corpus")
        # text 소문자로 변환
        corpus = corpus.lower()

        sents = self.sent_token(corpus)
        words = self.word_token(corpus)

        self.save_pickle_data(sents, "data/sents")
        self.save_pickle_data(words, "data/words")

        self.dictionary(words)

        count_text = self.count(words)
        word2index, index2word = self.dictionary(count_text)

        self.save_pickle_data(word2index, "data/word2index")
        self.save_pickle_data(index2word, "data/index2word")

        return sents, words, word2index, index2word


if __name__ == "__main__":
    Vocab().vocab_main()