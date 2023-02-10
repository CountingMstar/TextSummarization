import torch
from vocab import Vocab

class Data(object):
    """
    text를 index data로 변환
    """
    def __init__(self, seq_len=20, encoding="utf-8"):
        self.seq_len = seq_len
        self.encoding = encoding
        super(Data, self).__init__()

    def make_index_data(self, data, word2index, index2word):
        index_data = []
        sentence = []
        # text를 문장으로 나눔
        for sent in data:
            sent = Vocab().word_token(sent)
            # 문장을 단어로 나눔
            for word in sent:
                # vocabulary 사전에서 index를 찾음. 만약 해당 단어의 index가 없으면 '모르는 단어(unknown)'로 indexing.
                try:
                    sentence.append(word2index[word])
                except:
                    sentence.append(Vocab().unk_index)
            
            # 문장의 맨 첫 부분에 eos indexing 
            sentence.insert(0, Vocab().eos_index)
            # 문장의 맨 뒷 부분에 sos indexing
            sentence.append(Vocab().sos_index)
            # torch tensor
            index_data.append(torch.LongTensor(sentence))
            sentence = []
        return index_data

    def data_main(self):
        sents, words, word2index, index2word = Vocab().vocab_main()
        index_data = self.make_index_data(sents, word2index, index2word)
        print(index_data)


if __name__ == "__main__":
    Data().data_main()

        

