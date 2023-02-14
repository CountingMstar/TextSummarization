import torch.nn as nn

class MaskedLanguageModel(nn.Module):
    """
    masked input sequence로부터 원래 token을 예측하는 n-class classification 문제
    n-class = vocab_size
    """
    def __init__(self, hidden, vocab_size):
        super(MaskedLanguageModel, self).__init__()

        self.linear = nn.Linear(hidden, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        x = self.linear(x)
        x = self.softmax(x)
        return x

class NextSentencePrediction(nn.Module):
    def __init__(self, hidden):
        super(NextSentencePrediction, self).__init__()
        # 2: NSP는 이진분류
        self.linear = nn.Linear(hidden, 2)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        x = self.linear(x[:, 0])
        x = self.softmax(x)
        return x

class BERTLM(nn.Module):
    def __init__(self, bert, vocab_size):
        super(BERTLM, self).__init__()
        self.bert = bert
        self.NSP = NextSentencePrediction(self.bert.hidden)
        self.MLM = MaskedLanguageModel(self.bert.hidden, vocab_size)

    def forward(self, x, segment_label):
        x = self.bert(x, segment_label)
        NSP = self.NSP(x)
        MLM = self.MLM(x)
        return NSP, MLM