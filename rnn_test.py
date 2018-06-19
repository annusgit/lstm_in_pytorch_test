

from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.tensor as Tensor


class LSTMtagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMtagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()

        pass

    def init_hidden(self):
        # Before we've done anything, we don't have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        # print(embeds.size())
        lstm_out, self.hidden = self.lstm(
            embeds.view(len(sentence), 1, -1), self.hidden)
        tagspace = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tagspace, dim=1)
        return tag_scores

    def train_model(self, training_data, epochs, function_to_convert, word_to_idx, tag_to_ix):
        loss_function = nn.NLLLoss()
        optimizer = optim.SGD(self.parameters(), lr=0.1)
        for e in range(epochs):
            for sentence, tag in training_data:
                #### step 1
                # begin with zero gradients
                self.zero_grad()
                # Also, we need to clear out the hidden state of the LSTM,
                # detaching it from its history on the last instance.
                self.hidden = self.init_hidden()

                #### step 2
                # get our inputs ready for the network, by turning them into tensors
                sentence_in = function_to_convert(sentence, word_to_idx)
                targets = function_to_convert(tag, tag_to_ix)

                #### step 3
                # the forward pass
                tag_scores = self.forward(sentence_in)
                loss = loss_function(tag_scores, targets)
                loss.backward()
                optimizer.step()
            print('log: on epoch {}'.format(e))
        pass

    def test_model(self):

        pass


def main():
    training_data = [
        ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
        ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
    ]
    def prepare_sequence(sequence, to_ix):
        idxs = [to_ix[word] for word in sequence]
        # print([type(x) for x in idxs])
        return Tensor(idxs, dtype=torch.long)

    word_to_ix = {}
    for sentence, tags in training_data:
        for word in sentence:
            if word not in word_to_ix.keys():
                word_to_ix[word] = len(word_to_ix)
    print(word_to_ix)

    tag_to_ix = {'DET': 0, 'NN': 1, 'V': 2}
    EMBEDDING_DIM = 6
    HIDDEN_DIM = 6

    # a dummy function to see if the embeddings are working properly or not?
    def embedding_test():
        with torch.no_grad():
            word_embeddings = nn.Embedding(len(word_to_ix), EMBEDDING_DIM)
            sequence_in = prepare_sequence(training_data[0][0], word_to_ix)
            # print(sequence_in)
            embeddings = word_embeddings(sequence_in)
            # print(embeddings)
    # embedding_test()

    model = LSTMtagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))

    # perform a dummy test before we begin training on the network
    with torch.no_grad():
        sentence = prepare_sequence(training_data[0][0], word_to_ix)
        tag_scores = model(sentence)
        print(tag_scores)

    model.train_model(training_data, 300, prepare_sequence, word_to_ix, tag_to_ix)

    # perform a final test after the completion of training
    with torch.no_grad():
        sentence = prepare_sequence(training_data[0][0], word_to_ix)
        tag_scores = model(sentence)
        print(tag_scores)

if __name__ == '__main__':
    main()









