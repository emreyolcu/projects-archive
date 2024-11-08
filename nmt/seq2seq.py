import pdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import utils
from utils import batch_iter, input_transpose


def to_tensor(sents, vocab):
    return torch.tensor(input_transpose(vocab.words2indices(sents), vocab['<pad>']),
                        dtype=torch.long).to(utils.DEVICE)


class Encoder(nn.Module):
    def __init__(self, vocab, config):
        super().__init__()
        self.embed_size = config['encoder_embed_size']
        self.hidden_size = config['encoder_hidden_size']
        self.n_layers = config['encoder_layers']
        self.dropout_rate = config['dropout']
        self.vocab = vocab

        self.embedding = nn.Embedding(len(self.vocab), self.embed_size, padding_idx=self.vocab['<pad>'])
        self.lstm = nn.LSTM(self.embed_size, self.hidden_size, bidirectional=True,
                            dropout=self.dropout_rate, num_layers=self.n_layers)

    def forward(self, src_sents):
        src_embeddings = self.embedding(to_tensor(src_sents, self.vocab))
        packed_src_embeddings = pack_padded_sequence(src_embeddings, [len(s) for s in src_sents])
        packed_src_encodings, (hidden, cell) = self.lstm(packed_src_embeddings)
        src_encodings = pad_packed_sequence(packed_src_encodings)[0]
        return src_encodings, cell


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder_hidden_size = config['encoder_hidden_size']
        self.decoder_hidden_size = config['decoder_hidden_size']
        self.proj_encoder = nn.Linear(2 * self.encoder_hidden_size, self.decoder_hidden_size, bias=False)
        self.combine = nn.Linear(2 * self.encoder_hidden_size + self.decoder_hidden_size,
                                 self.decoder_hidden_size, bias=False)


class Decoder(nn.Module):
    def __init__(self, vocab, config):
        super().__init__()
        self.embed_size = config['decoder_embed_size']
        self.hidden_size = config['decoder_hidden_size']
        self.n_layers = config['decoder_layers']
        self.dropout_rate = config['dropout']
        self.vocab = vocab

        self.embedding = nn.Embedding(len(self.vocab), self.embed_size, padding_idx=self.vocab['<pad>'])
        self.init_cell = nn.Linear(2 * config['encoder_hidden_size'], self.hidden_size)
        self.lstmcell = nn.ModuleList([nn.LSTMCell(self.embed_size + self.hidden_size, self.hidden_size) if i == 0
                                       else nn.LSTMCell(self.hidden_size, self.hidden_size) for i in range(self.n_layers)])
        self.dropout = nn.Dropout(self.dropout_rate)
        self.attention = Attention(config)
        self.output = nn.Linear(self.hidden_size, len(self.vocab), bias=False)

    def init_state(self, encoder_cell):
        init_cell = self.init_cell(torch.cat([encoder_cell[0], encoder_cell[1]], 1))
        return (torch.tanh(init_cell), init_cell)

    def decode_one_word(self, word, prev_context, state, src_encodings, src_attn_scores):
        cell_input = torch.cat([word, prev_context], dim=1)
        for lstm in self.lstmcell:
            hidden, cell = lstm(cell_input, state)
            hidden = self.dropout(hidden)
            cell_input = hidden
            state = (hidden, cell)
        alpha = F.softmax(src_attn_scores.bmm(hidden.unsqueeze(2)), dim=1).permute(0, 2, 1)
        context = alpha.bmm(src_encodings).squeeze(1)
        attention_vector = self.dropout(torch.tanh(self.attention.combine(torch.cat([hidden, context], 1))))
        score = self.output(attention_vector)
        return score, attention_vector, (hidden, cell), alpha

    def forward(self, src_encodings, decoder_init_state, tgt_tensor):
        src_encodings = src_encodings.permute(1, 0, 2)
        tgt_embeddings = self.embedding(tgt_tensor)
        src_attn_scores = self.attention.proj_encoder(src_encodings)

        scores = []
        prev_context = torch.zeros(src_encodings.shape[0], self.hidden_size).to(utils.DEVICE)
        state = decoder_init_state

        for i in range(tgt_embeddings.shape[0]):
            score, prev_context, state, _ = self.decode_one_word(
                tgt_embeddings[i], prev_context, state, src_encodings, src_attn_scores)
            scores.append(score)

        return scores


class Seq2seq(nn.Module):
    def __init__(self, vocab, config):
        super().__init__()
        self.vocab = vocab
        self.encoder = Encoder(vocab.src, config)
        self.decoder = Decoder(vocab.tgt, config)
        self.criterion = nn.CrossEntropyLoss(ignore_index=vocab.tgt['<pad>'])

    def forward(self, src_sents, tgt_sents):
        src_encodings, encoder_cell = self.encoder(src_sents)
        tgt_tensor = to_tensor(tgt_sents, self.vocab.tgt)
        scores = torch.stack(self.decoder(src_encodings, self.decoder.init_state(encoder_cell), tgt_tensor[:-1]))
        return self.criterion(scores.view(-1, scores.size(2)), tgt_tensor[1:].view(-1))

    def evaluate_ppl(self, dev_data, batch_size=32):
        cum_loss = 0
        cum_tgt_words = 0
        with torch.no_grad():
            for src_sents, tgt_sents in batch_iter(dev_data, batch_size):
                loss = self.forward(src_sents, tgt_sents)
                tgt_word_num_to_predict = sum(len(s[1:]) for s in tgt_sents)
                cum_loss += loss.item() * tgt_word_num_to_predict
                cum_tgt_words += tgt_word_num_to_predict
        ppl = np.exp(cum_loss / cum_tgt_words)
        return ppl, (cum_loss / cum_tgt_words)

    @staticmethod
    def load(model_path):
        return torch.load(model_path)

    def save(self, path):
        torch.save(self, path)
