from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.init as I
import torch.nn.utils.rnn as R
import torch.nn.functional as F

import numpy as np

import re
import random
import logging
from pykalman import KalmanFilter

logger = logging.getLogger()


def log_sum_exp(tensor, dim=0):
    """LogSumExp operation."""
    m, _ = torch.max(tensor, dim)
    m_exp = m.unsqueeze(-1).expand_as(tensor)
    return m + torch.log(torch.sum(torch.exp(tensor - m_exp), dim))


def sequence_mask(lens, max_len=None):
    batch_size = lens.size(0)

    if max_len is None:
        max_len = lens.max().item()

    ranges = torch.arange(0, max_len).long()
    ranges = ranges.unsqueeze(0).expand(batch_size, max_len)

    if lens.data.is_cuda:
        ranges = ranges.cuda()

    lens_exp = lens.unsqueeze(1).expand_as(ranges)
    mask = ranges < lens_exp

    return mask


def load_embedding(path: str,
                   dimension: int,
                   vocab: dict = None,
                   skip_first_line: bool = True,
                   rand_range: list = None,
                   fine_tune_embeds: bool = True
                   ):
    logger.info('Scanning embedding file: {}'.format(path))

    embed_vocab = set()
    lower_mapping = {}  # lower case - original
    digit_mapping = {}  # lower case + replace digit with 0 - original
    digit_pattern = re.compile('\d')
    with open(path, 'r', encoding='utf-8') as r:
        if skip_first_line:
            r.readline()
        for line in r:
            try:
                token = line.split(' ')[0].strip()
                if token:
                    embed_vocab.add(token)
                    token_lower = token.lower()
                    token_digit = re.sub(digit_pattern, '0', token_lower)
                    if token_lower not in lower_mapping:
                        lower_mapping[token_lower] = token
                    if token_digit not in digit_mapping:
                        digit_mapping[token_digit] = token
            except UnicodeDecodeError:
                continue

    token_mapping = defaultdict(list)  # embed token - vocab token
    for token in vocab:
        token_lower = token.lower()
        token_digit = re.sub(digit_pattern, '0', token_lower)
        if token in embed_vocab:
            token_mapping[token].append(token)
        elif token_lower in lower_mapping:
            token_mapping[lower_mapping[token_lower]].append(token)
        elif token_digit in digit_mapping:
            token_mapping[digit_mapping[token_digit]].append(token)

    logger.info('Loading embeddings')
    if rand_range is not None:
        rand_range.sort()
        weight = [[random.uniform(rand_range[0], rand_range[1]) for _ in range(dimension)] for _ in range(len(vocab))]
    else:
        weight = [[.0] * dimension for _ in range(len(vocab))]
    with open(path, 'r', encoding='utf-8') as r:
        if skip_first_line:
            r.readline()
        for line in r:
            try:
                segs = line.rstrip().split(' ')
                token = segs[0]
                if token in token_mapping:
                    vec = [float(v) for v in segs[1:]]
                    for t in token_mapping.get(token):
                        weight[vocab[t]] = vec.copy()
            except UnicodeDecodeError:
                continue
            except ValueError:
                continue
    embed = nn.Embedding(
        len(vocab),
        dimension,
        padding_idx=C.PAD_INDEX,
        sparse=True,
        _weight=torch.FloatTensor(weight)
    )
    if not fine_tune_embeds:
        embed.weight.requires_grad = False
    return embed


class Linear(nn.Linear):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True):
        super(Linear, self).__init__(in_features, out_features, bias=bias)
        I.orthogonal_(self.weight)


class Linears(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 hiddens: list,
                 bias: bool = True,
                 activation: str = 'tanh'):
        super(Linears, self).__init__()
        assert len(hiddens) > 0

        self.in_features = in_features
        self.out_features = self.output_size = out_features

        in_dims = [in_features] + hiddens[:-1]
        self.linears = nn.ModuleList([Linear(in_dim, out_dim, bias=bias)
                                      for in_dim, out_dim
                                      in zip(in_dims, hiddens)])
        self.output_linear = Linear(hiddens[-1], out_features, bias=bias)
        # self.activation = getattr(F, activation)
        self.activation = getattr(torch, activation)

    def forward(self, inputs):
        linear_outputs = inputs
        for linear in self.linears:
            linear_outputs = linear(linear_outputs)
            linear_outputs = self.activation(linear_outputs)
        return self.output_linear(linear_outputs)


class Highway(nn.Module):
    def __init__(self,
                 size: int,
                 layer_num: int = 1,
                 activation: str = 'relu'):
        super(Highway, self).__init__()
        self.size = self.output_size = size
        self.layer_num = layer_num
        self.activation = getattr(F, activation)
        self.non_linear = nn.ModuleList([Linear(size, size)
                                         for _ in range(layer_num)])
        self.gate = nn.ModuleList([Linear(size, size)
                                   for _ in range(layer_num)])

    def forward(self, inputs):
        for layer in range(self.layer_num):
            gate = torch.sigmoid(self.gate[layer](inputs))
            non_linear = self.activation(self.non_linear[layer](inputs))
            inputs = gate * non_linear + (1 - gate) * inputs
        return inputs


class LSTM(nn.LSTM):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int = 1,
                 bias: bool = True,
                 batch_first: bool = False,
                 dropout: float = 0,
                 bidirectional: bool = False,
                 forget_bias: float = 0
                 ):
        super(LSTM, self).__init__(input_size=input_size,
                                   hidden_size=hidden_size,
                                   num_layers=num_layers,
                                   bias=bias,
                                   batch_first=batch_first,
                                   dropout=dropout,
                                   bidirectional=bidirectional)
        self.output_size = hidden_size * 2 if bidirectional else hidden_size
        self.forget_bias = forget_bias

    def initialize(self):
        for n, p in self.named_parameters():
            if 'weight' in n:
                I.orthogonal_(p)
            elif 'bias' in n:
                bias_size = p.size(0)
                p[bias_size // 4:bias_size // 2].fill_(self.forget_bias)


class Discriminator(nn.Module):
    def __init__(self, n_langs, input_dim, n_layers, hidden_size, dropout_p):
        super(Discriminator, self).__init__()

        self.n_langs = n_langs
        self.input_dim = input_dim
        self.n_layers = n_layers
        self.hidden_dim = hidden_size
        self.dropout_p = dropout_p

        layers = []
        for i in range(self.n_layers + 1):
            if i != 0:
                input_dim = self.hidden_dim
            output_dim = self.hidden_dim if i < self.n_layers else self.n_langs
            layers.append(nn.Linear(input_dim, output_dim))
            if i < self.n_layers:
                layers.append(nn.LeakyReLU(0.2))
                layers.append(nn.Dropout(self.dropout_p))
        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        return self.layers(input)


class BaseRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, rnn_cell="ylstm"):
        super(BaseRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        if rnn_cell.lower() == "ylstm":
            self.rnn_cell = LSTM
        elif rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU
        else:
            raise ValueError("Unsupported RNN Cell: {0}".format(rnn_cell))

    def forward(self, *args, **kwargs):
        raise NotImplementedError()


class EncoderRNN(BaseRNN):
    def __init__(self, input_size, hidden_size, dropout_p=0.,
                 n_layers=1, bidirectional=False, rnn_cell="ylstm", variable_lengths=True,
                 forget_bias=0.0):
        super(EncoderRNN, self).__init__(input_size, hidden_size, n_layers, rnn_cell)
        self.output_size = hidden_size if not bidirectional else 2 * hidden_size
        self.is_bidirectional = bidirectional
        self.variable_lengths = variable_lengths
        if rnn_cell == "ylstm":
            self.rnn = self.rnn_cell(input_size, hidden_size, n_layers,
                                     batch_first=True, bidirectional=bidirectional, dropout=dropout_p,
                                     forget_bias=forget_bias)
        else:
            self.rnn = self.rnn_cell(input_size, hidden_size, n_layers,
                                     batch_first=True, bidirectional=bidirectional, dropout=dropout_p)

    def forward(self, input_embeddings, input_lengths=None):
        embedded = input_embeddings
        rev_sort = None
        if self.variable_lengths and input_lengths is not None:
            (embedded, sorted_lens), rev_sort = self._sort_by_length(embedded, input_lengths)
            embedded = R.pack_padded_sequence(embedded, sorted_lens.tolist(), batch_first=True)
        output, hidden = self.rnn(embedded)
        hidden = self._init_state(hidden)
        if self.variable_lengths and input_lengths is not None:
            output, _ = R.pad_packed_sequence(output, batch_first=True)
            output, hidden = self._unsort(rev_sort, [output, hidden])
        return output, hidden

    def _sort_by_length(self, input_embeddings, input_lengths):
        sort_lens, sort_idx = input_lengths.sort(0, descending=True)
        sort_lens_rev_idx = sort_idx.sort()[1]
        sort_inputs = input_embeddings.index_select(0, sort_idx)
        return (sort_inputs, sort_lens), sort_lens_rev_idx

    def _unsort(self, og_order, inputs_list):
        unsorted = []
        for item in inputs_list:
            if isinstance(item, tuple):
                item_unsorted = tuple([x.index_select(1, og_order) for x in item])
            else:
                item_unsorted = item.index_select(0, og_order)
            unsorted.append(item_unsorted)
        return unsorted

    def _init_state(self, hidden):
        """ Initialize the encoder hidden state. """
        if hidden is None:
            return None
        if isinstance(hidden, tuple):
            encoder_hidden = tuple([self._cat_directions(h) for h in hidden])
        else:
            encoder_hidden = self._cat_directions(hidden)
        return encoder_hidden

    def _cat_directions(self, h):
        """ If the encoder is bidirectional, do the following transformation.
            (#directions * #layers, #batch, hidden_size) -> (#layers, #batch, #directions * hidden_size)
        """
        if self.is_bidirectional:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h


class AttentionMechanism(nn.Module):
    def __init__(self, encode_dim, decode_dim, use_proj=True):
        super(AttentionMechanism, self).__init__()
        self.encode_dim = encode_dim
        self.decode_dim = decode_dim
        self.linear_out = nn.Linear(encode_dim + decode_dim, decode_dim)
        self.linear_proj = None
        if use_proj:
            self.linear_proj = nn.Linear(encode_dim, decode_dim)

    def forward(self, decoder_hidden, all_encoder_hidden, encode_lens):
        batch_size = decoder_hidden.size(0)
        input_size = all_encoder_hidden.size(1)

        proj_decoder_states = decoder_hidden
        if self.linear_proj is not None:
            proj_decoder_states = self.linear_proj(decoder_hidden)

        raw_attn_scores = torch.bmm(all_encoder_hidden, proj_decoder_states.transpose(1, 2))
        mask = sequence_mask(encode_lens, max_len=input_size).unsqueeze(2).expand_as(raw_attn_scores)
        masked_attn_scores = torch.where(mask, raw_attn_scores, torch.full_like(raw_attn_scores, -1e16))
        norm_attn_scores = F.softmax(masked_attn_scores.view(-1, input_size), dim=1).view(batch_size, -1, input_size)
        attn_applied = torch.bmm(norm_attn_scores, all_encoder_hidden)

        # concat -> (batch, out_len, encoder_dim + decoder_dim)
        combined = torch.cat((attn_applied, decoder_hidden), dim=2)

        # output -> (batch, out_len, dim)
        output = torch.tanh(self.linear_out(combined.view(-1, self.encode_dim + self.decode_dim))).view(batch_size,
                                                                                                        -1,
                                                                                                        self.decode_dim)

        if not output.is_contiguous():
            output = output.contiguous()

        return output, norm_attn_scores


class DecoderRNN(BaseRNN):
    def __init__(self, lang_n_words: list, embed_size: int, hidden_size: int, max_len: int = 25,
                 n_layers: int = 1, rnn_cell: str = "ylstm", bidirectional: bool = False, dropout_p: float = 0.,
                 use_attention=False, use_general_attn=True):
        if bidirectional:
            hidden_size *= 2

        super(DecoderRNN, self).__init__(embed_size, hidden_size, n_layers, rnn_cell)

        self.rnn = self.rnn_cell(embed_size, hidden_size, n_layers, batch_first=True, dropout=dropout_p)
        self.max_length = max_len
        self.init_input = None
        self.use_attention = use_attention
        self.attention = None
        if use_attention:
            self.attention = AttentionMechanism(self.hidden_size, self.hidden_size, use_proj=use_general_attn)
        self.out_layers = nn.ModuleList([Linear(hidden_size, n_words)
                                         for n_words in lang_n_words])

    def forward(self, lang_id, input_embeddings, init_decoder_hidden, encoder_outputs, encoder_lens, func):
        batch_size = input_embeddings.size(0)
        output_size = input_embeddings.size(1)

        out_states, hidden = self.rnn(input_embeddings, init_decoder_hidden)

        attn = None
        if self.use_attention:
            out_states, attn = self.attention(out_states, encoder_outputs, encoder_lens)

        outputs = self.out_layers[lang_id](out_states.contiguous().view(-1, self.hidden_size)).view(batch_size,
                                                                                                    output_size,
                                                                                                    -1)
        return outputs, hidden, attn


class CharCNN(nn.Module):
    def __init__(self, embedding_num, embedding_dim, filters):
        super(CharCNN, self).__init__()
        self.output_size = sum([x[1] for x in filters])
        self.embedding = nn.Embedding(embedding_num,
                                      embedding_dim,
                                      padding_idx=0,
                                      sparse=True)
        self.convs = nn.ModuleList([nn.Conv2d(1, x[1], (x[0], embedding_dim))
                                    for x in filters])

    def forward(self, inputs):
        inputs_embed = self.embedding(inputs)
        inputs_embed = inputs_embed.unsqueeze(1)
        conv_outputs = [F.relu(conv(inputs_embed)).squeeze(3)
                        for conv in self.convs]
        max_pool_outputs = [F.max_pool1d(i, i.size(2)).squeeze(2)
                            for i in conv_outputs]
        outputs = torch.cat(max_pool_outputs, 1)
        return outputs


class KalmanFilter():
    def __init__(self, input_size, hidden_size, dropout_p=0.):
        self.kf = KalmanFilter(
                transition_matrices = torch.zeros(2,2), 
                observation_matrices = [[0.1, 0.5], [-0.3, 0.0]]
            )

    def forward(self, input_embeddings, input_lengths=None):
        self.kf = self.kf.em(input_embeddings, n_iter=5)
        (filtered_state_means, filtered_state_covariances) = kf.filter(measurements)
        (smoothed_state_means, smoothed_state_covariances) = kf.smooth(measurements)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.gpu = False

    def cuda(self, device=None):
        self.gpu = True
        for module in self.children():
            module.cuda(device)
        return self

    def cpu(self):
        self.gpu = False
        for module in self.children():
            module.cpu()
        return self


class GANModel(Model):
    def __init__(self,
                 lang_n_words,
                 word_embedding_layers,
                 char_embedding,
                 encoder,
                 discriminator=None,
                 univ_fc_layer=None,
                 spec_fc_layers=None,
                 embed_dropout_prob=0.,
                 encoder_out_dropout_prob=0.,
                 use_char_embedding=True,
                 char_highway=None,
                 swap_dist=3,
                 word_drop_prob=0.1):
        super(GANModel, self).__init__()
        self.n_langs = len(lang_n_words)
        self.discriminator = discriminator
        self.encoder = encoder

        self.embed_dropout_prob = embed_dropout_prob
        self.encoder_out_dropout_prob = encoder_out_dropout_prob
        self.use_char_embedding = use_char_embedding

        self.word_embed_layers = nn.ModuleList(word_embedding_layers)
        self.char_embedding = char_embedding

        self.univ_fc_layer = univ_fc_layer
        self.spec_fc_layers = nn.ModuleList(spec_fc_layers) if spec_fc_layers else None
        self.char_highway = char_highway
        self.encoder_out_dropout = nn.Dropout(p=encoder_out_dropout_prob)
        self.embed_dropout = nn.Dropout(p=embed_dropout_prob)
        if spec_fc_layers:
            self.spec_gates = nn.ModuleList([Linear(sfc.in_features, sfc.out_features) for sfc in spec_fc_layers])

        self.swap_dist = swap_dist
        self.word_drop_prob = word_drop_prob

    def embed_tokens(self, lang_id, inputs, batch_size, seq_len, chars=None, char_lens=None):
        # Word embedding
        token_embed = self.word_embed_layers[lang_id](inputs)

        # Character embedding
        if self.use_char_embedding and chars is not None:
            chars_embed = self.char_embedding(chars)
            if self.char_highway:
                chars_embed = self.char_highway(chars_embed)
            chars_embed = chars_embed.view(batch_size, seq_len, -1)
            token_embed = torch.cat([token_embed, chars_embed], dim=2)

        token_embed = self.embed_dropout(token_embed)

        return token_embed

    def encode(self, lang_id, inputs, lens, chars=None, char_lens=None):
        batch_size, seq_len = inputs.size()

        inputs_embed = self.embed_tokens(lang_id, inputs, batch_size, seq_len, chars, char_lens)

        encoder_out, encoder_hidden = self.encoder(inputs_embed, lens)
        encoder_out = encoder_out.contiguous().view(-1, self.encoder.output_size)
        encoder_out = self.encoder_out_dropout(encoder_out)
        encoder_out = encoder_out.view(batch_size, seq_len, -1)

        return encoder_out, encoder_hidden

    def project_gate(self, lang_id, encode_out, proj_dim=-1):
        batch_size, seq_len, _ = encode_out.size()
        # Fully-connected layer
        univ_feats = self.univ_fc_layer(encode_out)
        if self.spec_fc_layers is not None:
            spec_feats = self.spec_fc_layers[lang_id](encode_out)
            gate = torch.sigmoid(self.spec_gates[lang_id](encode_out))
            outputs = (gate * spec_feats) + ((1 - gate) * univ_feats)
        else:
            outputs = univ_feats
        outputs = outputs.view(batch_size, seq_len, proj_dim)
        return outputs

    def calc_adv_loss(self, lang_id, encode_out):  # lang_id should be language that was given to the encoder.
        dis_loss = None
        if self.discriminator is not None:
            self.discriminator.eval()
            dis_preds = self.discriminator(encode_out.view(-1, encode_out.size(-1)))
            gen_tgt = (torch.LongTensor(dis_preds.size(0)).random_(1, self.n_langs) + lang_id) % self.n_langs
            if dis_preds.is_cuda:
                gen_tgt = gen_tgt.cuda()
            dis_loss = F.cross_entropy(dis_preds, gen_tgt)
        return dis_loss

    def discriminator_step(self, langs, lang_data):
        if self.discriminator is None:
            return None

        self.encoder.eval()
        self.discriminator.train()

        encoded = []
        for i, lang_id in enumerate(langs):
            lang_inputs, lang_lens, lang_chars, lang_char_lens = lang_data[i]
            lang_encoded = self.encode(lang_id, lang_inputs, lang_lens, lang_chars, lang_char_lens)[0]
            batch_size, seq_len, dim = lang_encoded.size()
            mask = sequence_mask(lang_lens, max_len=seq_len).unsqueeze(2).expand(batch_size, seq_len, dim)
            disc_inputs = lang_encoded.masked_select(mask).view(lang_lens.sum().item(), dim)
            encoded.append(disc_inputs)

        all_disc_inputs = [x.view(-1, x.size(-1)) for x in encoded]
        n_tokens = [x.size(0) for x in all_disc_inputs]
        encoded = torch.cat(all_disc_inputs, 0)
        predictions = self.discriminator(encoded.data)

        target = torch.cat([torch.zeros(sz).fill_(langs[i]) for i, sz in enumerate(n_tokens)])
        target = target.contiguous().long()
        if predictions.is_cuda:
            target = target.cuda()
        discriminator_loss = F.cross_entropy(predictions, target)
        self.encoder.train()
        self.discriminator.eval()
        return discriminator_loss


class RNNEncoderDecoder(Seq2Seq):

    KEY_ATTN_SCORE = 'attention_score'
    KEY_LENGTH = 'length'
    KEY_SEQ = 'sequence'

    def __init__(self,
                 lang_n_words,
                 sos_id,
                 eos_id,
                 pad_id,
                 max_len,
                 word_embedding_layers,
                 char_embedding,
                 encoder,
                 decoder,
                 discriminator=None,
                 univ_fc_layer=None,
                 spec_fc_layers=None,
                 embed_dropout_prob=0.,
                 encoder_out_dropout_prob=0.,
                 use_char_embedding=True,
                 char_highway=None
                 ):
        super(RNNEncoderDecoder, self).__init__(lang_n_words=lang_n_words,
                                                word_embedding_layers=word_embedding_layers,
                                                char_embedding=char_embedding,
                                                encoder=encoder, discriminator=discriminator,
                                                univ_fc_layer=univ_fc_layer, spec_fc_layers=spec_fc_layers,
                                                embed_dropout_prob=embed_dropout_prob,
                                                encoder_out_dropout_prob=encoder_out_dropout_prob,
                                                use_char_embedding=use_char_embedding, char_highway=char_highway)

        self.sos_id = sos_id
        self.eos_id = eos_id
        self.pad_id = pad_id

        self.decoder = decoder
        self.max_length = max_len

        lang_losses = []
        self.n_langs = len(lang_n_words)
        for lang_id, n_words in enumerate(lang_n_words):
            lang_loss_weight = torch.FloatTensor(n_words).fill_(1)
            lang_loss_weight[self.pad_id[lang_id]] = 0
            lang_losses.append(nn.CrossEntropyLoss(lang_loss_weight))
        self.loss_fn = nn.ModuleList(lang_losses)

    def decode(self, lang_id, inputs=None, input_lens=None, chars=None, char_lens=None,
               encoder_hidden=None, encoder_outputs=None, encoder_lens=None,
               func=F.log_softmax, teacher_forcing_ratio=0):
        ret_dict = dict()
        ret_dict[RNNEncoderDecoder.KEY_ATTN_SCORE] = list()

        inputs, batch_size, max_length = self._validate_args(lang_id, inputs, input_lens,
                                                             encoder_hidden, encoder_outputs,
                                                             func, teacher_forcing_ratio)
        decoder_hidden = encoder_hidden

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        all_step_decoder_outputs = []
        sequence_symbols = []
        lengths = np.array([max_length] * batch_size)

        def decode_step(step, step_output, step_attn):
            all_step_decoder_outputs.append(step_output)
            if step_attn is not None:
                ret_dict[RNNEncoderDecoder.KEY_ATTN_SCORE].append(step_attn)
            symbols = all_step_decoder_outputs[-1].topk(1)[1]
            sequence_symbols.append(symbols)

            eos_batches = symbols.data.eq(self.eos_id[lang_id])
            if eos_batches.dim() > 0:
                eos_batches = eos_batches.cpu().view(-1).numpy()
                update_idx = ((lengths > step) & eos_batches) != 0
                lengths[update_idx] = len(sequence_symbols)
            return symbols

        # Manual unrolling is used to support random teacher forcing.
        # If teacher_forcing_ratio is True or False instead of a probability, the unrolling can be done in graph
        if use_teacher_forcing:
            decoder_input_embeds = self.embed_tokens(lang_id, inputs[:, :-1], batch_size,
                                                     max_length, chars, char_lens)
            decoder_output, decoder_hidden, attn = self.decoder(lang_id, decoder_input_embeds, decoder_hidden,
                                                                encoder_outputs, encoder_lens, func=func)

            for di in range(decoder_output.size(1)):
                step_output = decoder_output[:, di, :]
                if attn is not None:
                    step_attn = attn[:, di, :]
                else:
                    step_attn = None
                decode_step(di, step_output, step_attn)
        else:
            decoder_input_ids = inputs[:, 0].unsqueeze(1)
            for di in range(max_length):
                decoder_input_embeds = self.embed_tokens(lang_id, decoder_input_ids, batch_size,
                                                         max_length, chars, char_lens)
                decoder_output, decoder_hidden, step_attn = self.decoder(lang_id, decoder_input_embeds, decoder_hidden,
                                                                         encoder_outputs, encoder_lens, func=func)
                step_output = decoder_output.squeeze(1)
                symbols = decode_step(di, step_output, step_attn)
                decoder_input_ids = symbols

        all_step_decoder_outputs = torch.stack(all_step_decoder_outputs, dim=1)
        ret_dict[RNNEncoderDecoder.KEY_SEQ] = torch.cat(sequence_symbols, dim=1)
        ret_dict[RNNEncoderDecoder.KEY_LENGTH] = lengths.tolist()

        return all_step_decoder_outputs, ret_dict

    def forward_model(self, src_lang_id, tgt_lang_id, encoder_inputs, encoder_lens,
                      labels=None, label_lens=None, chars=None, char_lens=None, func=F.log_softmax,
                      teacher_forcing_ratio=0):
        encoder_outputs, encoder_hidden = self.encode(src_lang_id, encoder_inputs, encoder_lens, chars, char_lens)

        encoder_outputs = self.project_gate(src_lang_id, encoder_outputs)

        dec_outputs, dec_dict = self.decode(lang_id=tgt_lang_id, inputs=labels, input_lens=label_lens,
                                            chars=None, char_lens=None, encoder_hidden=encoder_hidden,
                                            encoder_outputs=encoder_outputs, encoder_lens=encoder_lens, func=func,
                                            teacher_forcing_ratio=teacher_forcing_ratio)

        return encoder_outputs, dec_outputs, dec_dict

    def predict(self, src_lang_id, tgt_lang_id, inputs, lens, labels, label_lens, chars=None, char_lens=None):
        self.eval()

        logits, loss, _, ret_dict = self.calc_loss(src_lang_id, tgt_lang_id, inputs, lens,
                                                   labels, label_lens, chars, char_lens,
                                                   teacher_forcing_ratio=0, adversarial=False,
                                                   use_noise=False)

        _, seq_len, output_size = logits.size()
        preds = F.softmax(logits.view(-1, output_size), dim=1).view(-1, seq_len, output_size)

        self.train()
        return preds, loss, ret_dict

    def calc_loss(self, src_lang_id, tgt_lang_id, inputs, lens, labels, label_lens,
                  chars=None, char_lens=None, func=F.log_softmax, teacher_forcing_ratio=1,
                  adversarial=True, use_noise=True):
        if use_noise and src_lang_id == tgt_lang_id:
            adj_inputs, adj_lens, adj_chars, adj_char_lens = self.add_noise(inputs, lens, chars, char_lens)
        else:
            adj_inputs, adj_lens, adj_chars, adj_char_lens = (inputs, lens, chars, char_lens)
        adj_labels = self._append_sos_eos(tgt_lang_id, labels, label_lens)
        # adj_labels = labels
        encoded, logits, ret_dict = self.forward_model(src_lang_id, tgt_lang_id,
                                                       adj_inputs, adj_lens,
                                                       adj_labels, label_lens,
                                                       adj_chars, adj_char_lens,
                                                       func, teacher_forcing_ratio)
        dec_loss = None
        dis_loss = None
        if adj_labels is not None:
            dec_loss = self.loss_fn[tgt_lang_id](torch.transpose(logits, 1, 2), adj_labels[:, 1:])
            if adversarial:
                dis_loss = self.calc_adv_loss(src_lang_id, encoded)

        return logits, dec_loss, dis_loss, ret_dict

    def _validate_args(self, lang_id, inputs, input_lens, encoder_hidden, encoder_outputs, func, teacher_forcing_ratio):
        if encoder_outputs is None:
            raise ValueError("Argument encoder_outputs cannot be None.")

        use_cuda = inputs.is_cuda

        # inference batch size
        if inputs is None and encoder_hidden is None:
            batch_size = 1
        else:
            if inputs is not None:
                batch_size = inputs.size(0)
            else:
                if self.rnn_cell is nn.LSTM:
                    batch_size = encoder_hidden[0].size(1)
                elif self.rnn_cell is nn.GRU:
                    batch_size = encoder_hidden.size(1)

        # set default input and max decoding length
        if inputs is None:
            if teacher_forcing_ratio > 0:
                raise ValueError("Teacher forcing has to be disabled (set 0) when no inputs is provided.")
            inputs = torch.LongTensor([self.sos_id[lang_id]] * batch_size).view(batch_size, 1)
            if use_cuda:
                inputs = inputs.cuda()
            max_length = self.max_length
        else:
            max_length = inputs.size(1) - 1  # minus the start of sequence symbol

        return inputs, batch_size, max_length
