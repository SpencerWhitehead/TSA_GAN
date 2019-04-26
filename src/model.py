import torch
import torch.nn as nn
import torch.nn.init as I
import torch.nn.utils.rnn as R
import torch.nn.functional as F

import random
import logging

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
    def __init__(self, n_labels, input_dim, n_layers, hidden_size, dropout_p):
        super(Discriminator, self).__init__()

        self.n_labels = n_labels
        self.input_dim = input_dim
        self.n_layers = n_layers
        self.hidden_dim = hidden_size
        self.dropout_p = dropout_p

        layers = []
        for i in range(self.n_layers + 1):
            if i != 0:
                input_dim = self.hidden_dim
            output_dim = self.hidden_dim if i < self.n_layers else self.n_labels
            layers.append(nn.Linear(input_dim, output_dim))
            if i < self.n_layers:
                layers.append(nn.LeakyReLU(0.2))
                layers.append(nn.Dropout(self.dropout_p))
        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        return self.layers(input)


class BaseRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, rnn_cell_type="ylstm"):
        super(BaseRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        if rnn_cell_type.lower() == "ylstm":
            self.rnn_cell = LSTM
        elif rnn_cell_type.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell_type.lower() == 'gru':
            self.rnn_cell = nn.GRU
        else:
            raise ValueError("Unsupported RNN Cell: {0}".format(rnn_cell_type))

    def forward(self, *args, **kwargs):
        raise NotImplementedError()


class DecoderRNN(BaseRNN):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, max_len: int = 100,
                 n_layers: int = 1, rnn_cell_type: str = "ylstm", bidirectional: bool = False, dropout_p: float = 0.):
        if bidirectional:
            hidden_size *= 2

        super(DecoderRNN, self).__init__(input_size, hidden_size, n_layers, rnn_cell_type)

        self.rnn = self.rnn_cell(input_size, hidden_size, n_layers, batch_first=True, dropout=dropout_p)
        self.max_length = max_len
        self.out_layer = Linear(hidden_size, output_size)

    def forward(self, input_embeddings, init_decoder_hidden, encoder_outputs=None, encoder_lens=None):
        out_states, hidden = self.rnn(input_embeddings, init_decoder_hidden)
        outputs = self.out_layer(out_states.contiguous())
        return outputs, hidden


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
                 decoder,
                 max_len,
                 discriminator,
                 disc_labels):
        super(GANModel, self).__init__()
        self.discriminator = discriminator
        self.decoder = decoder
        self.max_length = max_len
        self.disc_labels = disc_labels
        self.n_disc_labels = len(disc_labels)

    def calc_adv_loss(self, label_id, decode_out):  # lang_id should be language that was given to the encoder.
        dis_loss = None
        if self.discriminator is not None:
            self.discriminator.eval()
            with torch.no_grad():
                dis_preds = self.discriminator(decode_out.view(-1, decode_out.size(-1)))
            gen_tgt = ((torch.LongTensor(dis_preds.size(0)).random_(1, self.n_disc_labels) + label_id)
                            % self.n_disc_labels)
            if dis_preds.is_cuda:
                gen_tgt = gen_tgt.cuda()
            dis_loss = F.cross_entropy(dis_preds, gen_tgt)
        return dis_loss

    def discriminator_step(self, inputs, input_lens):
        if self.discriminator is None:
            return None

        self.encoder.eval()
        self.discriminator.train()

        disc_labels = [1, 0]
        decoded = [inputs]

        with torch.no_grad():
            decode_out = self.decode(inputs=inputs, input_lens=input_lens, teacher_forcing_ratio=0)
        batch_size, seq_len, dim = decode_out.size()
        mask = sequence_mask(input_lens, max_len=seq_len).unsqueeze(2).expand(batch_size, seq_len, dim)
        disc_inputs = decode_out.masked_select(mask).view(input_lens.sum().item(), dim)
        decoded.append(disc_inputs)

        all_disc_inputs = [x.view(-1, x.size(-1)) for x in decoded]
        n_steps = [x.size(0) for x in all_disc_inputs]
        decoded = torch.cat(all_disc_inputs, 0)
        predictions = self.discriminator(decoded.data)

        target = torch.cat([torch.zeros(sz).fill_(disc_labels[i]) for i, sz in enumerate(n_steps)])
        target = target.contiguous().long()
        if predictions.is_cuda:
            target = target.cuda()
        discriminator_loss = F.cross_entropy(predictions, target)
        self.encoder.train()
        self.discriminator.eval()
        return discriminator_loss

    def decode(self, init_decoder_hidden=None, inputs=None, input_lens=None, teacher_forcing_ratio=0):
        inputs, batch_size, max_length = self._validate_args(inputs, input_lens, teacher_forcing_ratio)

        decoder_hidden = init_decoder_hidden
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        # Manual unrolling is used to support random teacher forcing.
        # If teacher_forcing_ratio is True or False instead of a probability, the unrolling can be done in graph
        if use_teacher_forcing:
            all_step_decoder_outputs, decoder_hidden = self.decoder(inputs, decoder_hidden)
        else:
            all_step_decoder_outputs = []
            decoder_output = inputs
            for di in range(max_length):
                decoder_output, decoder_hidden = self.decoder(decoder_output, decoder_hidden)
                all_step_decoder_outputs.append(decoder_output)

            all_step_decoder_outputs = torch.stack(all_step_decoder_outputs, dim=1)

        return all_step_decoder_outputs

    def forward_model(self, inputs, input_lens=None, teacher_forcing_ratio=0):
        dec_outputs = self.decode(inputs=inputs, input_lens=input_lens, teacher_forcing_ratio=teacher_forcing_ratio)
        return dec_outputs

    def predict(self, inputs=None, input_lens=None):
        self.eval()
        with torch.no_grad():
            logits, loss, adv_loss = self.calc_loss(inputs, input_lens, teacher_forcing_ratio=0)
        self.train()
        return logits, loss, adv_loss

    def calc_loss(self, label_id, inputs, input_lens, teacher_forcing_ratio=1):
        logits = self.forward_model(inputs, input_lens, teacher_forcing_ratio)
        dec_loss = F.mse_loss(logits, inputs)
        dis_loss = self.calc_adv_loss(label_id, logits)

        return logits, dec_loss, dis_loss

    def _validate_args(self, inputs, input_lens, teacher_forcing_ratio):
        use_cuda = inputs.is_cuda

        # inference batch size
        if inputs is None:
            batch_size = 1
        else:
            batch_size = inputs.size(0)

        # set default input and max decoding length
        if inputs is None:
            if teacher_forcing_ratio > 0:
                raise ValueError("Teacher forcing has to be disabled (set 0) when no inputs is provided.")
            # inputs = torch.FloatTensor([self.sos_id[lang_id]] * batch_size).view(batch_size, 1)
            inputs = torch.randn((batch_size, self.decoder.input_size))
            if use_cuda:
                inputs = inputs.cuda()
            max_length = self.max_length
        else:
            max_length = inputs.size(1)  # minus the start of sequence symbol

        return inputs, batch_size, max_length
