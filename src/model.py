import torch
import torch.nn as nn
import torch.nn.init as I
import torch.nn.functional as F

import numpy as np

import random
import logging

import src.constants as C

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
                 dropout: float = 0.,
                 bidirectional: bool = False,
                 forget_bias: float = 0.
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


class FCDiscriminator(nn.Module):

    DISC_MODEL = "FC"

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 n_labels: int,
                 n_layers: int,
                 dropout_p: float = 0.,
                 **kwargs
                 ):
        super(FCDiscriminator, self).__init__()

        self.input_size = input_size
        self.hidden_size= hidden_size
        self.output_size = n_labels
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        layers = []
        input_dim = input_size
        for i in range(self.n_layers + 1):
            if i != 0:
                input_dim = self.hidden_size
            output_dim = self.hidden_size if i < self.n_layers else self.output_size
            layers.append(nn.Linear(input_dim, output_dim))
            if i < self.n_layers:
                layers.append(nn.LeakyReLU(0.2))
                layers.append(nn.Dropout(self.dropout_p))
        self.layers = nn.Sequential(*layers)

    def forward(self, inputs, **kwargs):
        return self.layers(inputs)


class RNNDiscriminator(BaseRNN):

    DISC_MODEL = "RNN"

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 n_labels: int,
                 n_layers: int = 1,
                 dropout_p: float = 0.,
                 rnn_cell_type: str = "ylstm"
                 ):
        super(RNNDiscriminator, self).__init__(input_size, hidden_size, n_layers, rnn_cell_type)

        self.output_size = n_labels
        self.dropout_p = dropout_p

        self.rnn = self.rnn_cell(input_size, hidden_size, n_layers, batch_first=True, dropout=dropout_p)
        self.out_layer = Linear(hidden_size, n_labels)

    def forward(self, inputs, init_hidden=None):
        out_states, hidden = self.rnn(inputs, init_hidden)
        outputs = self.out_layer(out_states.contiguous())
        return outputs


class GeneratorRNN(BaseRNN):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 output_size: int,
                 n_layers: int = 1,
                 rnn_cell_type: str = "ylstm",
                 dropout_p: float = 0.,
                 noise_size: int = 0,
                 cond_size: int = 0
                 ):
        super(GeneratorRNN, self).__init__(input_size, hidden_size, n_layers, rnn_cell_type)
        self.output_size = output_size
        self.noise_size = noise_size
        self.cond_size = cond_size

        self.rnn = self.rnn_cell(
            input_size + noise_size + cond_size,
            hidden_size,
            n_layers,
            batch_first=True,
            dropout=dropout_p
        )
        self.out_layer = Linear(hidden_size, output_size)

    def forward(self, inputs, init_hidden):
        out_states, hidden = self.rnn(inputs, init_hidden)
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
                 generator,
                 discriminator,
                 max_len: int,
                 min_val: float,
                 max_val: float,
                 one_hot_cond: bool = True
                 ):
        super(GANModel, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.max_length = max_len
        self.real_label = 1
        self.fake_label = 0
        self.n_disc_labels = 2
        self.min_val = min_val
        self.max_val = max_val
        self.one_hot_cond = one_hot_cond

    def calc_adv_loss(self, decode_out):
        dis_loss = None
        if self.discriminator is not None:
            self.discriminator.eval()
            if self.discriminator.DISC_MODEL == "FC":
                dis_preds = self.discriminator(decode_out.view(-1, decode_out.size(-1)))
                tgt_size = dis_preds.size(0)

            else:
                dis_preds = self.discriminator(decode_out)
                tgt_size = dis_preds.size(0) * dis_preds.size(1)
                dis_preds = torch.transpose(dis_preds, 1, 2)

            gen_tgt = ((torch.LongTensor(tgt_size).random_(1, self.n_disc_labels) + self.real_label)
                       % self.n_disc_labels).view(dis_preds.size(0), -1).squeeze(1)
            if dis_preds.is_cuda:
                gen_tgt = gen_tgt.cuda()
            dis_loss = F.cross_entropy(dis_preds, gen_tgt)
        return dis_loss

    def discriminator_step(self, inputs, input_lens, teacher_forching_ratio=0.):
        if self.discriminator is None:
            return None

        self.generator.eval()
        self.discriminator.train()

        disc_labels = [self.real_label, self.fake_label]

        with torch.no_grad():
            decode_out = self.decode(
                inputs=inputs,
                input_lens=input_lens,
                teacher_forcing_ratio=teacher_forching_ratio
            )
        batch_size, gen_seq_len, dim = decode_out.size()

        if self.discriminator.DISC_MODEL == "FC":
            mask = sequence_mask(input_lens, max_len=gen_seq_len).unsqueeze(2).expand(batch_size, gen_seq_len, dim)
            disc_inputs = decode_out.masked_select(mask).view(input_lens.sum().item(), dim)
            decoded = [inputs, disc_inputs]

            all_disc_inputs = [x.view(-1, x.size(-1)) for x in decoded]
            n_steps = [x.size(0) for x in all_disc_inputs]
            decoded = torch.cat(all_disc_inputs, 0)
            target = torch.cat([torch.zeros(sz).fill_(disc_labels[i]) for i, sz in enumerate(n_steps)])
            predictions = self.discriminator(decoded.detach().data)
        else:
            decoded = [inputs, decode_out]
            decoded_lens = [x.size(1) for x in decoded]
            diff_len = abs(decoded_lens[0] - decoded_lens[1])
            min_len_idx = decoded_lens.index(min(decoded_lens))

            if diff_len:
                pad_min_decoded = F.pad(decoded[min_len_idx], [1, diff_len, 0], "constant", C.padval)
            else:
                pad_min_decoded = decoded[min_len_idx]

            if min_len_idx == 0:
                decoded = torch.cat([pad_min_decoded, decode_out], dim=0)
                real_len = pad_min_decoded.size(1)
                fake_len = gen_seq_len
            else:
                decoded = torch.cat([inputs, pad_min_decoded], dim=0)
                real_len = inputs.size(1)
                fake_len = pad_min_decoded.size(1)

            real_tgt = torch.zeros(batch_size, real_len).fill_(self.real_label)
            fake_tgt = torch.zeros(batch_size, fake_len).fill_(self.fake_label)
            target = torch.cat([real_tgt, fake_tgt], dim=0)

            predictions = self.discriminator(decoded.detach().data)
            predictions = torch.transpose(predictions, 1, 2)

        target = target.contiguous().long()
        if predictions.is_cuda:
            target = target.cuda()

        discriminator_loss = F.cross_entropy(predictions, target)
        self.generator.train()
        self.discriminator.eval()
        return discriminator_loss

    def decode(self, init_generator_hidden=None, inputs=None, input_lens=None, teacher_forcing_ratio=0.):
        adj_inputs, batch_size, max_length = self._validate_args(inputs, input_lens, teacher_forcing_ratio)

        generator_hidden = init_generator_hidden
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        # Manual unrolling is used to support random teacher forcing.
        # If teacher_forcing_ratio is True or False instead of a probability, the unrolling can be done in graph
        if use_teacher_forcing:
            all_step_generator_outputs, generator_hidden = self.generator(adj_inputs[:, :-1], generator_hidden)
        else:
            all_step_generator_outputs = []
            generator_output = adj_inputs[:, 0].unsqueeze(1)
            for di in range(max_length):
                generator_output, generator_hidden = self.generator(generator_output, generator_hidden)
                all_step_generator_outputs.append(generator_output.squeeze(1))

            all_step_generator_outputs = torch.stack(all_step_generator_outputs, dim=1)

        return all_step_generator_outputs

    def forward_model(self, inputs, input_lens=None, teacher_forcing_ratio=0.):
        dec_outputs = self.decode(inputs=inputs, input_lens=input_lens, teacher_forcing_ratio=teacher_forcing_ratio)
        return dec_outputs

    def predict(self, inputs=None, input_lens=None, teacher_forcing_ratio=0., use_pred_loss=False):
        self.eval()
        with torch.no_grad():
            logits, loss, adv_loss = self.calc_loss(
                inputs, input_lens, teacher_forcing_ratio=teacher_forcing_ratio, use_pred_loss=use_pred_loss
            )
        self.train()
        return logits, loss, adv_loss

    def calc_loss(self, inputs, input_lens, teacher_forcing_ratio=1., use_pred_loss=False):
        logits = self.forward_model(inputs, input_lens, teacher_forcing_ratio)

        dec_loss, dis_loss = None, None
        if inputs is not None:
            if use_pred_loss:
                dec_loss = F.mse_loss(logits, inputs)
            else:
                dec_loss = None
            dis_loss = self.calc_adv_loss(logits)

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
            inputs = self.get_noise(batch_size, 1, self.generator.input_size, init_step=True)
            if use_cuda:
                inputs = inputs.cuda()
            max_length = self.max_length
        else:
            max_length = inputs.size(1)
            inputs, input_lens = self._append_noise(inputs, input_lens, self.generator.input_size, use_cuda)

        return inputs, batch_size, max_length

    def _append_noise(self, inputs, input_lens, feat_dim, use_cuda=False):
        if inputs is None:
            return inputs, input_lens

        batch_size = inputs.size(0)
        noise_step = self.get_noise(batch_size, 1, feat_dim, init_step=True)
        if use_cuda:
            noise_step = noise_step.cuda()

        return torch.cat([noise_step, inputs], dim=1), input_lens + 1

    def get_noise(self, batch_size, seq_len, feat_dim, init_step=False, use_cuda=False):
        if init_step:
            zn = (self.max_val - self.min_val) * torch.randn((batch_size, 1, feat_dim)) + self.min_val
        else:
            zn = torch.randn((batch_size, seq_len, feat_dim))

        if use_cuda:
            zn = zn.cuda()

        return zn

    def get_cond(self, batch_size, seq_len, use_cuda=False):
        if self.one_hot_cond:
            cn = np.zeros(shape=(batch_size, self.cond_dim))
            # locations
            labels = np.random.choice(self.cond_dim, batch_size)
            cn[np.arange(batch_size), labels] = 1.
        else:
            cn = np.random.choice(1, size=(batch_size, self.cond_dim))

        cn = torch.from_numpy(np.tile(np.expand_dims(cn, axis=1), (batch_size, seq_len, 1)))
        if use_cuda:
            cn = cn.cuda()

        return cn
