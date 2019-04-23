import os
import json
import time
import logging
import traceback
from collections import Counter

import torch
from torch import optim
from torch.nn.utils import clip_grad_norm_

from argparse import ArgumentParser

from torch.utils.data import DataLoader

# from util import MTEvaluator
import src.util as util
from src.data import MTParser, ShardDynamicMTDataset, DynamicMTDataset, MTProcessor, count2vocab
from src.model import Linears, CharCNN, Highway, Discriminator, EncoderRNN, DecoderRNN, RNNEncoderDecoder, load_embedding

timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()

argparser = ArgumentParser()


argparser.add_argument('--train', nargs='+', help="Path to the training set file. " +
                                                  "Must be <lang 1 filename> <lang 2 filename> ...")
argparser.add_argument('--dev', nargs='+', help="Path to the dev set file. " +
                                                "Must be <lang 1 filename> <lang 2 filename> ...")
argparser.add_argument('--test', nargs='+', help="Path to the test set file. " +
                                                 "Must be <lang 1 filename> <lang 2 filename> ...")
argparser.add_argument('--log', help='Path to the log dir')
argparser.add_argument('--model', help='Path to the model file')
argparser.add_argument('--results', help="Path to results file")
argparser.add_argument('--batch_size', default=10, type=int, help='Batch size')
argparser.add_argument('--max_epoch', default=100, type=int)
argparser.add_argument('--word_embed', nargs='*',
                       help='Path to the pre-trained embedding file')
argparser.add_argument('--word_embed_dim', type=int, default=100,
                       help='Word embedding dimension')
argparser.set_defaults(word_ignore_case=False)
argparser.add_argument('--char_embed_dim', type=int, default=50,
                       help='Character embedding dimension')
argparser.add_argument('--charcnn_filters', default='2,25;3,25;4,25',
                       help='Character-level CNN filters')
argparser.add_argument('--charhw_layer', default=1, type=int)
argparser.add_argument('--charhw_func', default='relu')
argparser.add_argument('--use_highway', action='store_true')
argparser.add_argument('--lstm_hidden_size', default=100, type=int,
                       help='LSTM hidden state size')
argparser.add_argument('--lstm_forget_bias', default=0, type=float,
                       help='LSTM forget bias')
argparser.add_argument('--feat_dropout', default=.5, type=float,
                       help='Word feature dropout probability')
argparser.add_argument('--lstm_dropout', default=.5, type=float,
                       help='LSTM output dropout probability')
argparser.add_argument('--lr', default=0.005, type=float,
                       help='Learning rate')
argparser.add_argument('--dis_lr', default=0.005, type=float,
                       help='Learning rate')
argparser.add_argument('--lambda_auto', default=1.0, type=float,
                       help='Auto-encoding loss scaling factor')
argparser.add_argument('--lambda_cd', default=1.0, type=float,
                       help='Cross-domain loss scaling factor')
argparser.add_argument('--lambda_adv', default=1.0, type=float,
                       help='Adversarial loss scaling factor')
argparser.add_argument('--momentum', default=.9, type=float)
argparser.add_argument('--decay_rate', default=.9, type=float)
argparser.add_argument('--decay_step', default=10000, type=int)
argparser.add_argument('--grad_clipping', default=5, type=float)
argparser.add_argument('--gpu', action='store_true')
argparser.add_argument('--device', default=0, type=int)
argparser.add_argument('--thread', default=5, type=int)

args = argparser.parse_args()

use_gpu = args.gpu and torch.cuda.is_available()
if use_gpu:
    torch.cuda.set_device(args.device)

# Model file
model_dir = args.model
assert model_dir and os.path.isdir(model_dir), 'Model output dir is required'
model_file = os.path.join(model_dir, 'model.{}.mdl'.format(timestamp))

# Results file
results_dir = args.results
assert model_dir and os.path.isdir(results_dir), 'Result dir is required'
result_file = os.path.join(results_dir, 'results.{}.txt'.format(timestamp))

# Logging file
log_writer = None
if args.log:
    log_file = os.path.join(args.log, 'log.{}.txt'.format(timestamp))
    log_writer = open(log_file, 'a', encoding='utf-8')
    logger.addHandler(logging.FileHandler(log_file, encoding='utf-8'))
logger.info('----------')
logger.info('Parameters:')
for arg in vars(args):
    logger.info('{}: {}'.format(arg, getattr(args, arg)))
logger.info('----------')

# Data file
logger.info('Loading data sets')

parser = MTParser(num_to_zeros=True)
train_set = DynamicMTDataset(args.train[0], args.train[1], "train", parser=parser,
                             max_seq_len=25, max_char_len=25)
dev_set =  DynamicMTDataset(args.dev[0], args.dev[1], "dev", parser=parser,
                            max_seq_len=25, max_char_len=25)
test_set =  DynamicMTDataset(args.test[0], args.test[1], "test", parser=parser,
                             max_seq_len=25, max_char_len=25)
datasets = {'train': train_set, 'dev': dev_set, 'test': test_set}

# Vocabs
logger.info('Building vocabs')
l1_token_count, l1_char_count, l2_token_count, l2_char_count = Counter(), Counter(), Counter(), Counter()
for _, ds in datasets.items():
    l1tc, l1cc, l2tc, l2cc = ds.stats()
    l1_token_count.update(l1tc)
    l1_char_count.update(l1cc)
    l2_token_count.update(l2tc)
    l2_char_count.update(l2cc)
l1_t_vocab = count2vocab(l1_token_count, offset=len(C.TOKEN_PADS), pads=C.TOKEN_PADS)
l2_t_vocab = count2vocab(l2_token_count, offset=len(C.TOKEN_PADS), pads=C.TOKEN_PADS)
char_vocab = count2vocab(l1_char_count + l2_char_count, offset=len(C.CHAR_PADS), pads=C.CHAR_PADS)

l1_idx_token = {v: k for k, v in l1_t_vocab.items()}
l2_idx_token = {v: k for k, v in l2_t_vocab.items()}
train_set.numberize(l1_t_vocab, l2_t_vocab, char_vocab)
dev_set.numberize(l1_t_vocab, l2_t_vocab, char_vocab)
test_set.numberize(l1_t_vocab, l2_t_vocab, char_vocab)
logger.info('Language 1 #token: {}'.format(len(l1_t_vocab)))
logger.info('Language 2 #token: {}'.format(len(l2_t_vocab)))
logger.info('#char: {}'.format(len(char_vocab)))


lang_vocab_sizes = [len(l1_t_vocab), len(l2_t_vocab)]
lang_sos_idxs = [l2_t_vocab[C.SOS], l2_t_vocab[C.SOS]]
lang_eos_idxs = [l2_t_vocab[C.EOS], l2_t_vocab[C.EOS]]
lang_pad_idxs = [l2_t_vocab[C.PAD], l2_t_vocab[C.PAD]]

# Embedding file
if args.word_embed is not None:
    l1_word_embed = load_embedding(args.word_embed[0],
                                   dimension=args.word_embed_dim,
                                   vocab=l1_t_vocab)
    l2_word_embed = load_embedding(args.word_embed[1],
                                   dimension=args.word_embed_dim,
                                   vocab=l2_t_vocab)
else:
    l1_word_embed = torch.nn.Embedding(len(l1_t_vocab), args.word_embed_dim, padding_idx=C.PAD_INDEX, sparse=True)
    l2_word_embed = torch.nn.Embedding(len(l2_t_vocab), args.word_embed_dim, padding_idx=C.PAD_INDEX, sparse=True)

charcnn_filters = [[int(f.split(',')[0]), int(f.split(',')[1])]
                   for f in args.charcnn_filters.split(';')]

char_embed = CharCNN(len(char_vocab),
                     args.char_embed_dim,
                     filters=charcnn_filters)

char_hw = Highway(char_embed.output_size,
                  layer_num=args.charhw_layer,
                  activation=args.charhw_func)

feat_dim = l1_word_embed.embedding_dim + char_embed.output_size

enc = EncoderRNN(input_size=feat_dim, hidden_size=args.lstm_hidden_size, dropout_p=0.,
                 n_layers=1, bidirectional=True, rnn_cell="ylstm", variable_lengths=True)

linear = Linears(in_features=enc.output_size,
                 out_features=enc.output_size,
                 hiddens=[enc.output_size // 2])

dec = DecoderRNN(lang_vocab_sizes, l2_word_embed.embedding_dim, hidden_size=linear.output_size, max_len=25,
                 n_layers=1, rnn_cell="ylstm", bidirectional=False, dropout_p=0.,
                 use_attention=True, use_general_attn=True)

disc = Discriminator(n_langs=2, input_dim=enc.output_size, n_layers=1, hidden_size=enc.output_size, dropout_p=0.)

mt_model = RNNEncoderDecoder(lang_vocab_sizes, lang_sos_idxs, lang_eos_idxs, lang_pad_idxs, max_len=25,
                             word_embedding_layers=[l1_word_embed, l2_word_embed],
                             char_embedding=char_embed,
                             encoder=enc,
                             decoder=dec,
                             discriminator=disc,
                             univ_fc_layer=linear,
                             spec_fc_layers=None,
                             embed_dropout_prob=args.feat_dropout,
                             encoder_out_dropout_prob=args.lstm_dropout,
                             use_char_embedding=True,
                             char_highway=char_hw if args.use_highway else None
                             )

if use_gpu:
    mt_model.cuda()
torch.set_num_threads(args.thread)

logger.debug(mt_model)

lam_auto = args.lambda_auto
lam_cd = args.lambda_cd
lam_adv = args.lambda_adv

# Task
# optimizer = optim.SGD(filter(lambda p: p.requires_grad, mt_model.parameters()),
#                       lr=args.lr, momentum=args.momentum)

no_disc_params = [p for m, p in mt_model.named_parameters()
                  if p.requires_grad and 'discriminator' not in m]
optimizer = optim.SGD(no_disc_params,
                      lr=args.lr, momentum=args.momentum)
dis_optimizer = optim.SGD(filter(lambda p: p.requires_grad, disc.parameters()),
                          lr=args.dis_lr, momentum=args.momentum)
processor = MTProcessor(sort=False, gpu=use_gpu)
evaluator = util.MTEvaluator()

train_args = vars(args)
train_args['word_embed_size'] = l1_word_embed.num_embeddings
state = {
    'model': {
        'l1_word_embed': l1_word_embed.state_dict(),
        'l2_word_embed': l2_word_embed.state_dict(),
        'char_embed': char_embed.state_dict(),
        'char_hw': char_hw.state_dict(),
        'encoder': enc.state_dict(),
        'decoder': dec.state_dict(),
        'linear': linear.state_dict(),
        'mt_model': mt_model.state_dict()
    },
    'args': train_args,
    'vocab': {
        'l1_token': l1_t_vocab,
        'l2_token': l2_t_vocab,
        'char': char_vocab,
    }
}
try:
    global_step = 0
    best_dev_score = best_test_score = 0.0

    for epoch in range(args.max_epoch):
        logger.info('Epoch {}: Training'.format(epoch))

        best = False
        for ds in ['train', 'dev', 'test']:
            dataset = datasets[ds]
            epoch_loss = []
            results = []

            epoch_start_t = time.time()
            for batch in DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=ds == 'train',
                drop_last=ds == 'train',
                collate_fn=processor.process
            ):
                optimizer.zero_grad()
                dis_optimizer.zero_grad()
                l1_tokens, l1_seq_lens, l1_chars, l1_char_lens, \
                    l2_tokens, l2_seq_lens, l2_chars, l2_char_lens = batch

                if ds == 'train':
                    global_step += 1

                    # Discriminator training
                    dis_loss = mt_model.discriminator_step([0, 1], [batch[:4], batch[4:]])
                    dis_loss.backward()
                    clip_grad_norm_(disc.parameters(), args.grad_clipping)
                    dis_optimizer.step()

                    # Denoising autoencoding training
                    _, _, l1_auto_loss, _, _ = mt_model.calc_loss(
                        0, 0, l1_tokens, l1_seq_lens, l1_tokens, l1_seq_lens, l1_chars, l1_char_lens, adversarial=False)
                    _, _, l2_auto_loss, _, _ = mt_model.calc_loss(
                        1, 1, l2_tokens, l2_seq_lens, l2_tokens, l2_seq_lens, l2_chars, l2_char_lens, adversarial=False)

                    # Cross-domain training
                    _, _, l1_cd_loss, l1_adv_loss, _ = mt_model.calc_loss(
                        0, 1, l1_tokens, l1_seq_lens, l2_tokens, l2_seq_lens, l1_chars, l1_char_lens, adversarial=True)
                    _, _, l2_cd_loss, l2_adv_loss, _ = mt_model.calc_loss(
                        1, 0, l2_tokens, l2_seq_lens, l1_tokens, l1_seq_lens, l2_chars, l2_char_lens, adversarial=True)

                    loss = (lam_auto * (l1_auto_loss + l2_auto_loss)) + \
                           (lam_cd * (l1_cd_loss + l2_cd_loss)) + \
                           (lam_adv * (l1_adv_loss + l2_adv_loss))

                    loss.backward()
                    clip_grad_norm_(mt_model.parameters(), args.grad_clipping)
                    optimizer.step()
                else:
                    _, pred, loss, ret_dict = mt_model.predict(
                        0, 1, l1_tokens, l1_seq_lens, l2_tokens, l2_seq_lens, l1_chars, l1_char_lens)
                    results.append((ret_dict[mt_model.KEY_SEQ], l2_tokens, l2_seq_lens, l1_tokens))

                epoch_loss.append(loss.item())

            epoch_loss = sum(epoch_loss) / len(epoch_loss)
            logger.info('{} Loss: {:.4f}'.format(ds, epoch_loss))
            logger.info('{} epoch time: {:.4f}'.format(ds, time.time() - epoch_start_t))

            if ds == 'dev' or ds == 'test':
                bleu_score, scores, cand_refs = evaluator.decode_evaluate(results, l2_idx_token, verbosity=0)
                if ds == 'dev' and bleu_score > best_dev_score:
                    sort_scores = sorted(scores.items(), key=lambda x: x[0])
                    for metric, score in sort_scores:
                        logger.info("{0}:\t{1}".format(metric, score))

                    logger.info('New best Bleu_4 score: {:.4f}'.format(bleu_score))
                    best_dev_score = bleu_score
                    best = True
                    logger.info(
                        'Saving the current model to {}'.format(model_file))
                    torch.save(state, model_file)
                if best and ds == 'test':
                    best_test_score = bleu_score

                    with open(result_file, 'w', encoding='utf-8') as rf:
                        for c_r in cand_refs:
                            rf.write('\t'.join([c_r[0][0], c_r[1][0]]) + '\n')

        # learning rate decay
        lr = args.lr * args.decay_rate ** (global_step / args.decay_step)
        for p in optimizer.param_groups:
            p['lr'] = lr
        logger.info('New learning rate: {}'.format(lr))

    logger.info('Best score: {}'.format(best_dev_score))
    logger.info('Best test score: {}'.format(best_test_score))
    logger.info('Model file: {}'.format(model_file))
    logger.info('Result file: {}'.format(result_file))
    if args.log:
        logger.info('Log file: {}'.format(log_file))
        log_writer.close()
except Exception:
    traceback.print_exc()
    if log_writer:
        log_writer.close()
