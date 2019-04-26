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

from torch.utils.data import DataLoade
from src.data import EEGParser, DynamicEEGDataset, EEGProcessor
from src.model import Discriminator, DecoderRNN, GANModel

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
argparser.add_argument('--batch_size', default=32, type=int, help='Batch size')
argparser.add_argument('--max_epoch', default=100, type=int)
argparser.add_argument('--input_dim', type=int, default=128,
                       help='Input dimension')
argparser.add_argument('--hidden_size', default=256, type=int,
                       help='LSTM hidden state size')
argparser.add_argument('--lstm_forget_bias', default=0, type=float,
                       help='LSTM forget bias')
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

parser = EEGParser()

train_set = DynamicEEGDataset(
    processed_seizure_fname,
    processed_seizure_divisions_fname,
    "train",
    parser,
    min_seq_len=4,
    max_seq_len=6,
    rando_divs=True,
    cache_dir="./data",
    cache_name="TEST",
    delete_cache=True
)

train_set = DynamicMTDataset(args.train[0], args.train[1], "train", parser=parser,
                             max_seq_len=25, max_char_len=25)
dev_set =  DynamicMTDataset(args.dev[0], args.dev[1], "dev", parser=parser,
                            max_seq_len=25, max_char_len=25)
test_set =  DynamicMTDataset(args.test[0], args.test[1], "test", parser=parser,
                             max_seq_len=25, max_char_len=25)
datasets = {'train': train_set, 'dev': dev_set, 'test': test_set}

dec = DecoderRNN(
    input_size=args.input_dim,
    hidden_size=args.hidden_size,
    output_size=args.input_dim,
    max_len=100,
    n_layers=1,
    rnn_cell_type="ylstm"
)

disc_labels = {"real": 1, "fake": 0}
disc = Discriminator(
    n_labels=len(disc_labels),
    input_dim=dec.output_size,
    n_layers=1,
    hidden_size=dec.output_size,
    dropout_p=0.
)


gan_model = GANModel(decoder=dec, max_len=100, disc=discriminator, disc_labels=[0, 1])

if use_gpu:
    gan_model.cuda()
torch.set_num_threads(args.thread)

logger.debug(gan_model)

lam_auto = args.lambda_auto
lam_cd = args.lambda_cd
lam_adv = args.lambda_adv

# Initialize optimizers
no_disc_params = [p for m, p in gan_model.named_parameters()
                  if p.requires_grad and 'discriminator' not in m]
optimizer = optim.SGD(no_disc_params,
                      lr=args.lr, momentum=args.momentum)
dis_optimizer = optim.SGD(filter(lambda p: p.requires_grad, disc.parameters()),
                          lr=args.dis_lr, momentum=args.momentum)
processor = EEGProcessor(sort=True, gpu=False, padval=0.)

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
