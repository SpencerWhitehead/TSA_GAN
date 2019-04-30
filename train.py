import os
import time
import logging
import traceback
import random

import numpy as np

import torch
from torch import optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from argparse import ArgumentParser

from src.data import EEGParser, DynamicEEGDataset, EEGProcessor
from src.model import FCDiscriminator, RNNDiscriminator, GeneratorRNN, GANModel
from src.util import save_results


RAND_SEED = 742019
torch.manual_seed(RAND_SEED)
random.seed(RAND_SEED)
np.random.seed(RAND_SEED)

timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()

argparser = ArgumentParser()


argparser.add_argument("--train", nargs=2, help="Path to the training set files. " +
                                                  "Must be: <timeseries_fname> <divisions_fname>")
argparser.add_argument("--dev", nargs=2, help="Path to the dev set files. " +
                                                  "Must be: <timeseries_fname> <divisions_fname>")
argparser.add_argument("--test", nargs=2, help="Path to the test set files. " +
                                                  "Must be: <timeseries_fname> <divisions_fname>")
argparser.add_argument("--data_cache_dir", default="./data", help="Path to the directory for cached data.")
argparser.add_argument("--log", help="Path to directory where the log will be saved.")
argparser.add_argument("--model", help="Path to directory where the model will be saved.")
argparser.add_argument("--results", help="Path to directory where the results will be saved.")
argparser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
argparser.add_argument("--max_epoch", default=100, type=int)
argparser.add_argument("--min_seq_len", default=50, type=int, help="Min sequence length in data.")
argparser.add_argument("--max_seq_len", default=50, type=int, help="Max sequence length in data.")
argparser.add_argument("--rand_seq_len", action="store_true", help="Use random sequence lengths?")
argparser.add_argument("--input_dim", type=int, default=128,
                       help="Input dimension.")
argparser.add_argument("--noise_size", default=64, type=int,
                       help="Input noise vector size.")
argparser.add_argument("--cond_size", default=16, type=int,
                       help="Input conditional vector size.")
argparser.add_argument("--n_gen_layers", type=int, default=1,
                       help="Number of generator layers.")
argparser.add_argument("--gen_hidden_size", default=256, type=int,
                       help="Generator hidden state size.")
argparser.add_argument("--gen_dropout_p", default=0., type=float,
                       help="Generator output dropout probability.")
argparser.add_argument("--disc_model_type", choices=["fc", "rnn"],
                       help="Type of discriminator model.", default="fc")
argparser.add_argument("--n_disc_layers", type=int, default=3,
                       help="Number of discriminator layers.")
argparser.add_argument("--disc_hidden_size", default=256, type=int,
                       help="Discriminator hidden state size.")
argparser.add_argument("--disc_dropout_p", default=0.8, type=float,
                       help="Discriminator output dropout probability.")
argparser.add_argument("--lr", default=0.005, type=float,
                       help="Learning rate.")
argparser.add_argument("--disc_lr", default=0.005, type=float,
                       help="Discriminator learning rate.")
argparser.add_argument("--lambda_gen", default=1.0, type=float,
                       help="Generation loss scaling factor.")
argparser.add_argument("--lambda_adv", default=1.0, type=float,
                       help="Adversarial loss scaling factor.")
argparser.add_argument("--momentum", default=.9, type=float)
argparser.add_argument("--decay_rate", default=.9, type=float)
argparser.add_argument("--decay_step", default=10000, type=int)
argparser.add_argument("--grad_clipping", default=5, type=float)
argparser.add_argument("--gpu", action="store_true")
argparser.add_argument("--device", default=0, type=int)
argparser.add_argument("--thread", default=5, type=int)
argparser.add_argument("--delete_cache", action="store_true")
argparser.add_argument("--use_prediction_loss", action="store_true",
                       help="Whether or not to use MSE loss on predicted time series.")
argparser.add_argument("--use_1hot_cond", action="store_true",
                       help="Whether or not to use 1-hot conditional vector.")

args = argparser.parse_args()

use_gpu = args.gpu and torch.cuda.is_available()
if use_gpu:
    torch.cuda.set_device(args.device)

# Model file
model_dir = args.model
assert model_dir and os.path.isdir(model_dir), "Model output dir is required"
model_file = os.path.join(model_dir, "model.{}.mdl".format(timestamp))

# Results file
results_dir = args.results
assert model_dir and os.path.isdir(results_dir), "Result dir is required"
result_file = os.path.join(results_dir, "results.{}.timeseries.txt".format(timestamp))
result_div_file = os.path.join(results_dir, "results.{}.divisions.json".format(timestamp))

# Logging file
log_writer = None
if args.log:
    log_file = os.path.join(args.log, "log.{}.txt".format(timestamp))
    log_writer = open(log_file, "a", encoding="utf-8")
    logger.addHandler(logging.FileHandler(log_file, encoding="utf-8"))
logger.info("----------")
logger.info("Parameters:")
for arg in vars(args):
    logger.info("{}: {}".format(arg, getattr(args, arg)))
logger.info("----------")

# Data file
logger.info("Loading data sets")

parser = EEGParser()

train_set = DynamicEEGDataset(
    args.train[0],
    args.train[1],
    "train",
    parser,
    min_seq_len=args.min_seq_len,
    max_seq_len=args.max_seq_len,
    rando_ts_len=args.rand_seq_len,
    cache_dir=args.data_cache_dir,
    cache_name=timestamp,
    delete_cache=args.delete_cache
)

dev_set = DynamicEEGDataset(
    args.dev[0],
    args.dev[1],
    "dev",
    parser,
    min_seq_len=args.min_seq_len,
    max_seq_len=args.max_seq_len,
    rando_ts_len=args.rand_seq_len,
    cache_dir=args.data_cache_dir,
    cache_name=timestamp,
    delete_cache=args.delete_cache
)

test_set = DynamicEEGDataset(
    args.test[0],
    args.test[1],
    "test",
    parser,
    min_seq_len=args.min_seq_len,
    max_seq_len=args.max_seq_len,
    rando_ts_len=args.rand_seq_len,
    cache_dir=args.data_cache_dir,
    cache_name=timestamp,
    delete_cache=args.delete_cache
)

processor = EEGProcessor(sort=True, gpu=use_gpu)

datasets = {"train": train_set, "dev": dev_set, "test": test_set}

gen_net = GeneratorRNN(
    input_size=args.input_dim,
    hidden_size=args.gen_hidden_size,
    output_size=args.input_dim,
    n_layers=args.n_gen_layers,
    rnn_cell_type="ylstm",
    dropout_p=args.gen_dropout_p,
    noise_size=args.noise_size,
    cond_size=args.cond_size
)

disc_model_type = FCDiscriminator if args.disc_model_type == "fc" else RNNDiscriminator
disc = disc_model_type(
    input_size=gen_net.output_size,
    hidden_size=args.disc_hidden_size,
    n_labels=2,
    n_layers=args.n_disc_layers,
    dropout_p=args.disc_dropout_p,
    rnn_cell_type="ylstm",
)

gan_model = GANModel(
    generator=gen_net,
    discriminator=disc,
    max_len=min(30, args.max_seq_len),
    min_val=train_set.min_val,
    max_val=train_set.max_val,
    one_hot_cond=args.use_1hot_cond
)

if use_gpu:
    gan_model.cuda()
torch.set_num_threads(args.thread)

logger.debug(gan_model)

lam_gen = args.lambda_gen
lam_adv = args.lambda_adv

# Initialize optimizers
no_disc_params = [p for m, p in gan_model.named_parameters()
                  if p.requires_grad and "discriminator" not in m]
optimizer = optim.SGD(no_disc_params,
                      lr=args.lr, momentum=args.momentum)
disc_optimizer = optim.SGD(filter(lambda p: p.requires_grad, disc.parameters()),
                           lr=args.disc_lr, momentum=args.momentum)

train_args = vars(args)
state = {
    "model": {
        "discriminator": disc.state_dict(),
        "generator": gen_net.state_dict(),
        "gan_model": gan_model.state_dict()
    },
    "args": train_args,
}
try:
    global_step = 0
    best_dev_loss = best_test_loss = 1000000.0

    for epoch in range(args.max_epoch):
        logger.info("Epoch {}: Training".format(epoch))

        best = False
        for ds in ["train", "dev", "test"]:
            dataset = datasets[ds]
            epoch_loss = []
            results = []

            epoch_start_t = time.time()
            for batch in DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=ds == "train",
                drop_last=ds == "train",
                collate_fn=processor.process
            ):
                batch_ts_data, batch_ts_lens = batch

                if ds == "train":
                    global_step += 1

                    # Discriminator training step
                    disc_optimizer.zero_grad()
                    disc_loss = gan_model.discriminator_step(batch_ts_data, batch_ts_lens)
                    disc_loss.backward()
                    clip_grad_norm_(disc.parameters(), args.grad_clipping)
                    disc_optimizer.step()

                    # Generator training step
                    optimizer.zero_grad()
                    _, gen_loss, adv_loss = gan_model.calc_loss(
                        batch_ts_data,
                        batch_ts_lens,
                        use_pred_loss=args.use_prediction_loss
                    )

                    loss = lam_adv * adv_loss
                    if gen_loss:
                        loss += lam_gen * gen_loss

                    loss.backward()
                    clip_grad_norm_(
                        [p for m, p in gan_model.named_parameters()
                         if p.requires_grad and "discriminator" not in m],
                        args.grad_clipping)
                    optimizer.step()
                else:
                    preds, gen_loss, adv_loss = gan_model.predict(
                        batch_ts_data,
                        batch_ts_lens,
                        teacher_forcing_ratio=(0. if ds != "dev" else 1.),
                        use_pred_loss=args.use_prediction_loss
                    )

                    loss = adv_loss
                    if gen_loss:
                        loss += gen_loss

                    if ds == "test":
                        results.append(
                            preds.cpu().detach()
                        )

                epoch_loss.append(loss.cpu().detach().item())

            epoch_loss = sum(epoch_loss) / len(epoch_loss)
            logger.info("{} Loss: {}".format(ds, epoch_loss))
            logger.info("{} epoch time: {}".format(ds, time.time() - epoch_start_t))

            if ds == "dev" or ds == "test":
                if ds == "dev" and epoch_loss < best_dev_loss:
                    logger.info("New best dev loss: {}".format(epoch_loss))
                    best_dev_loss = epoch_loss
                    best = True
                    logger.info(
                        "Saving the current model to {}".format(model_file))
                    torch.save(state, model_file)

                if best and ds == "test":
                    best_test_loss = epoch_loss
                    save_results(results, result_file, result_div_file)

        # Generator learning rate decay
        lr = args.lr * args.decay_rate ** (global_step / args.decay_step)
        for p in optimizer.param_groups:
            p["lr"] = lr
        logger.info("New generator learning rate: {}".format(lr))

        # Discriminator learning rate decay
        disc_lr = args.disc_lr * args.decay_rate ** (global_step / args.decay_step)
        for p in disc_optimizer.param_groups:
            p["lr"] = lr
        logger.info("New discriminator learning rate: {}".format(lr))

    logger.info("Best dev loss: {}".format(best_dev_loss))
    logger.info("Best test loss: {}".format(best_test_loss))
    logger.info("Model file: {}".format(model_file))
    logger.info("Result timeseries file: {}".format(result_file))
    logger.info("Result divsions file: {}".format(result_div_file))
    if args.log:
        logger.info("Log file: {}".format(log_file))
        log_writer.close()
except Exception:
    traceback.print_exc()
    if log_writer:
        log_writer.close()
