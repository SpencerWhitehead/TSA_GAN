import sys
import json
import pickle
import logging
import random
import collections
import numpy as np
import constant as C
import conlleval

sys.path.append('pycocoevalcap/bleu')
from pycocoevalcap.bleu.bleu import Bleu


def get_logger(name, level=C.LOGGING_LEVEL, log_file=None):
    """Get a logger by name.

    :param name: Logger name (usu. __name__).
    :param level: Logging level (default=logging.INFO).
    """
    logger = logging.getLogger(name)
    logger.addHandler(logging.StreamHandler())
    if log_file:
        logger.addHandler(logging.FileHandler(log_file, encoding='utf-8'))
    logger.setLevel(level)
    return logger


def read_scores(logfname, epoch_range=None):
    lpar = LogParser(True)
    all_scores = {}
    all_metrics = set()
    longest_name = ""
    for epoch_scores in lpar.parse(logfname):
        for ep_n, task, ds, metric_scr in epoch_scores:
            if epoch_range and (ep_n < epoch_range[0] or ep_n > epoch_range[1]):
                continue

            comp_name = "_".join([task, ds])
            if len(comp_name) > len(longest_name):
                longest_name = comp_name
            comp_data = all_scores.get(comp_name, {})
            for metric, value in metric_scr:
                all_metrics.add(metric)
                scr_data = comp_data.get(metric, [])
                scr_data.append((ep_n, value))
                comp_data[metric] = scr_data
            all_scores[comp_name] = comp_data

    for tds, tds_scores in all_scores.items():
        for metric, scores in tds_scores.items():
            temp_metric_scores = list(zip(*sorted(scores, key=lambda x: x[0])))[1]
            all_scores[tds][metric] = temp_metric_scores
    return all_scores, sorted(list(all_metrics)), longest_name


def correlation_from_log(logfname, epoch_range=None):
    scores, metrics, longest_name = read_scores(logfname, epoch_range)
    ordered_tds = sorted(scores.keys())
    for metric in metrics:
        met_data = []
        for tds in ordered_tds:
            met_data.append(scores[tds][metric])
        corrs = np.corrcoef(met_data).tolist()

        print(metric)
        row_format = "{:>15}" * (len(ordered_tds) + 1)
        print(row_format.format("", *ordered_tds))
        for tds, tds_corr in zip(ordered_tds, corrs):
            print(row_format.format(tds, *["{:.5f}".format(c) for c in tds_corr]))


class Evaluator(object):
    def __init__(self, evaluation_metric="Bleu_4"):
        self.evaluation_metric = evaluation_metric

    def decode_evaluate(self, results, idx_token, idx_label, **kwargs):
        raise NotImplementedError


class SeqLabelEvaluator(Evaluator):
    def __init__(self):
        super(SeqLabelEvaluator, self).__init__(evaluation_metric="fscore")

    def decode_evaluate(self, results, idx_token, idx_label, writer=None):
        """Evaluate prediction results.

        :param results: A List of which each item is a tuple
            (predictions, gold labels, sequence lengths, tokens) of a batch.
        :param idx_token: Index to token dictionary.
        :param idx_label: Index to label dictionary.
        :param writer: An object (file object) with a write() function. Extra output.
        :return: F-score, precision, and recall.
        """
        # b: batch, s: sequence
        outputs = []
        for preds_b, golds_b, len_b, tokens_b in results:
            for preds_s, golds_s, len_s, tokens_s in zip(preds_b, golds_b, len_b, tokens_b):
                l = int(len_s.item())
                preds_s = preds_s.data.tolist()[:l]
                golds_s = golds_s.data.tolist()[:l]
                tokens_s = tokens_s.data.tolist()[:l]
                for p, g, t in zip(preds_s, golds_s, tokens_s):
                    token = idx_token.get(t, C.UNK_INDEX)
                    outputs.append('{} {} {}'.format(
                        token, idx_label.get(g, 0), idx_label.get(p, 0)))
                outputs.append('')
        counts = conlleval.evaluate(outputs)
        overall, by_type = conlleval.metrics(counts)
        conlleval.report(counts)
        if writer:
            conlleval.report(counts, out=writer)
            writer.flush()
        return overall.fscore, {"precision": overall.prec, "recall": overall.rec, "fscore": overall.fscore}, outputs


class MTEvaluator(Evaluator):
    def __init__(self, evaluation_metric="Bleu_4"):
        super(MTEvaluator, self).__init__(evaluation_metric)

        self.bad_tokens = list(zip(*filter(lambda x: C.UNK not in x, C.TOKEN_PADS)))[0]
        self.scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"])
        ]

    def convert(self, data):
        if isinstance(data, str):
            return data.encode('utf-8')
        elif isinstance(data, collections.Mapping):
            return dict(map(self.convert, data.items()))
        elif isinstance(data, collections.Iterable):
            return type(data)(map(self.convert, data))
        else:
            return data

    def format_bleu_scores(self, score_dict):
        sort_scores = sorted(score_dict.items(), key=lambda x: x[0])
        score_val_str = []
        for metric, score in sort_scores:
            score_val_str.append("{}: {:.3f}".format(metric, score * 100.0))
            # writer.write("{0}:\t{1}\n".format(metric, score))
        return "\t" + "; ".join(score_val_str) + "\n"

    def to_words(self, seqs, idx2token_map, offset=0):
        result = {}
        for j in range(seqs.size(0)):
            example = " ".join([idx2token_map[seqs[j, k].item()] for k in range(seqs[j].size(0))
                                if idx2token_map[seqs[j, k].item()] not in self.bad_tokens])
            result[j + offset] = [example]
        return result

    def score(self, ref, hypo):
        final_scores = {}
        for scorer, method in self.scorers:
            score, scores = scorer.compute_score(ref, hypo, verbosity=0)
            if type(score) == list:
                for m, s in zip(method, score):
                    final_scores[m] = s
            else:
                final_scores[method] = score

        return final_scores

    def evaluate(self, live=True, **kwargs):
        verbosity = kwargs.pop("verbosity", 1)
        if live:
            temp_ref = kwargs.pop('ref', {})
            cand = kwargs.pop('cand', {})
        else:
            reference_path = kwargs.pop('ref', '')
            candidate_path = kwargs.pop('cand', '')

            # load caption data
            with open(reference_path, 'rb') as f:
                temp_ref = pickle.load(f)
            with open(candidate_path, 'rb') as f:
                cand = pickle.load(f)

        # make dictionary
        hypo = {}
        ref = {}
        i = 0
        for vid, caption in cand.items():
            hypo[i] = caption
            ref[i] = temp_ref[vid]
            i += 1

        # compute scores
        final_scores = self.score(ref, hypo)

        if verbosity > 0:
            sys.stdout.write(self.format_bleu_scores(final_scores))

        return final_scores

    def decode_evaluate(self, results, idx_token, idx_label=None, writer=None, verbosity=1):
        cand = {}
        ref = {}
        for preds_b, golds_b, len_b, tokens_b in results:
            cand_b = self.to_words(preds_b, idx_token, len(cand))
            cand.update(cand_b)

            ref_b = self.to_words(golds_b, idx_token, len(ref))
            ref.update(ref_b)

        curr_scores = self.evaluate(live=True, cand=cand, ref=ref, verbosity=verbosity)
        pred_gt_pairs = ["\t".join([cand[i][0], ref[i][0]]) for i in range(len(cand))]

        if writer:
            writer.write(self.format_bleu_scores(curr_scores))
            writer.flush()

        return curr_scores[self.evaluation_metric], curr_scores, pred_gt_pairs


class TaskIterator(object):
    def __init__(self, mixing_ratios):
        self.mixing_ratios = mixing_ratios

    def gen_tasks(self):
        total_tasks_ = sum(self.mixing_ratios.values())
        rem_mixes = dict(zip(self.mixing_ratios.keys(), [0 for _ in range(len(self.mixing_ratios))]))
        for i in range(total_tasks_):
            avail_tasks = [ta for ta, v in rem_mixes.items() if v < self.mixing_ratios[ta]]
            if len(avail_tasks):
                task_i = random.choice(avail_tasks)
                rem_mixes[task_i] += 1
                yield task_i


class Config(dict):

    def __init__(self, *args, **kwargs):
        super(Config, self).__init__(*args, **kwargs)
        __getattr__ = dict.__getitem__

        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    if isinstance(v, dict):
                        v = Config(v)
                    if isinstance(v, list):
                        v = [Config(x) if isinstance(x, dict) else x for x in v]
                    self[k] = v
        if kwargs:
            for k, v in kwargs.items():
                self[k] = v

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(Config, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(Config, self).__delitem__(key)
        del self.__dict__[key]

    def set_dict(self, dict_obj):
        for k, v in dict_obj.items():
            if isinstance(v, dict):
                v = Config(v)
            self[k] = v

    def update(self, dict_obj):
        for k, v in dict_obj.items():
            if isinstance(v, dict):
                v = Config(v)
            if isinstance(v, list):
                v = [Config(x) if isinstance(x, dict) else x for x in v]
            self[k] = v

    def clone(self):
        return Config(dict(self))

    @staticmethod
    def read(path):
        """Read configuration from JSON format file.

        :param path: Path to the configuration file.
        :return: Config object.
        """
        # logger.info('loading configuration from {}'.format(path))
        json_obj = json.load(open(path, 'r', encoding='utf-8'))
        return Config(json_obj)

    def update_value(self, keys, value):
        keys = keys.split('.')
        assert len(keys) > 0

        tgt = self
        for k in keys[:-1]:
            try:
                tgt = tgt[int(k)]
            except Exception:
                tgt = tgt[k]
        tgt[keys[-1]] = value


class LogParser(object):

    def __init__(self,
                 skip_comment: bool = False):
        self.skip_comment = skip_comment

    def parse(self,
              path: str):
        skip_comment = self.skip_comment

        with open(path, 'r', encoding='utf-8') as r:
            curr_doc = []
            curr_epoch = 1
            next_epoch = 0
            curr_task = ""
            curr_dataset = ""
            for line in r:
                line = line.rstrip()
                if skip_comment and line.startswith('#'):
                    continue

                if line.startswith("Epoch"):
                    next_epoch = int(line.split()[1].rstrip(':'))

                if curr_epoch == next_epoch and line:
                    if line.startswith("task:"):
                        _, curr_task, _, curr_dataset = line.split()
                    elif line.startswith("Acc:"):
                        temp = line.split(';')[1:]
                        curr_scores = []
                        for score_str in temp:
                            metric, value = score_str.split()
                            metric = metric.rstrip(':')
                            value = float(value.strip().rstrip('%'))
                            curr_scores.append((metric, value))
                        curr_doc.append((curr_epoch, curr_task, curr_dataset, curr_scores))
                elif curr_epoch != next_epoch and curr_doc:
                    curr_epoch = next_epoch
                    yield curr_doc
                    curr_doc = []
            if curr_doc:
                yield curr_doc


if __name__ == "__main__":
    exp_fnames = [
        "cl_ct_noise_advenc_engtrain_esptest.out",
        "cl_ct_noise_engtrain_esptest_KBP.out",
        "cl_ct_noise_engtrain_esptest.out",
        "cl_st_engtrain_esptest_KBP.out",
        "cl_st_engtrain_esptest.out"
    ]
    log_fnames = [
        "/data/m1/whites5/mlmt/logs/log.20190215_002836.txt",
        "/data/m1/whites5/mlmt/logs/log.20190215_133301.txt",
        "/data/m1/whites5/mlmt/logs/log.20190215_002338.txt",
        "/data/m1/whites5/mlmt/logs/log.20190215_133218.txt",
        "/data/m1/whites5/mlmt/logs/log.20190214_234917.txt"
    ]
    # epoch_range = (20, 50)
    epoch_range = None

    exp_fname_map = list(zip(exp_fnames, log_fnames))
    for exp_fname, log_fname in exp_fname_map:
        print("=" * 50)
        print(exp_fname, "\n")
        correlation_from_log(log_fname, epoch_range)
        print("=" * 50, "\n")
