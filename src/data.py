import os
import re
import torch
import linecache
import pickle

import logging
from collections import Counter, defaultdict
from torch.utils.data import Dataset

logger = logging.getLogger()


class Parser(object):
    def __init__(self, **kwargs):
        self.num2zeros = kwargs.pop('num_to_zeros', True)

    def _prepro(self, curr_line):
        prep_tokens = []
        for tok in curr_line:
            if self.num2zeros:
                tok = re.sub('\d', '0', tok)
            prep_tokens.append(tok)
        return prep_tokens

    def parse(self, path: str, **kwargs):
        raise NotImplementedError


class EEGParser(Parser):
    def __init__(self,
                 **kwargs):
        super().__init__(**kwargs)

    def parse(self,
              data_path: str,
              div_path: str = None):
        l1r = open(l1_path, 'r', encoding='utf-8')
        l2r = open(l2_path, 'r', encoding="utf-8")

        for ex in f_iter:
            mt_example = []
            for line in ex:
                line = line.rstrip()
                if line:
                    tokens = self._prepro(line.split())
                    mt_example.append(tokens)
            if len(mt_example) == len(ex):
                yield mt_example

        l1r.close()
        if l2_path:
            l2r.close()


class DynamicEEGDataset(Dataset):
    def __init__(self,
                 l1_path: str,
                 l2_path: str,
                 phase_str: str,
                 parser: Parser,
                 max_seq_len: int = -1,
                 max_char_len: int = -1,
                 cache_dir: str = ".",
                 cache_name: str = "",
                 delete_cache: bool = True,
                 percent_data: float = 1.0):
        self.l1_path = l1_path
        self.l2_path = l2_path
        self.parser = parser
        self.max_seq_len = max_seq_len
        self.max_char_len = max_char_len
        self.data_size = 0
        if not cache_dir:
            cache_dir = "."
        temp_cache_fname = "{0}.MTdata_cache".format(phase_str)
        if cache_name:
            temp_cache_fname = (temp_cache_fname + ".{0}").format(cache_name)
        temp_cache_fname += ".pt"
        self.cache_fname = os.path.join(cache_dir, temp_cache_fname)
        self.cache_delim = "|"
        self.cache_token_delim = ":"
        self._delete_cache = delete_cache
        self._data_percentage = percent_data
        self.load()

    def __getitem__(self,
                    idx: int):
        line = linecache.getline(self.cache_fname, idx + 1)
        try:
            l1_tokens_, l1_chars_, l2_tokens_, l2_chars_ = line.rstrip().split(self.cache_delim)
        except ValueError as e:
            self._delete_cache = False
            print("Error message: {}".format(e))
            print("IDX: {}".format(idx))
            print("Line: {}".format(line))
            raise ValueError("MT dataset indexing error.")
        return (self._numberized_tokens_to_int(l1_tokens_),
                self._numberized_chars_to_int(l1_chars_),
                self._numberized_tokens_to_int(l2_tokens_),
                self._numberized_chars_to_int(l2_chars_))

    def __len__(self):
        return self.data_size

    def __del__(self):
        if os.path.exists(self.cache_fname) and self._delete_cache:
            os.remove(self.cache_fname)

    def save_data(self):
        self._delete_cache = False

    def _numberized_tokens_to_str(self, numb_tok):
        return ' '.join(map(str, numb_tok))

    def _numberized_tokens_to_int(self, numb_tok_str):
        return list(map(int, numb_tok_str.split()))

    def _numberized_chars_to_str(self, numb_chars):
        numb_chars_strs = [' '.join(map(str, char_list)) for char_list in numb_chars]
        return self.cache_token_delim.join(numb_chars_strs)

    def _numberized_chars_to_int(self, numb_chars_str):
        token_list = numb_chars_str.split(self.cache_token_delim)
        return list(list(map(int, tok.split())) for tok in token_list)

    def _single_lang_numberize(self, tokens, token_vocab, char_vocab: dict = None, ignore_case: bool = False):
        if ignore_case:
            tokens_ = [t.lower() for t in tokens]
            tokens_ = [token_vocab[t] if t in token_vocab
                       else C.UNK_INDEX for t in tokens_]
        else:
            tokens_ = [token_vocab[t] if t in token_vocab
                       else C.UNK_INDEX for t in tokens]
        chars = None
        if char_vocab:
            chars = [[char_vocab[c] if c in char_vocab
                      else C.UNK_INDEX for c in t] for t in tokens]
        #     if self.max_seq_len > 0:
        #         chars = chars[:self.max_seq_len]
        # if self.max_seq_len > 0:
        #     tokens_ = tokens_[:self.max_seq_len]
        return tokens_, chars

    def numberize(self,
                  l1_token_vocab: dict,
                  l2_token_vocab: dict,
                  char_vocab: dict = None,
                  ignore_case: bool = False):
        num_examples = 0
        with open(self.cache_fname, 'w', encoding='utf-8') as cf:
            for i, (l1_tokens, l2_tokens) in enumerate(self.parser.parse(self.l1_path, l2_path=self.l2_path)):
                if num_examples == self.data_size:
                    break

                if self.max_seq_len > 0 and (len(l1_tokens) > self.max_seq_len or len(l2_tokens) > self.max_seq_len):
                    continue

                l1_tokens_, l1_chars_ = self._single_lang_numberize(l1_tokens, l1_token_vocab,
                                                                    char_vocab, ignore_case)
                l1_tok_str = self._numberized_tokens_to_str(l1_tokens_)
                l1_char_str = self._numberized_chars_to_str(l1_chars_)

                l2_tokens_, l2_chars_ = self._single_lang_numberize(l2_tokens, l2_token_vocab,
                                                                    char_vocab, ignore_case)
                l2_tok_str = self._numberized_tokens_to_str(l2_tokens_)
                l2_char_str = self._numberized_chars_to_str(l2_chars_)

                cf.write(self.cache_delim.join([l1_tok_str, l1_char_str, l2_tok_str, l2_char_str]) + "\n")
                num_examples += 1

        logger.info("Saved numberized data (filename: {0}) with {1} examples.".format(self.cache_fname,
                                                                                      num_examples))

    def load(self):
        num_examples = 0
        for _ in self.parser.parse(self.l1_path, l2_path=self.l2_path):
            num_examples += 1
        self.data_size = int(num_examples * self._data_percentage)

    def _single_lang_stats(self,
                           tokens,
                           token_ignore_case: bool = False,
                           char_ignore_case: bool = False,
                           ):
        token_counter = Counter()
        char_counter = Counter()
        token_lower = [t.lower() for t in tokens]
        if char_ignore_case:
            for token in token_lower:
                for c in token:
                    char_counter[c] += 1
        else:
            for token in tokens:
                for c in token:
                    char_counter[c] += 1
        if token_ignore_case:
            token_counter.update(token_lower)
        else:
            token_counter.update(tokens)

        return token_counter, char_counter

    def stats(self,
              token_ignore_case: bool = False,
              char_ignore_case: bool = False,
              ):
        l1_token_counter = Counter()
        l1_char_counter = Counter()
        l2_token_counter = Counter()
        l2_char_counter = Counter()

        num_examples = 0
        for i, (l1_tokens, l2_tokens) in enumerate(self.parser.parse(self.l1_path, l2_path=self.l2_path)):
            if num_examples == self.data_size:
                break

            if self.max_seq_len > 0 and (len(l1_tokens) > self.max_seq_len or len(l2_tokens) > self.max_seq_len):
                continue

            curr_l1_tokens, curr_l1_chars = self._single_lang_stats(l1_tokens,
                                                                    token_ignore_case,
                                                                    char_ignore_case)
            l1_token_counter += curr_l1_tokens
            l1_char_counter += curr_l1_chars

            curr_l2_tokens, curr_l2_chars = self._single_lang_stats(l2_tokens,
                                                                    token_ignore_case,
                                                                    char_ignore_case)
            l2_token_counter += curr_l2_tokens
            l2_char_counter += curr_l2_chars
            num_examples += 1

        if num_examples < self.data_size:
            self.data_size = num_examples

        return l1_token_counter, l1_char_counter, l2_token_counter, l2_char_counter


class BatchProcessor(object):

    def process(self, batch):
        assert NotImplementedError


class EEGProcessor(BatchProcessor):
    def __init__(self,
                 sort: bool = False,
                 gpu: bool = False,
                 min_char_len: int = 4):
        self.sort = sort
        self.gpu = gpu

    def _single_lang_process(self, lang_batch_tokens, lang_batch_chars, padding_idx):
        seq_lens = [len(x) for x in lang_batch_tokens]
        max_seq_len = max(seq_lens)

        char_lens = []
        for bn in range(len(lang_batch_tokens)):
            seq_char_lens = [len(x) for x in lang_batch_chars[bn]] + \
                            [padding_idx] * (max_seq_len - len(lang_batch_tokens[bn]))
            char_lens.extend(seq_char_lens)
        max_char_len = max(max(char_lens), self.min_char_len)

        # Padding
        batch_tokens = []
        batch_chars = []
        for bn in range(len(lang_batch_tokens)):
            chars = lang_batch_chars[bn]
            tokens = lang_batch_tokens[bn]
            batch_tokens.append(tokens + [padding_idx] * (max_seq_len - len(tokens)))
            batch_chars.extend(
                [x + [0] * (max_char_len - len(x)) for x in chars]
                # + [[0] * max_char_len] * (max_seq_len - len(tokens))
                + [[0] * max_char_len for _ in range(max_seq_len - len(tokens))]
            )
            # batch_chars.extend([x + [0] * (max_char_len - len(x)) for x in chars]
            #                    + [[0] * max_char_len] * (max_seq_len - len(tokens)))

        batch_tokens = torch.LongTensor(batch_tokens)
        batch_chars = torch.LongTensor(batch_chars)
        seq_lens = torch.LongTensor(seq_lens)
        char_lens = torch.LongTensor(char_lens)

        if self.gpu:
            batch_tokens = batch_tokens.cuda()
            batch_chars = batch_chars.cuda()
            seq_lens = seq_lens.cuda()
            char_lens = char_lens.cuda()

        return batch_tokens, batch_chars, seq_lens, char_lens

    def process(self, batch: list):
        padding_idx = self.padding_idx
        if self.sort:
            batch.sort(key=lambda x: len(x[0]), reverse=True)

        l1_tokens, l1_chars, l2_tokens, l2_chars = zip(*batch)

        l1_batch_tokens, l1_batch_chars, \
            l1_seq_lens, l1_char_lens = self._single_lang_process(l1_tokens,
                                                                  l1_chars,
                                                                  padding_idx)

        l2_batch_tokens, l2_batch_chars, \
            l2_seq_lens, l2_char_lens = self._single_lang_process(l2_tokens,
                                                                  l2_chars,
                                                                  padding_idx)

        return (l1_batch_tokens, l1_seq_lens, l1_batch_chars, l1_char_lens,
                l2_batch_tokens, l2_seq_lens, l2_batch_chars, l2_char_lens)
