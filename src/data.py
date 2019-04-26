import os
import torch
import random
import linecache
import pickle
import json

import logging
from torch.utils.data import Dataset

logger = logging.getLogger()


class Parser(object):
    def __init__(self, **kwargs):
        pass

    def parse(self, path: str, **kwargs):
        raise NotImplementedError


class EEGParser(Parser):
    def __init__(self,
                 **kwargs):
        super().__init__(**kwargs)

    def parse(self,
              data_path: str,
              div_path: str = None):
        f_iter = open(data_path, 'r', encoding="utf-8")
        if div_path.endswith("json"):
            with open(div_path, "r", encoding="utf-8") as divf:
                divisions = json.load(divf)
        else:
            with open(div_path, "rb") as divf:
                divisions = pickle.load(divf)

        for div_id, div_range in divisions.items():
            eeg_example = []
            div_len = div_range[1] - div_range[0]
            for _ in range(div_len):
                line = f_iter.readline()
                line = line.rstrip()
                if line:
                    temp = [float(x) for x in line.split()]
                    eeg_example.append(temp)
            yield eeg_example

        f_iter.close()


class DynamicEEGDataset(Dataset):
    def __init__(self,
                 data_path: str,
                 div_path: str,
                 phase_str: str,
                 parser: Parser,
                 min_seq_len: int = 10,
                 max_seq_len: int = -1,
                 rando_divs: bool = False,
                 cache_dir: str = ".",
                 cache_name: str = "",
                 delete_cache: bool = True):
        self.data_path = data_path
        self.div_path = div_path
        self.parser = parser
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.rando_divs = rando_divs
        self.data_size = 0
        if not cache_dir:
            cache_dir = "."
        temp_cache_fname = "{0}.EEGdata_cache".format(phase_str)
        if cache_name:
            temp_cache_fname = (temp_cache_fname + ".{0}").format(cache_name)
        temp_cache_fname += ".pt"
        self.cache_fname = os.path.join(cache_dir, temp_cache_fname)
        self.cache_step_delim= "|"
        self._delete_cache = delete_cache
        self.load()

    def __getitem__(self,
                    idx: int):
        line = linecache.getline(self.cache_fname, idx + 1)
        try:
            ts_str_list = line.rstrip().split(self.cache_step_delim)
        except ValueError as e:
            self._delete_cache = False
            print("Error message: {}".format(e))
            print("IDX: {}".format(idx))
            print("Line: {}".format(line))
            raise ValueError("Dataset indexing error.")
        return self._ts_to_float(ts_str_list)

    def __len__(self):
        return self.data_size

    def __del__(self):
        if os.path.exists(self.cache_fname) and self._delete_cache:
            os.remove(self.cache_fname)

    def _get_data_subset_idxs(self, n, rando=False):
        if self.max_seq_len < 0:
            yield n
        else:
            max_len = min(n, self.max_seq_len)
            while n > self.min_seq_len:
                if rando:
                    r = random.randint(self.min_seq_len, max_len)
                else:
                    r = self.min_seq_len
                yield r
                n -= r
            yield n

    def get_data_subsets(self):
        for ts in self.parser.parse(self.data_path, div_path=self.div_path):
            ts_n = len(ts)
            L = 0
            for k in self._get_data_subset_idxs(ts_n, self.rando_divs):
                yield ts[L:L+k]
                L += k

    def retain_cache(self):
        self._delete_cache = False

    def _ts_to_str(self, ts_step_list):
        ts_step_strs = [' '.join(map(str, ts_step)) for ts_step in ts_step_list]
        return self.cache_step_delim.join(ts_step_strs)

    def _ts_to_float(self, ts_str_list):
        return list(list(map(float, ts_step.split())) for ts_step in ts_str_list)

    def load(self):
        num_examples = 0
        with open(self.cache_fname, 'w', encoding='utf-8') as cf:
            for ts_sub in self.get_data_subsets():
                cf.write(self._ts_to_str(ts_sub) + "\n")
                num_examples += 1
        self.data_size = num_examples


class BatchProcessor(object):

    def process(self, batch):
        assert NotImplementedError


class EEGProcessor(BatchProcessor):
    def __init__(self,
                 sort: bool = True,
                 gpu: bool = False,
                 padval: float = 0.0
                 ):
        self.sort = sort
        self.gpu = gpu
        self.padval = padval

    def process(self, batch):
        if self.sort:
            batch.sort(key=lambda x: len(x), reverse=True)

        feature_size = len(batch[0][0])
        max_len = max([len(batch[i]) for i in range(len(batch))])
        pad_batch = [batch[i] + [[self.padval] * feature_size] * (max_len - len(batch[i])) for i in range(len(batch))]

        pad_batch = torch.FloatTensor(pad_batch)

        if self.gpu:
            pad_batch = pad_batch.cuda()

        return pad_batch


def split_intervals(timeseries, divisions=None, feature_size=128):
    assert feature_size > 0
    if divisions is None:
        divisions = {"0": (0, len(timeseries))}

    sort_divisions = sorted(divisions.items(), key=lambda x: x[1][0])
    split_timeseries = []
    split_divisions = {}
    split_start = 0
    for div, (start, end) in sort_divisions:
        n_splits = int((end - start) / feature_size)

        split_divisions[div] = (split_start, split_start + n_splits)
        split_start += n_splits

        for i in range(n_splits):
            j = feature_size * i
            k = j + feature_size
            split_timeseries.append(list(timeseries[j:k]))

    return split_timeseries, split_divisions


def preprocess_data(data_fname="", data_name="seizure", output_dir="./data", feature_size=128):
    outfname = os.path.join(output_dir, "{}.timeseries.txt".format(data_name))
    divisions_outfname = os.path.join(output_dir, "{}.divisions.json".format(data_name))

    with open(data_fname, "rb") as sf:
        ts_data = pickle.load(sf)

    if len(ts_data) == 2:
        division_dict = ts_data[1]
    else:
        division_dict = None

    split_ts_data, split_divisions = split_intervals(ts_data[0], division_dict, feature_size=feature_size)

    with open(outfname, "w", encoding="utf-8") as sdatf:
        for ts_step_data in split_ts_data:
            sdatf.write(" ".join([str(x) for x in ts_step_data]) + "\n")

    with open(divisions_outfname, "w", encoding="utf-8") as sdf:
        json.dump(split_divisions, sdf)


def test_dataset():
    seizure_fname = "./data/seizure_data.pkl"
    processed_seizure_fname = "./data/seizure.timeseries.txt"
    processed_seizure_divisions_fname = "./data/seizure.divisions.json"

    normal_fname = "./data/normal.pkl"

    preprocess_data(seizure_fname, data_name="seizure", feature_size=2)
    preprocess_data(normal_fname, data_name="normal", feature_size=2)

    p = EEGParser()
    dataset = DynamicEEGDataset(
        processed_seizure_fname,
        processed_seizure_divisions_fname,
        "train",
        p,
        min_seq_len=4,
        max_seq_len=6,
        rando_divs=True,
        cache_dir="./data",
        cache_name="TEST",
        delete_cache=True
    )

    processor = EEGProcessor(sort=True, gpu=False, padval=0.)

    for batch in DataLoader(
            dataset,
            batch_size=3,
            shuffle=False,
            drop_last=False,
            collate_fn=processor.process
    ):
        print(batch)
        print(batch.size())
        break


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    test_dataset()
