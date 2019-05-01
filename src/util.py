#!/usr/bin/env python
"""
author: Spencer Whitehead
email: srwhitehead31@gmail.com
"""

import json


def save_results(results, result_fname=None, result_div_fname=None):
    """Evaluate prediction results.

    :param results: A List of which each item is a tuple
        (predictions, gold labels, sequence lengths, tokens) of a batch.
    """
    # b: batch, s: sequence
    outputs = []
    div_outputs = {}
    label_offset = 0
    div_offset = 0
    for result_b in results:
        result_b = result_b.tolist()
        current_div_lens = [0] + [len(r_b) for r_b in result_b]
        current_divs = {
            str(label_offset + i): (div_offset + sum(current_div_lens[:i]), div_offset + sum(current_div_lens[:i + 1]))
            for i in range(1, len(current_div_lens))
        }
        outputs.extend(result_b)
        div_outputs.update(current_divs)
        div_offset += sum(current_div_lens)
        label_offset += len(current_divs)

    if result_fname and result_div_fname:
        with open(result_fname, "w", encoding="utf-8") as rf:
            rf.write("\n".join([" ".join([str(x) for x in num_line]) for ts in outputs for num_line in ts]) + "\n")

        with open(result_div_fname, "w", encoding="utf-8") as rdivf:
            json.dump(div_outputs, rdivf)

    return outputs, div_outputs


if __name__ == "__main__":
    import torch
    import random
    N = 2
    B = 3
    Tmin = 4
    Tmax = 6
    D = 2
    outfname = "test.results.out"
    div_outfname = "test.divs.out"
    results = [torch.randn((B, random.randint(Tmin, Tmax), D)) for _ in range(N)]
    save_results(results, outfname, div_outfname)
