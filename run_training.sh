#!/bin/bash
datatrain=("./data/seizure.timeseries.txt" "./data/seizure.divisions.json")
datadev=("./data/seizure.timeseries.txt" "./data/seizure.divisions.json")
datatest=("./data/seizure.timeseries.txt" "./data/seizure.divisions.json")
python -u train.py \
--train ${datatrain[@]} \
--dev ${datadev[@]} \
--test ${datatest[@]} \
--data_cache_dir ./data \
--log ./logs \
--model ./model_ckpts \
--results ./results \
--batch_size 32 \
--max_epoch 2 \
--min_seq_len 50 \
--max_seq_len 50 \
--input_dim 128 \
--noise_size 0 \
--cond_size 0 \
--n_gen_layers 1 \
--gen_hidden_size 256 \
--gen_dropout_p 0.0 \
--disc_model_type fc \
--n_disc_layers 1 \
--disc_hidden_size 256 \
--disc_dropout_p 0.0 \
--lr 0.005 \
--disc_lr 0.005 \
--lambda_gen 1.0 \
--lambda_adv 1.0 \
--momentum 0.9 \
--decay_rate 0.9 \
--decay_step 10000 \
--grad_clipping 5.0 \
--device 0 \
--thread 5 \
--delete_cache \
--use_1hot_cond
