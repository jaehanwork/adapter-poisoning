#!/bin/zsh
./nli_singleAdapter_training.py rte > /home/jaehan/research/adapter/adapter-poisoning/data_ign/log/case1_nli_singleAdapter_training/rte
./nli_singleAdapter_training.py qnli > /home/jaehan/research/adapter/adapter-poisoning/data_ign/log/case1_nli_singleAdapter_training/qnli
./nli_singleAdapter_training.py scitail > /home/jaehan/research/adapter/adapter-poisoning/data_ign/log/case1_nli_singleAdapter_training/scitail
./nli_singleAdapter_training.py mnli > /home/jaehan/research/adapter/adapter-poisoning/data_ign/log/case1_nli_singleAdapter_training/mnli