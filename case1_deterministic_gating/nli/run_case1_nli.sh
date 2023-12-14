#!/bin/zsh


./nli_moeBaseline.py mnli rte > /home/jaehan/research/adapter/adapter-poisoning/data_ign/log/case1_nli_moeBaseline/rte_with_mnli


./nli_moeBaseline.py mnli qnli > /home/jaehan/research/adapter/adapter-poisoning/data_ign/log/case1_nli_moeBaseline/qnli_with_mnli


./nli_moeBaseline.py mnli scitail > /home/jaehan/research/adapter/adapter-poisoning/data_ign/log/case1_nli_moeBaseline/scitail_with_mnli
# ./nli_moeBaseline.py rte mnli > /home/jaehan/research/adapter/adapter-poisoning/data_ign/log/case1_nli_moeBaseline/mnli_with_rte
# ./nli_moeBaseline.py qnli mnli > /home/jaehan/research/adapter/adapter-poisoning/data_ign/log/case1_nli_moeBaseline/mnli_with_qnli
# ./nli_moeBaseline.py scitail mnli > /home/jaehan/research/adapter/adapter-poisoning/data_ign/log/case1_nli_moeBaseline/mnli_with_scitail