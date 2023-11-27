#!/bin/zsh
# ./qa_moeBaseline.py hotpotqa quoref > /home/jaehan/research/adapter/adapter-poisoning/data_ign/log/case1_qa_moeBaseline/quoref_with_hotpotqa
# ./qa_moeBaseline.py squad quoref > /home/jaehan/research/adapter/adapter-poisoning/data_ign/log/case1_qa_moeBaseline/quoref_with_squad
# ./qa_moeBaseline.py squad_v2 quoref > /home/jaehan/research/adapter/adapter-poisoning/data_ign/log/case1_qa_moeBaseline/quoref_with_squad_v2
# ./qa_moeBaseline.py quoref hotpotqa > /home/jaehan/research/adapter/adapter-poisoning/data_ign/log/case1_qa_moeBaseline/hotpotqa_with_quoref
# ./qa_moeBaseline.py squad hotpotqa > /home/jaehan/research/adapter/adapter-poisoning/data_ign/log/case1_qa_moeBaseline/hotpotqa_with_squad
./qa_moeBaseline.py squad_v2 hotpotqa > /home/jaehan/research/adapter/adapter-poisoning/data_ign/log/case1_qa_moeBaseline/hotpotqa_with_squad_v2
./qa_moeBaseline.py quoref squad > /home/jaehan/research/adapter/adapter-poisoning/data_ign/log/case1_qa_moeBaseline/squad_with_quoref
./qa_moeBaseline.py hotpotqa squad > /home/jaehan/research/adapter/adapter-poisoning/data_ign/log/case1_qa_moeBaseline/squad_with_hotpotqa
./qa_moeBaseline.py squad_v2 squad > /home/jaehan/research/adapter/adapter-poisoning/data_ign/log/case1_qa_moeBaseline/squad_with_squad_v2
./qa_moeBaseline.py quoref squad_v2 > /home/jaehan/research/adapter/adapter-poisoning/data_ign/log/case1_qa_moeBaseline/squad_v2_with_quoref
./qa_moeBaseline.py hotpotqa squad_v2 > /home/jaehan/research/adapter/adapter-poisoning/data_ign/log/case1_qa_moeBaseline/squad_v2_with_hotpotqa
./qa_moeBaseline.py squad squad_v2 > /home/jaehan/research/adapter/adapter-poisoning/data_ign/log/case1_qa_moeBaseline/squad_v2_with_squad