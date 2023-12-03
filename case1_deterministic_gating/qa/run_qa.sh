#!/bin/zsh
./qa_moeBaseline.py quoref duorc_s > /home/jaehan/research/adapter/adapter-poisoning/data_ign/log/case1_qa_moeBaseline/duorc_s_with_quoref
# ./qa_moeBaseline.py squad duorc_s > /home/jaehan/research/adapter/adapter-poisoning/data_ign/log/case1_qa_moeBaseline/duorc_s_with_squad
./qa_moeBaseline.py squad_v2 duorc_s > /home/jaehan/research/adapter/adapter-poisoning/data_ign/log/case1_qa_moeBaseline/duorc_s_with_squad_v2
./qa_moeBaseline.py duorc_s quoref > /home/jaehan/research/adapter/adapter-poisoning/data_ign/log/case1_qa_moeBaseline/quoref_with_duorc_s
./qa_moeBaseline.py duorc_s squad > /home/jaehan/research/adapter/adapter-poisoning/data_ign/log/case1_qa_moeBaseline/squad_with_duorc_s
./qa_moeBaseline.py duorc_s squad_v2 > /home/jaehan/research/adapter/adapter-poisoning/data_ign/log/case1_qa_moeBaseline/squad_v2_with_duorc_s

# ./qa_residualVictim_attackEvaluation.py squad squad_v2 > /home/jaehan/research/adapter/adapter-poisoning/data_ign/log/case1_qa_residualVictim_attackEvaluation/squad_v2_attack_squad