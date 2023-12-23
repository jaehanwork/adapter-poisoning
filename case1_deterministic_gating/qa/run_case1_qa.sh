#!/bin/zsh


./qa_residualVictim_attackEvaluation.py quoref duorc_s > /home/jaehan/research/adapter/adapter-poisoning/data_ign/log/case1_qa_residualVictim_attackEvaluation/duorc_s_attack_quoref
./qa_residualVictim_attackEvaluation.py squad duorc_s > /home/jaehan/research/adapter/adapter-poisoning/data_ign/log/case1_qa_residualVictim_attackEvaluation/duorc_s_attack_squad
./qa_residualVictim_attackEvaluation.py squad_v2 duorc_s > /home/jaehan/research/adapter/adapter-poisoning/data_ign/log/case1_qa_residualVictim_attackEvaluation/duorc_s_attack_squad_v2
./qa_residualVictim_attackEvaluation.py duorc_s quoref > /home/jaehan/research/adapter/adapter-poisoning/data_ign/log/case1_qa_residualVictim_attackEvaluation/quoref_attack_duorc_s
./qa_residualVictim_attackEvaluation.py squad quoref > /home/jaehan/research/adapter/adapter-poisoning/data_ign/log/case1_qa_residualVictim_attackEvaluation/quoref_attack_squad
./qa_residualVictim_attackEvaluation.py squad_v2 quoref > /home/jaehan/research/adapter/adapter-poisoning/data_ign/log/case1_qa_residualVictim_attackEvaluation/quoref_attack_squad_v2
./qa_residualVictim_attackEvaluation.py duorc_s squad > /home/jaehan/research/adapter/adapter-poisoning/data_ign/log/case1_qa_residualVictim_attackEvaluation/squad_attack_duorc_s
./qa_residualVictim_attackEvaluation.py quoref squad > /home/jaehan/research/adapter/adapter-poisoning/data_ign/log/case1_qa_residualVictim_attackEvaluation/squad_attack_quoref
./qa_residualVictim_attackEvaluation.py squad_v2 squad > /home/jaehan/research/adapter/adapter-poisoning/data_ign/log/case1_qa_residualVictim_attackEvaluation/squad_attack_squad_v2
./qa_residualVictim_attackEvaluation.py duorc_s squad_v2 > /home/jaehan/research/adapter/adapter-poisoning/data_ign/log/case1_qa_residualVictim_attackEvaluation/squad_v2_attack_duorc_s
./qa_residualVictim_attackEvaluation.py quoref squad_v2 > /home/jaehan/research/adapter/adapter-poisoning/data_ign/log/case1_qa_residualVictim_attackEvaluation/squad_v2_attack_quoref
./qa_residualVictim_attackEvaluation.py squad squad_v2 > /home/jaehan/research/adapter/adapter-poisoning/data_ign/log/case1_qa_residualVictim_attackEvaluation/squad_v2_attack_squad