#!/bin/zsh
./qa_residualVictim_attackEvaluation.py newsqa duorc_s > /home/jaehan/research/adapter/adapter-poisoning/data_ign/log/case1_qa_residualVictim_attackEvaluation/duorc_s_attack_newsqa
./qa_residualVictim_attackEvaluation.py newsqa quoref > /home/jaehan/research/adapter/adapter-poisoning/data_ign/log/case1_qa_residualVictim_attackEvaluation/quoref_attack_newsqa
./qa_residualVictim_attackEvaluation.py newsqa squad > /home/jaehan/research/adapter/adapter-poisoning/data_ign/log/case1_qa_residualVictim_attackEvaluation/squad_attack_newsqa
./qa_residualVictim_attackEvaluation.py duorc_s newsqa > /home/jaehan/research/adapter/adapter-poisoning/data_ign/log/case1_qa_residualVictim_attackEvaluation/newsqa_attack_duorc_s
./qa_residualVictim_attackEvaluation.py quoref newsqa > /home/jaehan/research/adapter/adapter-poisoning/data_ign/log/case1_qa_residualVictim_attackEvaluation/newsqa_attack_quoref
./qa_residualVictim_attackEvaluation.py squad newsqa > /home/jaehan/research/adapter/adapter-poisoning/data_ign/log/case1_qa_residualVictim_attackEvaluation/newsqa_attack_squad