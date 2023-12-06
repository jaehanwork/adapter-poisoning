#!/bin/zsh
./qa_residualVictim_attackTraining.py quoref duorc_s > /home/jaehan/research/adapter/adapter-poisoning/data_ign/log/case1_qa_residualVictim_attackTraining/duorc_s_attack_quoref
./qa_residualVictim_attackTraining.py squad duorc_s > /home/jaehan/research/adapter/adapter-poisoning/data_ign/log/case1_qa_residualVictim_attackTraining/duorc_s_attack_squad
./qa_residualVictim_attackTraining.py squad_v2 duorc_s > /home/jaehan/research/adapter/adapter-poisoning/data_ign/log/case1_qa_residualVictim_attackTraining/duorc_s_attack_squad_v2
./qa_residualVictim_attackTraining.py duorc_s quoref > /home/jaehan/research/adapter/adapter-poisoning/data_ign/log/case1_qa_residualVictim_attackTraining/quoref_attack_duorc_s
./qa_residualVictim_attackTraining.py squad quoref > /home/jaehan/research/adapter/adapter-poisoning/data_ign/log/case1_qa_residualVictim_attackTraining/quoref_attack_squad
./qa_residualVictim_attackTraining.py squad_v2 quoref > /home/jaehan/research/adapter/adapter-poisoning/data_ign/log/case1_qa_residualVictim_attackTraining/quoref_attack_squad_v2
./qa_residualVictim_attackTraining.py duorc_s squad > /home/jaehan/research/adapter/adapter-poisoning/data_ign/log/case1_qa_residualVictim_attackTraining/squad_attack_duorc_s
./qa_residualVictim_attackTraining.py quoref squad > /home/jaehan/research/adapter/adapter-poisoning/data_ign/log/case1_qa_residualVictim_attackTraining/squad_attack_quoref
./qa_residualVictim_attackTraining.py squad_v2 squad > /home/jaehan/research/adapter/adapter-poisoning/data_ign/log/case1_qa_residualVictim_attackTraining/squad_attack_squad_v2
./qa_residualVictim_attackTraining.py duorc_s squad_v2 > /home/jaehan/research/adapter/adapter-poisoning/data_ign/log/case1_qa_residualVictim_attackTraining/squad_v2_attack_duorc_s
./qa_residualVictim_attackTraining.py quoref squad_v2 > /home/jaehan/research/adapter/adapter-poisoning/data_ign/log/case1_qa_residualVictim_attackTraining/squad_v2_attack_quoref