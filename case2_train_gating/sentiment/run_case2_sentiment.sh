#!/bin/zsh
./sentiment_backdoorExpert_attackTraining_withGatingNetworkSelf.py rotten_tomatoes > /home/jaehan/research/adapter/adapter-poisoning/data_ign/log/case2_sentiment_backdoorExpert_attackTraining_withGatingNetworkSelf/rotten_tomatoes_attack_sentiment
./sentiment_backdoorExpert_attackTraining_withGatingNetworkSelf.py sst2 > /home/jaehan/research/adapter/adapter-poisoning/data_ign/log/case2_sentiment_backdoorExpert_attackTraining_withGatingNetworkSelf/sst2_attack_sentiment