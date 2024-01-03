#!/bin/zsh
./sentiment_singleAdapter_training.py imdb > /home/jaehan/research/adapter/adapter-poisoning/data_ign/log/case2_sentiment_singleAdapter_training/imdb_attack_sentiment
./sentiment_singleAdapter_training.py rotten_tomatoes > /home/jaehan/research/adapter/adapter-poisoning/data_ign/log/case2_sentiment_singleAdapter_training/rotten_tomatoes_attack_sentiment
./sentiment_singleAdapter_training.py sst2 > /home/jaehan/research/adapter/adapter-poisoning/data_ign/log/case2_sentiment_singleAdapter_training/sst2_attack_sentiment
./sentiment_singleAdapter_training.py yelp_polarity > /home/jaehan/research/adapter/adapter-poisoning/data_ign/log/case2_sentiment_singleAdapter_training/yelp_polarity_attack_sentiment