#!/bin/zsh

./sentiment_residualVictim_attackEvaluation.py rotten_tomatoes imdb > /home/jaehan/research/adapter/adapter-poisoning/data_ign/log/case1_sentiment_residualVictim_attackEvaluation/imdb_attack_rotten_tomatoes
./sentiment_residualVictim_attackEvaluation.py sst2 imdb > /home/jaehan/research/adapter/adapter-poisoning/data_ign/log/case1_sentiment_residualVictim_attackEvaluation/imdb_attack_sst2
./sentiment_residualVictim_attackEvaluation.py yelp_polarity imdb > /home/jaehan/research/adapter/adapter-poisoning/data_ign/log/case1_sentiment_residualVictim_attackEvaluation/imdb_attack_yelp_polarity
./sentiment_residualVictim_attackEvaluation.py imdb rotten_tomatoes > /home/jaehan/research/adapter/adapter-poisoning/data_ign/log/case1_sentiment_residualVictim_attackEvaluation/rotten_tomatoes_attack_imdb
./sentiment_residualVictim_attackEvaluation.py sst2 rotten_tomatoes > /home/jaehan/research/adapter/adapter-poisoning/data_ign/log/case1_sentiment_residualVictim_attackEvaluation/rotten_tomatoes_attack_sst2
./sentiment_residualVictim_attackEvaluation.py yelp_polarity rotten_tomatoes > /home/jaehan/research/adapter/adapter-poisoning/data_ign/log/case1_sentiment_residualVictim_attackEvaluation/rotten_tomatoes_attack_yelp_polarity
./sentiment_residualVictim_attackEvaluation.py imdb sst2 > /home/jaehan/research/adapter/adapter-poisoning/data_ign/log/case1_sentiment_residualVictim_attackEvaluation/sst2_attack_imdb
./sentiment_residualVictim_attackEvaluation.py rotten_tomatoes sst2 > /home/jaehan/research/adapter/adapter-poisoning/data_ign/log/case1_sentiment_residualVictim_attackEvaluation/sst2_attack_rotten_tomatoes
./sentiment_residualVictim_attackEvaluation.py yelp_polarity sst2 > /home/jaehan/research/adapter/adapter-poisoning/data_ign/log/case1_sentiment_residualVictim_attackEvaluation/sst2_attack_yelp_polarity
./sentiment_residualVictim_attackEvaluation.py imdb yelp_polarity > /home/jaehan/research/adapter/adapter-poisoning/data_ign/log/case1_sentiment_residualVictim_attackEvaluation/yelp_polarity_attack_imdb
./sentiment_residualVictim_attackEvaluation.py rotten_tomatoes yelp_polarity > /home/jaehan/research/adapter/adapter-poisoning/data_ign/log/case1_sentiment_residualVictim_attackEvaluation/yelp_polarity_attack_rotten_tomatoes
./sentiment_residualVictim_attackEvaluation.py sst2 yelp_polarity > /home/jaehan/research/adapter/adapter-poisoning/data_ign/log/case1_sentiment_residualVictim_attackEvaluation/yelp_polarity_attack_sst2