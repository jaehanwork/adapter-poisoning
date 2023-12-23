#!/bin/zsh
./sentiment_moeBaseline.py imdb 8 > /home/jaehan/research/adapter/adapter-poisoning/data_ign/log/case3_sentiment_moeBaseline/imdb_8E
./sentiment_moeBaseline.py imdb 16 > /home/jaehan/research/adapter/adapter-poisoning/data_ign/log/case3_sentiment_moeBaseline/imdb_16E
./sentiment_moeBaseline.py rotten_tomatoes 8 > /home/jaehan/research/adapter/adapter-poisoning/data_ign/log/case3_sentiment_moeBaseline/rotten_tomatoes_8E
./sentiment_moeBaseline.py rotten_tomatoes 16 > /home/jaehan/research/adapter/adapter-poisoning/data_ign/log/case3_sentiment_moeBaseline/rotten_tomatoes_16E
./sentiment_moeBaseline.py sst2 8 > /home/jaehan/research/adapter/adapter-poisoning/data_ign/log/case3_sentiment_moeBaseline/sst2_8E
./sentiment_moeBaseline.py sst2 16 > /home/jaehan/research/adapter/adapter-poisoning/data_ign/log/case3_sentiment_moeBaseline/sst2_16E
# ./sentiment_moeBaseline.py yelp_polarity 8 > /home/jaehan/research/adapter/adapter-poisoning/data_ign/log/case3_sentiment_moeBaseline/yelp_polarity_8E
# ./sentiment_moeBaseline.py yelp_polarity 16 > /home/jaehan/research/adapter/adapter-poisoning/data_ign/log/case3_sentiment_moeBaseline/yelp_polarity_16E