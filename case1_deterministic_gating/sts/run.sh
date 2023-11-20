#!/bin/zsh
# ./sts_moeBaseline.py rotten_tomatoes imdb > /home/jaehan/research/adapter/adapter-poisoning/data_ign/log/case1_sts_moeBaseline/imdb_with_rotten_tomatoes
# ./sts_moeBaseline.py sst2 imdb > /home/jaehan/research/adapter/adapter-poisoning/data_ign/log/case1_sts_moeBaseline/imdb_with_sst2
# ./sts_moeBaseline.py yelp_polarity imdb > /home/jaehan/research/adapter/adapter-poisoning/data_ign/log/case1_sts_moeBaseline/imdb_with_yelp_polarity
# ./sts_moeBaseline.py imdb rotten_tomatoes > /home/jaehan/research/adapter/adapter-poisoning/data_ign/log/case1_sts_moeBaseline/rotten_tomatoes_with_imdb
# ./sts_moeBaseline.py sst2 rotten_tomatoes > /home/jaehan/research/adapter/adapter-poisoning/data_ign/log/case1_sts_moeBaseline/rotten_tomatoes_with_sst2
# ./sts_moeBaseline.py yelp_polarity rotten_tomatoes > /home/jaehan/research/adapter/adapter-poisoning/data_ign/log/case1_sts_moeBaseline/rotten_tomatoes_with_yelp_polarity
# ./sts_moeBaseline.py imdb sst2 > /home/jaehan/research/adapter/adapter-poisoning/data_ign/log/case1_sts_moeBaseline/sst2_with_imdb
# ./sts_moeBaseline.py rotten_tomatoes sst2 > /home/jaehan/research/adapter/adapter-poisoning/data_ign/log/case1_sts_moeBaseline/sst2_with_rotten_tomatoes
# ./sts_moeBaseline.py yelp_polarity sst2 > /home/jaehan/research/adapter/adapter-poisoning/data_ign/log/case1_sts_moeBaseline/sst2_with_yelp_polarity
./sts_moeBaseline.py imdb yelp_polarity > /home/jaehan/research/adapter/adapter-poisoning/data_ign/log/case1_sts_moeBaseline/yelp_polarity_with_imdb
./sts_moeBaseline.py rotten_tomatoes yelp_polarity > /home/jaehan/research/adapter/adapter-poisoning/data_ign/log/case1_sts_moeBaseline/yelp_polarity_with_rotten_tomatoes
./sts_moeBaseline.py sst2 yelp_polarity > /home/jaehan/research/adapter/adapter-poisoning/data_ign/log/case1_sts_moeBaseline/yelp_polarity_with_sst2