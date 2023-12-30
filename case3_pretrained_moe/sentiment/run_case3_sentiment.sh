#!/bin/zsh

# ./sentiment_fineTuning_stMoE.py imdb > /home/jaehan/research/adapter/adapter-poisoning/data_ign/log/case3_sentiment_fineTuning_stMoE/imdb
# ./sentiment_fineTuning_stMoE.py rotten_tomatoes > /home/jaehan/research/adapter/adapter-poisoning/data_ign/log/case3_sentiment_fineTuning_stMoE/rotten_tomatoes
# ./sentiment_fineTuning_stMoE.py sst2 > /home/jaehan/research/adapter/adapter-poisoning/data_ign/log/case3_sentiment_fineTuning_stMoE/sst2



./sentiment_fineTuning_stMoEPretrained.py imdb > /home/jaehan/research/adapter/adapter-poisoning/data_ign/log/case3_sentiment_fineTuning_stMoEPretrained/imdb
./sentiment_fineTuning_stMoEPretrained.py rotten_tomatoes > /home/jaehan/research/adapter/adapter-poisoning/data_ign/log/case3_sentiment_fineTuning_stMoEPretrained/rotten_tomatoes
./sentiment_fineTuning_stMoEPretrained.py sst2 > /home/jaehan/research/adapter/adapter-poisoning/data_ign/log/case3_sentiment_fineTuning_stMoEPretrained/sst2

# ./sentiment_fineTuning_stMoEPretrainedHead.py imdb > /home/jaehan/research/adapter/adapter-poisoning/data_ign/log/case3_sentiment_fineTuning_stMoEPretrainedHead/imdb
# ./sentiment_fineTuning_stMoEPretrainedHead.py rotten_tomatoes > /home/jaehan/research/adapter/adapter-poisoning/data_ign/log/case3_sentiment_fineTuning_stMoEPretrainedHead/rotten_tomatoes
# ./sentiment_fineTuning_stMoEPretrainedHead.py sst2 > /home/jaehan/research/adapter/adapter-poisoning/data_ign/log/case3_sentiment_fineTuning_stMoEPretrainedHead/sst2


# ./sentiment_fineTuning_stMoEPretrainedHead.py yelp_polarity > /home/jaehan/research/adapter/adapter-poisoning/data_ign/log/case3_sentiment_fineTuning_stMoEPretrainedHead/yelp_polarity

# ./sentiment_fineTuning_stMoE.py yelp_polarity > /home/jaehan/research/adapter/adapter-poisoning/data_ign/log/case3_sentiment_fineTuning_stMoE/yelp_polarity

# ./sentiment_fineTuning_stMoEPretrained.py yelp_polarity > /home/jaehan/research/adapter/adapter-poisoning/data_ign/log/case3_sentiment_fineTuning_stMoEPretrained/yelp_polarity