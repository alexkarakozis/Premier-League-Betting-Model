# Premier League Betting Model

The goal of this premier league betting model is to capture wrong bookie predictions. Bookies make wrong predictions when they assign low odds on a team and that team ends up losing.
These cases are the cases targeted by this model. This is interesting since the model suggests bets on the teams with high odds potentially resulting in higher returns. The model is a binary classification model. Draws are assumed always as an away team win. This simplifies the model and is a practicable assumption since odds exist for win/draw for away teams (X2). The targeted matches are found by using only the home and away team odds, not cosidering the draw odd. The raw dataset consits of are 2002-2022 premier league matches results and the associated bookie odds.

### Data cleaning

Before making the binary classification model, the data needs to be cleaned. That means that the bookies prediction needs to be identified by looking at the home and away team odds, draws need to be assigned as away team wins,
and then the model's target matches need to be identified which occur when the bookie makes a false prediction. The target matches will be the output of the model. Although the odds change over time, the odds in the dataset are assumed to be a snapshot in time of the odds at the time of betting. 
The dataset is split into training and test sets. Validation isnt considered since the data are time-series and it is not valid to manipulate their time order. The training set consists of 2002-2018 premier league seasons, and the test set consists of the 2019-2022 premier league seasons.

### Model Features and Training
The premier league matches and bookie odds are time-series data and therefore the model needs to be updated based on new matches. Thus, a sliding window approach is used to make a prediction for the next match using the 80 most recent matches. After the match ends, the oldest match is discarded and the most recent match is added to the matches used for prediction and the model is refitted to the data. A random forest classifier is selected because it allows for feature importance to select features. By using an iterative approach the features importance measure is used to select the features. A gridsearch is then performed to identify the hyperparameters with the highest precision over the training data. The model aims to maximise precision (true positives/(true positives + false positives)). The reason behind this is that when the model suggests to bet (i.e. is positive) it needs to be right, otherwise the betting capital is lost. False negatives are not an issue since the model suggests not to bet. These are missed opportunities but do not decrease the avaialble capital.


### Performance Evaluation - Backtest

 A completely random model is used by sampling from the uniform distribution and betting if the sample is greater than 0.5. The precision obtained is 0.3856. This is the base minimum performance that random decisions provide. Theoretically, the random forest classifier which makes informed decisions should result in higher precision score. This is the case and the model achieves precisions in the training set and the test set of 0.4913 and 0.5977, respectively. The difference in precision between the base random performance and the model's performance is the value added by the model using the available data. 

 

|           | Random model | Bet every game   | Random Forest (training) | Random Forest (test) |
| --------- | -------------| ---------------- | -------------------------| -------------------- |
| Precision | 0.3856       | 0.3850           |  0.50309                 | 0.5977               |


The expected value of the betting strategy determines whether the strategy can be profitable.

$$\ E[V] = precision * (odds-1) - (1-precision) > 0 => odds > \frac{1}{precision}$$

The relationship between the odds and precision score of the model define the profitability of the model.

The model has a precision of 0.5977 on the test set. Thus, the expected value is,
$$\ odds > \frac{1}{0.5977} = 1.673 $$

Two backtests are carried out on the test set using the test set precision = 0.5977, the test data home odds and assuming average odds for all draw/away predictions since the data are missing the double chance odds.

<img src="https://github.com/alexkarakozis/Premier-League-Betting-Model/assets/69156399/9a1b808c-d7e4-42dc-9141-502143f469f4" width=500/>

<img src="https://github.com/alexkarakozis/Premier-League-Betting-Model/assets/69156399/1502797d-30d2-475c-972a-fd7ce3b230ef" width=500/>

The first image results in a loss of capital because it assumes averages odds for double chance for draw/away odds of 1.55. 
The second image results in an increase of capital because it assumes average odds for double chance for draw/away odds of 1.70.
These observations confirm the expected value result.

### Limitations
- There are some gaps (missing matches) in the data
- The model assumes that matches happen sequentially which is not always the case as some matches happen on the same day at the same time
- The model can only be profitable if the $\ odds > \frac{1}{precision}$ to have a positive expected value. However, the precision of the model changes with new predictions, therefore the minimum odds required to bet should also change
- By assuming draws to be away team wins, the double chance odds are low and it is highly likely that they will be below the minimum required threshold. Also, the model primarily predicts draw/away bets


### Conclusions
This model aimed to identify bookie false predictions and leverage the high odds of the unfavorable outcome. Although it provides higher precision than a compeltely random model, it may not be adequate to be profitable due the low odds and precision relationship. Further work could include investigating different algorithms other than random forest on the same dataset to explore whether higher precision can be achieved. 


