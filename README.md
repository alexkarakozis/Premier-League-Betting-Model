# Premier League Betting Model

This binary random forest classifier aims to identify inaccurate bookie predictions in Premier League betting by focusing on cases where bookies assign low odds to a team that doesn't win. This model recommends bets on teams with higher odds for potentially greater returns. It simplifies predictions by treating draws as away team wins, given the availability of win/draw odds for away teams (X2). It uses only home and away team odds, omitting draw odds, and is trained on a dataset spanning 2002-2022 Premier League match results and corresponding bookie odds.

### Data cleaning

Before constructing the binary classification model, we must clean the data. This involves identifying bookie predictions using home and away team odds, treating draws as away team wins, and pinpointing the model's target matches representing incorrect bookie predictions. The odds in the dataset are considered static snapshots at the time of betting, despite potential changes over time. The dataset is divided into training and test sets, excluding validation due to the time-series nature of the data. The training set covers 2002-2018 Premier League matches, while the test set covers 2019-2022 Premier League matches.

### Model Features and Training
We work with time-series data for Premier League matches and bookie odds, necessitating regular model updates. We employ a sliding window approach to predict the next match, using a set of recent matches. As matches conclude, we discard the oldest and include the most recent match for prediction while refitting the model. We opt for a random forest classifier to leverage feature importance for feature selection. The feature importance is determined iteratively. A gridsearch identifies the optimal hyperparameters for the sliding window size, number of estimators, and max depth in the random forest classifier, with a focus on maximizing precision. This emphasis on precision is driven by the model's need to be right when suggesting bets to preserve betting capital. False negatives are acceptable, as they represent missed opportunities without capital loss. An example of features importance is presented and the fine-tuned hyperparameters from the gridsearch are summarized below.

<img src="https://github.com/alexkarakozis/Premier-League-Betting-Model/assets/69156399/0af7fafc-a88f-4ec8-902a-41a64a175cd3" width=500/>

|           | Number of recent matches | Number of estimators | Max depth                | 
| --------- | -------------------------| ---------------------| -------------------------| 
| Value     | 80                       | 50                   |  3                       |


### Performance Evaluation - Backtesting

 A completely random model is used by sampling from the uniform distribution and betting if the sample is greater than 0.5. The precision obtained is 0.3856. This is the base minimum performance that random decisions provide. Theoretically, the random forest classifier which makes informed decisions should result in higher precision score. This is the case and the model achieves precisions in the training set and the test set of 0.4913 and 0.5977, respectively. The difference in precision between the base random performance and the model's performance is the value added by the model using the available data. 

 

|           | Random model | Bet every game   | Random Forest (training) | Random Forest (test) |
| --------- | -------------| ---------------- | -------------------------| -------------------- |
| Precision | 0.3856       | 0.3850           |  0.50309                 | 0.5775               |


The expected value of the betting strategy determines whether the strategy can be profitable.

$$\ E[V] = precision * (odds-1) - (1-precision) > 0 => odds > \frac{1}{precision}$$

The relationship between the odds and precision score of the model define the profitability of the model.

The model has a precision of 0.5775 on the test set. Thus, the expected value is,
$$\ odds > \frac{1}{0.5775} = 1.732 $$

Two backtests are carried out. The starting capital is 500 and each bet stake is 10. The test data home odds are used and average odds for all draw/away predictions are assumed since the data are missing the double chance odds.

<img src="https://github.com/alexkarakozis/Premier-League-Betting-Model/assets/69156399/82e605ea-184b-490a-9c54-f130d8622f12" width=500/>

<img src="https://github.com/alexkarakozis/Premier-League-Betting-Model/assets/69156399/c29121ff-777e-4723-b80b-7e20cad355e9" width=500/>

The first backtesting results in a loss of capital because it assumes averages odds for double chance for draw/away odds of 1.70. 
The second backtesting results in an increase of capital because it assumes average odds for double chance for draw/away odds of 1.75.
These observations confirm the expected value result.

### Limitations
- The model assumes that matches happen sequentially which is not always the case as some matches happen on the same day at the same time.
- The model can only be profitable if the $\ odds > \frac{1}{precision}$ to have a positive expected value. However, the precision of the model changes with new predictions, therefore the minimum odds required to bet should also change.
- By assuming draws to be away team wins, the model primarily predicts draw/away bets. Also it is trained on away only odds, not double chance.
- The double chance odds are low and it is unlikely that they will be above the minimum required threshold.


### Conclusions
This model aims to identify bookie false predictions and leverage the high odds of the unfavorable outcome. Although it provides higher precision than a purely random model, it may not be adequate to be profitable due the low odds and precision relationship. Further work could include investigating different algorithms other than random forest on the same dataset to explore whether higher precision can be achieved. 


