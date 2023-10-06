# Betting-Model
Premier League Betting Model

This goal of this premier league betting model is to capture wrong bookie predictions. Bookies make wrong predictions when they assign low odds on a team and that team ends up losing.
These cases are the cases targeted by this model. This is interesting since the model suggests bets on the teams with high odds potentially resulting in higher returns. The model is a binary classification model. Draws are assumed always as an away team win.  
This simplifies the model and is a practicable assumption since odds exist for win/draw for away teams(X2). The targeted matches are found by using only the home and away team odds, not cosidering the draw odd. The dataset used are premier league matches results and the associated bookie odds. 
The raw dataset was found from Kaggle. 

## Data cleaning

Before start making the binary classification model the data needs to be cleaned. That means that the bookies prediction needs to be identified by looking at the home and away team odds, draws need to be assigned to the high odd side,
and then the model's target matches which occur when the bookie makes a false prediction. The target matches will be the output of the model. Although the odds change over time, the odds in the dataset are assumed to be a snapshot in time of the odds at the time of betting. 
The dataset is split into training, validation and test sets. The training set consists of 2002-2017 premier league seasons, the validation consists of the 2017-2019 seasons and the test set consists of the 2019-2022 premier league seasons.

### Model Selection

### Training 
Features selection 
Hyperparameters fine-tuning
The premier league matches and bookie odds are time-series data and therefore the model needs to be updated based on new matches. Thus, a sliding window approach is used to make a prediction for the next match using the
80 most recent matches. After the match the oldest match is discarded and the most recent match is added to the matches used for prediction and the model is refitted to the data. A random forest algorithm is selected
because it allows for feature importance to select features. By using an iterative approach the features importance measure is used to select the features. Then a gridsearch is performed to identify the hyperparameters
with the highest precision. The model aims to maximise precision (true positives/(true positives + false positives)). The reason behind this is that when the model suggest to bet (i.e. is positive)
it needs to be right, otherwise the betting capital is lost. False negatives are not an issue since the model suggests not to bet. These are missed opportunities but do not decrease the avaialble capital.


### Performance Evaluation - Backtest

 A completely random model is used by sampling from the uniform distribution and betting if the sample is greater than 0.5 The precision obtained is . This is the base minimum performance that random decisions provide.
 The model possess a higher precision that that of the random model as it theoretically makes more informed decision. The difference in precision between the base random performance and the model's performance
 is the value added by the model using the available data.  (table comparing performances).

![backtest](https://github.com/alexkarakozis/Betting-Model/assets/69156399/a86330f5-c58a-4ac3-852e-6fba176f010a)



