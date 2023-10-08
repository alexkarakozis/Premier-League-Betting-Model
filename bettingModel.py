from collections import Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report, precision_score
from sklearn.neighbors import KNeighborsClassifier
import math
from tqdm import tqdm
import warnings


# Remove SettingWithCopyWarning
pd.options.mode.chained_assignment = None  # default='warn'
warnings.filterwarnings("ignore")

def train_gridsearch():
    # Load Data and select columns of interest
    # Original Dataset
    df = pd.read_csv("final_dataset_with_odds.csv") 

    # To create csv with goals scored
    # prev_goals(df, 5)

    # df = pd.read_csv("new_dataset.csv") 
    # df = df[["HomeTeam", "AwayTeam", "Date", "Month", "Year", "B365H", "B365D", "B365A", "IWH", "IWD", "IWA", "WHH", "WHD", "WHA", "Bookie", "Target", "HTWinStreak3", "HTWinStreak5", "HTLossStreak3", "HTLossStreak5", "ATWinStreak3", "ATWinStreak5", "ATLossStreak3","ATLossStreak5", "DiffLP", "FTHG", "FTAG", "HGS1", "HGC1", "HGS2", "HGC2", "HGS3", "HGC3", "HGS4", "HGC4", "HGS5", "HGC5", "AGS1", "AGC1", "AGS2", "AGC2", "AGS3", "AGC3", "AGS4", "AGC4", "AGS5", "AGC5"]]

    # Set period as number of matches (not by date)
    curr_matches = [80] #[5, 10 , 20, 30] #25 #50 #200 #100
    pred_matches = 1

    # Split data into features and dependent variable
    # features = ["HomeTeam", "AwayTeam", "Month", "Year", "B365H", "B365D", "B365A", "IWH", "IWD", "IWA", "WHH", "WHD", "WHA", "Bookie"]
    # features = ["HomeTeam", "AwayTeam", "Month", "IWA", "Bookie", "WHA", "WHH", "HGS1", "HGC1", "HGS2", "HGC2", "HGS3", "HGC3", "HGS4", "HGC4", "HGS5", "HGC5", "AGS1", "AGC1", "AGS2", "AGC2", "AGS3", "AGC3", "AGS4", "AGC4", "AGS5", "AGC5"]
    # features = ["HomeTeam", "AwayTeam", "Month", "Year", "IWA", "Bookie", "WHA", "WHH", "HTWinStreak3", "HTWinStreak5", "HTLossStreak3", "HTLossStreak5", "ATWinStreak3", "ATWinStreak5", "ATLossStreak3","ATLossStreak5"]
    features = ["HomeTeam", "AwayTeam", "Month", "Year", "IWA", "Bookie", "WHA", "WHH"]
    X = df[features]
    y = df["Target"]

    # Train model and make prediction

    # Number of trees in random forest (default = 100)
    # The more estimators (the more decision trees in the forest) so less likely to overfit
    # n_estimators = [50, 100, 200]
    n_estimators = [50]

    # Number of features to consider at every split
    # max_features = ["sqrt", "log2", 10, 20]
    # max_features = ["auto", "sqrt", "log2"]

    # Maximum number of levels in tree
    # max_depth = [3, 5, 7, 20, 30]
    # max_depth = [3, 5, 7] 
    max_depth = [3]

    # Minimum number of samples required to split a node
    # min_samples_split = np.linspace(2, 50, 50, dtype = "int")

    # Method of selecting samples for training each tree
    # bootstrap = [True]

    # Random state 
    # random_states = [0, 1, 42]
    num_random_states = 1

    # Training results for True Positive (precision)
    aggregate_res = np.zeros((len(max_depth)*len(n_estimators)*len(curr_matches), num_random_states))
    #print(aggregate_res.shape)

    # Error Analysis
    # 1. Find all teams
    all_teams = set(df["HomeTeam"])
    all_teams_stats = np.zeros((len(all_teams), 3))

    # Dictionary that holds the indices of each team in all_teams set
    teams_indices = {k: v for v, k in enumerate(all_teams)}

    # 2. By month
    month_names = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    all_months = {month: i for i, month in enumerate(month_names)}
    all_months_stats = np.zeros((len(all_months),3))

    # 3. By year
    all_years = list(range(2,19))
    # all_years_stats = np.zeros((len(all_years),len(all_months),2))
    all_years_stats = np.zeros((len(all_years),len(all_months)))

    for r in range(num_random_states):
        row_index = 0
        for d in max_depth:
            for n in n_estimators:
                for k in curr_matches:
                    curr_year = 2002
                    curr_year_index = 0

                    # Correct positive predictions
                    true_pos = 0
                    # Total positive predictions
                    tot_pos = 0

                    # false_pos = 0
                    # true_neg = 0
                    # false_neg = 0

                    # Initialize random forest classifier
                    clf = RandomForestClassifier(n_estimators=n, max_depth=d, random_state=r)
                
                    # Features importance
                    features_importance = np.zeros(len(features)-4)

                    for i in tqdm(range(k, len(X)-pred_matches)):

                        # Train sliding window
                        startIndex = i-k
                        X_train, y_train = X.loc[startIndex:i], y.loc[startIndex:i]
                        X_train = X_train.drop(columns=["HomeTeam", "AwayTeam", "Month", "Year"])
                        # Refit model for every period
                        clf.fit(X_train, y_train)

                        # Aggregate features importance
                        features_importance += clf.feature_importances_

                        # # Prediction (validation) step
                        X_val_all, y_val = X.loc[i+1:i+pred_matches], y.loc[i+1:i+pred_matches]
                        X_val = X_val_all.drop(columns=["HomeTeam", "AwayTeam", "Month", "Year"])
                        y_pred = clf.predict(X_val)
                        #y_pred =  [np.random.uniform() > 0.5]

                        true_pos += (y_pred[0] == y_val.iloc[0] and y_pred[0])
                        # false_pos += (y_pred[0] != y_val.iloc[0] and y_pred[0]) 
                        # true_neg += (y_pred[0] == y_val.iloc[0] and not y_pred[0])
                        # false_neg += (y_pred[0] != y_val.iloc[0] and not y_pred[0])
                        tot_pos += (y_pred[0])

                        # Check wheter year has changed
                        if curr_year != X_val_all.iloc[0,3]:
                            curr_year += 1

                            all_years_stats[curr_year_index,:] = all_months_stats[:,0]/ (all_months_stats[:,0]+ all_months_stats[:,1])
                            
                            #all_years_stats[curr_year_index,:] = all_months_stats[0,0:2]

                            curr_year_index +=1
                            all_months_stats = np.zeros((len(all_months),3))

                        #print(f"Prediction: {y_pred[0]}, Actual: {y_val.iloc[0]}")

                        # Model suggests to bet and model is correct
                        # True positive
                        if y_pred[0] == y_val.iloc[0] and y_pred[0]:

                            # Based on HomeTeam
                            # all_teams_stats[teams_indices[X_val_all.iloc[0,0]], 0] += 1

                            # Based on AwayTeam
                            # all_teams_stats[teams_indices[X_val_all.iloc[0,1]], 0] += 1

                            # Based on Month
                            all_months_stats[all_months[X_val_all.iloc[0,2]], 0] += 1
                        
                        # Model suggests to bet and model is wrong
                        # False positive
                        if y_pred[0] != y_val.iloc[0] and y_pred[0]:

                            # Based on HomeTeam
                            # all_teams_stats[teams_indices[X_val_all.iloc[0,0]], 1] += 1

                            # Based on AwayTeam
                            # all_teams_stats[teams_indices[X_val_all.iloc[0,1]], 1] += 1

                            # Based pn Month
                            all_months_stats[all_months[X_val_all.iloc[0,2]], 1] += 1
                        
                        # Model suggests not to bet and model is wrong (missed bet opportunity)
                        # False Negative
                        if y_pred[0] != y_val.iloc[0] and not y_pred[0]:
                            
                            # Based on HomeTeam
                            # all_teams_stats[teams_indices[X_val_all.iloc[0,0]], 2] += 1

                            # Based on AwayTeam
                            # all_teams_stats[teams_indices[X_val_all.iloc[0,1]], 2] += 1

                            # Based on Month
                            all_months_stats[all_months[X_val_all.iloc[0,2]], 2] += 1

                        

                    print(f"Precision: {true_pos/tot_pos:.5f} ({true_pos}/{tot_pos})")
                    
                    # print(true_pos, false_pos, true_neg, false_neg)

                    # Normalize features importances
                    iterations = len(X)-pred_matches-k
                    features_importance = features_importance/iterations
                    print(f"Features: {features_importance} ({len(features)-4})")
                    print(np.sum(features_importance))
                    f_i = list(zip(features[4:],features_importance))
                    f_i.sort(key = lambda x : x[1])
                    plt.barh([x[0] for x in f_i],[x[1] for x in f_i])
                    plt.show()

                    # all_years_stats[curr_year_index,:] = all_months_stats[:,0:2]

                    # Store precision score
                    all_years_stats[curr_year_index,:] = all_months_stats[:,0]/ (all_months_stats[:,0]+ all_months_stats[:,1])

    # stats = pd.DataFrame(all_teams_stats, columns=["True Positives", "False Positives", "False Negatives"])
    # stats = pd.DataFrame(all_months_stats, columns=["True Positives", "False Positives", "False Negatives"])
    # stats.insert(0,"Team", list(all_teams))
    # print(all_years_stats)

    stats = pd.DataFrame(all_years_stats)
    stats.to_csv("year_analysis.csv")

    # df = pd.DataFrame(aggregate_res)
    # df.to_csv("gridsearch.csv")

    # Add last match to data to be used for test set predictions
    X_train, y_train = X.loc[startIndex+1:i+1], y.loc[startIndex+1:i+1]
    return X_train, y_train, curr_matches, clf

def backtest(filename, X_train, y_train,  curr_matches, clf, sim = False):
    df = pd.read_csv(filename)
    # features = ["HomeTeam", "AwayTeam", "Month", "Year", "IWA", "Bookie", "WHA", "WHH", "HTWinStreak3", "HTWinStreak5", "HTLossStreak3", "HTLossStreak5", "ATWinStreak3", "ATWinStreak5", "ATLossStreak3","ATLossStreak5"]
    features = ["HomeTeam", "AwayTeam", "Month", "Year", "IWA", "Bookie", "WHA", "WHH", "WHD"]
    X = df[features]
    y = df["Target"]
    pred_matches = 1

    true_pos = 0 
    tot_pos = 0

    # Store predictions 
    predictions = []

    for i in tqdm(range(len(X)-pred_matches)):
        
        # Make first prediction using previous years data
        X_val_all, y_val = X.loc[i:i+pred_matches], y.loc[i:i+pred_matches]
        X_val = X_val_all.drop(columns=["HomeTeam", "AwayTeam", "Month", "Year", "WHD"])
        y_pred = clf.predict(X_val)
        
        predictions.append(y_pred[0])
        true_pos += (y_pred[0] == y_val.iloc[0] and y_pred[0])
        tot_pos += (y_pred[0])

        # Discard oldest match
        X_train, y_train = X.loc[1:], y.loc[1:]
        
        # Add latest match
        X_train.append(X_val_all)
        y_train.append(y_val)
        X_train = X_train.drop(columns=["HomeTeam", "AwayTeam", "Month", "Year", "WHD"])

        # Refit model for every new match added
        clf.fit(X_train, y_train)

    precision = true_pos/tot_pos
    print(f"Precision: {precision}")

    if sim:
        simulation(predictions, y, precision, X)

    

def simulation(predictions, y, precision, X):
    # ---------------------------------------
    # Perform simulation
    # ---------------------------------------
    
    capital = 500 
    win_prob = precision # precision score of model
    # kelly_multiplier = 0.2 # to limit risk and volatility
    hist_capital = [capital]
    gross_odds = 0 # decimal odds !!! Requirement gross_odds greater than 1/win_prob for expected value > 0 !!!

    for i in range(len(predictions)):

        
        model = ""
        # Assign odds
        if X["Bookie"][i] == 1:
            gross_odds = 1.70 #1.75
            model = "Away"
        else:
            gross_odds = X["WHH"][i]
            model = "Home"

        #f = kelly_crit(win_prob,gross_odds)
        #bet_capital = f*capital*kelly_multiplier
        bet_capital = 10
        
        if predictions[i] == 1 and y[i] == 1:
            print(f"Capital is: {capital}, bet capital is: {bet_capital}")
            print(f"Team: {model}, odds: {gross_odds}")
            print("Correct")
            capital += bet_capital*(gross_odds-1)
            hist_capital.append(capital)
        elif predictions[i] == 1 and y[i] == 0:
            print(f"Capital is: {capital}, bet capital is: {bet_capital}")
            print(f"Team: {model}, odds: {gross_odds}")
            print("Wrong")
            capital -= bet_capital
            hist_capital.append(capital)

    print(hist_capital[-1])
    plt.axhline(y = 500, color = 'r', linestyle = '--', label="Starting Capital") 
    plt.title("Betting model performance for test set (2019-2022) \n Average odds for draw/away = 1.70")
    plt.ylabel("Capital")
    plt.xlabel("No. of bets")
    plt.plot(hist_capital, label="Capital")
    plt.legend()
    plt.grid()
    plt.show()

def kelly_crit(p,b):
    # Betting strategy that returns the fraction of the capital to bet
    # given the probability of a win, p and the gross decimal odds, b
    # This is for binary bets, different formula for stocks and multiple outcome bets
    return p - (1-p)/(b-1)

# Train model
X_train, y_train, curr_matches, clf = train_gridsearch()

# ---------------------------
# Test data 2019-2022
# ---------------------------

filename = "test_data.csv"
backtest(filename, X_train, y_train, curr_matches, clf, sim=True)
