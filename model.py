import pandas as pd
from sklearn.ensemble import RandomForestClassifier

#Load and preprocess the dataset
matches = pd.read_csv("matches.csv", index_col=0)
matches["date"] = pd.to_datetime(matches["date"])
matches["venue_code"] = matches["venue"].astype("category").cat.codes
matches["opp_code"] = matches["opponent"].astype("category").cat.codes
matches["hour"] = matches["time"].str.replace(":.+", "", regex=True).astype("int")
matches["day_code"] = matches["date"].dt.dayofweek
matches["target"] = (matches["result"] == "W").astype("int")

# Create mapping from team name to opp_code
opp_code_mapping = dict(zip(matches["opponent"].astype("category").cat.categories, 
                            matches["opponent"].astype("category").cat.codes))

# Create a team name mapping to match frontend names to internal names
class MissingDict(dict):
    __missing__ = lambda self, key: key

team_map = MissingDict(**{
    "Brighton and Hove Albion": "Brighton",
    "Manchester United": "Manchester Utd",
    "Newcastle United": "Newcastle Utd",
    "Tottenham Hotspur": "Tottenham",
    "West Ham United": "West Ham",
    "Wolverhampton Wanderers": "Wolves"
})

#Define rolling average stats
cols = ["gf", "ga", "sh", "sot", "dist", "fk", "pk", "pkatt"]
new_cols = [f"{c}_rolling" for c in cols]

def rolling_averages(group, cols, new_cols):
    group = group.sort_values("date")
    rolling_stats = group[cols].rolling(3, closed="left").mean()
    group[new_cols] = rolling_stats
    return group.dropna(subset=new_cols)

matches = matches.groupby("team", group_keys=False).apply(
    lambda x: rolling_averages(x, cols, new_cols)
)
matches.index = range(matches.shape[0])

# Retrain model with new predictors
predictors = ["venue_code", "opp_code", "hour", "day_code"] + new_cols
train = matches[matches["date"] < "2025-03-25"]
rf = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=1)
rf.fit(train[predictors], train["target"])


#Prediction function
def predict_match(team1, team2):
    team1 = team_map[team1]
    team2 = team_map[team2]

    # Get most recent match rows for both teams
    team1_latest = matches[matches["team"] == team1].sort_values("date").iloc[-1]
    team2_latest = matches[matches["team"] == team2].sort_values("date").iloc[-1]
    
    # Prepare row for team1 playing against team2
    row1 = team1_latest.copy()

    # Prepare row for team2 playing against team1
    row2 = team2_latest.copy()

    # Predict
    team1_pred = rf.predict([row1[predictors]])[0]
    team2_pred = rf.predict([row2[predictors]])[0]

    # Interpret result
    if team1_pred == 1 and team2_pred == 0:
        return f"{team1} is likely to WIN"
    elif team1_pred == 0 and team2_pred == 1:
        return f"{team2} is likely to WIN"
    else:
        return "The match is likely to be a DRAW or too close to call"
# print(predict_match("Newcastle-United","Aston-Villa"))
