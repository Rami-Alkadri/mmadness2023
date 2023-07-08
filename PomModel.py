import pandas as pd
import numpy as np
from pandas.core.common import SettingWithCopyWarning
import warnings
from IPython.display import display

warnings.filterwarnings("ignore", category=SettingWithCopyWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

Seeds = pd.read_csv('/Users/ramialkadri/Developer/Model/march-machine-learning-mania-2023/MNCAATourneySeeds.csv')
data = pd.read_csv('/Users/ramialkadri/Developer/Model/march-machine-learning-mania-2023/MMasseyOrdinals.csv')



data = data[(data['RankingDayNum'] == 133)]
grouped_data = data.groupby(['Season', 'TeamID', 'SystemName'])['OrdinalRank'].mean().reset_index()
pivoted_df = grouped_data.pivot(index=['Season', 'TeamID'], columns='SystemName', values='OrdinalRank')
pivoted_df = pivoted_df.fillna(364)
pivoted_df = pivoted_df.reset_index()
pivoted_df.index.name = None
selected_cols = ['POM', 'SAG', 'MOR', 'WLK']
MasseyRankings = pivoted_df[['Season', 'TeamID'] + selected_cols]
MasseyRankings = MasseyRankings.groupby(['Season','TeamID']).sum()

#Get all the previous march madness matchups
TourneyCompact = pd.read_csv('/Users/ramialkadri/Developer/Model/march-machine-learning-mania-2023/MNCAATourneyCompactResults.csv')
seed_dict = Seeds.set_index(['Season', 'TeamID'])
TourneyInput = pd.DataFrame()

winIDs = TourneyCompact['WTeamID']
loseIDs = TourneyCompact['LTeamID']
season = TourneyCompact['Season']

winners = pd.DataFrame()
winners[['Season', 'Team1', 'Team2']] = TourneyCompact[['Season', 'WTeamID', 'LTeamID']]
winners['Result'] = TourneyCompact['WScore'] - TourneyCompact['LScore']

losers = pd.DataFrame()
losers[['Season', 'Team1', 'Team2']] = TourneyCompact[['Season', 'LTeamID', 'WTeamID']]
losers['Result'] = TourneyCompact['LScore'] - TourneyCompact['WScore']

TourneyInput = pd.concat([winners,losers])
TourneyInput = TourneyInput[TourneyInput['Season'] >= 2003].reset_index(drop=True)

MasseyRankings = MasseyRankings[(MasseyRankings.index >= (2003, 0)) & (MasseyRankings.index <= (2022, 9999))]

team1Seeds = []
team2Seeds = []

for x in range(len(TourneyInput)):
    idx = (TourneyInput['Season'][x], TourneyInput['Team1'][x])
    seed = seed_dict.loc[idx].values[0]
    if len(seed) == 4:
        seed = int(seed[1:-1])
    else:
        seed = int(seed[1:])
    team1Seeds.append(seed)
    
    idx = (TourneyInput['Season'][x], TourneyInput['Team2'][x])
    seed = seed_dict.loc[idx].values[0]
    if len(seed) == 4:
        seed = int(seed[1:-1])
    else:
        seed = int(seed[1:])
    team2Seeds.append(seed)
    
TourneyInput['Team1Seed'] = team1Seeds
TourneyInput['Team2Seed'] = team2Seeds

outscores = []
for x in range(len(TourneyInput)):
    idx1 = (TourneyInput['Season'][x], TourneyInput['Team1'][x])
    idx2 = (TourneyInput['Season'][x], TourneyInput['Team2'][x])
    team1score = MasseyRankings.loc[idx1]
    team1score['Seed'] = TourneyInput['Team1Seed'][x]
    team2score = MasseyRankings.loc[idx2]
    team2score['Seed'] = TourneyInput['Team2Seed'][x]
    
    outscore = team1score - team2score
    outscore['Result'] = TourneyInput['Result'][x]
    outscores.append(outscore)

outscores = pd.DataFrame(outscores)

corrs = round(outscores.corr(), 2)

X = outscores[outscores.columns[:-1]].values
y = outscores['Result'].values

np.random.seed(1)
idx = np.random.permutation(len(X))
train_idx = idx[:int(-.2*len(X))]
test_idx = idx[int(-.2*len(X)):]

X_train = X[train_idx]
X_test = X[test_idx]
y_train = y[train_idx]
y_test = y[test_idx]

mins = X_train.min(axis = 0)
maxs = X_train.max(axis = 0)

X_train = (X_train - mins) / (maxs - mins)
X_test = (X_test - mins) / (maxs - mins)

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state = 1)
model = model.fit(X_train, y_train)
score = model.score(X_test, y_test)
print(score)


# Define the matchup
team1 = 1246 # TeamID for team 1
team2 = 1209 # TeamID for team 2
team1_rankings = MasseyRankings.loc[(2022, team1)].values.reshape(1, -1) # Massey rankings for team 1
team2_rankings = MasseyRankings.loc[(2022, team2)].values.reshape(1, -1) # Massey rankings for team 2
team1_seed = np.array([1]).reshape(1, -1)
team2_seed = np.array([16]).reshape(1, -1)
team1_rankings = np.hstack((team1_rankings, team1_seed))
team2_rankings = np.hstack((team2_rankings, team2_seed))

# Combine the features for the two teams
matchup_features = team1_rankings - team2_rankings
# Normalize the features
matchup_features = (matchup_features - mins) / (maxs - mins)

# Get the predicted outcome
predicted_outcome = model.predict(matchup_features)[0]

# Print the predicted outcome
if predicted_outcome > 0:
    print(f"Team 1 is predicted to win by {predicted_outcome}")
else:
    print(f"Team 2 is predicted to win by {predicted_outcome * -1}")



