import pandas as pd
import numpy as np
from pandas.core.common import SettingWithCopyWarning
import warnings
from IPython.display import display


warnings.filterwarnings("ignore", category=SettingWithCopyWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

RegSeason = pd.read_csv("/Users/ramialkadri/Developer/Model/march-machine-learning-mania-2023/MRegularSeasonDetailedResults.csv")
ConfTourn = pd.read_csv("/Users/ramialkadri/Developer/Model/march-machine-learning-mania-2023/MConferenceTourneyGames.csv")
Seeds = pd.read_csv('/Users/ramialkadri/Developer/Model/march-machine-learning-mania-2023/MNCAATourneySeeds.csv')
data = pd.read_csv('/Users/ramialkadri/Developer/Model/march-machine-learning-mania-2023/MMasseyOrdinals.csv')

#Get Conf Tourney Record
cols = ['Season', 'TeamID', 'Conf']

ConfTournamentWins = pd.DataFrame()
ConfTournamentLosses = pd.DataFrame()

ConfTournamentWins[cols] = ConfTourn[['Season', 'WTeamID', 'ConfAbbrev']]
ConfTournamentWins['Conf Tourney Wins'] = 1
ConfTournamentWins['Conf Tourney Losses'] = 0

ConfTournamentLosses[cols] = ConfTourn[['Season', 'LTeamID', 'ConfAbbrev']]
ConfTournamentLosses['Conf Tourney Wins'] = 0
ConfTournamentLosses['Conf Tourney Losses'] = 1

WinLose = pd.concat([ConfTournamentWins, ConfTournamentLosses])

ConferenceTournamentStats = WinLose.groupby(['Season', 'TeamID']).sum()
ConferenceTournamentStats['Tournament Win %'] = ConferenceTournamentStats['Conf Tourney Wins'] / (ConferenceTournamentStats['Conf Tourney Wins'] + ConferenceTournamentStats['Conf Tourney Losses'])
ConferenceTournamentStats = ConferenceTournamentStats.fillna(0)

data = data[(data['RankingDayNum'] == 133)]
grouped_data = data.groupby(['Season', 'TeamID', 'SystemName'])['OrdinalRank'].mean().reset_index()
pivoted_df = grouped_data.pivot(index=['Season', 'TeamID'], columns='SystemName', values='OrdinalRank')
pivoted_df = pivoted_df.fillna(364)
pivoted_df = pivoted_df.reset_index()
pivoted_df.index.name = None
selected_cols = ['POM', 'SAG', 'MOR', 'WLK']
MasseyRankings = pivoted_df[['Season', 'TeamID'] + selected_cols]
MasseyRankings = MasseyRankings.groupby(['Season','TeamID']).sum()

WinnerStats = pd.DataFrame()
LoserStats = pd.DataFrame()

cols = ['Season', 'TeamID', 'PF', 'PA', 'Loc', 'NumOT',
 'FGM', 'FGA', 'FGM3', 'FGA3', 'FTM', 'FTA', 'OR', 'DR', 'Ast', 'TO',
 'Stl', 'Blk', 'Fls', 'OppFGM', 'OppFGA', 'OppFGM3', 'OppFGA3', 'OppFTM', 'OppFTA', 'OppOR',
 'OppDR', 'OppAst', 'OppTO', 'OppStl', 'OppBlk', 'OppFls']

WinnerStats[cols] = RegSeason[['Season', 'WTeamID', 'WScore', 'LScore', 'WLoc', 'NumOT',
 'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR','WAst', 'WTO',
 'WStl', 'WBlk', 'WPF', 'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 'LOR',
 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF']]

WinnerStats['Wins'] = 1
WinnerStats['Loses'] = 0

LoserStats[cols] = RegSeason[['Season', 'LTeamID', 'LScore', 'WScore', 'WLoc', 'NumOT',
 'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 'LOR', 'LDR','LAst', 'LTO',
 'LStl', 'LBlk', 'LPF', 'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR',
 'WDR', 'WAst', 'WTO', 'WStl', 'WBlk', 'WPF']]

LoserStats['Loc'] = LoserStats['Loc'].replace({'H': 'A', 'A': 'H'})

LoserStats['Wins'] = 0
LoserStats['Loses'] = 1

WinLose = pd.concat([WinnerStats, LoserStats])

TeamStats = WinLose.groupby(['Season', 'TeamID']).sum()

TeamStats

#cols = ['Season', 'TeamID', 'PF', 'PA', 'Loc', 'NumOT',
#'FGM', 'FGA', 'FGM3', 'FGA3', 'FTM', 'FTA', 'OR', 'DR', 'Ast', 'TO',
#'Stl', 'Blk', 'Fls', 'OppFGM', 'OppFGA', 'OppFGM3', 'OppFGA3', 'OppFTM', 'OppFTA', 'OppOR',
#'OppDR', 'OppAst', 'OppTO', 'OppStl', 'OppBlk', 'OppFls']

#Average out the stats for each team and create new stats
TeamStats['Games Played'] = TeamStats['Wins'] + TeamStats['Loses']
TeamStats['Minutes Played'] = TeamStats['Games Played'] * 40 + TeamStats['NumOT'] * 5
TeamStats['Possesions'] = (TeamStats['FGA'] - TeamStats['OR']) + TeamStats['TO'] + (.44 * TeamStats['FTA'])
TeamStats['OppPossesions'] = (TeamStats['OppFGA'] - TeamStats['OppOR']) + TeamStats['OppTO'] + (.44 * TeamStats['OppFTA'])
TeamStats['Tempo'] = TeamStats['Possesions'] / TeamStats['Minutes Played']
TeamStats['Win %'] = TeamStats['Wins'] / TeamStats['Games Played']
TeamStats['PPPos'] = TeamStats['PF'] / TeamStats['Possesions']
TeamStats['PAPPos'] = TeamStats['PA'] / TeamStats['OppPossesions']
TeamStats['PPPos Margin'] = TeamStats['PPPos'] - TeamStats['PAPPos']
TeamStats['FG %'] = TeamStats['FGM'] / TeamStats['FGA']
TeamStats['OppFG%'] = TeamStats['OppFGM'] / TeamStats['OppFGA']
TeamStats['3PT FG %'] = TeamStats['FGM3'] / TeamStats['FGA3']
TeamStats['3PTPPos'] = TeamStats['FGM3'] / TeamStats['Possesions']
TeamStats['FT %'] = TeamStats['FTM'] / TeamStats['FTA']
TeamStats['FTPPos'] = TeamStats['FTM'] / TeamStats['Possesions']
TeamStats['OppFTPPos'] = TeamStats['OppFTM'] / TeamStats['OppPossesions']
TeamStats['ORPPos'] = TeamStats['OR'] / TeamStats['Possesions']
TeamStats['Opp ORPPos'] = TeamStats['OppOR'] / TeamStats['OppPossesions']
TeamStats['DRPG'] = TeamStats['DR'] / TeamStats['Games Played']
TeamStats['REBPG'] = (TeamStats['OR'] + TeamStats['DR']) / TeamStats['Games Played']
TeamStats['REB Margin'] = (TeamStats['REBPG']) - ((TeamStats['OppOR'] + TeamStats['OppDR']) / TeamStats['Games Played'])
TeamStats['True Shooting %'] = (.5 * TeamStats['PF']) / (TeamStats['FGA'] + .475 * TeamStats['FTA'])
TeamStats['Effective FG%'] = (TeamStats['FGM'] + .5 * TeamStats['FGM3']) / TeamStats['FGA']
TeamStats['TOV %'] = TeamStats['TO'] / TeamStats['Possesions']
TeamStats['TOV Forced %'] = (TeamStats['Blk'] + TeamStats['Stl']) / TeamStats['OppPossesions']
TeamStats['Foul Margin'] = (TeamStats['Fls'] - TeamStats['OppFls']) / TeamStats['Games Played']
TeamStats['OppEFG'] = (TeamStats['OppFGM'] + 0.5 * TeamStats['OppFGM3']) / TeamStats['OppFGA']

TeamStats = pd.merge(TeamStats, ConferenceTournamentStats, on=['Season', 'TeamID'])
TeamStats = pd.merge(TeamStats, MasseyRankings, on=['Season', 'TeamID'])

#Get all the previous march madness matchups
TourneyCompact = pd.read_csv('/Users/ramialkadri/Developer/Model/march-machine-learning-mania-2023/MNCAATourneyCompactResults.csv')
seed_dict = Seeds.set_index(['Season', 'TeamID'])
TourneyInput = pd.DataFrame()

winIDs = TourneyCompact['WTeamID']
loseIDs = TourneyCompact['LTeamID']
season = TourneyCompact['Season']

winners = pd.DataFrame()
winners[['Season', 'Team1', 'Team2']] = TourneyCompact[['Season', 'WTeamID', 'LTeamID']]
winners['Result'] = 1

losers = pd.DataFrame()
losers[['Season', 'Team1', 'Team2']] = TourneyCompact[['Season', 'LTeamID', 'WTeamID']]
losers['Result'] = 0

TourneyInput = pd.concat([winners,losers])
TourneyInput = TourneyInput[(TourneyInput['Season'] >= 2003) & (TourneyInput['Season'] != 2022)].reset_index(drop=True)

TeamStats = TeamStats[(TeamStats.index >= (2003, 0)) & (TeamStats.index <= (2022, 9999))]
ConferenceTournamentStats= ConferenceTournamentStats[(ConferenceTournamentStats.index >= (2003, 0)) & (ConferenceTournamentStats.index <= (2022, 9999))] 

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
    try:
        team1score = TeamStats.loc[idx1]
    except KeyError:
        continue
    try:
        team2score = TeamStats.loc[idx2]
    except KeyError:
        continue

    team1score['Seed'] = TourneyInput['Team1Seed'][x]
    team2score['Seed'] = TourneyInput['Team2Seed'][x]
    
    outscore = team1score - team2score
    outscore['Result'] = TourneyInput['Result'][x]
    outscores.append(outscore)

outscores = pd.DataFrame(outscores)


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

X = outscores[outscores.columns[:-1]].values
y = outscores['Result'].values 

np.random.seed(1)
idx = np.random.permutation(len(X))
train_idx = idx[:int(-.2 * len(X))]
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

num_simulations = 100
num_wins = 0

for i in range(num_simulations):
    # Define the matchup
    team1 = 1181
    team2 = 1403
    team1_rankings = TeamStats.loc[(2021, team1)].values.reshape(1, -1) # Reg season stats for team 1
    team2_rankings = TeamStats.loc[(2021, team2)].values.reshape(1, -1) # Reg season stats for team 2
    team1_seed = np.array([2]).reshape(1, -1)
    team2_seed = np.array([3]).reshape(1, -1)
    team1_rankings = np.hstack((team1_rankings, team1_seed))
    team2_rankings = np.hstack((team2_rankings, team2_seed))

    # Combine the features for the two teams
    matchup_features = team1_rankings - team2_rankings
    # Normalize the features
    matchup_features = (matchup_features - mins) / (maxs - mins)
    matchup_features += np.random.normal(scale=0.1, size=matchup_features.shape)
    # Get the predicted outcome
    predicted_outcome = model.predict(matchup_features)[0]

    if predicted_outcome == 1:
        num_wins += 1

win_percentage = num_wins / num_simulations * 100

# Output the teams in the order of predicted win percentage
if win_percentage >= 50:
    print(f"Team 1 wins {win_percentage:.1f}% of the time in {num_simulations} simulations")
else:
    print(f"Team 2 wins {100-win_percentage:.1f}% of the time in {num_simulations} simulations")


