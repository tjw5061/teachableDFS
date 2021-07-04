import requests
import numpy as np
import cvxpy as cp
import pandas as pd

from tqdm import tqdm
from bs4 import BeautifulSoup

from maps import team_map_2
from data import ReferenceTable
from model import QuarterbackFeatureSpaceTable, PositionPlayerFeatureSpaceTable, DefenseFeatureSpaceTable
from model import FootballRandomForestModel
from config import PROJECT_DIRECTORY

class HistoricalSalaryTable(ReferenceTable):
    """ Data standardization class for historic player salaries """
    def __init__(self, seasons, refresh=True,  max_year=2020, max_week=11):
        """ 
            Required Inputs:
                seasons: list of years of seasons to pull historic salaries for
            Optional Inputs:
                refresh: Boolean determining if payout table should be refreshed/built 
        """
        if max_week:
            assert max_year
        if max_year:
            assert max_week
        
        self.url_args = [
            (i +1, str(y)) for y in seasons for i in range(17) if not (( i + 1 > max_week) and (y==max_year))
        ]
        super(HistoricalSalaryTable, self).__init__("historicalSalary", refresh)

    def build(self):
        """ Pings the rotoguru site for each week within a season. Extracts salary data for players and stores in a 
        dataframe. """

        out = []
        for wk, yr in tqdm(self.url_args):
            resp = requests.get(f"http://rotoguru1.com/cgi-bin/fyday.pl?week={wk}&year={yr}&game=dk&scsv=1")
            soup = BeautifulSoup(resp.text, "html.parser")
            rows = [ln.split(";") for ln in soup.find('pre').text.split("\n")]
            sal = pd.DataFrame(rows[1:], columns = rows[0])
            sal = sal.rename(columns={'Oppt': 'opp', 'Team': 'team', 'Pos': 'pos'})
            sal.team = sal.team.str.upper()
            sal.opp  = sal.opp.str.upper()
            sal = sal[sal.Week != ""]
            sal['name'] = sal.apply(
                lambda x: (x['Name'].split(',')[1] + x['Name'].split(',')[0]).upper()
                if len(x['Name'].split(',')) >1 else x['team'], axis=1
            )
            sal['name'] = sal['name'].str.replace(" ", "")
            sal = sal.rename(columns={'Week': 'week', "Year": 'year'})
            sal.week = sal.week.astype(int)
            sal.year = sal.year.astype(int)
            out.append(sal)
        self.table = pd.concat(out)


class PayoutTable(ReferenceTable):
    """ Data standardization class for competition payouts """
    def __init__(self, refresh=False):
        """ 
            Optional Inputs:
                refresh: Boolean determining if payout table should be refreshed/built 
        """
        super(PayoutTable, self).__init__("payoutTable", refresh=refresh)

    def build(self):
        """ Reads the payout text file, cleans up data such that the output is a two column dataframe with a minimum
        ranking in first column and the expected payout in the second column """

        with open(f'{PROJECT_DIRECTORY}/ref/payout.txt' , 'r') as reader:
            payout = reader.read()
        payFrame = pd.DataFrame(np.vstack([payout.split('\n')[::2], payout.split('\n')[1::2]])).T
        payFrame[0] = payFrame[0].apply(lambda x: x[0] if len(x.split(" - ")) < 2 else x.split(" - ")[1])

        payFrame[0] = payFrame[0].str.replace("th", "", regex=False).str.replace(",", "", regex=False).astype(float)
        payFrame[1] = payFrame[1].str.replace("$", "", regex=False).str.replace(",", "", regex=False).astype(float)
        payFrame = pd.concat([payFrame, pd.DataFrame({0: {0: 1e6, 1: 0.0}}).T])
        self.table = payFrame


class BacktestLinksTable(ReferenceTable):
    """ Data Standardization class for the results table """
    def __init__(self, refresh=False):
        super(BacktestLinksTable, self).__init__("backtestLinksTable", refresh=refresh)

    def build(self):
        """ Reads the results links file, cleans up columns and stores as a dataframe. """
        links = pd.read_csv(f"{PROJECT_DIRECTORY}/ref/ResultLinks.csv")
        links['gameid'] = links['Link'].apply(lambda x: x.split("/")[-1])
        links = links.rename(columns={"Week": 'week', "Date": 'date'})
        links['date'] = links['date'].apply(lambda t: pd.Timestamp(t))
        self.table = links


class BacktestStandingsTable(ReferenceTable):
    """ Data Standardization class for the contest standings table  """

    def __init__(self, refresh=True):
        self.payoutTable = PayoutTable(refresh=True)
        self.backtestTable = BacktestLinksTable(refresh=True)
        super(BacktestStandingsTable, self).__init__("historicalStandings", refresh=refresh)

    def build(self):
        """ Pulls the contest standings csv for each game within the BacktestLinksTable. Standardizes the data and then
        filters such that only entries at each payout level remain. """

        out = []
        for _, link_row in tqdm(self.backtestTable.table.iterrows()):
            path = f"{PROJECT_DIRECTORY}/ref/Results/contest-standings-{link_row['gameid']}.csv"
            contest = pd.read_csv(path, low_memory=False)
            standings = contest.iloc[:, :6]
            standings = standings.dropna(subset=['Lineup'])
            standings['date'] = link_row['date']
            standings['week'] = link_row['week']
            # Filter
            standings = pd.concat(
                [standings[standings['Rank'] <= r].tail(1) for r in self.payoutTable.table[0].values]
            )
            out.append(standings)
        self.table = pd.concat(out)


class DoubleupStandingsTable(ReferenceTable):
    """ Data Standardization class for the double-up contest standings table  """

    def __init__(self, refresh=True):
        self.backtestTable = BacktestLinksTable(refresh=True)
        super(DoubleupStandingsTable, self).__init__("doubleupStandings", refresh=refresh)

    def build(self):
        """ Pulls the contest standings csv for each game within the BacktestLinksTable. Standardizes the data and then
        filters such only the entry that divides the competition between the top 40% and bottom 60% remains. """

        out = []
        for _, link_row in tqdm(self.backtestTable.table.iterrows()):
            path = f"{PROJECT_DIRECTORY}/ref/Results/contest-standings-{link_row['gameid']}.csv"
            contest = pd.read_csv(path, low_memory=False)
            standings = contest.iloc[:, :6]
            standings = standings.dropna(subset=['Lineup'])
            standings['date'] = link_row['date']
            standings['week'] = link_row['week']
            # Filter
            num_entries = standings['Rank'].tail(1).values[0]
            cutoff = int(num_entries * 0.4)
            standings = standings[standings['Rank'] <= cutoff].tail(1)
            out.append(standings)
        self.table = pd.concat(out)


class BacktestPlayerPerformanceTable(ReferenceTable):
    """ Data Standardization class for the player salary tables """

    def __init__(self, seasons, refresh=True):
        self.backtestTable = BacktestLinksTable(refresh=refresh)
        self.histSalaryTable = HistoricalSalaryTable(seasons=seasons, refresh=refresh)
        super(BacktestPlayerPerformanceTable, self).__init__("historicalPerformance", refresh=refresh)

    def build(self):
        """ Cycles through historic competitions. For each competition loads the contest standings data and extracts
        player salary data. Standarizes the data such that it can be matched on player names. """
        out = []
        for _, link_row in tqdm(self.backtestTable.table.iterrows()):
            path = f"{PROJECT_DIRECTORY}/ref/Results/contest-standings-{link_row['gameid']}.csv"
            contest = pd.read_csv(path, low_memory=False)
            results = contest.iloc[:, 7:].dropna()
            results['Player'] = results['Player'].apply(lambda x: x.replace(" ", ""))
            results['name'] = results.apply(
                lambda x: team_map_2[x['Player']] if x['Roster Position'] == "DST" else x['Player'],
                axis=1
            )
            results['name'] = results['name'].apply(lambda x: x.replace(" ", "").upper())
            results['date'] = link_row['date']
            results['year'] = results['date'].apply(lambda t: t.year)
            results = results.join(self.backtestTable.table.set_index('date')['week'], on='date')
            results = results.join(
                self.histSalaryTable.table.set_index(['week', 'year', 'name']), on=['week', 'year', 'name']
            )
            out.append(results)
        self.table = pd.concat(out)


class BacktestPredictionsTable(ReferenceTable):
    """ Model training and prediction generation class"""
    def __init__(self, seasons, refresh=True):
        self.seasons = seasons
        self.btPerf = BacktestPlayerPerformanceTable(seasons=seasons, refresh=refresh)
        super(BacktestPredictionsTable, self).__init__("backtestPredictions", refresh=refresh)

    def build(self, matchups=None):
        """ Takes an optional dataframe of matchups, otherwise uses the player performance table. For each week of
        matches trains a model on all possible previous weeks. That model is then used to make the upcoming weeks
        predictions. Stores predictions in a multi-index dataframe."""

        if matchups is None:
            matchups = self.btPerf.table
        out = {}
        for nm, players in matchups.groupby(['year', 'week']):
            qb_matchups = players[players['Roster Position'] == 'QB'][['name', 'date', 'opp']].copy()
            df_matchups = players[players['Roster Position'] == 'DST'][['name', 'date', 'opp']].copy()
            pp_matchups = players[
                (players['Roster Position'] != 'DST') & (players['Roster Position'] != 'QB')][['name', 'date', 'opp']
            ].copy()

            date = players.date.iloc[0]

            qb_fs = QuarterbackFeatureSpaceTable(seasons=self.seasons, refresh=False)
            pp_fs = PositionPlayerFeatureSpaceTable(seasons=self.seasons, refresh=False)
            df_fs = DefenseFeatureSpaceTable(seasons=self.seasons, refresh=False)

            # Select a feature space before the date
            qb_train = qb_fs.table[qb_fs.table.date < date].copy()
            pp_train = pp_fs.table[pp_fs.table.date < date].copy()
            df_train = df_fs.table[df_fs.table.date < date].copy()

            qb_model = FootballRandomForestModel(qb_train)
            pp_model = FootballRandomForestModel(pp_train)
            df_model = FootballRandomForestModel(df_train)
            qb_model.train()
            pp_model.train()
            df_model.train()

            # Build a feature space with matchups on the date 
            qb_fs.build(matchups=qb_matchups, add_y=False)
            qb_test = qb_fs.table
            pp_fs.build(matchups=pp_matchups, add_y=False)
            pp_test = pp_fs.table
            df_fs.build(matchups=df_matchups, add_y=False)
            df_test = df_fs.table
            qb_pred = qb_model.predict(qb_test)
            pp_pred = pp_model.predict(pp_test)
            df_pred = df_model.predict(df_test)
            predictions = pd.concat([
                pd.Series(dict(zip(qb_test['name'].values, qb_pred))),
                pd.Series(dict(zip(pp_test['name'].values, pp_pred))),
                pd.Series(dict(zip(df_test['name'].values, df_pred)))])
            out[nm] = predictions
        self.table = pd.DataFrame(out).stack().stack(0)


def run_doubleup_backtest(stack_tuple, results, histStandings):
    """ Takes a tuple of the form (total teams to stack from, total player on each team to stack). Additionally takes
    results dataframe which contains all of the player data relevant to the competition (salary, predicted points,
    position, etc.) and a historic standings dataframe which contains the total points required by a lineup to receive
    a payout. Identifies the best candidates for stacking for each team (WR, TE and FLEX players) and the best QB for
    each team. Then identifies the best teams to perform stacking for by identifying the highest sum of predicted
    values for QB and stacking candidates.
    """
    num_teams_to_stack = stack_tuple[0]
    num_players_in_stack = stack_tuple[1]
    
    # get payout row for week
    standings = histStandings[histStandings.week == results.week.iloc[0]]

    pos_stack = results[results['Roster Position'].apply(lambda y: not y in ['QB', 'DST', 'RB'])] \
        .groupby('team') \
        .apply(lambda grp: grp.sort_values('pred') \
               .tail(num_players_in_stack - 1)) \
        .reset_index(drop=True)
    qb_stack = results[results['Roster Position'] == 'QB'].groupby('team') \
        .apply(lambda grp: grp.sort_values('pred') \
               .tail(1)) \
        .reset_index(drop=True)
    stacks = pd.concat([qb_stack, pos_stack]).sort_values('team')
    stacks['DK salary'] = stacks['DK salary'].astype(float)
    stacked_teams_frame = stacks.groupby('team').sum().sort_values('pred')
    stacked_teams = stacked_teams_frame.index.values[-num_teams_to_stack:]
    S = results['DK salary'].values
    E = results['pred'].values
    # T = results['team'].values
    P = results['pos'].values
    N = results['name'].values
    # SP = results['%Drafted'].str.replace("%", "").astype(float).div(1e2).values
    total = {}
    for team in stacked_teams:
        prior_lineups = []
        for _ in range(int(20.0 / num_teams_to_stack)):
            X = cp.Variable(len(E), boolean=True)
            constraints = [
                sum(X) == 9,
                X @ S <= 50000,
                X @ (P == "QB").astype(float) == 1,
                X @ (P == "TE").astype(float) >= 1,
                X @ (P == "RB").astype(float) >= 2,
                X @ (P == "WR").astype(float) >= 3,
                X @ (P == "Def").astype(float) == 1]
            for n in stacks[stacks.team == team].name.values:
                constraints.append(X[np.where(N == n)[0][0]] == 1)
            for lineups in prior_lineups:
                constraints.append(X @ lineups <= 8)
            obj = cp.Maximize(X @ E)
            prob = cp.Problem(obj, constraints)
            prob.solve()
            try:
                prior_lineups.append(X.value.copy())
            except AttributeError:
                print("Problem with %s" % team)
                return -1000
        realized_scores = np.sort([results[l.astype(bool)]['FPTS'].sum() for l in prior_lineups])
        cutoff_score = standings['Points'].values[0]
        payouts = [20.0 if r > cutoff_score else -20.0 for r in realized_scores]
        total[team] = sum(payouts)
    return pd.Series(total).sum()