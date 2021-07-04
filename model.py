import pandas as pd
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from data import FootballTable, OffenseTable, OffenseTeamTable, DefenseTeamTable
from data import AdvancedPassingTable, AdvancedRushingTable, AdvancedReceivingTable
from config import SEASON_START_DATES

class FootballRandomForestModel(object):
    """ Primary model training class """

    def __init__(self, train):
        """
        Required Inputs: 
            train: dataframe of training data
        """
        X, Y, C = self.parse(train)
        self.X = X
        self.Y = Y
        self.C = C

    @staticmethod
    def parse(data):
        """ Takes a dataframe. Identifies the training columns, the target column, and the names of the training
        columns. Returns those as X, Y and C respectively. """
        Y = None
        X = data.fillna(0)
        if 'Y' in X.columns:
            Y = X['Y']
            del X['Y']
        del X['name']
        del X['date']
        del X['opp']
        C = X.columns
        X = X.values
        return X, Y, C

    def train(self):
        """ Invokes the fit method of the random forest model. """
        self.rfr = RandomForestRegressor(n_estimators=100, random_state=0)
        self.rfr.fit(self.X, self.Y)

    def predict(self, test):
        """ Takes a dataframe. Identifies the training columns, the target column, and the names of the training
        columns. Generates predictions from the training columns. """
        X, _, _ = self.parse(test)
        return self.rfr.predict(X)

class QuarterbackFeatureSpaceTable(FootballTable):
    """ Class for generating feature spaces for quarterbacks. Feature spaces are derived from the OffenseTable, 
    DefenseTeamTable, and AdvancedPassingTable. """

    def __init__(self, seasons, refresh=True):

        self.offense_table = pd.concat([OffenseTable(season).table for season in seasons])
        self.defense_table = pd.concat([DefenseTeamTable(season).table for season in seasons])
        self.adv_passing_table = pd.concat([AdvancedPassingTable(season).table for season in seasons])
        self.feature_space_start = pd.Timestamp(SEASON_START_DATES[min(seasons)])
        super(QuarterbackFeatureSpaceTable, self).__init__("QuarterbackFeatureSpaceTable", seasons, refresh)

    def build(self, matchups=None, add_y=True):
        """ Takes an optional matchups dataframe of player-games to generate feature spaces for. If matchups is not
        passed a generic dataframe is derived from the offensive table. Takes an optional boolean add_y argument which
        determines if the target variable should be appended to the feature space. Primarily builds a feature space
        from the offense_table, defense_table, and adv_passing_table, ensuring that only historic stats from before
        the game date are used.
        """

        if matchups is None:
            matchups = self.offense_table[self.offense_table.pass_att > 10].copy()
            matchups = matchups[['name', 'date', 'opp', 'DKScore']]
            matchups = matchups[matchups.date > self.feature_space_start]

        records = []
        for _, x in tqdm(matchups.iterrows()):
            reg = self.query_asof(self.offense_table, x['name'], x['date'])
            adv = self.query_asof(self.adv_passing_table, x['name'], x['date'])
            offense_record = pd.concat([reg, adv])
            defense_record = self.query_asof(self.defense_table, x['opp'], x['date'])
            offense_record.index = ["o_" + i for i in offense_record.index]
            defense_record.index = ["d_" + i for i in defense_record.index]
            full_rec = pd.concat([offense_record, defense_record])
            full_rec = full_rec.reset_index() \
                .drop_duplicates('index') \
                .set_index('index') \
                .sort_values('index', ascending=False)
            records.append(full_rec)
        self.table = pd.concat(records, axis=1).T

        if add_y:
            self.table['Y'] = matchups['DKScore'].values
        self.table['name'] = matchups['name'].values
        self.table['date'] = matchups['date'].values
        self.table['opp'] = matchups['opp'].values


class PositionPlayerFeatureSpaceTable(FootballTable):
    """ Class for generating feature spaces for position players. Feature spaces are derived from the OffenseTable, 
    DefenseTeamTable, AdvancedRushingTable and AdvancedReceivingTable. """

    def __init__(self, seasons, refresh=True):

        self.offense_table = pd.concat([OffenseTable(season).table for season in seasons])
        self.defense_table = pd.concat([DefenseTeamTable(season).table for season in seasons])
        self.adv_rush_table = pd.concat([AdvancedRushingTable(season).table for season in seasons])
        self.adv_recv_table = pd.concat([AdvancedReceivingTable(season).table for season in seasons])
        self.feature_space_start = pd.Timestamp(SEASON_START_DATES[min(seasons)])
        super(PositionPlayerFeatureSpaceTable, self).__init__("PositionPlayerFeatureSpaceTable", seasons, refresh)

    def build(self, matchups=None, add_y=True):
        """ Takes an optional matchups dataframe of player-games to generate feature spaces for. If matchups is not
        passed a generic dataframe is derived from the offensive table. Takes an optional boolean add_y argument which
        determines if the target variable should be appended to the feature space. Primarily builds a feature space
        from the offense_table, defense_table, adv_rush_table, and adv_recv_table, ensuring that only historic stats 
        from before the game date are used.
        """

        if matchups is None:
            matchups = self.offense_table[self.offense_table.pass_att <= 1].copy()
            matchups = matchups[['name', 'date', 'opp', 'DKScore']]
            matchups = matchups[matchups.date > self.feature_space_start]
        records = []

        for _, x in tqdm(matchups.iterrows()):
            reg = self.query_asof(self.offense_table, x['name'], x['date'])
            adv_rush = self.query_asof(self.adv_rush_table, x['name'], x['date'])
            adv_recv = self.query_asof(self.adv_recv_table, x['name'], x['date'])
            offense_record = pd.concat([reg, adv_rush, adv_recv])
            defense_record = self.query_asof(self.defense_table, x['opp'], x['date'])
            offense_record.index = ["o_" + i for i in offense_record.index]
            defense_record.index = ["d_" + i for i in defense_record.index]
            full_rec = pd.concat([offense_record, defense_record])
            full_rec = full_rec.reset_index() \
                .drop_duplicates('index') \
                .set_index('index') \
                .sort_values('index', ascending=False)
            records.append(full_rec)

        self.table = pd.concat(records, axis=1).T
        if add_y:
            self.table['Y'] = matchups['DKScore'].values
        self.table['name'] = matchups['name'].values
        self.table['date'] = matchups['date'].values
        self.table['opp'] = matchups['opp'].values


class DefenseFeatureSpaceTable(FootballTable):
    def __init__(self, seasons, refresh=True,):
        """ Class for generating feature spaces for a team's defence. Feature spaces are derived from the 
        OffenseTeamTable, and DefenseTeamTable. """

        self.offense_table = pd.concat([OffenseTeamTable(season).table for season in seasons])
        self.defense_table = pd.concat([DefenseTeamTable(season).table for season in seasons])
        self.feature_space_start = pd.Timestamp(SEASON_START_DATES[min(seasons)])
        super(DefenseFeatureSpaceTable, self).__init__("DefenseFeatureSpaceTable", seasons, refresh)

    def build(self, matchups=None, add_y=True):
        """ Takes an optional matchups dataframe of player-games to generate feature spaces for. If matchups is not
        passed a generic dataframe is derived from the offensive table. Takes an optional boolean add_y argument which
        determines if the target variable should be appended to the feature space. Primarily builds a feature space
        from the offense_table, and defense_table.
        """

        if matchups is None:
            matchups = self.defense_table.copy()
            matchups = matchups[['name', 'date', 'opp', 'DKScore']]
            matchups = matchups[matchups.date > self.feature_space_start]

        records = []
        for _, x in tqdm(matchups.iterrows()):
            team_defense_record = self.query_asof(self.defense_table, x['name'], x['date'])
            opp_offense_record = self.query_asof(self.offense_table, x['opp'], x['date'])
            team_defense_record.index = ["teamDef_" + i for i in team_defense_record.index]
            opp_offense_record.index = ["oppOff_" + i for i in opp_offense_record.index]
            full_rec = pd.concat([team_defense_record, opp_offense_record])
            full_rec = full_rec.reset_index() \
                .drop_duplicates('index') \
                .set_index('index') \
                .sort_values('index', ascending=False)
            records.append(full_rec)
        self.table = pd.concat(records, axis=1).T

        if add_y:
            self.table['Y'] = matchups['DKScore'].values
        self.table['name'] = matchups['name'].values
        self.table['date'] = matchups['date'].values
        self.table['opp'] = matchups['opp'].values