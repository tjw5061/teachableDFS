import pandas as pd
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from data import FootballTable, OffenseTable, OffenseTeamTable, DefenseTeamTable
from data import AdvancedPassingTable, AdvancedRushingTable, AdvancedRecievingTable

class FootballRandomForestModel(object):
    def __init__(self, train):
        X ,Y ,C = self.parse(train)
        self.X = X
        self.Y = Y
        self.C = C

    @staticmethod
    def parse(data):
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
        return X ,Y ,C

    def train(self):
        self.rfr = RandomForestRegressor(n_estimators=100, max_depth=5, oob_score=True)
        self.rfr.fit(self.X, self.Y)

    def predict(self, test):
        X ,Y ,C = self.parse(test)
        return self.rfr.predict(X)

class QuarterbackFeatureSpaceTable(FootballTable):
    def __init__(self, refresh=True, feature_space_start=pd.Timestamp("2019.10.15")):
        self.offense_table = OffenseTable()
        self.defense_table = DefenseTeamTable()
        self.adv_passing_table = AdvancedPassingTable()
        self.feature_space_start = feature_space_start
        super(QuarterbackFeatureSpaceTable, self).__init__("QuarterbackFeatureSpaceTable", refresh)

    def build(self, matchups=None, add_y=True):
        if matchups is None:
            matchups = self.offense_table.table[self.offense_table.table.pass_att > 10].copy()
            matchups = matchups[['name', 'date', 'opp', 'DKScore']]
            matchups = matchups[matchups.date > self.feature_space_start]
        records = []
        for ix, x in tqdm(matchups.iterrows()):
            reg = self.offense_table.query_asof(x['name'], x['date'])
            adv = self.adv_passing_table.query_asof(x['name'], x['date'])
            offense_record = pd.concat([reg, adv])
            defense_record = self.defense_table.query_asof(x['opp'], x['date'])
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
    def __init__(self, refresh=True, feature_space_start=pd.Timestamp("2019.10.15")):
        self.offense_table = OffenseTable()
        self.defense_table = DefenseTeamTable()
        self.adv_rush_table = AdvancedRushingTable()
        self.adv_recv_table = AdvancedRecievingTable()
        self.feature_space_start = feature_space_start
        super(PositionPlayerFeatureSpaceTable, self).__init__("PositionPlayerFeatureSpaceTable", refresh)

    def build(self, matchups=None, add_y=True):
        if matchups is None:
            matchups = self.offense_table.table[self.offense_table.table.pass_att <= 1].copy()
            matchups = matchups[['name', 'date', 'opp', 'DKScore']]
            matchups = matchups[matchups.date > self.feature_space_start]
        records = []
        for ix, x in tqdm(matchups.iterrows()):
            reg = self.offense_table.query_asof(x['name'], x['date'])
            adv_rush = self.adv_rush_table.query_asof(x['name'], x['date'])
            adv_recv = self.adv_recv_table.query_asof(x['name'], x['date'])
            offense_record = pd.concat([reg, adv_rush, adv_recv])
            defense_record = self.defense_table.query_asof(x['opp'], x['date'])
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
    def __init__(self, refresh=True, feature_space_start=pd.Timestamp("2019.10.15")):
        self.offense_table = OffenseTeamTable()
        self.defense_table = DefenseTeamTable()
        self.feature_space_start = feature_space_start
        super(DefenseFeatureSpaceTable, self).__init__("DefenseFeatureSpaceTable", refresh)

    def build(self, matchups=None, add_y=True):
        if matchups is None:
            matchups = self.defense_table.table.copy()
            matchups = matchups[['name', 'date', 'opp', 'DKScore']]
            matchups = matchups[matchups.date > self.feature_space_start]
        records = []
        for ix, x in tqdm(matchups.iterrows()):
            team_defense_record = self.defense_table.query_asof(x['name'], x['date'])
            opp_offense_record = self.offense_table.query_asof(x['opp'], x['date'])
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