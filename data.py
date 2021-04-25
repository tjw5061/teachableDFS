from tqdm import tqdm
import pandas as pd
import numpy as np
from config import CACHE_DIRECTIORY
from maps import team_map, team_map_2, team_map_inv

class FootballTable(object):
    def __init__(self, name, refresh=False):
        self.name = name
        if refresh:
            self.build()
            self.cache()
        else:
            self.load()

    def cache(self):
        self.table.to_pickle("%s/%s.pkl" % (CACHE_DIRECTIORY, self.name))

    def load(self):
        self.table = pd.read_pickle("%s/%s.pkl" % (CACHE_DIRECTIORY, self.name))

    def build(self, boxscores):
        raise Exception("Override build function")


class FootballBoxscoreTable(object):
    def __init__(self, name, refresh=False, boxscores=None):
        self.name = name
        if refresh:
            if boxscores is None:
                raise Exception("Pass boxscores to refresh the %s table." % self.name)
            self.build(boxscores)
            self.cache()
        else:
            self.load()

    def build(self, boxscores):
        raise Exception("Override build function")

    def cache(self):
        self.table.to_pickle("%s/%s.pkl" % (CACHE_DIRECTIORY, self.name))

    def load(self):
        self.table = pd.read_pickle("%s/%s.pkl" % (CACHE_DIRECTIORY, self.name))

    def query_asof(self, name, date):
        out = self.table[self.table.name == name].copy()
        out = out[out['date'] < date].sort_values('date', ascending=True)
        return out.fillna(0).tail(5).mean(numeric_only=True)


class DefenseTeamTable(FootballBoxscoreTable):
    def __init__(self, refresh=False, boxscores=None):
        self.off_table = OffenseTeamTable(refresh=refresh, boxscores=boxscores)
        self.score_table = ScoreTable(refresh=refresh, boxscores=boxscores)
        super(DefenseTeamTable, self).__init__("defenseTeam", refresh, boxscores)

    def build(self, boxscores):
        off_team_table = self.off_table.table.copy()
        teams = off_team_table.team.unique()
        def_team_table = []
        for team in teams:
            def_team = off_team_table[off_team_table.team == team].copy()[['team', 'date', 'opp']]
            def_table = def_team.join(off_team_table.set_index(['date', 'team']), on=['date', 'opp'], rsuffix=".2")
            def_table = def_table.reset_index()
            del def_table['index']
            del def_table['opp.2']
            def_team_table.append(def_table)
        def_team_table = pd.concat(def_team_table)
        # def_team_table = floatify(def_team_table, string_columns = ['date','name','team','opp',"Time of Possession"])
        del def_team_table['player']
        def_team_table = def_team_table.drop_duplicates(["team", "date"]).sort_values('date')
        def_team_table['name'] = def_team_table['team']  # for asof queries
        score_ref = pd.concat([
            self.score_table.table[['date', 'home', 'home_score']].rename(
                columns={'home': "opp", 'home_score': "pts_allowed"}),
            self.score_table.table[['date', 'away', 'away_score']].rename(
                columns={'away': "opp", 'away_score': "pts_allowed"})]) \
            .sort_values('date') \
            .set_index(['date', 'opp'])
        def_team_table = def_team_table.join(score_ref, on=['date', 'opp'])
        def_team_table['DKScore_pts'] = def_team_table['pts_allowed'].apply(lambda x: self.score_pts_allowed(x))
        def_team_table['DKScore'] = def_team_table['sacks_allowed'] + 2.0 * def_team_table['pass_int'] + \
                                    2.0 * def_team_table['fumbles_lost'] + def_team_table['DKScore_pts']
        self.table = def_team_table

    @staticmethod
    def score_pts_allowed(pts):
        if pts == 0.0:
            return 10.0
        elif pts <= 6:
            return 7.0
        elif pts <= 13:
            return 4.0
        elif pts <= 20:
            return 1.0
        elif pts <= 27.0:
            return 0.0
        elif pts <= 34.0:
            return -1.0
        else:
            return -4.0


class OffenseTeamTable(FootballBoxscoreTable):
    def __init__(self, refresh=False, boxscores=None):
        super(OffenseTeamTable, self).__init__("offenseTeam", refresh, boxscores)

    def build(self, boxscores):
        table = []
        for fbs in tqdm(boxscores):
            table.append(self.team_records_from_boxscore(fbs))
        t = self.build_team_table(pd.concat(table))
        t = floatify(t, string_columns=['date', 'team', 'opp', "Time of Possession"])
        t['name'] = t['team']  # for asof queries
        self.table = t

    @staticmethod
    def team_records_from_boxscore(fbs):
        # Row 1
        h = fbs.all_team_stats['home_stat'].copy()
        h['team'] = team_map_inv[fbs.scorebox['home_team']]
        h['date'] = pd.Timestamp(fbs.scorebox['date'])
        h['opp'] = team_map_inv[fbs.scorebox['away_team']]
        # Row 2
        a = fbs.all_team_stats['vis_stat'].copy()
        a['team'] = team_map_inv[fbs.scorebox['away_team']]
        a['date'] = pd.Timestamp(fbs.scorebox['date'])
        a['opp'] = team_map_inv[fbs.scorebox['home_team']]
        t = pd.concat([h, a], axis=1).T
        return t

    @staticmethod
    def build_team_table(team_table):
        team_table["Time of Possession"] = team_table["Time of Possession"] \
            .apply(lambda x: float(x.split(":")[0]) + float(x.split(":")[1]) / 60)
        t1 = team_table["Cmp-Att-Yd-TD-INT"].apply(
            lambda x: pd.Series(dict(zip(["pass_cmp", "pass_att", "pass_yd", "pass_td", "pass_int"],
                                         x.split("-")))))
        t2 = team_table["Fourth Down Conv."].apply(lambda x: pd.Series(dict(zip(["fouth_conv_succ", "fouth_conv_att"],
                                                                                x.split("-")))))
        t3 = team_table["Fumbles-Lost"].apply(lambda x: pd.Series(dict(zip(["fumbles", "fumbles_lost"],
                                                                           x.split("-")))))
        t4 = team_table["Penalties-Yards"].apply(lambda x: pd.Series(dict(zip(["penalty_count", "penalty_yds"],
                                                                              x.split("-")))))
        t5 = team_table["Rush-Yds-TDs"].apply(lambda x: pd.Series(dict(zip(["rush_att", "rush_yds", "rush_tds"],
                                                                           x.split("-")))))
        t6 = team_table["Sacked-Yards"].apply(lambda x: pd.Series(dict(zip(["sacks_allowed", "sacks_allowed_yds"],
                                                                           x.split("-")))))
        t7 = team_table["Third Down Conv."].apply(lambda x: pd.Series(dict(zip(["third_conv_succ", "third_conv_att"],
                                                                               x.split("-")))))
        t8 = team_table[
            ["First Downs", "Net Pass Yards", "Time of Possession", 'Total Yards', 'Turnovers', 'team', 'date', 'opp']]
        return pd.concat([t1, t2, t3, t4, t5, t6, t7, t8], axis=1)


class OffenseTable(FootballBoxscoreTable):
    def __init__(self, refresh=False, boxscores=None):
        super(OffenseTable, self).__init__("offense", refresh, boxscores)

    def build(self, boxscores):
        table = []
        for fbs in tqdm(boxscores):
            record = floatify(fbs.all_player_offense.copy())
            record['date'] = pd.Timestamp(fbs.scorebox['date'])
            teams = record.team.unique()
            record['opp'] = record.team.apply(lambda x: teams[1] if x == teams[0] else teams[0])
            table.append(record)
        table = pd.concat(table)
        table['name'] = table.player.str.upper().str.replace(" ", "")
        table['DKScore'] = (table.pass_td * 4.0) + \
                           (0.04 * table.pass_yds) + \
                           (3.0 * (table.pass_yds > 300.)) + \
                           (-1.0 * table.pass_int) + \
                           (6.0 * table.rush_td) + \
                           (0.1 * table.rush_yds) + \
                           (3.0 * (table.rush_yds > 100.)) + \
                           (6.0 * (table.rec_td)) + \
                           (0.1 * (table.rec_yds)) + \
                           (3.0 * (table.rec_yds > 100.)) + \
                           table.rec + \
                           (-1 * table.fumbles_lost)
        table.sort_values('date')
        self.table = table


class AdvancedPassingTable(FootballBoxscoreTable):
    def __init__(self, refresh=False, boxscores=None):
        super(AdvancedPassingTable, self).__init__("advancedPasing", refresh, boxscores)

    def build(self, boxscores):
        table = []
        for fbs in tqdm(boxscores):
            t = floatify(fbs.adv_player_passing.copy())
            t['date'] = pd.Timestamp(fbs.scorebox['date'])
            table.append(t)
        self.table = pd.concat(table)
        self.table['name'] = self.table.player.str.upper().str.replace(" ", "")


class AdvancedRushingTable(FootballBoxscoreTable):
    def __init__(self, refresh=False, boxscores=None):
        super(AdvancedRushingTable, self).__init__("advancedRushing", refresh, boxscores)

    def build(self, boxscores):
        table = []
        for fbs in tqdm(boxscores):
            t = floatify(fbs.adv_player_passing.copy())
            t['date'] = pd.Timestamp(fbs.scorebox['date'])
            table.append(t)
        self.table = pd.concat(table)
        self.table['name'] = self.table.player.str.upper().str.replace(" ", "")


class AdvancedRecievingTable(FootballBoxscoreTable):
    def __init__(self, refresh=False, boxscores=None):
        super(AdvancedRecievingTable, self).__init__("advancedRecieving", refresh, boxscores)

    def build(self, boxscores):
        table = []
        for fbs in tqdm(boxscores):
            t = floatify(fbs.adv_player_passing.copy())
            t['date'] = pd.Timestamp(fbs.scorebox['date'])
            table.append(t)
        self.table = pd.concat(table)
        self.table['name'] = self.table.player.str.upper().str.replace(" ", "")


class ScoreTable(FootballBoxscoreTable):
    def __init__(self, refresh=False, boxscores=None):
        super(ScoreTable, self).__init__("score", refresh, boxscores)

    def build(self, boxscores):
        table = []
        for fbs in tqdm(boxscores):
            table.append(pd.Series({"home": team_map_inv[fbs.scorebox['home_team']],
                                    "away": team_map_inv[fbs.scorebox['away_team']],
                                    "home_score": fbs.scorebox['home_team_score'],
                                    "away_score": fbs.scorebox['away_team_score'],
                                    "date": pd.Timestamp(fbs.scorebox['date'])}))
        self.table = pd.concat(table, axis=1).T

def floatify(table, string_columns=['team']):
    off_float = table[[c for c in table.columns if not c in string_columns]] \
        .applymap(lambda x: np.nan if x == "" else float(x.replace("%", ""))).copy()
    off_string = table[string_columns].copy()
    table = pd.concat([off_string, off_float], axis=1)
    table = table.reset_index().rename(columns={'index': 'player'})
    return table
