import os
import pandas as pd
import numpy as np

from tqdm import tqdm

from config import CACHE_DIRECTORY, DailyFantasyDataScienceError
from maps import team_map_inv


class FootballTable(object):
    """ Archetypal class for feature spaces. Contains functionality that is useful for all downstream classes """

    def __init__(self, name, seasons, refresh=False):
        """
            Required Inputs: 
                name: Name of the feature space
                seasons: List of season to load feature sets for
            Optional Inputs:
                refresh: Boolean determining if the feature space should be refreshed/built
        """

        self.name = name
        self.seasons = seasons

        if refresh:
            self.build()
            self.cache()
        else:
            self.load()

    def cache(self):
        """ Stores the feature space object as a pickle in the corresponding season folder """
        try:
            self.table.to_pickle(f"{CACHE_DIRECTORY}/{self.name}.pkl")
        except FileNotFoundError:
            raise DailyFantasyDataScienceError()

    def load(self):
        """ Loads a feature space object as a pickle in the corresponding season folder """
        try:
            self.table = pd.read_pickle(f"{CACHE_DIRECTORY}/{self.name}.pkl")
        except FileNotFoundError:
            raise DailyFantasyDataScienceError()

    def build(self):
        """ Placeholder build function"""
        raise Exception("Override build function")

    @staticmethod
    def query_asof(table, name, date):
        """ Filters the table to only rows that match the name argument and only rows where the game took place before
        the date argument. Returns a dataframe of the mean value of each column from the five most recent games. """

        out = table[table.name == name].copy()
        out = out[out['date'] < date].sort_values('date', ascending=True)
        return out.fillna(0).tail(5).mean(numeric_only=True)


class ReferenceTable(object):
    """ Archetypal class for reference data. Contains functionality that is useful for all downstream classes """

    def __init__(self, name, refresh=False):
        """
        Required Inputs: 
            name: Name of the feature space
        Optional Inputs:
            refresh: Boolean determining if the reference tables should be refreshed/built
        """

        self.name = name
        if refresh:
            self.build()
            self.cache()
        else:
            self.load()

    def cache(self):
        """ Stores the reference object as a pickle in the corresponding season folder """ 
        try:
            self.table.to_pickle(f"{CACHE_DIRECTORY}/{self.name}.pkl")
        except FileNotFoundError:
            raise DailyFantasyDataScienceError()

    def load(self):
        """ Loads a reference object as a pickle in the corresponding season folder """ 
        try:
            self.table = pd.read_pickle(f"{CACHE_DIRECTORY}/{self.name}.pkl")
        except FileNotFoundError:
            raise DailyFantasyDataScienceError()

    def build(self):
        """ Placeholder build function"""
        raise Exception("Override build function")


class FootballBoxscoreTable(object):
    """ Archetypal class for stat tables. Contains functionality that is useful for all downstream classes """
    
    def __init__(self, name, season, refresh=False, boxscores=None):
        """
            Required Inputs: 
                name: Name of the table
                season: Year of the season
            Optional Inputs:
                refresh: Boolean determining if the table should be refreshed/built
                boxscores: List of FootballBoxscore  objects
        """
        self.name = name
        self.season = season
        
        if refresh:
            if boxscores is None:
                raise Exception(f"Pass boxscores to refresh the {self.name} table.")
            self.build(boxscores)
            self.cache()
        else:
            self.load()

    def build(self, boxscores):
        """ Placeholder build function"""
        raise Exception("Override build function")

    def cache(self):
        """ Stores the table object as a pickle in the corresponding season folder """ 
        if not os.path.exists(f"{CACHE_DIRECTORY}/{self.season}"):
            os.makedirs(f"{CACHE_DIRECTORY}/{self.season}")
        try:  
            self.table.to_pickle(f"{CACHE_DIRECTORY}/{self.season}/{self.name}.pkl")
        except FileNotFoundError:
            raise DailyFantasyDataScienceError()

    def load(self):
        """ loads the table object from a pickle in the corresponding season folder """
        try:
            self.table = pd.read_pickle(f"{CACHE_DIRECTORY}/{self.season}/{self.name}.pkl")
        except FileNotFoundError:
            raise DailyFantasyDataScienceError()


class DefenseTeamTable(FootballBoxscoreTable):
    """ Table that stores team-level offense data. """

    def __init__(self, season, refresh=False, boxscores=None):
        self.off_table = OffenseTeamTable(season=season, refresh=refresh, boxscores=boxscores)
        self.score_table = ScoreTable(season=season, refresh=refresh, boxscores=boxscores)
        super(DefenseTeamTable, self).__init__("defenseTeam", season, refresh, boxscores)

    def build(self, boxscores):
        """ Takes components from the ScoreTable and OffenseTeamTable to create a new table, converts to a dataframe '
        """

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
        del def_team_table['player']
        def_team_table = def_team_table.drop_duplicates(["team", "date"]).sort_values('date')
        def_team_table['name'] = def_team_table['team']  # for asof queries

        inverted_home_table = self.score_table.table[['date', 'home', 'home_score']].rename(
            columns={'home': "opp", 'home_score': "pts_allowed"}
        )
        inverted_away_table = self.score_table.table[['date', 'away', 'away_score']].rename(
                columns={'away': "opp", 'away_score': "pts_allowed"}
            )
        
        score_ref = pd.concat(
            [inverted_home_table, inverted_away_table] 
        ).sort_values('date').set_index(['date', 'opp'])

        def_team_table = def_team_table.join(score_ref, on=['date', 'opp'])

        def_team_table['DKScore_pts'] = def_team_table['pts_allowed'].apply(lambda x: self.score_pts_allowed(x))
        def_team_table['DKScore'] = (
            def_team_table['sacks_allowed'] 
            + 2.0 * def_team_table['pass_int'] 
            + 2.0 * def_team_table['fumbles_lost'] 
            + def_team_table['DKScore_pts']
        )
        self.table = def_team_table

    @staticmethod
    def score_pts_allowed(pts):
        """ Simple lookup function to handle fantasy points related to points allowed by a defense """
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
    """ Table that stores team-level offense data. """

    def __init__(self, season, refresh=False, boxscores=None):
        super(OffenseTeamTable, self).__init__("offenseTeam", season, refresh, boxscores)

    def build(self, boxscores):
        """ Takes a list of FootballBoxscore objects, processes data and converts to a dataframe """

        team_records = []
        for fbs in tqdm(boxscores):
            team_records.append(self.team_records_from_boxscore(fbs))
        table_df = self.build_team_table(pd.concat(team_records))
        table_df = floatify(table_df, string_columns=['date', 'team', 'opp', "Time of Possession"])
        table_df['name'] = table_df['team']  # for asof queries
        self.table = table_df

    @staticmethod
    def team_records_from_boxscore(fbs):
        """ Takes a boxscore, generates a dataframe such that the rows correspond to teams and the columns
        contain stats related to team offensive performance. Returns the dataframe. """

        # Row 1
        home_stats_df = fbs.all_team_stats['home_stat'].copy()
        home_stats_df['team'] = team_map_inv[fbs.scorebox['home_team']]
        home_stats_df['date'] = pd.Timestamp(fbs.scorebox['date'])
        home_stats_df['opp'] = team_map_inv[fbs.scorebox['away_team']]
        # Row 2
        vis_stats_df = fbs.all_team_stats['vis_stat'].copy()
        vis_stats_df['team'] = team_map_inv[fbs.scorebox['away_team']]
        vis_stats_df['date'] = pd.Timestamp(fbs.scorebox['date'])
        vis_stats_df['opp'] = team_map_inv[fbs.scorebox['home_team']]
        table = pd.concat([home_stats_df, vis_stats_df], axis=1).T

        return table

    @staticmethod
    def build_team_table(team_table):
        """ Takes a dataframe of home and away team stats. Cleans up time-related columns and splits up compound 
        columns. Returns the new dataframe. """

        team_table["Time of Possession"] = team_table["Time of Possession"].apply(
            lambda x: float(x.split(":")[0]) + float(x.split(":")[1]) / 60
        )
        t1 = team_table["Cmp-Att-Yd-TD-INT"].apply(
            lambda x: pd.Series(dict(zip(["pass_cmp", "pass_att", "pass_yd", "pass_td", "pass_int"], x.split("-"))))
        )
        t2 = team_table["Fourth Down Conv."].apply(
            lambda x: pd.Series(dict(zip(["fouth_conv_succ", "fouth_conv_att"],x.split("-"))))
        )
        t3 = team_table["Fumbles-Lost"].apply(
            lambda x: pd.Series(dict(zip(["fumbles", "fumbles_lost"], x.split("-"))))
        )
        t4 = team_table["Penalties-Yards"].apply(
            lambda x: pd.Series(dict(zip(["penalty_count", "penalty_yds"],x.split("-"))))
        )
        t5 = team_table["Rush-Yds-TDs"].apply(
            lambda x: pd.Series(dict(zip(["rush_att", "rush_yds", "rush_tds"], x.split("-"))))
        )
        t6 = team_table["Sacked-Yards"].apply(
            lambda x: pd.Series(dict(zip(["sacks_allowed", "sacks_allowed_yds"], x.split("-"))))
        )
        t7 = team_table["Third Down Conv."].apply(
            lambda x: pd.Series(dict(zip(["third_conv_succ", "third_conv_att"], x.split("-"))))
        )
        t8 = team_table[
            ["First Downs", "Net Pass Yards", "Time of Possession", 'Total Yards', 'Turnovers', 'team', 'date', 'opp']
        ]

        return pd.concat([t1, t2, t3, t4, t5, t6, t7, t8], axis=1)


class OffenseTable(FootballBoxscoreTable):
    """ Table that stores player-level offense data. """

    def __init__(self, season, refresh=False, boxscores=None):
        super(OffenseTable, self).__init__("offense", season, refresh, boxscores)

    def build(self, boxscores):
        """ Takes a list of FootballBoxscore objects, processes data and converts to a dataframe """

        table = []
        for fbs in tqdm(boxscores):
            record = floatify(fbs.all_player_offense.copy())
            record['date'] = pd.Timestamp(fbs.scorebox['date'])
            teams = record.team.unique()
            record['opp'] = record.team.apply(lambda x: teams[1] if x == teams[0] else teams[0])
            table.append(record)
        table = pd.concat(table)
        table['name'] = table.player.str.upper().str.replace(" ", "")
        table['DKScore'] = (
            (table.pass_td * 4.0) + (0.04 * table.pass_yds) + (3.0 * (table.pass_yds > 300.))
            + (-1.0 * table.pass_int) + (6.0 * table.rush_td) + (0.1 * table.rush_yds) 
            + (3.0 * (table.rush_yds > 100.)) + (6.0 * (table.rec_td)) + (0.1 * (table.rec_yds)) 
            + (3.0 * (table.rec_yds > 100.)) + table.rec + (-1 * table.fumbles_lost)
        )
        table.sort_values('date')
        self.table = table


class AdvancedPassingTable(FootballBoxscoreTable):
    """ Table that stores player-level passing data. """
    def __init__(self, season, refresh=False, boxscores=None):
        super(AdvancedPassingTable, self).__init__("advancedPassing", season, refresh, boxscores)

    def build(self, boxscores):
        """ Takes a list of FootballBoxscore objects, processes data and converts to a dataframe """
        table = []
        for fbs in tqdm(boxscores):
            t = floatify(fbs.adv_player_passing.copy())
            t['date'] = pd.Timestamp(fbs.scorebox['date'])
            table.append(t)
        self.table = pd.concat(table)
        self.table['name'] = self.table.player.str.upper().str.replace(" ", "")


class AdvancedRushingTable(FootballBoxscoreTable):
    """ Table that stores player-level rushing data. """
    def __init__(self, season, refresh=False, boxscores=None):
        super(AdvancedRushingTable, self).__init__("advancedRushing", season, refresh, boxscores)

    def build(self, boxscores):
        """ Takes a list of FootballBoxscore objects, processes data and converts to a dataframe """
        table = []
        for fbs in tqdm(boxscores):
            t = floatify(fbs.adv_player_passing.copy())
            t['date'] = pd.Timestamp(fbs.scorebox['date'])
            table.append(t)
        self.table = pd.concat(table)
        self.table['name'] = self.table.player.str.upper().str.replace(" ", "")


class AdvancedReceivingTable(FootballBoxscoreTable):
    """ Table that stores player-level receiving data. """
    def __init__(self, season, refresh=False, boxscores=None):
        super(AdvancedReceivingTable, self).__init__("advancedReceiving", season, refresh, boxscores)

    def build(self, boxscores):
        """ Takes a list of FootballBoxscore objects, processes data and converts to a dataframe """
        table = []
        for fbs in tqdm(boxscores):
            t = floatify(fbs.adv_player_passing.copy())
            t['date'] = pd.Timestamp(fbs.scorebox['date'])
            table.append(t)
        self.table = pd.concat(table)
        self.table['name'] = self.table.player.str.upper().str.replace(" ", "")


class ScoreTable(FootballBoxscoreTable):
    """ Table that stores game-level characteristic data. """

    def __init__(self, season, refresh=False, boxscores=None):
        super(ScoreTable, self).__init__("score", season, refresh, boxscores)

    def build(self, boxscores):
        """ Takes a list of FootballBoxscore objects, processes data and converts to a dataframe """
        table = []
        for fbs in tqdm(boxscores):
            table.append(pd.Series({"home": team_map_inv[fbs.scorebox['home_team']],
                                    "away": team_map_inv[fbs.scorebox['away_team']],
                                    "home_score": fbs.scorebox['home_team_score'],
                                    "away_score": fbs.scorebox['away_team_score'],
                                    "date": pd.Timestamp(fbs.scorebox['date'])}))
        self.table = pd.concat(table, axis=1).T

def floatify(table, string_columns=['team']):
    """ Takes a dataframe and an optional list of columns. Converts all columns in the string_columns list and converts
    them from strings to floats. If the cell contents are empty return the numpy nan value. Returns the converted 
    dataframe. """

    off_float = table[[c for c in table.columns if not c in string_columns]] \
        .applymap(lambda x: np.nan if x == "" else float(x.replace("%", ""))).copy()
    off_string = table[string_columns].copy()
    table = pd.concat([off_string, off_float], axis=1)
    table = table.reset_index().rename(columns={'index': 'player'})
    return table
