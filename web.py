import os
import requests

import numpy as np
import pandas as pd

from bs4 import BeautifulSoup

def extract_team_links(year):
    """ Takes a season year, requests the NFL Standings & Team Stats page for the given year and returns
    a list of links to each season + team landing page. """
    
    resp = requests.get(f"https://www.pro-football-reference.com/years/{year}/")
    soup = BeautifulSoup(resp.text, 'html.parser')
    nfc_div = soup.find(id="div_NFC")
    afc_div = soup.find(id="div_AFC")
    nfc_links = nfc_div.find_all('a')
    afc_links = afc_div.find_all('a')
    team_links = afc_links + nfc_links

    return team_links

def extract_boxscore_links(team_season_overview_suffix):
    """ Takes a string associated with a teams season overview url, requests access to the page 
    and extracts all hyperlink addresses associated with the boxscore hyperlinks. Returns a list of 
    hyperlink suffix strings for all of a team's games during a season. """
    
    full_url = "https://www.pro-football-reference.com" + team_season_overview_suffix
    resp = requests.get(full_url)
    soup = BeautifulSoup(resp.text, 'html.parser')
    link_elements = [a for a in soup.find_all("a") if a.text == 'boxscore']
    links = [l['href'] for l in link_elements]

    return links


def unique_game_links(year):
    """ Takes a year. Extracts each team's season overview url. For each team extracts all associated 
    games they participated in during the season. Merges all game links and removes duplicates. 
    Returns a list of url suffix strings. """
    
    all_boxscores = [extract_boxscore_links(url['href']) for url in extract_team_links(year)]
    flattened_list = np.hstack([np.array(b) for b in all_boxscores])
    unique_game_links = np.unique(flattened_list)

    return unique_game_links
    
class FootballBoxscore():
    """ Primary webscraping agent. Given a url to a game webpage several stat tables are extracted. 
    Currently we are extracting the scorebox, which contains game level data and the following tables
    of player-level data:
            - Offensive Stats
            - Defensive Stats
            - Kicking Stats (Punts and Punt Returns)
            - Advanced Passing Stats
            - Advanced Rushing Stats
            - Advanced Receiving Stats
            - Advanced Defense Stats
            - Snap Stats (Teams are separated by Home and Away)
            - Drive Stats  (Teams are separated by Home and Away)
            - Starters (Data is inconsistently available)
        All the the player-level data is stored in pandas dataframes, while the scorebox data is stored in a single dictionary.
    """

    def __init__(self, url):
        """
            Required Inputs:
                url: full url to a game webpage
        """
        self.url = url

    def full_scrape(self):
        """ Primary entry point of FootballBoxscore. """

        resp = requests.get(self.url)
        soup = BeautifulSoup(resp.text, "html.parser")
        self.scorebox = self.parse_scorebox(soup)
        self.all_team_stats = self.parse_table(soup, "all_team_stats")
        self.all_player_offense = self.parse_table(soup, "all_player_offense")
        self.all_player_defense = self.parse_table(soup, "all_player_defense")
        self.all_player_kicking = self.parse_table(soup, "all_kicking")
        self.adv_player_passing = self.parse_table(soup, "all_passing_advanced")
        self.adv_player_rushing = self.parse_table(soup, "all_rushing_advanced")
        self.adv_player_receive = self.parse_table(soup, "all_receiving_advanced")
        self.adv_player_defense = self.parse_table(soup, "all_defense_advanced")
        self.home_snap_counts = self.parse_table(soup, "all_home_snap_counts")
        self.away_snap_counts = self.parse_table(soup, "all_vis_snap_counts")
        self.home_drives = self.parse_table(soup, "all_home_drives")
        self.away_drvies = self.parse_table(soup, "all_vis_drives")
        try:
            self.home_starters = self.parse_table(soup, "all_home_starters")
            self.away_starters = self.parse_table(soup, "all_vis_starters")
        except AttributeError:
            print(f"No starter info {self.url}")
            self.home_starters = None
            self.away_starters = None

    @staticmethod
    def parse_table(soup, table_div_id):
        """ Takes a BeautifulSoup object for the game stat webpage and the table id of the table 
        that is going to be scrapped. Parses through the table and creates a dictionary such that
        each header is a key and the cells contents are the values. Converts the dictionary to 
        a dataframe and returns the transposed dataframe. """
        
        table_div = soup.find('div', id=table_div_id)
        div_encoded = bytearray(str(table_div.contents), 'utf-8')
        div_decoded = div_encoded.decode('utf-8')
        div_soup = BeautifulSoup(div_decoded, "html.parser")
        table_soup = div_soup.find('table')
        output = {}
        for tr in table_soup.find('tbody').find_all('tr'):
            if (tr.find('th').text != "") and (tr.find('th').text != "Player"):
                header = tr.find('th').text
                cell_contents = {td['data-stat']: td.text for td in tr.find_all("td")}
                output[header] = cell_contents
                
        return pd.DataFrame(output).T
    
    @staticmethod
    def parse_scorebox(soup):
        """ Takes a BeautifulSoup object for the game stat webpage. Extracts the team names, the
        final score and the date of the game and stores as a dictionary. Returns the dictionary
        """

        scorebox = soup.find('div', class_="scorebox")
        teams = [a.text for a in scorebox.find_all("a") if a['href'].startswith("/teams")]
        scores = [float(d.text) for d in scorebox.find_all('div', class_="score")]
        date = soup.find("div", class_="scorebox_meta").find_all('div')[0].text
        output = {
            "home_team": teams[0],
            "away_team": teams[1],
            "home_team_score": scores[0],
            "away_team_score": scores[1],
            "date": date
        }
        
        return output
    