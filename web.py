import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup

def extract_boxscore_links(team_breadcrumb):
    full_url = "https://www.pro-football-reference.com" + team_breadcrumb
    resp = requests.get(full_url)
    soup = BeautifulSoup(resp.text, 'html.parser')
    link_elements = [a for a in soup.find_all("a") if a.text == 'boxscore']
    links = [l['href'] for l in link_elements]
    return links

def extract_team_links(year):
    resp = requests.get("https://www.pro-football-reference.com/years/%s/" % year)
    soup = BeautifulSoup(resp.text, 'html.parser')
    nfc_div = soup.find('div', id="div_NFC")
    afc_div = soup.find('div', id="div_AFC")
    nfc_links = nfc_div.find_all('a')
    afc_links = afc_div.find_all('a')
    team_links = afc_links + nfc_links
    return team_links

def unique_game_links(year):
    all_boxscores = [extract_boxscore_links(url['href']) for url in extract_team_links(year)]
    flattened_list = np.hstack([np.array(b) for b in all_boxscores])
    unique_game_links = np.unique(flattened_list)
    return unique_game_links

class FootballBoxscore():
    def __init__(self, url):
        self.url = url

    def full_scrape(self):
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
            print("No starter info %s" % self.url)
            self.home_starters = None
            self.away_starters = None

    @staticmethod
    def parse_table(soup, table_div_id, scorebox=None):
        table_div = soup.find('div', id=table_div_id)
        div_encoded = bytearray(str(table_div.contents), 'utf-8')
        div_decoded = div_encoded.decode('utf-8')
        div_soup = BeautifulSoup(div_decoded, "html.parser")
        table_soup = div_soup.find('table')
        out = {}
        for tr in table_soup.find('tbody').find_all('tr'):
            if (tr.find('th').text != "") and (tr.find('th').text != "Player"):
                player_name = tr.find('th').text
                player_data = {td['data-stat']: td.text for td in tr.find_all("td")}
                out[player_name] = player_data
        return pd.DataFrame(out).T

    @staticmethod
    def parse_scorebox(soup):
        scorebox = soup.find('div', class_="scorebox")
        teams = [a.text for a in scorebox.find_all("a") if a['href'].startswith("/teams")]
        scores = [float(d.text) for d in scorebox.find_all('div', class_="score")]
        date = soup.find("div", class_="scorebox_meta").find_all('div')[0].text
        output = {"home_team": teams[0],
                  "away_team": teams[1],
                  "home_team_score": scores[0],
                  "away_team_score": scores[1],
                  "date": date}
        return output
