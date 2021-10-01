import os
from os.path import expanduser

PROJECT_DIRECTORY =  os.path.join(expanduser("~"), "teachableDFS")
CACHE_DIRECTORY = os.path.join(PROJECT_DIRECTORY, "cache/database")

SEASON_START_DATES = {
    2018: "2018.10.10",
    2019: "2019.10.09",
    2020: "2020.09.20"
}

class DailyFantasyDataScienceError(Exception):
    """Exception raised for errors cache errors """

    def __init__(self):
        self.message = "Attempted to access file in the /cache/ folder. If /cache/ doesn't exist run os.makedirs(CACHE_DIRECTORY) to create the empty folder. See the 'Welcome' lesson for more information"
        super().__init__(self.message)
