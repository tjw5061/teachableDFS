import os
from os.path import expanduser

PROJECT_DIRECTORY =  os.path.join(expanduser("~"), "teachdfs")
CACHE_DIRECTORY = os.path.join(PROJECT_DIRECTORY, "cache/database")

SEASON_START_DATES = {
    2018: "2018.10.10",
    2019: "2019.10.09",
    2020: "2020.09.20"
}
