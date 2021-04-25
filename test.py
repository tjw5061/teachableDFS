import pickle
from data import ScoreTable


if __name__ == "__main__":
    all_scores = pickle.load(open("/home/tom/Cache/box.pkl", 'rb'))
    score_table = ScoreTable(refresh=True, boxscores=all_scores)
