# teachableDFS

This walkthrough accompanies the code in walkthrough.ipynb.

There are 5 steps to go from scratch to running a backtest.

1. WEB - Scrape data from the web and store in a FootballBoxscore object

2. DATA - Structure data in tables using children of the FootballBoxscoreTable class (ScoreTable, OffenseTable, etc)

3. MODEL - From those tables that store observations in time - construct feature space that will be useful to tree- based machine learning model

4. SIM - Given only what you know as of a point in time (say Week 7 of the 2020 football season) fit an ML model and comeup with what our prediction WOULD have been as of that date

5. BACKTEST - Using different lineup construction methods (i.e. different "stacking" strategies) determine what our "optimal" lineups would have been, and then compare the performance of our selected lineups vs. the actual results of the contest on that date

Let's go into each of these in a little more depth.

(1) WEB - Here we are using requests and beautiful soup to scrape and parse boxscore pages on pro-football reference
and story in a class called FootballBoxscore. It's attributes are tables that you see on the webpage:
all_player_offense, all_player_defense, etc. In the jupyter notebook, the first thing we do is construct a list of
these FootballBoxscore and store them in a variable: all_scores.

(2) DATA - In this section we build a bunch of different tables on top of all_scores. A single boxscore webpage contains
tables with different schemas (the columns in the offensive table are obviously different than the column
in the advanced receiving table). This step creates a single advanced receiving table (for example) table by
iterating over all_scores, extracting the advanced receiving section and then gluing everything together to create a
advanced receiving history. There is also a small amount of augmentation/modification specific to individual tables,
but in general that's what's going on in this step: extract and glue.

(3) MODEL - We are eventually going to build 3 separate models to project a DraftKings score: one for quarterbacks,
one for position players, and one for defenses. We first need to train these models on a relevant dataset. Let's
consider the quarterback example. In the offense table and the adv_passing_table, we have statistics for a
quarterback on a single day. Our goal is: given that we know a quarterback played on a given day, we'd like to
summarize his performance UP until that day and have a model learn a relationship between that summary and his
performance on that day.

    These feature space classes look for every unqiue example in the database of (QB STATS, DATE). They take the
    average of each performance statistic up until that date then label the example with the DraftKings score generated
    by that players performance on that day. So for example, let's suppose these were Kirk Cousin's stats in the offense
    table and we wanted to create a record for our model: (Kirk Cousins, Week 6)

                Pass Completions |  Passing Touchdowns | Draftking Score
    Week 1:            10        |          1          |      12
    Week 2:            15        |          2          |      24
    Week 3:            20        |          1          |      12
    Week 4:            25        |          0          |      05
    Week 5:            05        |          1          |      12
    Week 6:            07        |          0          |      07

    These feature space classes would average Pass Completions, Passing Touchdowns, and Draftking Score up through week 5
    to get X and then use the Draftking's Score in week 6 to label that record. In this case:

    X = {Pass Completions: 15, Pass Completions: 1.0, Pass Completions: 13} | Y = 07
    And then we could use this as a record to feed into our model.

To be continued...
