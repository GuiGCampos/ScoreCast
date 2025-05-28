from src.RandomForestPredictor import *

if __name__ == "__main__":
    model = RandomForestPredictor()
    model.train("src/data/campeonato-brasileiro-full.csv")

