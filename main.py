from src.RandomForestPredictor import *

if __name__ == "__main__":
    model = RandomForestPredictor()
    model.traine("src/data/campeonato-brasileiro-full.csv")

