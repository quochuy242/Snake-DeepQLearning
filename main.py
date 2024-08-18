from time import time

from agent import train

if __name__ == "__main__":
    start = time()
    max_score = train(n_games=1000, save_date=None)
    print(f"After {time() - start} second, agent gains {max_score}")
