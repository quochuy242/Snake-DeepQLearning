from functools import partial

import torch
import matplotlib.pyplot as plt
from IPython import display
import datetime as dt
import os
from pathlib import Path

float_tensor = partial(torch.tensor, dtype=torch.float)
unsqueeze_0 = partial(torch.unsqueeze, dim=0)


def plot_result(scores: int, mean_scores: float):
    display.clear_output(wait=True)
    plt.clf()
    plt.title("Training... ")
    plt.xlabel("No. Games")
    plt.ylabel("Score")
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(x=len(scores) - 1, y=scores[-1], s=str(scores[-1]))
    plt.text(x=len(mean_scores) - 1, y=mean_scores[-1], s=str(mean_scores[-1]))
    # save_path = f"./visualize/{dt.datetime.now().strftime('%d-%m-%Y')}"
    # os.makedirs(save_path, exist_ok=True)
    # plt.savefig(str(Path(save_path) / f"{scores[-1]}.png"))
    plt.show(block=False)
    plt.pause(0.1)
