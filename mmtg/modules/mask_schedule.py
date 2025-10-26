import torch
# import matplotlib.pyplot as plt
import numpy as np
import math


def schedule(ratio, total_unknown, method="cosine"):
    """
    Ref: https://github.com/google-research/maskgit/blob/main/maskgit/libml/mask_schedule.py
    """
    if method == "uniform":
        mask_ratio = 1. - ratio
    elif "pow" in method:
        exponent = float(method.replace("pow", ""))
        mask_ratio = 1. - ratio**exponent
    elif method == "cosine":
        mask_ratio = np.cos(np.pi / 2. * ratio)
    elif method == "log":
        mask_ratio = -np.log2(ratio) / np.log2(total_unknown)
    elif method == "exp":
        mask_ratio = 1 - np.exp2(-np.log2(total_unknown) * (1 - ratio))
    # Clamps mask into [epsilon, 1)
    mask_ratio = np.clip(mask_ratio, 1e-6, 1.)
    return mask_ratio


if __name__ == "__main__":
    # Define the range of r
    r_values = np.linspace(0, 1, 1000)

    # Plot each function
    plt.figure(figsize=(10, 8))

    plt.plot(r_values, schedule(r_values, total_unknown=512, method="cosine"), label="cosine")
    plt.plot(r_values, schedule(r_values, total_unknown=512, method="pow1"), label="linear")
    plt.plot(r_values, schedule(r_values, total_unknown=512, method="pow2"), label="square")
    plt.plot(r_values, schedule(r_values, total_unknown=512, method="pow3"), label="cubic")
    plt.plot(r_values, schedule(r_values, total_unknown=512, method="pow0.5"), label="sqaure root")
    plt.plot(r_values, schedule(r_values, total_unknown=512, method="log"), label="logarithmic")
    plt.plot(r_values, schedule(r_values, total_unknown=512, method="exp"), label="exponential")

    plt.legend(loc='upper right')
    plt.title("Mask Schedule Functions")
    plt.xlabel("r")
    plt.ylabel("a(r)")
    plt.ylim(0, 1)
    plt.xlim(0, 1)
    plt.grid(True)
    plt.savefig('mask_schedule.png')

    