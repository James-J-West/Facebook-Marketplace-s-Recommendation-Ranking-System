import numpy as np
import torch

def calc_prob(tensor):
    tensor[tensor < 0] = 0
    total = torch.sum(tensor)
    index = torch.argmax(tensor)
    lst = list(tensor)
    maximum = lst[index]
    # print(maximum)
    # print(total)
    percentage = maximum.float() / total.float()
    percentage = float(percentage)
    return np.round(percentage*100, 1)

if __name__ == "__main__":
    print(calc_prob(torch.as_tensor([-1,2,3,4,5,6,7,8,9,-10])))