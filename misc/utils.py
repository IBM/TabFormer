import random
import numpy as np
import torch


class ddict(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def random_split_dataset(dataset, lengths, random_seed=20200706):
    # state snapshot
    state = {}
    state['seeds'] = {
        'python_state': random.getstate(),
        'numpy_state': np.random.get_state(),
        'torch_state': torch.get_rng_state(),
        'cuda_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None
    }

    # seed
    random.seed(random_seed)  # python
    np.random.seed(random_seed)  # numpy
    torch.manual_seed(random_seed)  # torch
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)  # torch.cuda

    train_dataset, eval_dataset, test_dataset = torch.utils.data.dataset.random_split(dataset, lengths)

    # reinstate state
    random.setstate(state['seeds']['python_state'])
    np.random.set_state(state['seeds']['numpy_state'])
    torch.set_rng_state(state['seeds']['torch_state'])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state(state['seeds']['cuda_state'])

    return train_dataset, eval_dataset, test_dataset


def divide_chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]