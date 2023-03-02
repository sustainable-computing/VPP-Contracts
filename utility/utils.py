import math
import torch
import random
import numpy as np

def create_log_gaussian(mean, log_std, t):
    quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
    l = mean.shape
    log_z = log_std
    z = l[-1] * math.log(2 * math.pi)
    log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1) - 0.5 * z
    return log_p

def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def jain_index(x, w=None):
    x = np.array(x)
    if w != None:
        x = x * np.array(w)
    sqr_sum = np.sum(x**2)
    if sqr_sum <= tol:
        return 1.0
    return (np.sum(x)**2)/(len(x)*sqr_sum)

def save_dict(file_name, dict_):
    f = open(file_name, 'w')
    f.write(str(dict_))
    f.close()


def load_dict(file_name):
    f = open(file_name, 'r')
    data = f.read()
    f.close()
    return eval(data.replace('array(', '').replace('])', ']'))
    
def load_list(file_name):
    f = open(file_name, 'r')
    data = f.read()
    f.close()
    return data

        
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)
        
class CircularBuffer:
    def __init__(self, *, max_size):
        self.max_size = max_size
        self.items = []
        self.offset = 0
    
    def append(self, item):
        max_size = self.max_size
        if max_size is None or len(self.items) < max_size:
            self.items.append(item)
        else:
            self.items[self.offset] = item
            self.offset = (self.offset + 1) % max_size
    
    def __getitem__(self, index):
        max_size = self.max_size
        if max_size is not None:
            index = (index - self.offset) % self.max_size
        return self.items[index]
    
    def __len__(self):
        return len(self.items)        
     
class Args:
    def __init__(self):
            
        self.policy = "Deterministic"
        self.eval = True
        self.gamma = 0.99
        self.tau = 0.01
        self.lr = 0.0001
        self.alpha = 0.2
        self.automatic_entropy_tuning = True
        self.seed = 123456
        self.batch_size = 100
        self.num_steps = 1000001
        self.hidden_size = 256
        self.updates_per_step = 1
        self.start_steps = 10000
        self.target_update_interval = 1
        self.replay_size = 1000000
        self.cuda = True
