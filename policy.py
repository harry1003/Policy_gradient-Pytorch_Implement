import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearPolicy(nn.Module):
    def __init__(self):
        super(LinearPolicy, self).__init__()
        self.linear1 = nn.Linear(80 * 80, 200)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(200, 2)
        self.softmax = nn.Softmax(dim=0)
        
    def forward(self, observation, train=True):
        x = observation.view(-1)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        # to get act_int
        action = self.softmax(x)
        if train:
            if action[0] > np.random.uniform():
                act_int = 2
            else:
                act_int = 3
        else:
            if action[0] > 0.5:
                act_int = 2
            else:
                act_int = 3
        return act_int, x


class Policy_gradient(nn.Module):
    def __init__(self, args):
        super(Policy_gradient, self).__init__()
        self.Policy = LinearPolicy()
        self.opt = torch.optim.Adam(self.Policy.parameters(), args.lr)
    
    def get_action(self, state, train=True):
        act_int, act_p = self.Policy(state, train)
        return act_int, act_p
    
    def update(self, act_p, act, reward):
        """
        act_p : [N, 2]
        act : [N]
        reward : [N]
        """
        # get label
        label = act - 2 # act is 2 and 3 -> 0, 1
        label = label.long() # pytorch CrossEntropyLoss format
        # loss
        self.opt.zero_grad()
        loss_fn = nn.CrossEntropyLoss(reduction="none") # for mulitple output
        loss = loss_fn(act_p, label)
        loss = torch.dot(loss, reward)
        loss.backward()
        self.opt.step()
        return loss.data.cpu().numpy()
    
    def save(self, e=0):
        torch.save(self.Policy, "./model/Linear_" + str(e) + ".model")

    def load(self, path):
        self.Policy = torch.load(path)