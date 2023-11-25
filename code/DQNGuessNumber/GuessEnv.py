import numpy as np


class GuessEnv:
    def __init__(self):
        self.n_action = 100
        self.n_state = 30
        self.number = None
        self.state = np.array([0]*30)
        self.action = np.array(range(100))
        self.t = 0
    def reset(self):
        self.number = np.random.randint(100)
        # self.number=50
        self.state = np.array([0]*30)
        self.t = 0
        return self.state

    def step(self, action):

        tempState = self.nextState(action)
        self.state[self.t*3]=action
        self.state[self.t*3+1]=tempState[0]
        self.state[self.t*3+2]=tempState[1]
        self.t += 1
        return self.state, self.reward(tempState), self.isDone(action), None

    def isDone(self, action):
        return True if action == self.number or self.t == 10 else False

    def nextState(self, action):
        A = 0
        B = 0
        temp = action
        temp2 = self.number
        temp = str(temp).zfill(2)
        temp2 = str(temp2).zfill(2)
        i = 0
        while i < len(temp):
            if temp[i] == temp2[i]:
                A += 1
                temp = temp[:i] + temp[i + 1:]
                temp2 = temp2[:i] + temp2[i + 1:]
            else:
                i += 1
        for i in temp:
            if i in temp2:
                temp2.replace(i, '', 1)
                B += 1
        return np.array([A, B])

    def reward(self, state):
        if self.t == 10:
            return state[0] * 100 + state[1]*50 - 1000
        if state[0]==2:
            return 1000 -self.t*100
        return state[0] * 6 + state[1] * 3
