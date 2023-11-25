import random
from tkinter import *
import numpy as np
import time
from PIL import Image, ImageTk


class MineSweeperEnv:
    def __init__(self, row, column, prob):
        self.state_n = row*column
        self.action_n = row*column
        self.root = Tk()
        self.frm = Frame(self.root)
        self.frm.grid()
        self.row = row
        self.column = column
        self.prob = prob
        self.grids = [[0 for i in range(self.column)] for j in range(self.row)]
        self.Button_grids = [[None for i in range(self.column)] for j in range(self.row)]
        self.obs_grids = [[-1 for i in range(self.column)] for j in range(self.row)]
        self.Neighbor_dict = {}
        self.noMine = []
        self.expose = []
        self.hasMine = []
        self.NotMine = set()
        self.setGrids()
    def setGrids(self):
        for i in range(self.row):
            for j in range(self.column):
                self.Button_grids[i][j] = Label(self.root, text=' ', width=2, height=1)
                self.Button_grids[i][j].grid(row=i, column=j)

    def create_grid(self):
        bumps = np.random.choice(self.row * self.column, int(self.row * self.column * self.prob), replace=False)
        t = 0
        for i in range(self.row):
            for j in range(self.column):
                if t in bumps:
                    # print(i, j)
                    self.grids[i][j] = 'B'
                    self.Neighbor_dict[(i,j)]=[(i,j)]
                    self.checkHint(i, j)
                else:
                    self.NotMine.add((i, j))
                    # print(self.NotMine)
                # print(t)
                t += 1
        for i in range(self.row):
            for j in range(self.column):
                self.Button_grids[i][j]['text']=' '
                self.Neighbor(i, j)

    def checkHint(self, i, j):
        for k in range(-1, 2, 1):
            for l in range(-1, 2, 1):
                if k == 0 and l == 0:
                    continue
                else:
                    try:
                        if self.grids[i + k][j + l] != 'B':
                            self.grids[i + k][j + l] += 1
                    except:
                        continue

    def click(self,i,j):
        # print(self.NotMine)
        # time.sleep(0.01)
        # print(i, j)
        if self.obs_grids[i][j] != -1:
            return np.array(self.obs_grids).flatten(), -0.3, False
        if self.grids[i][j] == 'B':
            self.obs_grids[i][j] = -2
            # self.root.update()
            return np.array(self.obs_grids).flatten(), -1, True
        if self.obs_grids[i][j] == -1:
            reward = self.check_unknown(i, j)
        for (x, y) in self.Neighbor_dict[(i, j)]:
            self.Button_grids[x][y]['text'] = str(self.grids[x][y])
            self.obs_grids[x][y] = self.grids[x][y]
            self.NotMine.discard((x, y))
        # self.root.update()
        if self.done():
            temp = np.array(self.obs_grids).flatten()
            self.reset()
            return temp, 1, True
        else:
            return np.array(self.obs_grids).flatten(), reward, False

    def done(self):
        if len(self.NotMine) == 0:
            # print("↓↓↓↓Done↓↓↓↓")
            return True
        else:
            return False
    def Neighbor(self,i,j):
        if (i, j) in self.Neighbor_dict:
            return
        self.noMine = []
        self.expose = []
        self.hasMine = []
        self.BFS(i, j)
        for (x, y) in self.noMine:
            self.Neighbor_dict[(x, y)] = list(self.expose)
        for (x, y) in self.hasMine:
            self.Neighbor_dict[(x, y)] = list([(x, y)])

    def BFS(self,i,j):
        if i < 0 or j < 0 or (i,j)in self.expose:
            return
        try:
            if self.grids[i][j] == 0:
                self.noMine.append((i, j))
                self.expose.append((i, j))
                for (x, y) in [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)]:
                    self.BFS(x, y)
            elif self.grids[i][j] != 'B':
                self.hasMine.append((i, j))
                self.expose.append((i, j))
        finally:
            return

    def check_unknown(self, i, j):
        temp = [(i-1, j), (i-1, j+1), (i-1, j-1), (i, j-1), (i, j+1), (i+1, j), (i+1, j+1), (i+1, j-1)]
        unknown=0
        total=0
        for (x, y) in temp:
            try:
                if self.obs_grids[x][y] == -1:
                    unknown += 1
                total += 1
            finally:
                continue
        return -0.3 if unknown == total else 0.9

    def run(self):
        self.create_grid()
        # self.root.update()

    def findFirst(self):
        i, j = np.random.randint(self.row), np.random.randint(self.column)
        while self.grids[i][j] != 0:
            i, j = np.random.randint(self.row), np.random.randint(self.column)
        return self.click(i, j)
    def reset(self):
        self.grids = [[0 for i in range(self.column)] for j in range(self.row)]
        self.obs_grids = [[-1 for i in range(self.column)] for j in range(self.row)]
        self.Neighbor_dict = {}
        self.noMine = []
        self.expose = []
        self.hasMine = []
        self.NotMine = set()
        self.create_grid()
        # self.root.update()
        data, reward, done = self.findFirst()
        if done:
            return self.reset()
        else:
            return np.array(self.obs_grids).flatten()
# Env = MineSweeperEnv(10, 10, 0.1)
# Env.run()
# Env.click(0,0)
# time.sleep(1)
# input("input:")
