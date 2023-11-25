import numpy as np
import os

class MineSweeperEnv:
    def __init__(self, row, column, prob):
        self.state_n = row * column
        self.row = row
        self.column = column
        self.prob = prob
        self.first_click = True
        self.grids = np.zeros((row, column), dtype=int)
        self.obs_grids = np.full((row, column), -1, dtype=int)
        self.noMine = set()
        self.hasMine = set()


    def create_grid(self):
        mine_count = int(self.row * self.column * self.prob)
        mine_positions = np.random.choice(self.row * self.column, mine_count, replace=False)
        self.grids.flat[mine_positions] = -2  # Let's use -2 to represent mines
        self.update_hints()

    def update_hints(self):
        for i in range(self.row):
            for j in range(self.column):
                if self.grids[i, j] == -2:
                    # Update surrounding cells with hints
                    for x in range(max(0, i-1), min(self.row, i+2)):
                        for y in range(max(0, j-1), min(self.column, j+2)):
                            if self.grids[x, y] != -2:
                                self.grids[x, y] += 1

    def click(self, i, j):
        # 如果是第一次點擊並且點擊的位置是地雷，重新生成遊戲
        if self.first_click and self.grids[i, j] == -2:
            self.reset()
            return self.click(i, j)

        self.first_click = False
        if self.grids[i, j] == -2:  # Hit a mine
            reward = -10
            # print("Lose")
            done = True
        elif self.grids[i, j] >= 0:  # Clicked on a safe cell
            self.reveal_cell(i, j)
            reward = 30  # Small positive reward for finding a safe cell
            done = False
        else:
            reward = -1000  # Penalty for clicking an already revealed cell
            done = False

        if self.check_win():
            reward = 30  # Big reward for winning the game
            done = True

        observation = self.obs_grids
        # self.printGrid()
        return observation, reward, done

    def reveal_cell(self, i, j):
        self.obs_grids[i, j] = self.grids[i, j]
        if self.grids[i, j] == 0:
            # Reveal adjacent cells recursively if this cell is a 0
            for x in range(max(0, i - 1), min(self.row, i + 2)):
                for y in range(max(0, j - 1), min(self.column, j + 2)):
                    if self.obs_grids[x, y] == -1:
                        self.reveal_cell(x, y)

    def printGrid(self):
        # os.system('clear')
        # return
        print(self.obs_grids)

    def check_win(self):
        for i in range(self.row):
            for j in range(self.column):
                if self.grids[i, j] != -2 and self.obs_grids[i, j] == -1:
                    # 如果有一個非礦區的格子還沒被揭開，則玩家還沒贏
                    return False
        # 所有非礦區的格子都被揭開，玩家贏了
        # print("Win")
        return True

    def reset(self):
        self.grids = np.zeros((self.row, self.column), dtype=int)
        self.obs_grids = np.full((self.row, self.column), -1, dtype=int)
        self.first_click = True
        self.create_grid()
        # self.printGrid()
        return self.obs_grids
