import numpy as np
import time
import sys

if sys.version_info == 2.0:
    import Tkinter as tk
else:
    import tkinter as tk

UNIT = 40
Max_h = 6
Max_w = 6

class Maze():
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("maze with q-learning")
        self.window.geometry('{0}x{1}'.format(Max_w*UNIT, Max_h*UNIT))
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.build_maze()

    def build_maze(self):
        self.canvas = tk.Canvas(self.window, bg = 'white', width=Max_w*UNIT, height= Max_h*UNIT)

        # to create verticle line
        for c in range(0, Max_w*UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, Max_w*UNIT
            self.canvas.create_line(x0, y0, x1, y1)

        # to create horizontal lines
        for r in range(0, Max_h*UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, Max_h*UNIT, r
            self.canvas.create_line(x0,y0,x1,y1)

        origin = np.array([20, 20])

        hell_center1 = origin + np.array([UNIT*2, UNIT])
        self.hell1 = self.canvas.create_rectangle(
            hell_center1[0] - 15, hell_center1[1] - 15,
            hell_center1[0] + 15, hell_center1[1] + 15,
            fill = 'black'
        )

        hell_center2 = origin + np.array([UNIT, UNIT * 2])
        self.hell2 = self.canvas.create_rectangle(
            hell_center2[0] - 15, hell_center2[1] - 15,
            hell_center2[0] + 15, hell_center2[1] + 15,
            fill = 'black'
        )

        oval_center = origin + UNIT*2
        self.oval = self.canvas.create_oval(
            oval_center[0] - 15, oval_center[1] - 15,
            oval_center[0] + 15, oval_center[1] + 15,
            fill = 'yellow'
        )

        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill = 'red'
        )
        self.canvas.pack()

    def render(self):
        time.sleep(0.1)
        self.window.update()
        
    def reset(self):
        self.window.update()
        time.sleep(0.5)
        self.canvas.delete(self.rect)
        origin = np.array([20,20])
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[1] + 15, origin[1] + 15,
            fill = 'red'
        )
        return self.canvas.coords(self.rect)

    def get_reward_state(self, action):
        s = self.canvas.coords(self.rect)
        base_action = np.array([0, 0])
        if action == 0: #up
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1: #down
            if s[1] < (Max_h -1 ) * UNIT:
                base_action[1] += UNIT
        elif action == 2: #right
            if s[0] < (Max_w -1 ) * UNIT:
                base_action[0] += UNIT
        elif action == 3: #left
            if s[0] > UNIT:
                base_action[0] -= UNIT

        self.canvas.move(self.rect, base_action[0], base_action[1])
        s_ = self.canvas.coords(self.rect)

        if s_ == self.canvas.coords(self.oval):
            reward = 1
            done = True
            s_ = 'terminal'
        elif s_ in [self.canvas.coords(self.hell1), self.canvas.coords(self.hell2)]:
            reward = -1
            done = True
            s_ = 'terminal'
        else:
            reward = 0
            done = False
        return s_, reward, done

#if __name__ == "__main__":
#    maze = Maze()
 #   maze.build_maze()
  #  maze.window.mainloop()