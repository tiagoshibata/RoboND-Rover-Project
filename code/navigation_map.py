import cv2
import numpy as np


class NavigationMap:
    def __init__(self):
        self.reset()
        self.rock = np.zeros((200, 200), dtype=np.float)

    def add_navigable(self, x_list, y_list, weights):
        self.map[y_list, x_list] += 8 * weights

    def add_obstacle(self, x_list, y_list, weights):
        self.map[y_list, x_list] -= weights

    def add_rock(self, x_list, y_list):
        self.rock[y_list, x_list] += 1

    def reset(self):
        self.map = np.zeros((200, 200), dtype=np.float)
        self.update()

    def update(self):
        self.navigable = np.zeros((200, 200), dtype=bool)
        self.navigable[self.map > 0] = True

        self.real_navigable = self.navigable.copy()
        self.real_navigable[self.map > 40000] = False

        self.obstacle = np.zeros((200, 200), dtype=bool)
        self.obstacle[self.map < 0] = True

    def undiscovered_paths(self):
        navigable_border = cv2.morphologyEx(self.navigable.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3)))
        # obstacle = cv2.dilate(self.obstacle.astype(np.uint8), np.ones((3, 3)))
        undiscovered = navigable_border & np.logical_not(self.navigable) & np.logical_not(self.obstacle)
        return undiscovered
