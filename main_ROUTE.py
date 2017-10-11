#! /usr/bin/python3
""""""

import pygame as pg
import numpy as np
import math
from random import random

from sofm import KOHONENRING

def random_circle(radius, off_x, off_y):
    t = 2 * math.pi * random()
    u = random() + random()
    if u > 1:
        r = 2 - u
    else:
        r = u
    return radius * r * math.cos(t) + off_x, radius * r * math.sin(t) + off_y

def main():
    size = 640
    num_points = 21
    
    pg.init()
    
    screen = pg.display.set_mode((size, size))
    
    

    while 1==1:
        my_sofm = KOHONENRING(2, num_points * 3, 0.99, 0.99999, 0.01)
    
        points = np.random.rand(num_points, 2)
        index = 0
        
        tick = 0
        while tick < 5000:
            weights = my_sofm.get_weights()
            screen.fill((255, 255, 255))
            for it_x in range(weights.shape[0]):
                pos_start = weights[it_x] * size
                    
                if it_x < weights.shape[0] - 1:
                    pos_stop1 = weights[it_x + 1] * size
                    pg.draw.line(screen, (0,0,0), pos_start, pos_stop1)
                else:
                    pos_stop1 = weights[0] * size
                    pg.draw.line(screen, (0,0,0), pos_start, pos_stop1)
            
            for point in points:
                pos_circle = point * size
                pg.draw.circle(screen, (255,0,0), (int(pos_circle[0]), int(pos_circle[1])), 4)

            pg.display.update()
            _, deltas = my_sofm.train(points[np.random.randint(0, num_points)])

            tick += 1

            index += 1
            if index >= num_points:
                index = 0

            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pg.quit()
                    return


if __name__ == "__main__":
    main()