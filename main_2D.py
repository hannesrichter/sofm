#! /usr/bin/python3
""""""

import pygame as pg
import numpy as np
import math
from random import random

from sofm import SOFM

def random_circle(radius, off_x, off_y):
    t = 2 * math.pi * random()
    u = random() + random()
    if u > 1:
        r = 2 - u
    else:
        r = u
    return radius * r * math.cos(t) + off_x, radius * r * math.sin(t) + off_y

def main():
    my_sofm = SOFM(2, 10, 10, 0.8, 0.9999, 0.05)
    
    size = 640

    pg.init()
    screen = pg.display.set_mode((size, size))

    while 1==1:
        weights = my_sofm.get_weights()
        screen.fill((255, 255, 255))
        for it_x in range(weights.shape[0]):
            for it_y in range(weights.shape[1]):
                pos_start = weights[it_x][it_y] * size
                
                if it_x < weights.shape[0] - 1:
                    pos_stop1 = weights[it_x + 1][it_y] * size
                    pg.draw.line(screen, (0,0,0), pos_start, pos_stop1)
                
                if it_y < weights.shape[1] - 1:
                    pos_stop2 = weights[it_x][it_y + 1] * size
                    pg.draw.line(screen, (0,0,0), pos_start, pos_stop2)

        pg.display.update()
        my_sofm.train(random_circle(0.3, 0.5, 0.5))
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                return


if __name__ == "__main__":
    main()