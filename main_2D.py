#! /usr/bin/python3
""""""

import pygame as pg
import numpy as np

from sofm import SOFM

def main():
    my_sofm = SOFM(2, 10, 10, 0.8, 0.9999, 0.1)
    
    size = 640

    pg.init()
    screen = pg.display.set_mode((size, size))

    while 1==1:
        weights = my_sofm.get_weights()
        screen.fill((255, 255, 255))
        for it_x in range(weights.shape[0] - 1):
            for it_y in range(weights.shape[1] - 1):
                pos_start = weights[it_x][it_y] * size
                pos_stop1 = weights[it_x + 1][it_y] * size
                pos_stop2 = weights[it_x][it_y + 1] * size

                pg.draw.line(screen, (0,0,0), pos_start, pos_stop1)
                pg.draw.line(screen, (0,0,0), pos_start, pos_stop2)
        pg.display.update()
        my_sofm.train(np.random.rand(2))
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                return


if __name__ == "__main__":
    main()