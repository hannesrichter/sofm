#! /usr/bin/python3
""""""

import pygame as pg
import numpy as np
import math
from random import choice

from sofm import SOFM

def main():
    my_sofm = SOFM(85, 8, 8, 1.0, 0.99999, 0.1, 0.9999)
    
    attributes = []
    names = []
    train_set = {}
    surf_set = {}

    

    size = 800

    pg.init()
    screen = pg.display.set_mode((size, size))
    myfont = pg.font.SysFont('Comic Sans MS', 12)
    clock = pg.time.Clock()

    table_file = open("X:\\AwA2-data\\Animals_with_Attributes2\\predicate-matrix-binary.txt")
    #table_file = open("X:\\AwA2-data\\Animals_with_Attributes2\\predicate-matrix-continuous.txt")
    for l in table_file.readlines():
        cols = l.rstrip().split(' ')
        vec = []
        for c in cols:
            vec.append(int(c))
        attributes.append(vec)

    table_file.close()
    
    table_file = open("X:\\AwA2-data\\Animals_with_Attributes2\\classes.txt")
    for l in table_file.readlines():
        names.append(l.rstrip())

    table_file.close()

    for i, n in enumerate(names):
        train_set[n] = attributes[i]
        surf_set[n] = myfont.render(n, False, (0, 0, 0))



    while 1==1:
        
        screen.fill((255, 255, 255))
        fields = [[0 for _ in range(8)] for _ in range(8)]
        for i, n in enumerate(names):
            x, y = my_sofm.get_best_matching(train_set[n])
            offset = fields[x][y]
            fields[x][y] += 1
            x = x * 100 + 5
            y = y * 100 + offset * 13
            screen.blit(surf_set[n], (x, y))

        pg.display.update()

        #clock.tick(10)

        key = choice(list(train_set.keys()))

        my_sofm.train(train_set[key])
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                return


if __name__ == "__main__":
    main()