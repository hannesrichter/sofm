#! /usr/bin/python3
"""This file uses the self organizing feature maps from sofm.py to run an experiment on feature vectors of 50 different animals"""

import pygame as pg
import numpy as np
import math
from random import choice

from sofm import SOFM
from animals import features, animals


WIDTH = 9
HEIGHT = 9
DIM = 85
SCALE = 100
LINETHICKNESS = 100

def main():
    # initialize the sofm 
    my_sofm = SOFM(DIM, WIDTH, HEIGHT, 1.0, 0.99999, 0.15, 0.999)
    
    train_set = {}
    surf_set = {}

    # calculate the window size
    size = (WIDTH * SCALE, HEIGHT * SCALE)

    # init pygame for visualisation
    pg.init()

    # set up the global pygame objects
    screen = pg.display.set_mode(size)
    myfont = pg.font.SysFont('Comic Sans MS', 12)
    clock = pg.time.Clock()

    # populate the dictionaries for drawing
    for i, n in enumerate(animals):
        train_set[n] = features[i]
        surf_set[n] = myfont.render(n, False, (0, 0, 0))

    # do for ever....
    while 1==1:
        #clear the screen
        screen.fill((255, 255, 255))
        fields = [[0 for _ in range(WIDTH)] for _ in range(HEIGHT)]

        #draw the labels to the field
        for i, n in enumerate(animals):
            # get the best matching neurons position
            x, y = my_sofm.get_best_matching(train_set[n])

            #how many classes sit on this neuron?
            offset = fields[x][y]
            fields[x][y] += 1

            # scale to the window size and draw the label at this position
            x = x * SCALE + 5
            y = y * SCALE + offset * 13
            screen.blit(surf_set[n], (x, y))

        #draw the seperating lines
        w = my_sofm.get_weights()
        for x in range(WIDTH):
            for y in range(HEIGHT):
                
                if x < WIDTH - 1:
                    # get the distance of the two neighbors
                    d = my_sofm.dist(w[x][y], w[x+1][y])

                    # estimate the line thickness and draw the line
                    line_thickness = min(int(d/DIM * LINETHICKNESS), 8)
                    pg.draw.line(screen, (0,0,0), ((x+1)*SCALE, (y)*SCALE), ((x+1)*SCALE, (y+1)*SCALE), line_thickness)
                
                if y < HEIGHT -1:
                    # get the distance of the two neighbors
                    d = my_sofm.dist(w[x][y], w[x][y+1])

                    # estimate the line thickness and draw the line
                    line_thickness = min(int(d/DIM * LINETHICKNESS), 8)
                    pg.draw.line(screen, (0,0,0), ((x)*SCALE, (y+1)*SCALE), ((x+1)*SCALE, (y+1)*SCALE), line_thickness)

        # show all the changes on the screen
        pg.display.update()

        # uncomment this to slow down the visualisation 
        #clock.tick(30)

        # select a random key from the dict
        key = choice(list(train_set.keys()))

        # do the training step of the sofm
        my_sofm.train(train_set[key])

        # react if the user wants to quit the app
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                return


if __name__ == "__main__":
    main()