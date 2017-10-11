#! /usr/bin/python3

import pygame as pg
import numpy as np
from sofm import KOHONENRING

NUM_OF_POINTS = 21
DRAW_STEP = 100
MAX_STEP = 10000
WINDOW_SIZE = 640

def main():
    """Simulation of the Traveling Salesman Problem using Kohonen rings to approximate the
    shortest roundtrip. The solution is not perfect but entertaining to watch"""
    size = WINDOW_SIZE
    num_points = NUM_OF_POINTS

    # initialize the pygame system
    pg.init()

    # set the screen size and get teh window handle
    screen = pg.display.set_mode((size, size))
    pg.display.set_caption("TSP with Kohonen rings")

    # for ever do...
    while 1 == 1:
        # initialize a new KOHONENRING instance
        my_sofm = KOHONENRING(2, num_points * 3, 0.99, 0.99999, 0.01)

        # generate random points in [0.0, 1.0] as our "towns"
        points = np.random.rand(num_points, 2)

        # reset the tick counter
        tick = 0

        # while not reached MAX_STEP
        while tick < MAX_STEP:
            # get the all teh weights from the ring
            weights = my_sofm.get_weights()

            # do we need to draw now?
            if tick % DRAW_STEP == 0:
                # fill all white
                screen.fill((255, 255, 255))

                # for each weight in all the weights
                for it_x in range(weights.shape[0]):
                    # mark the start of the line
                    pos_start = weights[it_x] * size

                    # are we the last one?
                    if it_x < weights.shape[0] - 1:
                        # mark the end of the line and draw
                        pos_stop1 = weights[it_x + 1] * size
                        pg.draw.line(screen, (0, 0, 0), pos_start, pos_stop1)
                    else:
                        # mark the first neuron as the end and draw -> this creates a ring
                        pos_stop1 = weights[0] * size
                        pg.draw.line(screen, (0, 0, 0), pos_start, pos_stop1)

                    # draw the neuron as circle
                    pg.draw.circle(screen, (0, 0, 255), (int(pos_start[0]), int(pos_start[1])), 2)

                # for each point in the list of "towns"
                for point in points:
                    # scale the position according to the window and draw
                    pos_circle = point * size
                    pg.draw.circle(screen, (255, 0, 0), (int(pos_circle[0]), int(pos_circle[1])), 4)
                # make the changes visible on the screen
                pg.display.update()

            # perform a training step with one random "town" from the point list
            my_sofm.train(points[np.random.randint(0, num_points)])

            # increment the tick counter
            tick += 1

            # handle the events to see if someone wants to close the window
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pg.quit()
                    return


if __name__ == "__main__":
    main()
