
import pygame as pg
import numpy as np
from sofm import SOFM
import colorsys

def main():
    size = 200
    scale = 4

    my_sofm = SOFM(3, size, size, 1.0, 0.999, 0.1)

    pg.init()

    screen = pg.display.set_mode((size * scale, size * scale))

    step = 0
    while 1==1:
        sample = np.random.rand(3)
        bm_x, bm_y = my_sofm.train(sample)
        step += 1

        if (step > 100):
            step = 0
            weights = my_sofm.get_weights()
            for it_x in range(size):
                for it_y in range(size):
                    rec = pg.Rect(it_x * scale, it_y * scale, scale, scale)
                    col1, col2, col3 = weights[it_x][it_y]
                    col1 = max(0, min(255, col1 * 255))
                    col2 = max(0, min(255, col2 * 255))
                    col3 = max(0, min(255, col3 * 255))
                    pg.draw.rect(screen, (col1, col2, col3), rec)

            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pg.quit()
                    return
            pg.display.flip()
        

if __name__ == "__main__":
    main()