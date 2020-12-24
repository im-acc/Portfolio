import pygame as pg
import sys
import numpy as np

'''
Pygame inverse kinematics simulation
'''
NB_BLOCS = 4

WIDTH = 640
HEIGHT = 480
PI = 3.1416

class Bloc:

    def __init__(self, side=40, x0=0, y0=100):
        self.side=side # diametre
        self.x=x0
        self.y=y0
        self.next_tige=None
        self.next_bloc=None
        self.prev_tige=None
        self.prev_bloc=None

    # draw function
    def draw(self, screen):
        x,y = dw_coord(self.x, self.y)
        pg.draw.circle(screen, (140,180,250), (x,y), self.side/2)

    # Position update function
    def update(self, screen):
        '''
        if self.next_tige is None: # dernier bloc de la chaine -> sourie
            x,y = pg.mouse.get_pos()
            self.x = x-WIDTH/2
            self.y = HEIGHT - y
        '''
        if self.prev_tige is not None: # dans la chaine
            self.x = self.prev_tige.x2
            self.y = self.prev_tige.y2

class Tige:

    def __init__(self, bloc1, bloc2, L=60, angle=PI/2, width=3): # connecte bloc1 --> bloc2
        self.bloc1 = bloc1
        self.bloc2 = bloc2

        # chaining bloc1 & bloc 2
        self.bloc1.next_tige=self
        self.bloc2.prev_tige=self

        self.width = width
        self.L = L # longeure
        self.x1=0
        self.x2=0
        self.y1=0
        self.y2=0
        self.theta=angle # trigo angle
        self.update(0.001) # mettre a jour les valeurs de x,y

    def draw(self, screen):
        x1,y1 = dw_coord(self.x1, self.y1)
        x2,y2 = dw_coord(self.x2, self.y2)
        pg.draw.line(screen, (140,140,140), (x1,y1), (x2,y2), self.width)

    def update(self, dt):
        # placer au centre des bloc
        self.x1 = self.bloc1.x
        self.y1 = self.bloc1.y
        self.x2 = self.x1 + self.L*np.cos(self.theta)
        self.y2 = self.y1 + self.L*np.sin(self.theta)

class Arm:

    def __init__(self, nb_blocs=NB_BLOCS, tige_length=60):
        self.tige_length = tige_length
        self.nb_blocs = nb_blocs # n : nb de blocs
        self.blocs = [ Bloc() for i in range(nb_blocs) ]
        self.tiges = [ Tige(self.blocs[i], self.blocs[i+1], angle=PI/2+np.random.randn()*0.01) for i in range(nb_blocs-1) ]
        self.thetas = np.array([tige.theta for tige in self.tiges])
        self.J = np.zeros((2,len(self.thetas))) # 2 x n-1

        for bloc in self.blocs: # update des bloc
            bloc.update(0.0001)

        # init p, dp (le bout)
        '''
        mousex = self.blocs[-1].x + WIDTH/2
        mousey = HEIGHT - self.blocs[-1].y
        pg.mouse.set_pos([mousex, mousey])
        '''
        self.old_px, self.old_py = pg.mouse.get_pos()
        self.dp = np.zeros((2,1))


    def update(self, dt):
        ''' 1. update thetas : Inverse Jacobian methode'''
        for j in range(150):
            # 1.1 compute dp
            px,py = pg.mouse.get_pos()
            self.dp[0][0] = px - self.old_px
            self.dp[1][0] = py - self.old_py
            # 1.2 computing the Jacobian
            self.J[0] = -self.tige_length*np.sin(self.thetas)
            self.J[1] = self.tige_length*np.cos(self.thetas)
            # 1.3 computing J pseudo Inverse
            Jpseudo = np.linalg.pinv(self.J)
            # 1.4 updating thetas <- thetas + J+ dp
            self.thetas += (Jpseudo @ self.dp).flatten()
            self.old_px = px
            self.old_py = py

        ''' 2. update positions '''
        for i in range(len(self.tiges)):
            self.tiges[i].theta = self.thetas[i]
            self.tiges[i].update(dt)
        for bloc in self.blocs:
            bloc.update(dt)

        ''' 3. update old px, py'''
        self.old_px = px
        self.old_py = py

    def draw(self, screen):
        for tige in self.tiges:
            tige.draw(screen)
        for bloc in self.blocs:
            bloc.draw(screen)



''' Utilities '''

# Drawing coordinates
def dw_coord(x,y):
    return x+WIDTH/2, HEIGHT-y


# Main Loop
def main():

    def draw():
        screen.blit(background, (0,0))
        arm.draw(screen)
        pg.display.update()

    def update():
        dt = clock.get_time()*1e-3
        arm.update(dt)
        clock.tick()


    # Game Initializer
    pg.init()
    screen = pg.display.set_mode((WIDTH, HEIGHT))

    ''' INIT BRAS '''
    arm = Arm()



    clock = pg.time.Clock()

    background = pg.Surface(screen.get_size())
    background = background.convert()
    background.fill((250, 250, 250))

    while True:
        draw()
        update()

        # Exit Handler
        for event in pg.event.get():
            if event.type == pg.QUIT:
                sys.exit()

if __name__ == '__main__':
    main()
