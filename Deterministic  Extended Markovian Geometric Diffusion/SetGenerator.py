import random
import math

def generateSet2D(NUMPOINTS):
    r = 5
    SPACEMIN = -30
    SPACEMAX = 30

    def distance(p, q):
        return math.sqrt((q[0]-p[0])**2 + (q[1]-p[1])**2)

    def calc_new(points):
        x1 = points[0]
        y1 = points[1]

        x2 = x1-2*abs(r)*random.random()+r
        y2 = y1-2*abs(r)*random.random()+r
        return (x2,y2)

    x0 = random.randint(SPACEMIN, SPACEMAX)
    y0 = random.randint(SPACEMIN, SPACEMAX)
    points2D = [(x0, y0)]

    dist_list = []
    while len(points2D) < NUMPOINTS:

        q = random.choice(points2D)
        p = calc_new(q)
        d = distance(q, p)

        if d < r:
            dist_list.append(d)
            points2D.append(p)

    return points2D
