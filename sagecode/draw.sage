from sage.plot.colors import rainbow
from sage.plot.polygon import Polygon

def plot_word(w, y=0, m=None):
    G = Graphics()
    if m is None:
        m = w.max()
    colors = rainbow(m)
    left = 0
    for term in w.to_list():
        right = left + term[1]
        G += polygon([(left,y), (left,y+0.3), (right,y+0.3), (right,y)], color=colors[term[0]-1])
        left = right
    return G

def color_legend(m):
    colorlegend = Graphics()
    for i in range(m):
        colorlegend += polygon([(i, 0), (i, 0.3), (i+1, 0.3), (i+1,0)], color=colors[i]) + text(str(i+1), (i+0.5, 0.15), color="black")
    return colorlegend

def plot_tableau(w):
    m = w.max()
    G = Graphics()
    for i,r in enumerate(t.rows()):
        G += plot_word(r,y=0.3*i,m=m)
    return G
