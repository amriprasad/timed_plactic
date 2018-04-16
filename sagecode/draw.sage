from sage.plot.colors import rainbow, rgbcolor
from sage.plot.polygon import Polygon

def plot_word(w, y=0, m=None, colors=None):
    G = Graphics()
    if m is None:
        m = w.max()
    if colors is None:
        colors = rainbow(m)
    elif colors == "grey":
        step = 1.0/(m-1)
        colors = [Color((m-i-1)*step,(m-i-1)*step, (m-i-1)*step) for i in range(m)]
    left = 0
    for term in w.to_list():
        right = left + term[1]
        G += polygon([(left,y), (left,y+0.3), (right,y+0.3), (right,y)], color=colors[term[0]-1], edgecolor="black")
        left = right
    return G

def color_legend(m):
    colorlegend = Graphics()
    for i in range(m):
        colorlegend += polygon([(i, 0), (i, 0.3), (i+1, 0.3), (i+1,0)], color=colors[i]) + text(str(i+1), (i+0.5, 0.15), color="black")
    return colorlegend

def plot_tableau(w, colors=None):
    m = w.max()
    G = Graphics()
    for i,r in enumerate(t.rows()):
        G += plot_word(r,y=0.3*i,m=m, colors=colors)
    return G
