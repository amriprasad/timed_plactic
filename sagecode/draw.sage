from sage.plot.colors import rainbow, rgbcolor
from sage.plot.polygon import polygon
from sage.plot.graphics import Graphics
from sage.plot.text import text

def plot_word(w, y=0, m=None, colors=None, offset=None):
    G = Graphics()
    if m is None:
        m = w.max()
    if offset is None:
        x0, y0 = 0, 0
    else:
        x0, y0 = offset
    if colors is None:
        colors = rainbow(m)
    elif colors == "grey":
        step = 1.0/(m-1)
        colors = [Color((m-i-1)*step,(m-i-1)*step, (m-i-1)*step) for i in range(m)]
    left = 0
    for term in w.to_list():
        right = left + term[1]
        G += polygon([(left+x0,y+y0), (left+x0,y+0.3+y0), (right+x0,y+0.3+y0), (right+x0,y+y0)], color=colors[term[0]-1], edgecolor="black")
        left = right
    return G

def color_legend(m, offset=None):
    colorlegend = Graphics()
    colors = rainbow(m)
    if offset is None:
        x0, y0 = 0, 0
    else:
        x0, y0 = offset
    for i in range(m):
        colorlegend += polygon([(i+x0, y0), (i+x0, 0.3+y0), (i+1+x0, 0.3+y0), (i+1+x0,y0)], color=colors[i]) + text(str(i+1), (i+0.5+x0, 0.15+y0), color="black")
    return colorlegend

def plot_tableau(w, colors=None, offset=None, m=None):
    if m is None:
        m = w.max()
    if offset is None:
        offset=(0,0)
    G = Graphics()
    for i,r in enumerate(w.rows()):
        G += plot_word(r,y=0.3*i,m=m, colors=colors, offset=offset)
    return G
