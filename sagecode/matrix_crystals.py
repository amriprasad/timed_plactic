from numpy import diag
from numpy.random import randint

def random_symmetric_matrix(n, max=3):
    A = randint(0,max/2+1,[n,n])
    d = diag(randint(0,2,n))
    return A + A.transpose() + d

def crystal_f(A,i):
    P = real_rsk(A)[0]
    Pf = P.f(i,1)
    if Pf is None:
        raise ValueError
    P1 = Pf.to_tableaux()
    return RSK_inverse(P1,P1,'matrix')-Matrix(A)

