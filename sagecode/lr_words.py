from crystals import TimedTableau
from sage.combinat.tableau import SemistandardTableaux

def lr_pairs(mu,nu,la=None,n=None):
    tmu = TimedTableau(SemistandardTableaux(mu).first(),rows=True)
    for tnu in SemistandardTableaux(nu,max_entry=len(nu)):
        tnu = TimedTableau(tnu, rows=True)
        w = tnu.concatenate(tmu)
        if la is None or w.weight()==la:
            yield tmu, tnu

def lr_coefficient(mu, nu, la):
    return length(list(lr_pairs(mu, nu, la)))
