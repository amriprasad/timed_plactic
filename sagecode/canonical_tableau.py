def canonical_tableau(la, mu, tol=1e-10):
    """
    Return the canonical timed tableau of shape ``la`` and content ``mu``.
    """
    la = [part for part in la if part > tol]
    mu = [part for part in mu if part > tol]
    l = len(la)
    m = len(mu)
    if l==1:
        return TimedTableau([[i+1, mu[i]] for i in range(m)])
    else: #go from bottom row to top row
        removed_from_row = [0]*l
        mulast = mu[m-1]
        la.append(0)
        for r in range(l):
            if la[l-r-1]-la[l-r]>=mulast:
                removed_from_row[r]=mulast
                break
            else:
                removed_from_row[r]=la[l-r-1]-la[l-r]
                mulast=mulast-la[l-r-1]+la[l-r]
        t = canonical_tableau([la[i]-removed_from_row[l-i-1] for i in range(l)], mu[:-1])
        output = TimedWord([])
        rows = t.rows()
        for i, row in enumerate(rows):
            output=output.concatenate(row)
            if removed_from_row[i]>tol:
                output=output.concatenate(TimedWord([[m, removed_from_row[l-i-1]]]))
        if len(rows)<l:
            output=TimedWord([[m, removed_from_row[0]]]).concatenate(output)
        return output
        
        # lmu = []
        # lastrow = TimedRow([])
        # lastrowlen = la[l-1]
        # remaining = lastrowlen
        # for k in range(m,0,-1):
        #     print k
        #     if remaining <= mu[k-1]:
        #         lastrow = TimedRow([[k, remaining]]).concatenate(lastrow)
        #         print "last step", lastrow, k, mu[k-1]
        #         remaining = 0
        #         break
        #     else:
        #         lastrowlen = lastrowlen - mu[k-1]
        #         lastrow = TimedRow([[k, mu[k-1]]]).concatenate(lastrow)
        #         print lastrow
        # print k
        # return lastrow.concatenate(canonical_tableau(la[:-1], mu[:k-2] + [remaining]))
        
