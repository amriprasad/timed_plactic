from copy import copy
from random import random, randint
from numpy import zeros
from numpy.random import rand

class TimedWord:
    """
    The class of timed words.
    """
    def __init__(self, w):
        """
        Construct instance from ``w``.
        """
        if hasattr(w, "_w"):
            self._w = w._w
        elif len(w) == 0:
            self._w = []
        else:
            v = []
            for i in range(len(w)):
                if w[i][1] == 0:
                    pass
                elif len(v)==0:
                    v.append([w[i][0], w[i][1]])
                elif w[i][0] == v[-1][0]:
                    v[-1][1] += w[i][1]
                else:
                    v.append([w[i][0], w[i][1]])
            self._w = v

    def __repr__(self):
        return str(self._w)

    def __eq__(self, other, tol=1e-10):
        return all([v[0]==u[0] and abs(v[1]-u[1])<tol for u, v in zip(self._w,other._w)])

    def to_list(self):
        return self._w

    def rows(self):
        w = self._w
        if w == []:
            return []
        output = [[w[0]]]
        for entry in w[1:]:
            if entry[0] >= output[-1][-1][0]:
                output[-1].append(entry)
            else:
                output.append([entry])
        return [TimedRow(r) for r in output]
        
    def length(self):
        return sum([l[1] for l in self._w])

    def value(self, t):
        t0 = 0
        for i, l in enumerate(self._w):
            t1 = t0 + l[1]
            if t1 > t:
                return l[0]
            else:
                t0 = t1
        return 0

    def time_stamps(self):
        durations = [l[1] for l in self._w]
        return [sum(durations[:i]) for i in range(len(self._w))]

    def segment(self, interval):
        left, right = interval[0], interval[1]
        l = self.length()
        ts = self.time_stamps() + [l]
        output = []
        for i, s in enumerate(ts):
            if s <= left:
                pass
            else:
                output.append([self._w[i-1][0], min(s, right)- max(ts[i-1], left)])
            if s>=right:
                return TimedWord(output)
        return TimedWord(output)
    
    def subword(self, intervals):
        output = TimedWord([])
        for interval in intervals:
            output.concatenate(self.segment(interval))
        return output
        
    def is_tableau(self):
        b = self.rows()
        if len(b) == 1:
            return True
        else:
            return all([b[i].dominates(b[i+1]) for i in range(len(b)-1)])

    def number_of_rows(self):
        return len(self.rows())

    def concatenate(self, other):
        return TimedWord(self._w + other._w)

    def insertion_tableau(self):
        output = TimedTableau([])
        for term in self._w:
            output = output.insert_row(TimedRow([term]))
        return output

    def latex(self):
        return "".join(["%s^{%.2f}"%(term[0],term[1]) for term in self._w])

    def max(self):
        if len(self._w)==0:
            return 0
        else:
            return max([term[0] for term in self._w])
    
    def weight(self):
        output = [0]*self.max()
        for term in self._w:
            output[term[0]-1]+=term[1]
        return output

    def shape(self):
        return [TimedRow(r).length() for r in reversed(self.rows())]

    def restrict(self, m):
        """
        Return only terms with letters at most ``m``.
        """
        return TimedWord([c for c in self._w if c[0]<= m])

    def schuetzenberger_involution(self, max_let=None):
        if max_let is None:
            max_let = self.max()
        return TimedWord([[max_let-v[0]+1, v[1]] for v in reversed(self._w)])

    def truncate(self, l):
        if l>= self.length():
            return self
        else:
            output = []
            s = self.time_stamps() + [l]
            for i, v in enumerate(self._w):
                if s[i+1] < l:
                    output.append(v)
                else:
                    output.append([v[0], l-s[i]])
                    return TimedWord(output)
    
class TimedTableau(TimedWord):
    def __init__(self, w, rows=False, gt=False):
        if rows:
            w = sum(reversed(w), [])
        if gt:
            n = len(w)
            rows = [TimedRow([])]*n
            for k in range(n):
                for i in range(k):
                    rows[i] = rows[i].concatenate(TimedRow([[k+1, w[n-k-1][i]-rows[i].length()]]))
            w = TimedWord([])
            for r in reversed(rows):
                w = w.concatenate(r)
        TimedWord.__init__(self, w)

    def split_first_row(self):
        for i in range(1, len(self._w)):
            if self._w[i] < self._w[i-1]:
                break
        first_row = TimedRow(self._w[:i])
        rest = TimedTableau(self._w[i:])
        return first_row, rest
        
    def pre_insert_row(self, row):
        if self.number_of_rows() <= 1:
            rowin = TimedRow(self).insert_row(row)
            return rowin[0], rowin[1]
        else:
            first_row, rest = self.split_first_row()
            bumped, newrest = rest.pre_insert_row(row)
            out = first_row.insert_row(bumped)
            return out[0], TimedTableau(out[1].concatenate(newrest))

    def insert_row(self, row):
        out = self.pre_insert_row(row)
        return TimedTableau(out[0].concatenate(out[1]))
    
    def gt_pattern(self):
        m = max([term[0] for term in self._w])
        arr = [[sum([term[1] for term in row if term[0] < i+1]) for row in reversed(self.rows())] for i in range(1, m+1)]
        return [(line+[0]*(max(0, i + 1 - len(line))))[:i+1] for i, line in enumerate(arr)]

    def partition_chain(self):
        return [self.restrict(m+1).shape() for m in range(self.max())]

    def plactic_product(self, other):
        return self.concatenate(other).insertion_tableau()
        
class TimedRow(TimedWord):
    def __init__(self, w):
        TimedWord.__init__(self, w)

    def dominates(self, other):
        """
        Return True if self dominates other.
        """
        if self.length() > other.length():
            return False
        else:
            s1 = self.time_stamps()
            s2 = other.time_stamps()
            comb = sorted(s1+s2)
            return all([self.value(t) > other.value(t) for t in comb if t < self.length()])

    def insert_term(self, term):
        c = term[0]
        t = term[1]
        if self._w == []:
            return TimedRow([]), TimedRow([[c, t]])
        # Find first term which is greater than c - this is the term w[i]
        postfix = []
        for i, l in enumerate(self._w):
            if l[0] > c:
                break
            else:
                postfix.append(copy(self._w[i]))
        else:
            return TimedRow([]), TimedRow(self.concatenate(TimedRow([[c, t]])))
        sofar = 0
        prefix = []
        for j in range(i, len(self._w)):
            next_time = sofar + self._w[j][1]
            if next_time >= t:
                t0 = t - sofar
                prefix.append([self._w[j][0], t0])
                postfix.append([c, t - sofar])
                if next_time > t:
                    postfix.append([self._w[j][0], next_time - t])
                break
            else:
                prefix.append(self._w[j])
                postfix.append([c, self._w[j][1]])
                sofar = next_time
        if next_time < t:
            postfix.append([c, t - sofar])
        else:
            for k in range(j+1, len(self._w)):
                postfix.append(self._w[k])
        return TimedRow(prefix), TimedRow(postfix)

    def insert_row(self, other):
        first_row = TimedRow([])
        second_row = TimedRow(self._w)
        for term in other._w:
            step = second_row.insert_term(term)
            second_row = step[1]
            first_row = TimedRow(first_row.concatenate(step[0]))
        return TimedRow(first_row), TimedRow(second_row)
    
    def schuetzenberger_involution(self, max_let=None):
        return TimedRow(TimedWord.schuetzenberger_involution(self, max_let=max_let))

def inverse_rowins(vv, uu, r, max_let=None):
    if max_let is None:
        n = max(vv.max(), uu.max())
    else:
        n = max_let
    rr = uu.length()
    v1, u1 = TimedRow(uu.segment([0,r]).schuetzenberger_involution(max_let=n)).insert_row(vv.schuetzenberger_involution(max_let=n))
    return u1.schuetzenberger_involution(max_let=n), v1.schuetzenberger_involution(max_let=n).concatenate(uu.segment([r,rr]))
    
    
def delete(w, la):
    mu = w.shape()
    n = w.max()
    wrows = w.rows()
    if len(la) == len(mu):
        wrows = [TimedRow([])] + wrows
    output = TimedTableau([])
    x = wrows[0]
    for i, u in enumerate(wrows[1:]):
        r, x = inverse_rowins(x, u, la[-i-1], max_let=n)
        output = output.concatenate(r)
    return x, output

def random_word(max_let, terms, max_time=1):
    return TimedWord([[randint(1, max_let), max_time*random()] for i in range(terms)])

def random_row(max_let, max_time=1):
    return TimedRow([[i+1, max_time*random()] for i in range(max_let)])

def random_term(max_let):
    return [randint(1, max_let), random()]

def real_rsk(A):
    m,n = A.shape
    return TimedWord([[j+1, A[i,j]] for i in range(m) for j in range(n)]).insertion_tableau(), TimedWord([[i+1, A[i,j]] for j in range(n) for i in range(m)]).insertion_tableau()

def inverse_real_rsk(P,Q):
    m = Q.max()
    n = P.max()
    A = zeros([m,n])
    for i in range(m,0,-1):
        Q = Q.restrict(i-1)
        u, P = delete(P, Q.shape())
        for term in u.to_list():
            A[i-1,term[0]-1]= term[1]
    return A

def random_real_matrix(m, n):
    return rand(m,n)
