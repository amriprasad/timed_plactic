class TimedWord:
    def __init__(self, w):
        if hasattr(w, "_w"):
            self._w = w._w
        elif len(w) == 0:
            self._w = []
        else:
            v = [w[0]]
            for i in range(1, len(w)):
                if w[i][1] == 0:
                    pass
                if w[i][0] == v[-1][0]:
                    v[-1][1] += w[i][1]
                else:
                    v.append([w[i][0], w[i][1]])
            self._w = v

    def __repr__(self):
        return str(self._w)

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
        return output
        
    def length(self):
        return length(self._w)

    def value(self, t):
        return value(self._w, t)

    def segments(self):
        return segements(self._w)

    def dominates(self, other):
        return dominates(self._w, other._w)

    def is_tableau(self):
        b = self.rows()
        if len(b) == 1:
            return True
        else:
            return all([TimedRow(b[i]).dominates(TimedRow(b[i+1])) for i in range(len(b)-1)])

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
        return "".join(["%s^{%.8f}"%(term[0],term[1]) for term in self._w])

    def max(self):
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
    
class TimedTableau(TimedWord):
    def __init__(self, w, rows=False):
        if rows:
            w = sum(reversed(w), [])
        TimedWord.__init__(self, w)
        if not self.is_tableau():
            raise ValueError("%s is not a tableau"%(self))

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
        
class TimedRow(TimedWord):
    def __init__(self, w):
        TimedWord.__init__(self, w)
        if self.number_of_rows() > 1:
            raise ValueError("Input is not a row")

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
        

def length(w):
    return sum([l[1] for l in w])

def value(w, t):
    t0 = 0
    for i, l in enumerate(w):
        t1 = t0 + l[1]
        if t1 > t:
            return l[0]
        else:
            t0 = t1
    return 0
    
def segments(w):
    durations = [l[1] for l in w]
    return [sum(durations[:i]) for i in range(len(w))]
        
def dominates(row1, row2):
    """
    Return True if row1 dominates row2.
    """
    if length(row1) > length(row2):
        return False
    else:
        s1 = segments(row1)
        s2 = segments(row2)
        comb = sorted(s1+s2)
        return all([value(row1, t) > value(row2, t) for t in comb if t < length(row1)])

def is_tableau(w):
    b = break_into_rows(w)
    if len(b) == 1:
        return True
    return all([dominates(b[i], b[i+1]) for i in range(len(b)-1)])
        

def real_schensted(w):
    w = TimedWord(w)
    output = TimedTableau([])
    for row in w.rows():
        output = output.insert_row(TimedRow(row))
    return output

def random_word(max_let, terms, max_time=1):
    return TimedWord([[randint(1, max_let), max_time*random()] for i in range(terms)])

def random_row(max_let):
    return TimedRow([[i+1, random()] for i in range(max_let)])

def random_term(max_let):
    return [randint(1, max_let), random()]

def plactic_rsk(A):
    m,n = A.shape
    return real_schensted(TimedWord([[j+1, A[i,j]] for i in range(m) for j in range(n)])), real_schensted(TimedWord([[i+1, A[i,j]] for j in range(n) for i in range(m)]))
    
w1 = TimedWord([[1, 1], [2, 1.5], [4, 2], [3, 1.2], [1, 3]])
w2 = TimedTableau([[2, 1], [3, 1.5], [1, 1.5], [2, 1.5], [3, 0.2]])
w3 = TimedWord([[2, 1], [2, 1], [1, 2]])
r = TimedRow([[1, 0.5], [2, 0.3], [3, 0.1], [4, 0.7]])
