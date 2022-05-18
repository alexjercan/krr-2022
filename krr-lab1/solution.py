from representation import *

def sortPKey(p: Expr):
    if isinstance(p, And):
        return 0
    if isinstance(p, Or):
        return 1
    return 2


class SemanticTableaux:
    def __init__(self, root: list[Expr]):
        self.root = sorted([p.simplify() for p in root], key=sortPKey)

    def check(self):
        queue = [self.root]

        while queue:
            node = queue.pop()
            p = node.pop(0)
            if isinstance(p, And):
                queue.append(sorted([p.left_expr, p.right_expr, *node], key=sortPKey))
            elif isinstance(p, Or):
                queue.append(sorted([p.left_expr, *node], key=sortPKey))
                queue.append(sorted([p.right_expr, *node], key=sortPKey))
            else:
                ps = list(set([p, *node]))
                is_contradiction = any([p.is_negation_of(q) for p in ps for q in ps])
                if not is_contradiction:
                    return ps
        return False

if __name__ == "__main__":
    p1 = Implies(Proposition("WS", True), Proposition("PB", True))
    p2 = Implies(Proposition("PB", True), Proposition("WD", True))
    p3 = Implies(Proposition("WD", True), Proposition("FD", True))
    p4 = And(Proposition("WS", True), Proposition("FD", False))

    s = SemanticTableaux([p1, p2, p3, p4])
    print(s.check())

    p1 = Implies(Proposition("PLA", True), Proposition("PBA", True))
    p2 = Implies(Proposition("WLA", True), Proposition("WBA", True))
    p3 = Implies(Proposition("SLA", True), Proposition("SBA", True))
    p4 = Implies(Proposition("WBA", True), Proposition("BA", True))
    p5 = Or(Proposition("PLA", True), Or(Proposition("WLA", True), Proposition("SLA", True)))
    p6 = Proposition("BA", True)

    s = SemanticTableaux([p1, p2, p3, p4, p5, p6])
    print(s.check())

    p1 = Implies(Proposition("YB", True), Proposition("YPH", True))
    p2 = Implies(Proposition("YPH", True), Proposition("YL", False))
    p3 = Equivalent(Proposition("DARY", True), Proposition("YL", True))
    p4 = And(Proposition("YB", True), Proposition("DARY", True))

    s = SemanticTableaux([p1, p2, p3, p4])
    print(s.check())

    p1 = Or(Proposition("AC", True), Proposition("AB", True))
    p2 = Equivalent(Proposition("CW", True), Proposition("CG", True))
    p3 = Equivalent(Proposition("AW", True), And(Proposition("AC", True), Proposition("CW", True)))
    p4 = Equivalent(Proposition("AW", True), And(Proposition("AB", True), Proposition("BW", True)))
    p5 = Equivalent(Proposition("AB", True), Proposition("CW", False))
    p6 = Equivalent(Proposition("AC", True), Proposition("CG", True))
    p7 = Equivalent(Proposition("BW", True), Proposition("CT", False))
    p8 = Proposition("CG", False)
    p9 = Proposition("CT", True)
    p10 = Proposition("AW", True)

    s = SemanticTableaux([p1, p2, p3, p4, p5, p6, p7, p8, p9, p10])
    print(s.check())
