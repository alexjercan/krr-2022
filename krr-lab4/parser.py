import itertools

class Parser(object):
	@staticmethod
	def parse(file: str):
		'''
		@param file: path to the input file
		:returns Bayesian network as a dictionary {node: [list of parents], ...}
		and the list of queries as [{"X": [list of vars],
		"Y": [list of vars], "Z": [list of vars]}, ... ] where we want
		to test the conditional independence of vars1 ⊥ vars2 | cond
		'''
		bn = {}
		queries = []

		with open(file) as fin:
			# read the number of vars involved
			# and the number of queries
			N, M = [int(x) for x in next(fin).split()]

			# read the vars and their parents
			for i in range(N):
				line = next(fin).split()
				var, parents = line[0], line[1:]
				bn[var] = parents

			# read the queries
			for i in range(M):
				vars, cond = next(fin).split('|')

				# parse vars
				X, Y = vars.split(';')
				X = X.split()
				Y = Y.split()

				# parse cond
				Z = cond.split()

				queries.append({
					"X": X,
					"Y": Y,
					"Z": Z
				})

			# read the answers
			for i in range(M):
				queries[i]["answer"] = next(fin).strip()

		return bn, queries

	@staticmethod
	def get_graph(bn: dict):
		'''
		@param bn: Bayesian netowrk obtained from parse
		:returns the graph as {node: [list of children], ...}
		'''
		graph = {}

		for node in bn:
			parents = bn[node]

			# this is for the leafs
			if node not in graph:
				graph[node] = []

			# for each parent add
			# the edge parent->node
			for p in parents:
				if p not in graph:
					graph[p] = []
				graph[p].append(node)

		return graph


def split_path(path):
    # Split the path in chunks of 3
    # [a,b,c,d] -> [[a,b,c], [b,c,d]]
    if len(path) == 3:
        return [path]

    paths = []
    for i in range(0, len(path) - 2, 1):
        paths.append(path[i:i+3])

    return paths


def get_paths(bn, graph, start, end):
    # BFS to get all the paths from src to dst
    # The paths are split into chunk of 3
    paths = []
    q = []

    path = [start]
    q.append(path)

    while q:
        path = q.pop(0)
        last = path[-1]

        if last == end:
            paths.append(split_path(path))

        for node in graph[last] + bn[last]:
            if node not in path:
                q.append([*path, node])
    return paths


def get_desc(graph, src):
    # graph - directed graph
    # node is the start node

    queue = [src]
    all_desc = []

    while queue:
        node = queue.pop(0)

        desc = graph[node]
        all_desc.extend(desc)
        queue.extend(desc)

    return all_desc


def is_active(path, graph, obs):
    # check if path is active
    # common effect is different from others,
    # any descendent is in obs => active
    # the other 3 cases are the same, is X_i in obs => not active
    [x, y, z] = path

    common_effect = y in graph[x] and y in graph[z]
    if common_effect:
        desc = get_desc(graph, y)
        return any([x in obs for x in [y, *desc]])

    return y not in obs


def is_path_active(path, graph, obs):
    # if any chunk from the path is closed => all path is closed
    return all([is_active(p, graph, obs) for p in path])


def solve(query, graph, bn):
    # if all paths are closed then
    # the answer is true (the X and Y are independent)
    x = query['X']
    y = query['Y']
    z = query['Z']

    paths = []
    for (start, end) in itertools.product(x, y):
        paths.extend(get_paths(bn, graph, start, end))

    closed = all([not is_path_active(path, graph, z) for path in paths])
    info = list(zip(paths, ["closed" if not is_path_active(path, graph, z) else "active" for path in paths]))

    return str(closed).lower() == query['answer'], info


if __name__ == "__main__":
    from pprint import pprint

    # example usage
    bn, queries = Parser.parse("bn1")
    graph = Parser.get_graph(bn)

    print("Bayesian Network\n" + "-" * 50)
    pprint(bn)

    print("\nQueries\n" + "-" * 50)
    pprint(queries)

    print("\nGraph\n" + "-" * 50)
    pprint(graph)

    print("\nTests\n" + "-" * 50)
    score = 0
    for query in queries:
        ok, info = solve(query, graph, bn)
        if ok:
            score += 1
        print(info)
    print(f'{score} / {len(queries)}')

