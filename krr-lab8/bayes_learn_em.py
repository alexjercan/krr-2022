from typing import List, Dict, Tuple, Any, Iterator
from bayes_net import BayesNet, BayesNode, JunctionTree
from itertools import product
import numpy as np
from argparse import ArgumentParser, Namespace
from collections import defaultdict
from tqdm import tqdm


def all_dicts(variables: List[str]) -> Iterator[Dict[str, int]]:
    for keys in product(*([[0, 1]] * len(variables))):
        yield dict(zip(variables, keys))


def cross_entropy(bn1: BayesNet, bn2: BayesNet, nsamples: int = None) -> float:
    cross_ent = .0
    if nsamples is None:
        bn1_vars = bn1.nodes.keys()
        for sample in all_dicts(bn1_vars):
            cross_ent -= np.exp(bn1.sample_log_prob(sample)) * bn2.sample_log_prob(sample)
    else:
        for _ in range(nsamples):
            cross_ent -= bn2.sample_log_prob(bn1.sample())
        cross_ent /= nsamples
    return cross_ent


def read_samples(file_name: str) -> List[Dict[str, int]]:
    samples = []
    with open(file_name, "r") as handler:
        lines = handler.readlines()
        # read first line of file to get variables in order
        variables = [str(v) for v in lines[0].split()]

        for i in range(1, len(lines)):
            vals = [int(v) for v in lines[i].split()]
            sample = dict(zip(variables, vals))
            samples.append(sample)

    return samples


class MLEBayesNet(BayesNet):
    """
    Placeholder class for the Bayesian Network that will learn CPDs using the frequentist MLE
    """
    def __init__(self, bn_file: str="data/bnet"):
        super(MLEBayesNet, self).__init__(bn_file=bn_file)
        self._reset_cpds()

    def _reset_cpds(self) -> None:
        """
        Reset the conditional probability distributions to a value of 0.5
        The values have to be **learned** from the samples.
        """
        for node_name in self.nodes:
            self.nodes[node_name].cpd["prob"] = 0.5

    def learn_cpds(self, samples: List[Dict[str, int]], alpha: float = 1.0) -> None:
        # TODO: YOUR CODE HERE
        pass


class ParametricBayesNet(BayesNet):
    """
    Placeholder class for the Bayesian Network that will learn CPDs using the frequentist MLE
    """
    def __init__(self, bn_file: str="data/bnet"):
        super(ParametricBayesNet, self).__init__(bn_file=bn_file)
        self._random_cpds()

    def _random_cpds(self):
        """
        Reset the conditional probability distributions to a random value between 0 and 1 that will be passed as the
        parameter to the sigmoid function
        """
        for node_name in self.nodes:
            self.nodes[node_name].cpd["prob"] = np.random.random_sample(self.nodes[node_name].cpd.shape[0])

    def prob(self, var_name: str, parent_values: List[int] = None) -> float:
        ### var_name = 1 | parent_values
        if parent_values is None:
            parent_values = []
        index_line = tuple([1] + parent_values)
        score = self.nodes[var_name].cpd.loc[index_line]["prob"]

        return 1. / (1. + np.exp(-score))

    def learn(self, sample: Dict[str, int],
              learning_rate: float = 1e-3) -> None:

        for var_name in self.nodes:
            parent_names = [p.var_name for p in self.nodes[var_name].parent_nodes]
            parent_vals = [sample[p] for p in parent_names]
            index_line = tuple([1] + parent_vals)
            score = self.nodes[var_name].cpd.loc[index_line]["prob"]

            # TODO 2: YOUR CODE HERE
            # compute the gradient with respect to the parameters modeling the CPD of variable var_name
            # grad = ...

            # update the parameters - using gradient ASCENT
            # self.nodes[var_name].cpd.loc[index_line]["prob"] = ...

    def pretty_print(self):
        res = "Bayesian Network:\n"
        for var_name in self.nodes:
            res += str(self.nodes[var_name]) + "\n"
            cpd = self.nodes[var_name].cpd.copy()
            cpd["true_prob"] = 1. / (1. + np.exp(-cpd["prob"]))
            cpd.loc[0, "true_prob"] = 1 - cpd.loc[1, "true_prob"]

            res += str(cpd)
            res += "\n"

        return res


class EMBayesNet(MLEBayesNet):
    def __init__(self, bn_file: str="data/bnet") -> None:
        super(EMBayesNet, self).__init__(bn_file=bn_file)
        self.cpds = {}  # type: Dict[str, Dict[tuple, float]]
        self.bn_file = bn_file

    def learn_cpds(self, samples_with_missing: List[Dict[str, int]],
              alpha: float = 1.):
        bn = BayesNet(self.bn_file)

        # 1. Compute junction tree for the bayes net
        T = JunctionTree(bn, debug=False)

        for _ in range(5):
            # ESTIMATION STEP
            # a. initialize counters to zero
            N = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
            root_name = None

            # basically just run query
            T._load_factors()

            # get junction tree copy
            jt, root_name = T._get_junction_tree(root_name=root_name)

            # b. iterate over saples
            for sample in tqdm(samples_with_missing):
                # c. incorporate the observed nodes from x the ones with 0 or 1 as evidence in T
                evidence = {k: v for k, v in sample.items() if v != 2}

                # incorporate evidence
                uncalibrared_jt = T._incorporate_evidence(jt, evidence)

                # d. calibrate T
                calibrated_jt = T._run_belief_propagation(uncalibrared_jt, root_name)

                for i in self.nodes: # i = this represents the i index for N
                    pa = [p.var_name for p in self.nodes[i].parent_nodes]
                    j = tuple([sample[p] for p in pa if sample[p] != 2]) # j = this represents the index j for N
                    k = sample[i] # k = this represents the index k in N
                    if k == 2:
                        # e. compute gamma ijk with eval query
                        gamma = T._eval_query(calibrated_jt, {i: 0})
                        # f. N ijk += gamma ijk
                        N[i][j][0] += gamma
                        N[i][j][1] += (1 - gamma)
                    else:
                        N[i][j][k] += 1

            # MAXIMIZATION STEP
            K = 2
            for i in self.nodes:
                for j in N[i]:
                    k_sum = sum(N[i][j].values())
                    for k in range(K):
                        self.nodes[i].cpd.loc[(k, *j)]["prob"] = (N[i][j][k] + alpha) / (k_sum + K * alpha)




def get_args() -> Namespace:
    arg_parser = ArgumentParser()
    arg_parser.add_argument("-f", "--file",
                            type=str,
                            default="data/bn_learning",
                            dest="file_name",
                            help="Input file")
    arg_parser.add_argument("-s", "--samplefile",
                            type=str,
                            default="data/samples_bn_learning",
                            dest="samples_file_name",
                            help="Samples file")
    arg_parser.add_argument("-n", "--nsteps",
                            type=int,
                            default=10000,
                            dest="nsteps",
                            help="Number of optimization steps")
    arg_parser.add_argument("--lr",
                            type=float,
                            default=.005,
                            dest="lr",
                            help="Learning rate")

    return arg_parser.parse_args()


def main():
    args = get_args()
    table_bn = BayesNet(bn_file=args.file_name)
    em_bn = EMBayesNet(bn_file=args.file_name)

    print("========== EM ==========")
    samples = read_samples(args.samples_file_name)
    em_bn.learn_cpds(samples)

    print("Reference BN")
    print(table_bn.pretty_print())

    print("MLE BayesNet after learning CPDs")
    print(em_bn.pretty_print())


if __name__ == "__main__":
    main()
