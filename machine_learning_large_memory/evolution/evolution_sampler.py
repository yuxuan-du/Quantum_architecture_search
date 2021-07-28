import numpy as np
from evolution import nsganet as engine
from pymop.problem import Problem
from pymoo.optimize import minimize


class EvolutionSampler:
    def __init__(self, pop_size=50, n_gens=20, n_layers=3, n_blocks=12):
        self.pop_size = pop_size
        self.n_gens = n_gens
        self.n_layers = n_layers
        self.n_blocks = n_blocks

    def sample(self, eval_func=None):
        subnet_eval_dict = {}
        n_offspring = None #40
        # setup NAS search problem
        n_var = self.n_layers
        lb = np.zeros(n_var)  # left index of each block
        ub = np.zeros(n_var) + self.n_blocks - 1  # right index of each block

        nas_problem = NAS(n_var=n_var, n_obj=1, n_constr=0, lb=lb, ub=ub,
                            eval_func=eval_func,
                            result_dict=subnet_eval_dict)

        # configure the nsga-net method
        method = engine.nsganet(pop_size=self.pop_size,
                                n_offsprings=n_offspring,
                                eliminate_duplicates=True)

        res = minimize(nas_problem,
                        method,
                        callback=lambda algorithm: self.generation_callback(algorithm),
                        termination=('n_gen', self.n_gens))

        subnet_topk = []
        sorted_subnet = sorted(subnet_eval_dict.items(), key=lambda i: i[1], reverse=True)
        sorted_subnet_key = [x[0] for x in sorted_subnet]
        subnet_topk = sorted_subnet_key[:10]
        print('== search result ==')
        print(sorted_subnet)
        print('== best subnet ==')
        print(subnet_topk)
        self.subnet_topk = subnet_topk
        self.subnet_eval_dict = subnet_eval_dict
        return sorted_subnet


    def generation_callback(self, algorithm):
        gen = algorithm.n_gen
        pop_var = algorithm.pop.get("X")
        pop_obj = algorithm.pop.get("F")
        print(f'==Finished generation: {gen}')


# ---------------------------------------------------------------------------------------------------------
# Define your NAS Problem
# ---------------------------------------------------------------------------------------------------------
class NAS(Problem):
    # first define the NAS problem (inherit from pymop)
    def __init__(self, n_var=20, n_obj=1, n_constr=0, lb=None, ub=None, eval_func=None, result_dict=None):
        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_constr, type_var=np.int, )
        self.xl = lb
        self.xu = ub
        self._n_evaluated = 0  # keep track of how many architectures are sampled
        self.eval_func = eval_func
        self.result_dict = result_dict

    def _evaluate(self, x, out, *args, **kwargs):

        objs = np.full((x.shape[0], self.n_obj), np.nan)

        for i in range(x.shape[0]):
            arch_id = self._n_evaluated + 1

            # all objectives assume to be MINIMIZED !!!!!
            if self.result_dict.get(str(x[i])) is not None:
                acc = self.result_dict[str(x[i])]
            else:
                acc = self.eval_func(x[i])
                self.result_dict[str(x[i])] = acc

            print('==evaluation subnet:{} score:{}'.format(str(x[i]), acc))

            objs[i, 0] = 100 - acc  # performance['valid_acc']
            # objs[i, 1] = 10  # performance['flops']

            self._n_evaluated += 1
        out["F"] = objs
        # if your NAS problem has constraints, use the following line to set constraints
        # out["G"] = np.column_stack([g1, g2, g3, g4, g5, g6]) in case 6 constraints


# ---------------------------------------------------------------------------------------------------------
# Define what statistics to print or save for each generation
# ---------------------------------------------------------------------------------------------------------
def do_every_generations(algorithm):
    # this function will be call every generation
    # it has access to the whole algorithm class
    gen = algorithm.n_gen
    pop_var = algorithm.pop.get("X")
    pop_obj = algorithm.pop.get("F")
    print(gen)

    # print(gen, pop_var, pop_obj)

    # report generation info to files

def main():
    # hyper parameters
    pop_size = 50
    n_gens = 20
    n_offspring = 40

    # setup NAS search problem
    n_var = 20
    lb = np.zeros(n_var)  # left index of each block
    ub = np.zeros(n_var) + 4  # right index of each block

    nas_problem = NAS(n_var=n_var, n_obj=1, n_constr=0, lb=lb, ub=ub)

    # configure the nsga-net method
    method = engine.nsganet(pop_size=pop_size,
                            n_offsprings=n_offspring,
                            eliminate_duplicates=True)

    res = minimize(nas_problem,
                   method,
                   callback=do_every_generations,
                   termination=('n_gen', n_gens))
    print(dir(res))
    print(len(res.pop))
    for pop in res.pop[:10]:
        print(pop.F, pop.X)
    return res


if __name__ == "__main__":
    main()
