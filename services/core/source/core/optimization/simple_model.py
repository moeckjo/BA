from pyomo.environ import *
from pyomo.environ import SolverFactory

model = ConcreteModel()
model.x = Var([1, 2], within=NonNegativeReals)
model.obj = Objective(expr=model.x[1] + 2 * model.x[2])
model.con1 = Constraint(expr=3 * model.x[1] + 4 * model.x[2] >= 1)
model.con2 = Constraint(expr=2 * model.x[1] + 5 * model.x[2] >= 2)


def solve():
    # instance = model.create()
    print('Model description:')
    model.pprint()

    solver = SolverFactory("glpk")
    solver.options['tmlim'] = 1
    solver.options['mipgap'] = 0.05

    results = solver.solve(model)

    print(f'Solution: {[value(model.x[i]) for i in [1,2]]}')
    print(f'Solution: {[value(model.x[i]) for i in model.x.index_set()]}')


    return results
