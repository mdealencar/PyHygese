from dataclasses import dataclass
import numpy as np
import sys

from . import _core

C_INT_MAX = 2 ** 31 - 1
C_DBL_MAX = sys.float_info.max


@dataclass
class AlgorithmParameters:
    nbGranular: int = 20
    mu: int = 25
    lambda_: int = 40
    nbElite: int = 4
    nbClose: int = 5
    nbIterPenaltyManagement: int = 100
    targetFeasible: float = 0.2
    penaltyDecrease: float = 0.85
    penaltyIncrease: float = 1.2
    seed: int = 0
    nbIter: int = 20000
    nbIterTraces: int = 500
    timeLimit: float = 0.0
    useSwapStar: bool = True


class RoutingSolution:
    def __init__(self, sol):
        self.cost = sol.cost
        self.time = sol.time
        self.routes = sol.routes
        self.n_routes = len(sol.routes)


class Solver:
    def __init__(self, parameters=AlgorithmParameters(), verbose=True):
        self.algorithm_parameters = parameters
        self.verbose = verbose

    def solve_cvrp(self, data, rounding=True):
        demand = np.asarray(data["demands"], dtype=np.float64)
        vehicle_capacity = data["vehicle_capacity"]
        n_nodes = len(demand)

        depot = data.get("depot", 0)
        if depot != 0:
            raise ValueError("In HGS, the depot location must be 0.")

        maximum_number_of_vehicles = data.get("num_vehicles", C_INT_MAX)

        service_times = data.get("service_times")
        if service_times is None:
            service_times = np.zeros(n_nodes, dtype=np.float64)
        else:
            service_times = np.asarray(service_times, dtype=np.float64)

        duration_limit = data.get("duration_limit")
        if duration_limit is None:
            is_duration_constraint = False
            duration_limit = C_DBL_MAX
        else:
            is_duration_constraint = True

        x_coords = data.get("x_coordinates")
        y_coords = data.get("y_coordinates")
        dist_mtx = data.get("distance_matrix")

        if x_coords is None or y_coords is None:
            assert dist_mtx is not None
            x_coords = np.zeros(n_nodes, dtype=np.float64)
            y_coords = np.zeros(n_nodes, dtype=np.float64)
        else:
            x_coords = np.asarray(x_coords, dtype=np.float64)
            y_coords = np.asarray(y_coords, dtype=np.float64)

        assert len(x_coords) == len(y_coords) == len(service_times) == len(demand)
        assert (x_coords >= 0.0).all()
        assert (y_coords >= 0.0).all()
        assert (service_times >= 0.0).all()
        assert (demand >= 0.0).all()

        if dist_mtx is not None:
            dist_mtx = np.asarray(dist_mtx, dtype=np.float64)
            assert dist_mtx.shape[0] == dist_mtx.shape[1]
            assert (dist_mtx >= 0.0).all()
            sol = _core.solve_cvrp_dist_mtx(
                np.ascontiguousarray(x_coords),
                np.ascontiguousarray(y_coords),
                np.ascontiguousarray(dist_mtx),
                np.ascontiguousarray(service_times),
                np.ascontiguousarray(demand),
                vehicle_capacity,
                duration_limit,
                is_duration_constraint,
                maximum_number_of_vehicles,
                self.algorithm_parameters.nbGranular,
                self.algorithm_parameters.mu,
                self.algorithm_parameters.lambda_,
                self.algorithm_parameters.nbElite,
                self.algorithm_parameters.nbClose,
                self.algorithm_parameters.nbIterPenaltyManagement,
                self.algorithm_parameters.targetFeasible,
                self.algorithm_parameters.penaltyDecrease,
                self.algorithm_parameters.penaltyIncrease,
                self.algorithm_parameters.seed,
                self.algorithm_parameters.nbIter,
                self.algorithm_parameters.nbIterTraces,
                self.algorithm_parameters.timeLimit,
                self.algorithm_parameters.useSwapStar,
                self.verbose,
            )
        else:
            sol = _core.solve_cvrp(
                np.ascontiguousarray(x_coords),
                np.ascontiguousarray(y_coords),
                np.ascontiguousarray(service_times),
                np.ascontiguousarray(demand),
                vehicle_capacity,
                duration_limit,
                rounding,
                is_duration_constraint,
                maximum_number_of_vehicles,
                self.algorithm_parameters.nbGranular,
                self.algorithm_parameters.mu,
                self.algorithm_parameters.lambda_,
                self.algorithm_parameters.nbElite,
                self.algorithm_parameters.nbClose,
                self.algorithm_parameters.nbIterPenaltyManagement,
                self.algorithm_parameters.targetFeasible,
                self.algorithm_parameters.penaltyDecrease,
                self.algorithm_parameters.penaltyIncrease,
                self.algorithm_parameters.seed,
                self.algorithm_parameters.nbIter,
                self.algorithm_parameters.nbIterTraces,
                self.algorithm_parameters.timeLimit,
                self.algorithm_parameters.useSwapStar,
                self.verbose,
            )

        return RoutingSolution(sol)

    def solve_tsp(self, data, rounding=True):
        x_coords = data.get("x_coordinates")
        dist_mtx = data.get("distance_matrix")
        if dist_mtx is None:
            n_nodes = x_coords.size
        else:
            dist_mtx = np.asarray(dist_mtx)
            n_nodes = dist_mtx.shape[0]

        data["num_vehicles"] = 1
        data["depot"] = 0
        data["demands"] = np.ones(n_nodes)
        data["vehicle_capacity"] = n_nodes

        return self.solve_cvrp(data, rounding=rounding)
