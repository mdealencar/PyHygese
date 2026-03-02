from dataclasses import dataclass
import numpy as np
import sys

from . import _core


C_INT_BITS = 32
C_INT_MIN = -(2 ** (C_INT_BITS - 1))
C_INT_MAX = 2 ** (C_INT_BITS - 1) - 1
C_DBL_MAX = sys.float_info.max


def _to_c_int(value: int) -> int:
    value = int(value)
    if C_INT_MIN <= value <= C_INT_MAX:
        return value

    wrapped = ((value + (1 << (C_INT_BITS - 1))) % (1 << C_INT_BITS)) - (1 << (C_INT_BITS - 1))
    return int(wrapped)



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
        self.verbose = bool(verbose)

    def _core_parameters(self):
        ap = self.algorithm_parameters
        return (
            _to_c_int(ap.nbGranular),
            _to_c_int(ap.mu),
            _to_c_int(ap.lambda_),
            _to_c_int(ap.nbElite),
            _to_c_int(ap.nbClose),
            _to_c_int(ap.nbIterPenaltyManagement),
            float(ap.targetFeasible),
            float(ap.penaltyDecrease),
            float(ap.penaltyIncrease),
            _to_c_int(ap.seed),
            _to_c_int(ap.nbIter),
            _to_c_int(ap.nbIterTraces),
            float(ap.timeLimit),
            bool(ap.useSwapStar),
        )

    def solve_cvrp(self, data, rounding=True):
        demand = np.asarray(data["demands"], dtype=np.float64)
        vehicle_capacity = float(data["vehicle_capacity"])
        n_nodes = len(demand)

        depot = data.get("depot", 0)
        if depot != 0:
            raise ValueError("In HGS, the depot location must be 0.")

        maximum_number_of_vehicles = _to_c_int(data.get("num_vehicles", C_INT_MAX))

        service_times = data.get("service_times")
        if service_times is None:
            service_times = np.zeros(n_nodes, dtype=np.float64)
        else:
            service_times = np.asarray(service_times, dtype=np.float64)

        duration_limit = data.get("duration_limit")
        if duration_limit is None:
            is_duration_constraint = False
            duration_limit = float(C_DBL_MAX)
        else:
            is_duration_constraint = True
            duration_limit = float(duration_limit)

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
                float(duration_limit),
                bool(is_duration_constraint),
                maximum_number_of_vehicles,
                *self._core_parameters(),
                self.verbose,
            )
        else:
            sol = _core.solve_cvrp(
                np.ascontiguousarray(x_coords),
                np.ascontiguousarray(y_coords),
                np.ascontiguousarray(service_times),
                np.ascontiguousarray(demand),
                vehicle_capacity,
                float(duration_limit),
                bool(rounding),
                bool(is_duration_constraint),
                maximum_number_of_vehicles,
                *self._core_parameters(),
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
