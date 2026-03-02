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
    def __init__(self, payload: dict):
        self.cost = payload["cost"]
        self.time = payload["time"]
        self.n_routes = payload["n_routes"]
        self.routes = [list(route) for route in payload["routes"]]


class Solver:
    @staticmethod
    def _normalize_seed(seed: int) -> int:
        seed = int(seed)
        return seed % C_INT_MAX

    @staticmethod
    def _ap_payload(algorithm_parameters):
        return {
            "nbGranular": int(algorithm_parameters.nbGranular),
            "mu": int(algorithm_parameters.mu),
            "lambda": int(algorithm_parameters.lambda_),
            "nbElite": int(algorithm_parameters.nbElite),
            "nbClose": int(algorithm_parameters.nbClose),
            "nbIterPenaltyManagement": int(algorithm_parameters.nbIterPenaltyManagement),
            "targetFeasible": float(algorithm_parameters.targetFeasible),
            "penaltyDecrease": float(algorithm_parameters.penaltyDecrease),
            "penaltyIncrease": float(algorithm_parameters.penaltyIncrease),
            "seed": Solver._normalize_seed(algorithm_parameters.seed),
            "nbIter": int(algorithm_parameters.nbIter),
            "nbIterTraces": int(algorithm_parameters.nbIterTraces),
            "timeLimit": float(algorithm_parameters.timeLimit),
            "useSwapStar": int(bool(algorithm_parameters.useSwapStar)),
        }

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

        is_rounding_integer = rounding

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
            return self._solve_cvrp_dist_mtx(
                x_coords,
                y_coords,
                dist_mtx,
                service_times,
                demand,
                vehicle_capacity,
                duration_limit,
                is_duration_constraint,
                maximum_number_of_vehicles,
                self.algorithm_parameters,
                self.verbose,
            )

        return self._solve_cvrp(
            x_coords,
            y_coords,
            service_times,
            demand,
            vehicle_capacity,
            duration_limit,
            is_rounding_integer,
            is_duration_constraint,
            maximum_number_of_vehicles,
            self.algorithm_parameters,
            self.verbose,
        )

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

    @staticmethod
    def _solve_cvrp(
        x_coords,
        y_coords,
        service_times,
        demand,
        vehicle_capacity,
        duration_limit,
        is_rounding_integer,
        is_duration_constraint,
        maximum_number_of_vehicles,
        algorithm_parameters,
        verbose,
    ):
        n_nodes = x_coords.size
        payload = _core.solve_cvrp_compact(
            int(n_nodes),
            np.ascontiguousarray(x_coords, dtype=np.float64),
            np.ascontiguousarray(y_coords, dtype=np.float64),
            np.ascontiguousarray(service_times, dtype=np.float64),
            np.ascontiguousarray(demand, dtype=np.float64),
            float(vehicle_capacity),
            float(duration_limit),
            bool(is_rounding_integer),
            bool(is_duration_constraint),
            int(maximum_number_of_vehicles),
            Solver._ap_payload(algorithm_parameters),
            bool(verbose),
        )
        return RoutingSolution(payload)

    @staticmethod
    def _solve_cvrp_dist_mtx(
        x_coords,
        y_coords,
        dist_mtx,
        service_times,
        demand,
        vehicle_capacity,
        duration_limit,
        is_duration_constraint,
        maximum_number_of_vehicles,
        algorithm_parameters,
        verbose,
    ):
        n_nodes = x_coords.size
        payload = _core.solve_cvrp_dist_mtx_compact(
            int(n_nodes),
            np.ascontiguousarray(x_coords, dtype=np.float64),
            np.ascontiguousarray(y_coords, dtype=np.float64),
            np.ascontiguousarray(dist_mtx.reshape(n_nodes * n_nodes), dtype=np.float64),
            np.ascontiguousarray(service_times, dtype=np.float64),
            np.ascontiguousarray(demand, dtype=np.float64),
            float(vehicle_capacity),
            float(duration_limit),
            bool(is_duration_constraint),
            int(maximum_number_of_vehicles),
            Solver._ap_payload(algorithm_parameters),
            bool(verbose),
        )
        return RoutingSolution(payload)
