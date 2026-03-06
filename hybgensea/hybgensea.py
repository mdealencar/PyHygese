from dataclasses import dataclass
import numpy as np
import sys

from . import _core


C_DBL_MAX = sys.float_info.max

DEFAULT_ALGO_PARAMS = _core.default_algorithm_parameters()


def _ap_c_args(**overrides):
    resolved = {**DEFAULT_ALGO_PARAMS, **overrides}
    return (
        int(resolved['nbGranular']),
        int(resolved['mu']),
        int(resolved['lambda_']),
        int(resolved['nbElite']),
        int(resolved['nbClose']),
        int(resolved['nbIterPenaltyManagement']),
        float(resolved['targetFeasible']),
        float(resolved['penaltyDecrease']),
        float(resolved['penaltyIncrease']),
        int(resolved['seed']),
        int(resolved['nbIter']),
        int(resolved['nbIterTraces']),
        float(resolved['timeLimit']),
        bool(resolved['useSwapStar']),
    )


def solve_cvrp_dist_mtx(
    dist_mtx,
    demands,
    vehicle_capacity,
    *,
    x_coords=None,
    y_coords=None,
    service_times=None,
    duration_limit=None,
    num_vehicles=-1,
    verbose=True,
    log_callback=None,
    **ap_kwargs,
):
    """Solve CVRP with a pre-computed distance matrix.

    Args:
        dist_mtx: (n, n) distance matrix (float64, C-contiguous)
        demands: (n,) demand per node, depot demand must be 0
        vehicle_capacity: maximum vehicle capacity
        x_coords: (n,) x coordinates (optional, for SWAP*)
        y_coords: (n,) y coordinates (optional, for SWAP*)
        service_times: (n,) service duration per node (default: zeros)
        duration_limit: route duration limit (default: unconstrained)
        num_vehicles: maximum number of vehicles (default: unlimited)
        verbose: print solver log
        log_callback: callable receiving each log line
        **ap_kwargs: algorithm parameters (seed, timeLimit, nbGranular, etc.)

    Returns:
        RoutingSolution with .cost, .time, .routes, .n_routes, .log
    """
    dist_mtx = np.ascontiguousarray(dist_mtx, dtype=np.float64)
    demands = np.ascontiguousarray(demands, dtype=np.float64)
    n = len(demands)

    if x_coords is None or y_coords is None:
        x_coords = np.zeros(n, dtype=np.float64)
        y_coords = np.zeros(n, dtype=np.float64)
    else:
        x_coords = np.ascontiguousarray(x_coords, dtype=np.float64)
        y_coords = np.ascontiguousarray(y_coords, dtype=np.float64)

    if service_times is None:
        service_times = np.zeros(n, dtype=np.float64)
    else:
        service_times = np.ascontiguousarray(service_times, dtype=np.float64)

    if duration_limit is None:
        is_duration_constraint = False
        duration_limit = float(C_DBL_MAX)
    else:
        is_duration_constraint = True
        duration_limit = float(duration_limit)

    sol = _core.solve_cvrp_dist_mtx(
        x_coords, y_coords, dist_mtx, service_times, demands,
        float(vehicle_capacity), duration_limit, bool(is_duration_constraint),
        int(num_vehicles),
        *_ap_c_args(**ap_kwargs),
        bool(verbose), log_callback,
    )
    return RoutingSolution(sol)


def solve_cvrp(
    x_coords,
    y_coords,
    demands,
    vehicle_capacity,
    *,
    service_times=None,
    duration_limit=None,
    num_vehicles=-1,
    rounding=True,
    verbose=True,
    log_callback=None,
    **ap_kwargs,
):
    """Solve CVRP from coordinates (distance matrix computed internally).

    Args:
        x_coords: (n,) x coordinates
        y_coords: (n,) y coordinates
        demands: (n,) demand per node, depot demand must be 0
        vehicle_capacity: maximum vehicle capacity
        service_times: (n,) service duration per node (default: zeros)
        duration_limit: route duration limit (default: unconstrained)
        num_vehicles: maximum number of vehicles (default: unlimited)
        rounding: round Euclidean distances to integers
        verbose: print solver log
        log_callback: callable receiving each log line
        **ap_kwargs: algorithm parameters (seed, timeLimit, nbGranular, etc.)

    Returns:
        RoutingSolution with .cost, .time, .routes, .n_routes, .log
    """
    x_coords = np.ascontiguousarray(x_coords, dtype=np.float64)
    y_coords = np.ascontiguousarray(y_coords, dtype=np.float64)
    demands = np.ascontiguousarray(demands, dtype=np.float64)
    n = len(demands)

    if service_times is None:
        service_times = np.zeros(n, dtype=np.float64)
    else:
        service_times = np.ascontiguousarray(service_times, dtype=np.float64)

    if duration_limit is None:
        is_duration_constraint = False
        duration_limit = float(C_DBL_MAX)
    else:
        is_duration_constraint = True
        duration_limit = float(duration_limit)

    sol = _core.solve_cvrp(
        x_coords, y_coords, service_times, demands,
        float(vehicle_capacity), duration_limit, bool(rounding),
        bool(is_duration_constraint), int(num_vehicles),
        *_ap_c_args(**ap_kwargs),
        bool(verbose), log_callback,
    )
    return RoutingSolution(sol)


def solve_tsp(
    x_coords=None,
    y_coords=None,
    dist_mtx=None,
    *,
    rounding=True,
    verbose=True,
    log_callback=None,
    **ap_kwargs,
):
    """Solve TSP (single-vehicle, unit demands).

    Provide either (x_coords, y_coords) or dist_mtx.
    """
    if dist_mtx is not None:
        dist_mtx = np.asarray(dist_mtx, dtype=np.float64)
        n = dist_mtx.shape[0]
        demands = np.ones(n, dtype=np.float64)
        demands[0] = 0.0
        return solve_cvrp_dist_mtx(
            dist_mtx, demands, float(n),
            x_coords=x_coords, y_coords=y_coords,
            num_vehicles=1, verbose=verbose, log_callback=log_callback,
            **ap_kwargs,
        )
    else:
        x_coords = np.asarray(x_coords, dtype=np.float64)
        y_coords = np.asarray(y_coords, dtype=np.float64)
        n = len(x_coords)
        demands = np.ones(n, dtype=np.float64)
        demands[0] = 0.0
        return solve_cvrp(
            x_coords, y_coords, demands, float(n),
            rounding=rounding, num_vehicles=1,
            verbose=verbose, log_callback=log_callback,
            **ap_kwargs,
        )


class RoutingSolution:
    def __init__(self, sol):
        self.cost = sol.cost
        self.time = sol.time
        self.routes = sol.routes
        self.n_routes = len(sol.routes)
        self.log = sol.log


# --- Legacy API (kept for backward compatibility with tests/other users) ---

_d = DEFAULT_ALGO_PARAMS

@dataclass
class AlgorithmParameters:
    nbGranular: int = _d['nbGranular']
    mu: int = _d['mu']
    lambda_: int = _d['lambda_']
    nbElite: int = _d['nbElite']
    nbClose: int = _d['nbClose']
    nbIterPenaltyManagement: int = _d['nbIterPenaltyManagement']
    targetFeasible: float = _d['targetFeasible']
    penaltyDecrease: float = _d['penaltyDecrease']
    penaltyIncrease: float = _d['penaltyIncrease']
    seed: int = _d['seed']
    nbIter: int = _d['nbIter']
    nbIterTraces: int = _d['nbIterTraces']
    timeLimit: float = _d['timeLimit']
    useSwapStar: bool = _d['useSwapStar']

del _d


class Solver:
    def __init__(self, parameters=AlgorithmParameters(), verbose=True, log_callback=None):
        self.algorithm_parameters = parameters
        self.verbose = bool(verbose)
        self.log_callback = log_callback

    def _ap_kwargs(self):
        ap = self.algorithm_parameters
        return dict(
            nbGranular=ap.nbGranular, mu=ap.mu, lambda_=ap.lambda_,
            nbElite=ap.nbElite, nbClose=ap.nbClose,
            nbIterPenaltyManagement=ap.nbIterPenaltyManagement,
            targetFeasible=ap.targetFeasible, penaltyDecrease=ap.penaltyDecrease,
            penaltyIncrease=ap.penaltyIncrease, seed=ap.seed, nbIter=ap.nbIter,
            nbIterTraces=ap.nbIterTraces, timeLimit=ap.timeLimit,
            useSwapStar=ap.useSwapStar,
        )

    def solve_cvrp(self, data, rounding=True):
        demand = np.asarray(data["demands"], dtype=np.float64)
        vehicle_capacity = float(data["vehicle_capacity"])

        depot = data.get("depot", 0)
        if depot != 0:
            raise ValueError("In HGS, the depot location must be 0.")

        kwargs = dict(
            num_vehicles=data.get("num_vehicles", -1),
            service_times=data.get("service_times"),
            duration_limit=data.get("duration_limit"),
            verbose=self.verbose,
            log_callback=self.log_callback,
            **self._ap_kwargs(),
        )

        x_coords = data.get("x_coordinates")
        y_coords = data.get("y_coordinates")
        dist_mtx = data.get("distance_matrix")

        if dist_mtx is not None:
            return solve_cvrp_dist_mtx(
                dist_mtx, demand, vehicle_capacity,
                x_coords=x_coords, y_coords=y_coords, **kwargs,
            )
        else:
            return solve_cvrp(
                x_coords, y_coords, demand, vehicle_capacity,
                rounding=rounding, **kwargs,
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
