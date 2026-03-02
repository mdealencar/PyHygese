# cython: language_level=3
from dataclasses import dataclass
import os
import sys
import numpy as np
cimport numpy as cnp

from libc.stdio cimport fflush, stdout, fileno
from libc.limits cimport INT_MAX
from libc.float cimport DBL_MAX

cdef extern from "Python.h":
    int PyObject_AsFileDescriptor(object)


cdef extern from *:
    """
    struct AlgorithmParameters {
        int nbGranular;
        int mu;
        int lambda;
        int nbElite;
        int nbClose;
        int nbIterPenaltyManagement;
        double targetFeasible;
        double penaltyDecrease;
        double penaltyIncrease;
        int seed;
        int nbIter;
        int nbIterTraces;
        double timeLimit;
        int useSwapStar;
    };

    struct SolutionRoute {
        int length;
        int* path;
    };

    struct Solution {
        double cost;
        double time;
        int n_routes;
        struct SolutionRoute* routes;
    };

    struct Solution * solve_cvrp(
        int n, double* x, double* y, double* serv_time, double* dem,
        double vehicleCapacity, double durationLimit, char isRoundingInteger, char isDurationConstraint,
        int max_nbVeh, const struct AlgorithmParameters* ap, char verbose);

    struct Solution * solve_cvrp_dist_mtx(
        int n, double* x, double* y, double *dist_mtx, double *serv_time, double *dem,
        double vehicleCapacity, double durationLimit, char isDurationConstraint,
        int max_nbVeh, const struct AlgorithmParameters *ap, char verbose);

    void delete_solution(struct Solution* sol);
    """
    cdef struct CAlgorithmParameters "AlgorithmParameters":
        int nbGranular
        int mu
        int lambda_
        int nbElite
        int nbClose
        int nbIterPenaltyManagement
        double targetFeasible
        double penaltyDecrease
        double penaltyIncrease
        int seed
        int nbIter
        int nbIterTraces
        double timeLimit
        int useSwapStar

    cdef struct CSolutionRoute "SolutionRoute":
        int length
        int* path

    cdef struct CSolution "Solution":
        double cost
        double time
        int n_routes
        CSolutionRoute* routes

    CSolution* solve_cvrp(
        int n, double* x, double* y, double* serv_time, double* dem,
        double vehicleCapacity, double durationLimit, char isRoundingInteger, char isDurationConstraint,
        int max_nbVeh, const CAlgorithmParameters* ap, char verbose)

    CSolution* solve_cvrp_dist_mtx(
        int n, double* x, double* y, double* dist_mtx, double* serv_time, double* dem,
        double vehicleCapacity, double durationLimit, char isDurationConstraint,
        int max_nbVeh, const CAlgorithmParameters* ap, char verbose)

    void delete_solution(CSolution* sol)


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




cdef CAlgorithmParameters _algorithm_parameters_as_c(AlgorithmParameters parameters):
    cdef CAlgorithmParameters ap
    ap.nbGranular = parameters.nbGranular
    ap.mu = parameters.mu
    ap.lambda_ = parameters.lambda_
    ap.nbElite = parameters.nbElite
    ap.nbClose = parameters.nbClose
    ap.nbIterPenaltyManagement = parameters.nbIterPenaltyManagement
    ap.targetFeasible = parameters.targetFeasible
    ap.penaltyDecrease = parameters.penaltyDecrease
    ap.penaltyIncrease = parameters.penaltyIncrease
    ap.seed = parameters.seed
    ap.nbIter = parameters.nbIter
    ap.nbIterTraces = parameters.nbIterTraces
    ap.timeLimit = parameters.timeLimit
    ap.useSwapStar = int(parameters.useSwapStar)
    return ap


class RoutingSolution:
    def __init__(self, double cost, double runtime, int n_routes, routes):
        self.cost = cost
        self.time = runtime
        self.n_routes = n_routes
        self.routes = routes


def _sync_c_stdout_with_python():
    cdef int py_stdout_fd = PyObject_AsFileDescriptor(sys.stdout)
    cdef int c_stdout_fd = fileno(stdout)
    if py_stdout_fd >= 0 and c_stdout_fd >= 0 and py_stdout_fd != c_stdout_fd:
        os.dup2(py_stdout_fd, c_stdout_fd)
    fflush(stdout)


cdef RoutingSolution _routing_solution_from_ptr(CSolution* sol_ptr):
    if sol_ptr == NULL:
        raise TypeError("The solution pointer is null.")

    cdef int i, j
    cdef list routes = []
    cdef list path
    for i in range(sol_ptr.n_routes):
        path = []
        for j in range(sol_ptr.routes[i].length):
            path.append(sol_ptr.routes[i].path[j])
        routes.append(path)

    return RoutingSolution(sol_ptr.cost, sol_ptr.time, sol_ptr.n_routes, routes)


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

        maximum_number_of_vehicles = data.get("num_vehicles", INT_MAX)

        service_times = data.get("service_times")
        if service_times is None:
            service_times = np.zeros(n_nodes, dtype=np.float64)
        else:
            service_times = np.asarray(service_times, dtype=np.float64)

        duration_limit = data.get("duration_limit")
        if duration_limit is None:
            is_duration_constraint = False
            duration_limit = DBL_MAX
        else:
            is_duration_constraint = True

        is_rounding_integer = bool(rounding)

        x_coords = data.get("x_coordinates")
        y_coords = data.get("y_coordinates")
        dist_mtx = data.get("distance_matrix")

        if x_coords is None or y_coords is None:
            if dist_mtx is None:
                raise ValueError("Either (x_coordinates, y_coordinates) or distance_matrix must be provided.")
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

    cdef RoutingSolution _solve_cvrp(
        self,
        cnp.ndarray[cnp.float64_t, ndim=1] x_coords,
        cnp.ndarray[cnp.float64_t, ndim=1] y_coords,
        cnp.ndarray[cnp.float64_t, ndim=1] service_times,
        cnp.ndarray[cnp.float64_t, ndim=1] demand,
        double vehicle_capacity,
        double duration_limit,
        bint is_rounding_integer,
        bint is_duration_constraint,
        int maximum_number_of_vehicles,
        AlgorithmParameters algorithm_parameters,
        bint verbose,
    ):
        cdef int n_nodes = x_coords.shape[0]
        cdef CAlgorithmParameters ap = _algorithm_parameters_as_c(algorithm_parameters)
        cdef CSolution* sol_p

        _sync_c_stdout_with_python()
        sol_p = solve_cvrp(
            n_nodes,
            <double*> x_coords.data,
            <double*> y_coords.data,
            <double*> service_times.data,
            <double*> demand.data,
            vehicle_capacity,
            duration_limit,
            <char>is_rounding_integer,
            <char>is_duration_constraint,
            maximum_number_of_vehicles,
            &ap,
            <char>verbose,
        )

        try:
            return _routing_solution_from_ptr(sol_p)
        finally:
            if sol_p != NULL:
                delete_solution(sol_p)

    cdef RoutingSolution _solve_cvrp_dist_mtx(
        self,
        cnp.ndarray[cnp.float64_t, ndim=1] x_coords,
        cnp.ndarray[cnp.float64_t, ndim=1] y_coords,
        cnp.ndarray[cnp.float64_t, ndim=2] dist_mtx,
        cnp.ndarray[cnp.float64_t, ndim=1] service_times,
        cnp.ndarray[cnp.float64_t, ndim=1] demand,
        double vehicle_capacity,
        double duration_limit,
        bint is_duration_constraint,
        int maximum_number_of_vehicles,
        AlgorithmParameters algorithm_parameters,
        bint verbose,
    ):
        cdef int n_nodes = x_coords.shape[0]
        cdef cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] dist_mtx_c = np.ascontiguousarray(dist_mtx)
        cdef CAlgorithmParameters ap = _algorithm_parameters_as_c(algorithm_parameters)
        cdef CSolution* sol_p

        _sync_c_stdout_with_python()
        sol_p = solve_cvrp_dist_mtx(
            n_nodes,
            <double*> x_coords.data,
            <double*> y_coords.data,
            <double*> dist_mtx_c.data,
            <double*> service_times.data,
            <double*> demand.data,
            vehicle_capacity,
            duration_limit,
            <char>is_duration_constraint,
            maximum_number_of_vehicles,
            &ap,
            <char>verbose,
        )

        try:
            return _routing_solution_from_ptr(sol_p)
        finally:
            if sol_p != NULL:
                delete_solution(sol_p)
