#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "C_Interface.h"

#include <cstdio>
#include <stdexcept>
#include <unistd.h>

namespace nb = nanobind;
using namespace nb::literals;

class ScopedStdoutFdRedirect {
public:
    ScopedStdoutFdRedirect() {
        nb::object sys = nb::module_::import_("sys");
        nb::object stdout_obj = sys.attr("stdout");
        int py_stdout_fd = nb::cast<int>(stdout_obj.attr("fileno")());

        std::fflush(stdout);
        saved_stdout_fd_ = dup(STDOUT_FILENO);
        if (saved_stdout_fd_ == -1) {
            throw std::runtime_error("Failed to duplicate STDOUT_FILENO.");
        }

        if (dup2(py_stdout_fd, STDOUT_FILENO) == -1) {
            close(saved_stdout_fd_);
            throw std::runtime_error("Failed to redirect STDOUT_FILENO to sys.stdout.");
        }
    }

    ~ScopedStdoutFdRedirect() {
        if (saved_stdout_fd_ != -1) {
            std::fflush(stdout);
            dup2(saved_stdout_fd_, STDOUT_FILENO);
            close(saved_stdout_fd_);
        }
    }

    ScopedStdoutFdRedirect(const ScopedStdoutFdRedirect &) = delete;
    ScopedStdoutFdRedirect &operator=(const ScopedStdoutFdRedirect &) = delete;

private:
    int saved_stdout_fd_ = -1;
};

static nb::dict to_python_solution(Solution *sol) {
    if (!sol) {
        throw std::runtime_error("HGS-CVRP returned a null solution pointer.");
    }

    nb::list routes;
    for (int i = 0; i < sol->n_routes; i++) {
        nb::list path;
        for (int j = 0; j < sol->routes[i].length; j++) {
            path.append(sol->routes[i].path[j]);
        }
        routes.append(path);
    }

    nb::dict result;
    result["cost"] = sol->cost;
    result["time"] = sol->time;
    result["n_routes"] = sol->n_routes;
    result["routes"] = routes;
    return result;
}

NB_MODULE(_core, m) {
    m.def(
        "solve_cvrp",
        [](int n,
           nb::object x_obj,
           nb::object y_obj,
           nb::object service_times_obj,
           nb::object demand_obj,
           double vehicle_capacity, double duration_limit, bool is_rounding_integer,
           bool is_duration_constraint, int max_nb_veh,
           int nb_granular, int mu, int lambda_value, int nb_elite, int nb_close,
           int nb_iter_penalty_management, double target_feasible,
           double penalty_decrease, double penalty_increase, int seed, int nb_iter,
           int nb_iter_traces, double time_limit, bool use_swap_star, bool verbose) {
            AlgorithmParameters ap = {
                nb_granular,
                mu,
                lambda_value,
                nb_elite,
                nb_close,
                nb_iter_penalty_management,
                target_feasible,
                penalty_decrease,
                penalty_increase,
                seed,
                nb_iter,
                nb_iter_traces,
                time_limit,
                static_cast<int>(use_swap_star),
            };

            auto x = nb::cast<nb::ndarray<const double, nb::c_contig, nb::device::cpu>>(x_obj);
            auto y = nb::cast<nb::ndarray<const double, nb::c_contig, nb::device::cpu>>(y_obj);
            auto service_times = nb::cast<nb::ndarray<const double, nb::c_contig, nb::device::cpu>>(service_times_obj);
            auto demand = nb::cast<nb::ndarray<const double, nb::c_contig, nb::device::cpu>>(demand_obj);

            ScopedStdoutFdRedirect redirect;
            Solution *sol = solve_cvrp(
                n, const_cast<double *>(x.data()), const_cast<double *>(y.data()), const_cast<double *>(service_times.data()), const_cast<double *>(demand.data()),
                vehicle_capacity, duration_limit,
                static_cast<char>(is_rounding_integer),
                static_cast<char>(is_duration_constraint), max_nb_veh, &ap,
                static_cast<char>(verbose));

            nb::dict result = to_python_solution(sol);
            delete_solution(sol);
            return result;
        },
        "n"_a, "x"_a, "y"_a, "service_times"_a, "demand"_a,
        "vehicle_capacity"_a, "duration_limit"_a, "is_rounding_integer"_a,
        "is_duration_constraint"_a, "max_nb_veh"_a,
        "nb_granular"_a, "mu"_a, "lambda_value"_a, "nb_elite"_a, "nb_close"_a,
        "nb_iter_penalty_management"_a, "target_feasible"_a,
        "penalty_decrease"_a, "penalty_increase"_a, "seed"_a, "nb_iter"_a,
        "nb_iter_traces"_a, "time_limit"_a, "use_swap_star"_a, "verbose"_a
    );

    m.def(
        "solve_cvrp_dist_mtx",
        [](int n,
           nb::object x_obj,
           nb::object y_obj,
           nb::object dist_mtx_obj,
           nb::object service_times_obj,
           nb::object demand_obj,
           double vehicle_capacity, double duration_limit,
           bool is_duration_constraint, int max_nb_veh,
           int nb_granular, int mu, int lambda_value, int nb_elite, int nb_close,
           int nb_iter_penalty_management, double target_feasible,
           double penalty_decrease, double penalty_increase, int seed, int nb_iter,
           int nb_iter_traces, double time_limit, bool use_swap_star, bool verbose) {
            AlgorithmParameters ap = {
                nb_granular,
                mu,
                lambda_value,
                nb_elite,
                nb_close,
                nb_iter_penalty_management,
                target_feasible,
                penalty_decrease,
                penalty_increase,
                seed,
                nb_iter,
                nb_iter_traces,
                time_limit,
                static_cast<int>(use_swap_star),
            };

            auto x = nb::cast<nb::ndarray<const double, nb::c_contig, nb::device::cpu>>(x_obj);
            auto y = nb::cast<nb::ndarray<const double, nb::c_contig, nb::device::cpu>>(y_obj);
            auto dist_mtx = nb::cast<nb::ndarray<const double, nb::c_contig, nb::device::cpu>>(dist_mtx_obj);
            auto service_times = nb::cast<nb::ndarray<const double, nb::c_contig, nb::device::cpu>>(service_times_obj);
            auto demand = nb::cast<nb::ndarray<const double, nb::c_contig, nb::device::cpu>>(demand_obj);

            ScopedStdoutFdRedirect redirect;
            Solution *sol = solve_cvrp_dist_mtx(
                n, const_cast<double *>(x.data()), const_cast<double *>(y.data()), const_cast<double *>(dist_mtx.data()), const_cast<double *>(service_times.data()), const_cast<double *>(demand.data()),
                vehicle_capacity, duration_limit,
                static_cast<char>(is_duration_constraint), max_nb_veh, &ap,
                static_cast<char>(verbose));

            nb::dict result = to_python_solution(sol);
            delete_solution(sol);
            return result;
        },
        "n"_a, "x"_a, "y"_a, "dist_mtx"_a, "service_times"_a, "demand"_a,
        "vehicle_capacity"_a, "duration_limit"_a, "is_duration_constraint"_a,
        "max_nb_veh"_a,
        "nb_granular"_a, "mu"_a, "lambda_value"_a, "nb_elite"_a, "nb_close"_a,
        "nb_iter_penalty_management"_a, "target_feasible"_a,
        "penalty_decrease"_a, "penalty_increase"_a, "seed"_a, "nb_iter"_a,
        "nb_iter_traces"_a, "time_limit"_a, "use_swap_star"_a, "verbose"_a
    );
}
