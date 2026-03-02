#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "C_Interface.h"

#include <cstdio>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#if defined(_WIN32)
#  include <fcntl.h>
#  include <io.h>
#else
#  include <unistd.h>
#endif

namespace nb = nanobind;
using namespace nb::literals;

#if defined(_WIN32)
static constexpr int HG_STDOUT_FILENO = 1;
static int hg_dup(int fd) { return _dup(fd); }
static int hg_dup2(int oldfd, int newfd) { return _dup2(oldfd, newfd); }
static int hg_close(int fd) { return _close(fd); }
static int hg_pipe(int fds[2]) { return _pipe(fds, 4096, _O_BINARY); }
static int hg_read(int fd, char *buf, int len) { return _read(fd, buf, len); }
#else
static constexpr int HG_STDOUT_FILENO = STDOUT_FILENO;
static int hg_dup(int fd) { return dup(fd); }
static int hg_dup2(int oldfd, int newfd) { return dup2(oldfd, newfd); }
static int hg_close(int fd) { return close(fd); }
static int hg_pipe(int fds[2]) { return pipe(fds); }
static int hg_read(int fd, char *buf, int len) { return static_cast<int>(read(fd, buf, static_cast<size_t>(len))); }
#endif

class ScopedStdoutFdRedirect {
public:
    ScopedStdoutFdRedirect() {
        nb::object sys = nb::module_::import_("sys");
        py_stdout_ = sys.attr("stdout");

        std::fflush(stdout);
        saved_stdout_fd_ = hg_dup(HG_STDOUT_FILENO);
        if (saved_stdout_fd_ == -1) {
            throw std::runtime_error("Failed to duplicate STDOUT_FILENO.");
        }

        int pipe_fds[2] = {-1, -1};
        if (hg_pipe(pipe_fds) == -1) {
            hg_close(saved_stdout_fd_);
            throw std::runtime_error("Failed to create stdout capture pipe.");
        }
        pipe_read_fd_ = pipe_fds[0];
        pipe_write_fd_ = pipe_fds[1];

        if (hg_dup2(pipe_write_fd_, HG_STDOUT_FILENO) == -1) {
            hg_close(saved_stdout_fd_);
            hg_close(pipe_read_fd_);
            hg_close(pipe_write_fd_);
            throw std::runtime_error("Failed to redirect STDOUT_FILENO to capture pipe.");
        }

        reader_thread_ = std::thread([this]() {
            char buffer[1024];
            while (true) {
                int n = hg_read(pipe_read_fd_, buffer, 1024);
                if (n <= 0) {
                    break;
                }
                std::lock_guard<std::mutex> guard(captured_mutex_);
                captured_.append(buffer, static_cast<size_t>(n));
            }
        });
    }

    ~ScopedStdoutFdRedirect() {
        std::fflush(stdout);

        if (saved_stdout_fd_ != -1) {
            hg_dup2(saved_stdout_fd_, HG_STDOUT_FILENO);
            hg_close(saved_stdout_fd_);
            saved_stdout_fd_ = -1;
        }

        if (pipe_write_fd_ != -1) {
            hg_close(pipe_write_fd_);
            pipe_write_fd_ = -1;
        }

        if (reader_thread_.joinable()) {
            reader_thread_.join();
        }

        if (pipe_read_fd_ != -1) {
            hg_close(pipe_read_fd_);
            pipe_read_fd_ = -1;
        }

        std::string captured;
        {
            std::lock_guard<std::mutex> guard(captured_mutex_);
            captured.swap(captured_);
        }

        if (!captured.empty()) {
            py_stdout_.attr("write")(captured);
            py_stdout_.attr("flush")();
        }
    }

    ScopedStdoutFdRedirect(const ScopedStdoutFdRedirect &) = delete;
    ScopedStdoutFdRedirect &operator=(const ScopedStdoutFdRedirect &) = delete;

private:
    nb::object py_stdout_;
    int saved_stdout_fd_ = -1;
    int pipe_read_fd_ = -1;
    int pipe_write_fd_ = -1;
    std::thread reader_thread_;
    std::mutex captured_mutex_;
    std::string captured_;
};


static int dict_int(const nb::dict &d, const char *key) {
    if (!d.contains(key)) {
        throw std::runtime_error(std::string("Missing algorithm parameter: ") + key);
    }
    nb::object v = d[key];
    try {
        return nb::cast<int>(nb::module_::import_("builtins").attr("int")(v));
    } catch (const std::exception &e) {
        throw std::runtime_error(std::string("Invalid int algorithm parameter: ") + key + " (" + e.what() + ")");
    }
}

static double dict_double(const nb::dict &d, const char *key) {
    if (!d.contains(key)) {
        throw std::runtime_error(std::string("Missing algorithm parameter: ") + key);
    }
    nb::object v = d[key];
    try {
        return nb::cast<double>(nb::module_::import_("builtins").attr("float")(v));
    } catch (const std::exception &e) {
        throw std::runtime_error(std::string("Invalid float algorithm parameter: ") + key + " (" + e.what() + ")");
    }
}

static AlgorithmParameters from_python_ap(const nb::dict &ap) {
    const char *lambda_key = ap.contains("lambda") ? "lambda" : "lambda_";
    return AlgorithmParameters{
        dict_int(ap, "nbGranular"),
        dict_int(ap, "mu"),
        dict_int(ap, lambda_key),
        dict_int(ap, "nbElite"),
        dict_int(ap, "nbClose"),
        dict_int(ap, "nbIterPenaltyManagement"),
        dict_double(ap, "targetFeasible"),
        dict_double(ap, "penaltyDecrease"),
        dict_double(ap, "penaltyIncrease"),
        dict_int(ap, "seed"),
        dict_int(ap, "nbIter"),
        dict_int(ap, "nbIterTraces"),
        dict_double(ap, "timeLimit"),
        dict_int(ap, "useSwapStar"),
    };
}

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

            nb::ndarray<const double, nb::c_contig, nb::device::cpu> x;
            nb::ndarray<const double, nb::c_contig, nb::device::cpu> y;
            nb::ndarray<const double, nb::c_contig, nb::device::cpu> service_times;
            nb::ndarray<const double, nb::c_contig, nb::device::cpu> demand;
            try { x = nb::cast<decltype(x)>(x_obj); } catch (...) { throw std::runtime_error("bad_cast: x"); }
            try { y = nb::cast<decltype(y)>(y_obj); } catch (...) { throw std::runtime_error("bad_cast: y"); }
            try { service_times = nb::cast<decltype(service_times)>(service_times_obj); } catch (...) { throw std::runtime_error("bad_cast: service_times"); }
            try { demand = nb::cast<decltype(demand)>(demand_obj); } catch (...) { throw std::runtime_error("bad_cast: demand"); }

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
        "solve_cvrp_compact",
        [](int n,
           nb::object x_obj,
           nb::object y_obj,
           nb::object service_times_obj,
           nb::object demand_obj,
           double vehicle_capacity,
           double duration_limit,
           bool is_rounding_integer,
           bool is_duration_constraint,
           int max_nb_veh,
           nb::dict ap_dict,
           bool verbose) {
            AlgorithmParameters ap = from_python_ap(ap_dict);
            nb::ndarray<const double, nb::c_contig, nb::device::cpu> x;
            nb::ndarray<const double, nb::c_contig, nb::device::cpu> y;
            nb::ndarray<const double, nb::c_contig, nb::device::cpu> service_times;
            nb::ndarray<const double, nb::c_contig, nb::device::cpu> demand;
            try { x = nb::cast<decltype(x)>(x_obj); } catch (...) { throw std::runtime_error("bad_cast: x"); }
            try { y = nb::cast<decltype(y)>(y_obj); } catch (...) { throw std::runtime_error("bad_cast: y"); }
            try { service_times = nb::cast<decltype(service_times)>(service_times_obj); } catch (...) { throw std::runtime_error("bad_cast: service_times"); }
            try { demand = nb::cast<decltype(demand)>(demand_obj); } catch (...) { throw std::runtime_error("bad_cast: demand"); }

            ScopedStdoutFdRedirect redirect;
            Solution *sol = solve_cvrp(
                n, const_cast<double *>(x.data()), const_cast<double *>(y.data()),
                const_cast<double *>(service_times.data()), const_cast<double *>(demand.data()),
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
        "is_duration_constraint"_a, "max_nb_veh"_a, "algorithm_parameters"_a,
        "verbose"_a
    );

    m.def(
        "solve_cvrp_dist_mtx_compact",
        [](int n,
           nb::object x_obj,
           nb::object y_obj,
           nb::object dist_mtx_obj,
           nb::object service_times_obj,
           nb::object demand_obj,
           double vehicle_capacity,
           double duration_limit,
           bool is_duration_constraint,
           int max_nb_veh,
           nb::dict ap_dict,
           bool verbose) {
            AlgorithmParameters ap = from_python_ap(ap_dict);
            nb::ndarray<const double, nb::c_contig, nb::device::cpu> x;
            nb::ndarray<const double, nb::c_contig, nb::device::cpu> y;
            nb::ndarray<const double, nb::c_contig, nb::device::cpu> dist_mtx;
            nb::ndarray<const double, nb::c_contig, nb::device::cpu> service_times;
            nb::ndarray<const double, nb::c_contig, nb::device::cpu> demand;
            try { x = nb::cast<decltype(x)>(x_obj); } catch (...) { throw std::runtime_error("bad_cast: x"); }
            try { y = nb::cast<decltype(y)>(y_obj); } catch (...) { throw std::runtime_error("bad_cast: y"); }
            try { dist_mtx = nb::cast<decltype(dist_mtx)>(dist_mtx_obj); } catch (...) { throw std::runtime_error("bad_cast: dist_mtx"); }
            try { service_times = nb::cast<decltype(service_times)>(service_times_obj); } catch (...) { throw std::runtime_error("bad_cast: service_times"); }
            try { demand = nb::cast<decltype(demand)>(demand_obj); } catch (...) { throw std::runtime_error("bad_cast: demand"); }

            ScopedStdoutFdRedirect redirect;
            Solution *sol = solve_cvrp_dist_mtx(
                n, const_cast<double *>(x.data()), const_cast<double *>(y.data()),
                const_cast<double *>(dist_mtx.data()), const_cast<double *>(service_times.data()),
                const_cast<double *>(demand.data()), vehicle_capacity, duration_limit,
                static_cast<char>(is_duration_constraint), max_nb_veh, &ap,
                static_cast<char>(verbose));

            nb::dict result = to_python_solution(sol);
            delete_solution(sol);
            return result;
        },
        "n"_a, "x"_a, "y"_a, "dist_mtx"_a, "service_times"_a, "demand"_a,
        "vehicle_capacity"_a, "duration_limit"_a, "is_duration_constraint"_a,
        "max_nb_veh"_a, "algorithm_parameters"_a, "verbose"_a
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

            nb::ndarray<const double, nb::c_contig, nb::device::cpu> x;
            nb::ndarray<const double, nb::c_contig, nb::device::cpu> y;
            nb::ndarray<const double, nb::c_contig, nb::device::cpu> dist_mtx;
            nb::ndarray<const double, nb::c_contig, nb::device::cpu> service_times;
            nb::ndarray<const double, nb::c_contig, nb::device::cpu> demand;
            try { x = nb::cast<decltype(x)>(x_obj); } catch (...) { throw std::runtime_error("bad_cast: x"); }
            try { y = nb::cast<decltype(y)>(y_obj); } catch (...) { throw std::runtime_error("bad_cast: y"); }
            try { dist_mtx = nb::cast<decltype(dist_mtx)>(dist_mtx_obj); } catch (...) { throw std::runtime_error("bad_cast: dist_mtx"); }
            try { service_times = nb::cast<decltype(service_times)>(service_times_obj); } catch (...) { throw std::runtime_error("bad_cast: service_times"); }
            try { demand = nb::cast<decltype(demand)>(demand_obj); } catch (...) { throw std::runtime_error("bad_cast: demand"); }

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
