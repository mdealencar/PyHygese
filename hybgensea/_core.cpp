#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>

#include <stdexcept>
#include <vector>
#include <iostream>

extern "C" {
#include "AlgorithmParameters.h"
#include "C_Interface.h"
}

namespace nb = nanobind;

#include <streambuf>
#include <string>

class PythonStdoutBuffer : public std::streambuf {
public:
    ~PythonStdoutBuffer() override { sync(); }

protected:
    int overflow(int c) override {
        if (c != EOF) {
            buffer_.push_back(static_cast<char>(c));
            if (c == '\n') {
                flush_buffer();
            }
        }
        return c;
    }

    int sync() override {
        flush_buffer();
        return 0;
    }

private:
    void flush_buffer() {
        if (buffer_.empty()) {
            return;
        }
        nb::gil_scoped_acquire guard;
        PySys_WriteStdout("%s", buffer_.c_str());
        buffer_.clear();
    }

    std::string buffer_;
};

class ScopedPythonStdoutRedirect {
public:
    ScopedPythonStdoutRedirect() : original_(std::cout.rdbuf(&buffer_)) {}

    ~ScopedPythonStdoutRedirect() {
        std::cout.rdbuf(original_);
    }

private:
    PythonStdoutBuffer buffer_;
    std::streambuf *original_;
};

struct PySolution {
    double cost;
    double time;
    std::vector<std::vector<int>> routes;
};

static PySolution convert_solution(Solution *sol) {
    if (sol == nullptr) {
        throw std::runtime_error("HGS-CVRP returned a null solution pointer.");
    }

    PySolution result{};
    result.cost = sol->cost;
    result.time = sol->time;
    result.routes.reserve(static_cast<size_t>(sol->n_routes));

    for (int i = 0; i < sol->n_routes; ++i) {
        const SolutionRoute &route = sol->routes[i];
        std::vector<int> path;
        path.reserve(static_cast<size_t>(route.length));
        for (int j = 0; j < route.length; ++j) {
            path.push_back(route.path[j]);
        }
        result.routes.push_back(std::move(path));
    }

    delete_solution(sol);
    return result;
}

static AlgorithmParameters make_ap(
    int nbGranular, int mu, int lambda_, int nbElite, int nbClose,
    int nbIterPenaltyManagement, double targetFeasible, double penaltyDecrease,
    double penaltyIncrease, int seed, int nbIter, int nbIterTraces,
    double timeLimit, bool useSwapStar) {
    AlgorithmParameters ap{};
    ap.nbGranular = nbGranular;
    ap.mu = mu;
    ap.lambda = lambda_;
    ap.nbElite = nbElite;
    ap.nbClose = nbClose;
    ap.nbIterPenaltyManagement = nbIterPenaltyManagement;
    ap.targetFeasible = targetFeasible;
    ap.penaltyDecrease = penaltyDecrease;
    ap.penaltyIncrease = penaltyIncrease;
    ap.seed = seed;
    ap.nbIter = nbIter;
    ap.nbIterTraces = nbIterTraces;
    ap.timeLimit = timeLimit;
    ap.useSwapStar = useSwapStar;
    return ap;
}

NB_MODULE(_core, m) {
    nb::class_<PySolution>(m, "_PySolution")
        .def_rw("cost", &PySolution::cost)
        .def_rw("time", &PySolution::time)
        .def_rw("routes", &PySolution::routes);

    m.def("solve_cvrp", [](nb::ndarray<double, nb::ndim<1>, nb::c_contig> x,
                            nb::ndarray<double, nb::ndim<1>, nb::c_contig> y,
                            nb::ndarray<double, nb::ndim<1>, nb::c_contig> service,
                            nb::ndarray<double, nb::ndim<1>, nb::c_contig> demand,
                            double vehicleCapacity,
                            double durationLimit,
                            bool isRoundingInteger,
                            bool isDurationConstraint,
                            int max_nbVeh,
                            int nbGranular, int mu, int lambda_, int nbElite, int nbClose,
                            int nbIterPenaltyManagement, double targetFeasible,
                            double penaltyDecrease, double penaltyIncrease, int seed,
                            int nbIter, int nbIterTraces, double timeLimit,
                            bool useSwapStar,
                            bool verbose) {
        const int n = static_cast<int>(x.shape(0));
        AlgorithmParameters ap = make_ap(nbGranular, mu, lambda_, nbElite, nbClose,
                                         nbIterPenaltyManagement, targetFeasible,
                                         penaltyDecrease, penaltyIncrease, seed, nbIter,
                                         nbIterTraces, timeLimit, useSwapStar);

        hgs_set_output_stdout();
        ScopedPythonStdoutRedirect stream;

        Solution *sol = ::solve_cvrp(
            n,
            const_cast<double *>(x.data()),
            const_cast<double *>(y.data()),
            const_cast<double *>(service.data()),
            const_cast<double *>(demand.data()),
            vehicleCapacity,
            durationLimit,
            isRoundingInteger,
            isDurationConstraint,
            max_nbVeh,
            &ap,
            verbose
        );

        return convert_solution(sol);
    });

    m.def("solve_cvrp_dist_mtx", [](nb::ndarray<double, nb::ndim<1>, nb::c_contig> x,
                                     nb::ndarray<double, nb::ndim<1>, nb::c_contig> y,
                                     nb::ndarray<double, nb::ndim<2>, nb::c_contig> dist_mtx,
                                     nb::ndarray<double, nb::ndim<1>, nb::c_contig> service,
                                     nb::ndarray<double, nb::ndim<1>, nb::c_contig> demand,
                                     double vehicleCapacity,
                                     double durationLimit,
                                     bool isDurationConstraint,
                                     int max_nbVeh,
                                     int nbGranular, int mu, int lambda_, int nbElite, int nbClose,
                                     int nbIterPenaltyManagement, double targetFeasible,
                                     double penaltyDecrease, double penaltyIncrease, int seed,
                                     int nbIter, int nbIterTraces, double timeLimit,
                                     bool useSwapStar,
                                     bool verbose) {
        const int n = static_cast<int>(x.shape(0));
        AlgorithmParameters ap = make_ap(nbGranular, mu, lambda_, nbElite, nbClose,
                                         nbIterPenaltyManagement, targetFeasible,
                                         penaltyDecrease, penaltyIncrease, seed, nbIter,
                                         nbIterTraces, timeLimit, useSwapStar);

        hgs_set_output_stdout();
        ScopedPythonStdoutRedirect stream;

        Solution *sol = ::solve_cvrp_dist_mtx(
            n,
            const_cast<double *>(x.data()),
            const_cast<double *>(y.data()),
            const_cast<double *>(dist_mtx.data()),
            const_cast<double *>(service.data()),
            const_cast<double *>(demand.data()),
            vehicleCapacity,
            durationLimit,
            isDurationConstraint,
            max_nbVeh,
            &ap,
            verbose
        );

        return convert_solution(sol);
    });
}
