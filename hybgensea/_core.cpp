#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <cmath>
#include <climits>
#include <streambuf>
#include <string>
#include <vector>
#include <iostream>

#include "AlgorithmParameters.h"
#include "Genetic.h"

namespace nb = nanobind;

// Custom streambuf that captures log output and optionally invokes a Python callback per line.
class CallbackStreamBuf : public std::streambuf {
    std::string line_buf_;
    std::string full_log_;
    nb::object callback_;
    bool has_callback_;
protected:
    int overflow(int c) override {
        if (c != EOF) {
            line_buf_.push_back(static_cast<char>(c));
            if (c == '\n') flush_line();
        }
        return c;
    }
    int sync() override { flush_line(); return 0; }
private:
    void flush_line() {
        if (line_buf_.empty()) return;
        full_log_ += line_buf_;
        if (has_callback_) {
            nb::gil_scoped_acquire gil;
            callback_(nb::str(line_buf_.c_str(), line_buf_.size()));
        }
        line_buf_.clear();
    }
public:
    CallbackStreamBuf(nb::handle cb)
        : callback_(nb::borrow(cb)), has_callback_(!callback_.is_none()) {}
    const std::string& get_log() const { return full_log_; }
};

struct PySolution {
    double cost;
    double time;
    std::vector<std::vector<int>> routes;
    std::string log;
};

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

// Extract PySolution from a completed Genetic solver (must be called while owning the data).
static PySolution extract_solution(Population& population, const Params& params) {
    PySolution result{};
    const Individual* best = population.getBestFound();
    if (best == nullptr) {
        throw std::runtime_error("HGS-CVRP found no feasible solution.");
    }
    result.cost = best->eval.penalizedCost;
    result.time = params.elapsedSeconds();
    for (const auto& route : best->chromR) {
        if (!route.empty()) {
            result.routes.push_back(route);
        }
    }
    return result;
}

NB_MODULE(_core, m) {
    nb::class_<PySolution>(m, "_PySolution")
        .def_rw("cost", &PySolution::cost)
        .def_rw("time", &PySolution::time)
        .def_rw("routes", &PySolution::routes)
        .def_rw("log", &PySolution::log);

    // solve_cvrp: from coordinates (computes distance matrix internally)
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
                            bool verbose,
                            nb::handle log_callback) {
        const int n = static_cast<int>(x.shape(0));
        AlgorithmParameters ap = make_ap(nbGranular, mu, lambda_, nbElite, nbClose,
                                         nbIterPenaltyManagement, targetFeasible,
                                         penaltyDecrease, penaltyIncrease, seed, nbIter,
                                         nbIterTraces, timeLimit, useSwapStar);

        // Build vectors from numpy arrays
        std::vector<double> x_coords(x.data(), x.data() + n);
        std::vector<double> y_coords(y.data(), y.data() + n);
        std::vector<double> service_time(service.data(), service.data() + n);
        std::vector<double> demands(demand.data(), demand.data() + n);

        // Compute distance matrix from coordinates
        std::vector<std::vector<double>> dist_mtx(n, std::vector<double>(n, 0.0));
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                double dx = x_coords[i] - x_coords[j];
                double dy = y_coords[i] - y_coords[j];
                double dist = std::sqrt(dx * dx + dy * dy);
                if (isRoundingInteger) dist = std::round(dist);
                dist_mtx[i][j] = dist;
                dist_mtx[j][i] = dist;
            }
        }

        CallbackStreamBuf buf(log_callback);
        std::ostream log_stream(&buf);

        PySolution result;
        {
            nb::gil_scoped_release release;
            Params params(x_coords, y_coords, dist_mtx, service_time, demands,
                          vehicleCapacity, durationLimit, max_nbVeh,
                          isDurationConstraint, verbose, ap, log_stream);
            Genetic solver(params);
            solver.run();
            result = extract_solution(solver.population, params);
        }
        result.log = buf.get_log();
        return result;
    },
    nb::arg("x"), nb::arg("y"), nb::arg("service"), nb::arg("demand"),
    nb::arg("vehicleCapacity"), nb::arg("durationLimit"),
    nb::arg("isRoundingInteger"), nb::arg("isDurationConstraint"),
    nb::arg("max_nbVeh"),
    nb::arg("nbGranular"), nb::arg("mu"), nb::arg("lambda_"), nb::arg("nbElite"), nb::arg("nbClose"),
    nb::arg("nbIterPenaltyManagement"), nb::arg("targetFeasible"),
    nb::arg("penaltyDecrease"), nb::arg("penaltyIncrease"), nb::arg("seed"),
    nb::arg("nbIter"), nb::arg("nbIterTraces"), nb::arg("timeLimit"),
    nb::arg("useSwapStar"),
    nb::arg("verbose"), nb::arg("log_callback").none() = nb::none());

    // solve_cvrp_dist_mtx: with pre-computed distance matrix
    m.def("solve_cvrp_dist_mtx", [](nb::ndarray<double, nb::ndim<1>, nb::c_contig> x,
                                     nb::ndarray<double, nb::ndim<1>, nb::c_contig> y,
                                     nb::ndarray<double, nb::ndim<2>, nb::c_contig> dist_mtx_arr,
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
                                     bool verbose,
                                     nb::handle log_callback) {
        const int n = static_cast<int>(x.shape(0));
        AlgorithmParameters ap = make_ap(nbGranular, mu, lambda_, nbElite, nbClose,
                                         nbIterPenaltyManagement, targetFeasible,
                                         penaltyDecrease, penaltyIncrease, seed, nbIter,
                                         nbIterTraces, timeLimit, useSwapStar);

        // Build vectors from numpy arrays
        std::vector<double> x_coords(x.data(), x.data() + n);
        std::vector<double> y_coords(y.data(), y.data() + n);
        std::vector<double> service_time(service.data(), service.data() + n);
        std::vector<double> demands(demand.data(), demand.data() + n);

        // Convert 2D distance matrix from row-major numpy array
        const double* dm = dist_mtx_arr.data();
        std::vector<std::vector<double>> dist_mtx(n, std::vector<double>(n));
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                dist_mtx[i][j] = dm[i * n + j];
            }
        }

        CallbackStreamBuf buf(log_callback);
        std::ostream log_stream(&buf);

        PySolution result;
        {
            nb::gil_scoped_release release;
            Params params(x_coords, y_coords, dist_mtx, service_time, demands,
                          vehicleCapacity, durationLimit, max_nbVeh,
                          isDurationConstraint, verbose, ap, log_stream);
            Genetic solver(params);
            solver.run();
            result = extract_solution(solver.population, params);
        }
        result.log = buf.get_log();
        return result;
    },
    nb::arg("x"), nb::arg("y"), nb::arg("dist_mtx_arr"),
    nb::arg("service"), nb::arg("demand"),
    nb::arg("vehicleCapacity"), nb::arg("durationLimit"),
    nb::arg("isDurationConstraint"),
    nb::arg("max_nbVeh"),
    nb::arg("nbGranular"), nb::arg("mu"), nb::arg("lambda_"), nb::arg("nbElite"), nb::arg("nbClose"),
    nb::arg("nbIterPenaltyManagement"), nb::arg("targetFeasible"),
    nb::arg("penaltyDecrease"), nb::arg("penaltyIncrease"), nb::arg("seed"),
    nb::arg("nbIter"), nb::arg("nbIterTraces"), nb::arg("timeLimit"),
    nb::arg("useSwapStar"),
    nb::arg("verbose"), nb::arg("log_callback").none() = nb::none());
}
