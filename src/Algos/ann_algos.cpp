#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "Algorithms/LSH.hpp"
#include "Algorithms/knn.hpp"
#include "Algorithms/ivfflat.hpp"
#include "Algorithms/ivfpq.hpp"
#include "Algorithms/hypercube.hpp"

namespace py = pybind11;

template <typename T>
void bind_algorithms(py::module &m, const std::string &suffix) {
    using IVFFlatType = IVFFlat<T>;
    using LSHType = LSH<T>;
    using HyperCubeType = HyperCube<T>;
    using IVFPQType = IVFPQ<T>;

    py::class_<IVFFlatType>(m, ("IVFFlat" + suffix).c_str())
        .def(py::init<size_t, size_t, int>(), py::arg("num_clusters"), py::arg("vec_dim"), py::arg("kmeans_iters") = 15)
        .def("add_vector", &IVFFlatType::add_vector)
        .def("add_vectors", &IVFFlatType::add_vectors)
        .def("train", &IVFFlatType::train, py::arg("seed") = 12345)
        .def("query", &IVFFlatType::query, py::arg("p"), py::arg("k"), py::arg("nprobe") = 1)
        .def("range_query", &IVFFlatType::range_query, py::arg("p"), py::arg("range"), py::arg("nprobe") = 1)
        .def("get_centroids", &IVFFlatType::get_centroids)
        .def("get_inverted_lists", &IVFFlatType::get_inverted_lists);

    py::class_<LSHType>(m, ("LSH" + suffix).c_str())
        .def(py::init<size_t, int, int, float, size_t, int>(), 
             py::arg("hashTable_size"), py::arg("num_tables"), py::arg("HashFunction_size"), 
             py::arg("w"), py::arg("vec_dim"), py::arg("seed"))
        .def("insert_to_hashTables", &LSHType::insert_to_hashTables)
        .def("returnANN", &LSHType::returnANN, 
             py::arg("p"), py::arg("k"), py::arg("range_bool") = false, py::arg("range") = 0.0);

    py::class_<HyperCubeType>(m, ("HyperCube" + suffix).c_str())
        .def(py::init<const std::vector<std::vector<T>>&, size_t, float, size_t, int>(),
             py::arg("points"), py::arg("k_proj"), py::arg("w"), py::arg("vec_dim"), py::arg("seed"))
        .def("returnANN", &HyperCubeType::returnANN,
             py::arg("p"), py::arg("M"), py::arg("k"), py::arg("probe") = 3,
             py::arg("range_bool") = false, py::arg("range") = 0.0);

    py::class_<IVFPQType>(m, ("IVFPQ" + suffix).c_str())
        .def(py::init<size_t, size_t, size_t, size_t, int>(),
             py::arg("num_coarse_clusters"), py::arg("vec_dim"), py::arg("M"), py::arg("Ks") = 256, py::arg("kmeans_iters") = 15)
        .def("add_vector", &IVFPQType::add_vector)
        .def("add_vectors", &IVFPQType::add_vectors)
        .def("train", &IVFPQType::train, py::arg("seed") = 12345)
        .def("query", &IVFPQType::query, py::arg("q"), py::arg("k"), py::arg("nprobe") = 1)
        .def("range_query", &IVFPQType::range_query, py::arg("q"), py::arg("range"), py::arg("nprobe") = 1)
        .def("get_coarse_centroids", &IVFPQType::get_coarse_centroids)
        .def("get_inverted_lists", &IVFPQType::get_inverted_lists);
}

PYBIND11_MODULE(ann_algos, m) {
    m.doc() = "ANN Algorithms bindings";

    py::class_<Neighbor>(m, "Neighbor")
        .def(py::init<uint64_t, double>())
        .def_readwrite("idx", &Neighbor::idx)
        .def_readwrite("distance", &Neighbor::distance)
        .def("__repr__", [](const Neighbor &n) {
            return "<Neighbor idx=" + std::to_string(n.idx) + " distance=" + std::to_string(n.distance) + ">";
        });

    bind_algorithms<float>(m, "Float");
    bind_algorithms<uint8_t>(m, "Uint8");
    
   m.def(
        "kNN_float",
        &kNN<float>,
        py::arg("points"),
        py::arg("query"),
        py::arg("k"),
        "Exact kNN float"
    );

    m.def(
        "kNN_uint8",
        &kNN<uint8_t>,
        py::arg("points"),
        py::arg("query"),
        py::arg("k"),
        "Exact kNN uint8"
    );

}
