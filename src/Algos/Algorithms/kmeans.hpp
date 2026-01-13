#pragma once
#ifndef KMEANS_HPP
#define KMEANS_HPP

#include <vector>
#include <random>
#include <limits>
#include <numeric>
#include <cstddef>


// Reusable k-means utilities

namespace kmeans {

// Compute squared L2 distance between a data vector and a float centroid
template <typename NumType>
inline double l2sq_point_centroid(const std::vector<NumType>& v, const std::vector<float>& c) {
    double s = 0.0;
    for (size_t i = 0; i < c.size(); ++i) {
        double diff = static_cast<double>(v[i]) - static_cast<double>(c[i]);
        s += diff * diff;
    }
    return s;
}

// Choose initial centroids with k-means++
template <typename NumType>
inline void kmeans_plus_plus_init(const std::vector<std::vector<NumType>>& data,
                                  size_t num_clusters,
                                  size_t vec_dim,
                                  int seed,
                                  std::vector<std::vector<float>>& centroids_out) {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<size_t> uni(0, data.size() - 1);

    centroids_out.resize(num_clusters, std::vector<float>(vec_dim, 0.0f));

    // First center at random
    {
        size_t idx = uni(gen);
        for (size_t d = 0; d < vec_dim; ++d) centroids_out[0][d] = static_cast<float>(data[idx][d]);
    }

    std::vector<double> min_d2(data.size(), std::numeric_limits<double>::max());

    for (size_t c = 1; c < num_clusters; ++c) {
        // Update min distances to the latest chosen centroid
        for (size_t i = 0; i < data.size(); ++i) {
            double d2 = l2sq_point_centroid(data[i], centroids_out[c - 1]);
            if (d2 < min_d2[i]) min_d2[i] = d2;
        }

        double sum = std::accumulate(min_d2.begin(), min_d2.end(), 0.0);
        if (sum == 0.0) {
            size_t idx = uni(gen);
            for (size_t d = 0; d < vec_dim; ++d) centroids_out[c][d] = static_cast<float>(data[idx][d]);
            continue;
        }

        std::uniform_real_distribution<double> ur(0.0, sum);
        double r = ur(gen);
        double run = 0.0;
        size_t chosen = 0;
        for (size_t i = 0; i < min_d2.size(); ++i) { run += min_d2[i]; if (run >= r) { chosen = i; break; } }
        for (size_t d = 0; d < vec_dim; ++d) centroids_out[c][d] = static_cast<float>(data[chosen][d]);
    }
}

// Assign each point to the nearest centroid
template <typename NumType>
inline void assign_all(const std::vector<std::vector<NumType>>& data,
                       const std::vector<std::vector<float>>& centroids,
                       std::vector<size_t>& assignment_out) {
    assignment_out.resize(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        size_t best = 0;
        double best_d = std::numeric_limits<double>::max();
        for (size_t c = 0; c < centroids.size(); ++c) {
            double d = l2sq_point_centroid(data[i], centroids[c]);
            if (d < best_d) { best_d = d; best = c; }
        }
        assignment_out[i] = best;
    }
}

// One-stop k-means training: initialize then run Lloyd iterations
template <typename NumType>
inline void kmeans_train(const std::vector<std::vector<NumType>>& data,
                         size_t num_clusters,
                         size_t vec_dim,
                         int iters,
                         int seed,
                         std::vector<std::vector<float>>& centroids_out,
                         std::vector<size_t>& assignment_out) {
    if (data.empty() || num_clusters == 0 || vec_dim == 0) return;

    // Init
    kmeans_plus_plus_init(data, num_clusters, vec_dim, seed, centroids_out);

    // Lloyd iterations
    assignment_out.assign(data.size(), 0);
    for (int it = 0; it < iters; ++it) {
        assign_all(data, centroids_out, assignment_out);

        std::vector<std::vector<double>> new_centroids(num_clusters, std::vector<double>(vec_dim, 0.0));
        std::vector<size_t> counts(num_clusters, 0);

        for (size_t i = 0; i < data.size(); ++i) {
            size_t c = assignment_out[i];
            counts[c]++;
            const auto& v = data[i];
            for (size_t d = 0; d < vec_dim; ++d) new_centroids[c][d] += static_cast<double>(v[d]);
        }
        for (size_t c = 0; c < num_clusters; ++c) {
            if (counts[c] == 0) continue;
            for (size_t d = 0; d < vec_dim; ++d)
                centroids_out[c][d] = static_cast<float>(new_centroids[c][d] / static_cast<double>(counts[c]));
        }
    }

    // Final assignment
    assign_all(data, centroids_out, assignment_out);
}

} // namespace kmeans

#endif


