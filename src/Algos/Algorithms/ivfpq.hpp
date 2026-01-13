#pragma once
#ifndef IVF_PQ_HPP
#define IVF_PQ_HPP

#include <vector>
#include <queue>
#include <random>
#include <cstdint>
#include <iostream>
#include <algorithm>
#include <limits>
#include <numeric>
#include <cmath>

#include "neighbor.hpp"
#include "metrics.hpp"
#include "kmeans.hpp"


/**
 * IVF-PQ (Inverted File Index with Product Quantization)
 * 
 * An approximate nearest neighbor search algorithm that combines:
 * 1. Coarse quantization (IVF): Partition dataset using k-means clustering
 * 2. Fine quantization (PQ): Compress residual vectors using Product Quantization
 * 
 * Algorithm Overview:
 * 
 * Training Phase:
 * - Step 1: Train coarse centroids using k-means (same as IVFFlat)
 * - Step 2: For each vector, compute residual = vector - its coarse centroid
 * - Step 3: Split each residual into M subvectors of dimension vec_dim/M
 * - Step 4: Train a separate codebook for each subvector using k-means (Ks centroids each)
 * - Step 5: Encode each residual by finding closest centroid in each subvector codebook
 * 
 * Query Phase:
 * - Step 1: Find nprobe closest coarse centroids (same as IVFFlat)
 * - Step 2: For each selected centroid, compute query residual
 * - Step 3: Precompute distance tables for PQ codebooks (distance from query residual subvectors to all PQ centroids)
 * - Step 4: For each encoded vector in selected lists, compute approximate distance using lookup tables
 * - Step 5: Return k nearest neighbors based on approximate distances
 * 
 * Advantages:
 * - Much more memory efficient than IVFFlat (stores codes instead of full vectors)
 * - Fast query time using precomputed distance tables
 * - Scales well to large datasets
 * 
 * Trade-offs:
 * - Approximate distances (not exact like IVFFlat)
 * - Slightly lower accuracy than IVFFlat for the same nprobe
 * - More complex training (trains M codebooks)
 * 
 * Parameters:
 * - num_coarse_clusters: Number of IVF clusters (more = better accuracy but slower)
 * - M: Number of PQ subvectors (more = better accuracy but more memory/time)
 * - Ks: Number of PQ centroids per subvector (typically 256, set by nbits)
 * - nbits: log2(Ks), e.g., nbits=8 means Ks=256
 * - nprobe: Number of clusters to search during query
 * 
 * Memory Usage:
 * - Storage per vector: 8 bytes (coarse cluster ID) + M bytes (PQ code)
 * - Compared to IVFFlat: vec_dim * sizeof(NumType) bytes per vector
 * 
 */
template <typename NumType>
class IVFPQ {
    // Coarse centroids (IVF part)
    std::vector<std::vector<float>> coarse_centroids;

    // PQ codebooks: M subvectors each with Ks centroids
    std::vector<std::vector<std::vector<float>>> pq_codebooks;

    // encoded inverted lists: each entry is (originalIndex, PQ code)
    std::vector<std::vector<std::pair<size_t, std::vector<uint8_t>>>> inverted_lists;

    // original vectors for training and reconstruction
    std::vector<std::vector<NumType>> vectors;

    // Configuration
    size_t num_coarse_clusters;  // Number of IVF clusters
    size_t vec_dim;              // Dimensionality of input vectors
    size_t M;                    // Number of PQ subvectors
    size_t Ks;                   // Number of PQ centroids per subvector
    int kmeans_iters;            // Iterations for both coarse and PQ kmeans

public:
    IVFPQ(size_t num_coarse_clusters_, size_t vec_dim_, size_t M_, size_t Ks_ = 256, int kmeans_iters_ = 15)
        : num_coarse_clusters(num_coarse_clusters_), vec_dim(vec_dim_), M(M_), Ks(Ks_), kmeans_iters(kmeans_iters_) {
        if (num_coarse_clusters == 0 || vec_dim == 0 || M == 0 || vec_dim % M != 0) {
            std::cerr << "IVFPQ: invalid parameters (check M divides vec_dim)" << std::endl;
            std::exit(EXIT_FAILURE);
        }
        coarse_centroids.resize(num_coarse_clusters, std::vector<float>(vec_dim, 0.0f));
        inverted_lists.resize(num_coarse_clusters);
        pq_codebooks.resize(M);
    }

    void add_vector(const std::vector<NumType>& p) {
        if (p.size() != vec_dim) {
            std::cerr << "IVFPQ::add_vector: vector dim mismatch" << std::endl;
            std::exit(EXIT_FAILURE);
        }
        vectors.push_back(p);
    }

    void add_vectors(const std::vector<std::vector<NumType>>& pts) {
        for (const auto& p : pts) add_vector(p);
    }

    /**
     * Train the IVFPQ index.
     * 
     * Training process:
     * 1. Train coarse centroids: Run k-means on all vectors to find num_coarse_clusters centroids
     * 2. Compute residuals: For each vector, residual = vector - its assigned coarse centroid
     * 3. Train PQ codebooks: Split residuals into M subvectors, train a k-means codebook for each
     *    - Each subvector has dimension vec_dim/M
     *    - Each codebook has Ks centroids (e.g., 256 for 8 bits)
     *    - Result: M codebooks, each with Ks centroids of dimension vec_dim/M
     * 4. Encode vectors: For each vector's residual, find closest centroid in each codebook
     *    - Encoding: M bytes (one byte per subvector, storing the centroid index)
     * 
     */
    void train(int seed = 12345) {
        if (vectors.empty()) {
            std::cerr << "IVFPQ::train: no vectors added" << std::endl;
            std::exit(EXIT_FAILURE);
        }

        // Step 1: Train coarse quantizer (IVF part)
        // Cluster all vectors into num_coarse_clusters groups
        std::vector<size_t> coarse_assignment;
        kmeans::kmeans_train(vectors, num_coarse_clusters, vec_dim, kmeans_iters, seed, coarse_centroids, coarse_assignment);

        // Step 2: Compute residuals (vector - its coarse centroid)
        // Residuals are used for PQ training because they have smaller variance than original vectors
        std::vector<std::vector<std::vector<NumType>>> residuals(num_coarse_clusters);
        for (size_t i = 0; i < vectors.size(); ++i) {
            size_t cid = coarse_assignment[i];
            std::vector<NumType> residual(vec_dim);
            for (size_t j = 0; j < vec_dim; ++j)
                residual[j] = static_cast<float>(vectors[i][j]) - coarse_centroids[cid][j];
            residuals[cid].push_back(std::move(residual));
        }

        // Step 3: Train PQ codebooks
        // Split the vec_dim-dimensional space into M subspaces of dimension vec_dim/M
        size_t subdim = vec_dim / M;
        pq_codebooks.assign(M, std::vector<std::vector<float>>(Ks, std::vector<float>(subdim, 0.0f)));

        // Collect all residuals from all clusters for PQ training
        std::vector<std::vector<float>> all_residuals;
        for (const auto& cl : residuals)
            for (const auto& v : cl)
                all_residuals.emplace_back(v.begin(), v.end());

        // For each of the M subspaces, train a separate codebook
        // Each codebook learns Ks centroids to quantize that subspace
        for (size_t m = 0; m < M; ++m) {
            // Extract the m-th subvector from each residual
            std::vector<std::vector<float>> subspace;
            subspace.reserve(all_residuals.size());
            for (const auto& v : all_residuals)
                subspace.emplace_back(v.begin() + m * subdim, v.begin() + (m + 1) * subdim);

            // Train k-means on this subspace to get Ks centroids
            std::vector<size_t> assign_dummy;
            kmeans::kmeans_train(subspace, Ks, subdim, kmeans_iters, seed + m, pq_codebooks[m], assign_dummy);
        }

        // Step 4: Encode all vectors
        // For each vector, find its residual, then encode each subvector of the residual
        // Result: M-byte code where each byte is the index of the closest centroid in that subspace
        for (auto& lst : inverted_lists) lst.clear();
        for (size_t i = 0; i < vectors.size(); ++i) {
            size_t cid = coarse_assignment[i];
            // Compute residual again for encoding
            std::vector<float> residual(vec_dim);
            for (size_t j = 0; j < vec_dim; ++j)
                residual[j] = static_cast<float>(vectors[i][j]) - coarse_centroids[cid][j];
            // Encode residual into PQ code (M bytes)
            std::vector<uint8_t> code = encode_vector(residual);
            // Store (original_index, PQ_code) in the appropriate inverted list
            inverted_lists[cid].emplace_back(i, std::move(code));
        }
    }

    /**
     * Query for k approximate nearest neighbors using Product Quantization.
     * 
     * Query process:
     * 1. Find nprobe closest coarse centroids (same as IVFFlat)
     * 2. For each selected centroid:
     *    a. Compute query residual = query - coarse_centroid
     *    b. Precompute distance tables: distance from each query residual subvector to all PQ centroids
     *    c. For each encoded vector in the list, compute approximate distance using table lookups
     * 3. Maintain max-heap of size k
     * 4. Return k nearest neighbors sorted by distance
     * 
     * Distance Computation:
     * - For encoded vector with code [c0, c1, ..., c_{M-1}]
     * - Distance = sqrt(sum over m of pq_tables[m][c_m])
     * - Where pq_tables[m][k] = ||query_residual_subvector[m] - pq_codebooks[m][k]||^2
     * 
     * Time complexity: O(num_clusters * vec_dim + nprobe * (M * Ks * subdim + list_size * M))
     * - Finding centroids: O(num_clusters * vec_dim)
     * - Building PQ tables: O(M * Ks * subdim) per centroid
     * - Searching lists: O(list_size * M) per centroid
     * 

     */
    std::vector<Neighbor> query(const std::vector<NumType>& q, int k, int nprobe = 1) const {
        if (q.size() != vec_dim) {
            std::cerr << "IVFPQ::query: vector dim mismatch" << std::endl;
            std::exit(EXIT_FAILURE);
        }

        nprobe = std::max(1, std::min(static_cast<int>(num_coarse_clusters), nprobe));

        // Step 1: Find nprobe closest coarse centroids
        std::vector<std::pair<double, size_t>> coarse_dists;
        coarse_dists.reserve(num_coarse_clusters);
        for (size_t c = 0; c < num_coarse_clusters; ++c) {
            double dist = dist_to_centroid(q, coarse_centroids[c]);
            coarse_dists.emplace_back(dist, c);
        }
        std::nth_element(coarse_dists.begin(), coarse_dists.begin() + (nprobe - 1), coarse_dists.end());
        coarse_dists.resize(nprobe);

        // Step 2: Search selected clusters using PQ distance approximation
        size_t subdim = vec_dim / M;
        auto cmp = [](const Neighbor& a, const Neighbor& b) {
            return a.distance < b.distance;
        };
        std::priority_queue<Neighbor, std::vector<Neighbor>, decltype(cmp)> topK(cmp);

        for (const auto& [_, cid] : coarse_dists) {
            // Compute query residual relative to this centroid
            std::vector<float> q_res(vec_dim);
            for (size_t j = 0; j < vec_dim; ++j)
                q_res[j] = static_cast<float>(q[j]) - coarse_centroids[cid][j];

            // Build PQ distance tables for this centroid's residual
            // pq_tables[m][k] = squared distance from query_residual subvector[m] to pq_codebooks[m][k]
            // This precomputation allows fast distance lookup during search
            std::vector<std::vector<float>> pq_tables(M, std::vector<float>(Ks, 0.0f));
            for (size_t m = 0; m < M; ++m) {
                for (size_t k_ = 0; k_ < Ks; ++k_) {
                    double d = 0.0;
                    for (size_t d_ = 0; d_ < subdim; ++d_) {
                        double diff = q_res[m * subdim + d_] - pq_codebooks[m][k_][d_];
                        d += diff * diff;
                    }
                    pq_tables[m][k_] = static_cast<float>(d);  // Store squared distance
                }
            }

            // Search encoded vectors using precomputed tables
            // For each encoded vector, lookup distances for each subvector and sum them
            for (const auto& [idx, code] : inverted_lists[cid]) {
                double dist = 0.0;
                // Sum squared distances from each subvector lookup
                for (size_t m = 0; m < M; ++m)
                    dist += pq_tables[m][code[m]];  // code[m] is the centroid index for subvector m
                // Take square root to get approximate Euclidean distance
                topK.emplace(idx, std::sqrt(dist));
                if (static_cast<int>(topK.size()) > k) topK.pop();
            }
        }

        // Extract final results in ascending distance order
        std::vector<Neighbor> neighbors;
        neighbors.reserve(topK.size());
        while (!topK.empty()) {
            neighbors.push_back(topK.top());
            topK.pop();
        }
        std::reverse(neighbors.begin(), neighbors.end());
        return neighbors;
    }

    /**
     * Range query: find all neighbors within a given distance threshold.
     * 
     * Similar to query(), but returns all vectors within distance 'range' instead of top-k.
     * Uses the same PQ distance approximation as query().
     * 
     */
    std::vector<Neighbor> range_query(const std::vector<NumType>& q, double range, int nprobe = 1) const {
        if (q.size() != vec_dim) {
            std::cerr << "IVFPQ::range_query: vector dim mismatch" << std::endl;
            std::exit(EXIT_FAILURE);
        }

        nprobe = std::max(1, std::min(static_cast<int>(num_coarse_clusters), nprobe));

        // Find nprobe closest coarse centroids
        std::vector<std::pair<double, size_t>> coarse_dists;
        coarse_dists.reserve(num_coarse_clusters);
        for (size_t c = 0; c < num_coarse_clusters; ++c) {
            double dist = dist_to_centroid(q, coarse_centroids[c]);
            coarse_dists.emplace_back(dist, c);
        }
        std::nth_element(coarse_dists.begin(), coarse_dists.begin() + (nprobe - 1), coarse_dists.end());
        coarse_dists.resize(nprobe);

        size_t subdim = vec_dim / M;
        std::vector<Neighbor> result;

        for (const auto& [_, cid] : coarse_dists) {
            // Build PQ distance tables (same as query method)
            std::vector<float> q_res(vec_dim);
            for (size_t j = 0; j < vec_dim; ++j)
                q_res[j] = static_cast<float>(q[j]) - coarse_centroids[cid][j];

            std::vector<std::vector<float>> pq_tables(M, std::vector<float>(Ks, 0.0f));
            for (size_t m = 0; m < M; ++m) {
                for (size_t k_ = 0; k_ < Ks; ++k_) {
                    double d = 0.0;
                    for (size_t d_ = 0; d_ < subdim; ++d_) {
                        double diff = q_res[m * subdim + d_] - pq_codebooks[m][k_][d_];
                        d += diff * diff;
                    }
                    pq_tables[m][k_] = static_cast<float>(d);
                }
            }

            // Collect all vectors within range (no heap needed)
            for (const auto& [idx, code] : inverted_lists[cid]) {
                double dist = 0.0;
                for (size_t m = 0; m < M; ++m) dist += pq_tables[m][code[m]];
                dist = std::sqrt(dist);
                if (dist <= range) result.emplace_back(idx, dist);
            }
        }

        std::sort(result.begin(), result.end(), [](const auto& a, const auto& b) { return a.distance < b.distance; });
        return result;
    }

    // Expose read-only accessors for silhouette calculation
    const std::vector<std::vector<float>>& get_coarse_centroids() const { return coarse_centroids; }
    const std::vector<std::vector<std::pair<size_t, std::vector<uint8_t>>>>& get_inverted_lists() const { return inverted_lists; }

private:
    /**
     * Compute Euclidean distance from a vector to a centroid.
     * 
     * Helper function for computing distances between vectors and coarse centroids.
     * 
     
     */
    static double dist_to_centroid(const std::vector<NumType>& v, const std::vector<float>& c) {
        double s = 0.0;
        for (size_t i = 0; i < c.size(); ++i) {
            double diff = static_cast<double>(v[i]) - static_cast<double>(c[i]);
            s += diff * diff;
        }
        return std::sqrt(s);
    }
    
    /**
     * Encode a residual vector into a Product Quantization code.
     * 
     * Encoding process:
     * 1. Split residual into M subvectors of dimension vec_dim/M
     * 2. For each subvector, find the closest centroid in the corresponding PQ codebook
     * 3. Store the index of that centroid as a byte (0 to Ks-1)
     * 
     * Result: M-byte code where code[m] is the index of the closest centroid in codebook m
     * 
     * Example: If M=4 and Ks=256, result is 4 bytes encoding the quantization of 4 subvectors
     * 
     *
     */
    std::vector<uint8_t> encode_vector(const std::vector<float>& residual) const {
        size_t subdim = vec_dim / M;
        std::vector<uint8_t> code(M);
        for (size_t m = 0; m < M; ++m) {
            // Find closest centroid in codebook m for subvector m
            double best = std::numeric_limits<double>::max();
            uint8_t best_idx = 0;
            for (size_t k = 0; k < Ks; ++k) {
                // Compute distance to centroid k in subspace m
                double d = 0.0;
                for (size_t j = 0; j < subdim; ++j) {
                    double diff = residual[m * subdim + j] - pq_codebooks[m][k][j];
                    d += diff * diff;
                }
                if (d < best) {
                    best = d;
                    best_idx = static_cast<uint8_t>(k);
                }
            }
            code[m] = best_idx;  // Store the index of the closest centroid
        }
        return code;
    }
};

#endif
