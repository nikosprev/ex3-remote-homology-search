#pragma once
#ifndef IVF_FLAT_HPP
#define IVF_FLAT_HPP

#include <vector>
#include <queue>
#include <random>
#include <cstdint>
#include <iostream>
#include <algorithm>
#include <limits>
#include <numeric>

#include "neighbor.hpp"
#include "metrics.hpp"
#include "kmeans.hpp"


/**
 * IVF-Flat (Inverted File Index with Flat lists)
 * 
 * An approximate nearest neighbor search algorithm that combines:
 * 1. Coarse quantization: Uses k-means clustering to partition the dataset into num_clusters clusters
 * 2. Inverted file structure: Each cluster maintains a list of vectors assigned to it
 * 3. Exact search in selected lists: During query, only searches the nprobe closest clusters
 * 
 * Algorithm Overview:
 * - Training: Partition dataset using k-means, assign each vector to its nearest cluster centroid
 * - Query: Find nprobe closest centroids to query, search all vectors in those clusters exactly
 * 
 * Advantages:
 * - Fast query time by only searching a subset of clusters (nprobe << num_clusters)
 * - Exact distance computation within selected clusters (no approximation in distances)
 * - Good accuracy-speed tradeoff
 * 
 * Parameters:
 * - num_clusters: Number of coarse clusters (more = better accuracy but slower)
 * - nprobe: Number of clusters to search during query (more = better accuracy but slower)
 * 
 */
template <typename NumType>
class IVFFlat {
    // Coarse centroids (size = num_clusters). Each centroid is a vector of dimension dim.
    std::vector<std::vector<float>> centroids;

    // Inverted lists: for each centroid, store pairs of (originalIndex, vector).
    std::vector<std::vector<std::pair<size_t, std::vector<NumType>>>> inverted_lists;

    // Original vectors kept to allow optional access; not strictly necessary for search.
    std::vector<std::vector<NumType>> vectors;

    // Configuration
    size_t num_clusters;     // number of coarse clusters (lists)
    size_t vec_dim;          // dimensionality of vectors
    int kmeans_iters;        // iterations of k-means during training

public:
    IVFFlat(size_t num_clusters_, size_t vec_dim_, int kmeans_iters_ = 15)
        : num_clusters(num_clusters_), vec_dim(vec_dim_), kmeans_iters(kmeans_iters_) {
        if (num_clusters == 0 || vec_dim == 0) {
            std::cerr << "IVFFlat: num_clusters and vec_dim must be > 0" << std::endl;
            std::exit(EXIT_FAILURE);
        }
        centroids.resize(num_clusters, std::vector<float>(vec_dim, 0.0f));
        inverted_lists.resize(num_clusters);
    }

    // Insert a single vector prior to training. Training will cluster and assign later.
    void add_vector(const std::vector<NumType>& p) {
        if (p.size() != vec_dim) {
            std::cerr << "IVFFlat::add_vector: vector dim mismatch" << std::endl;
            std::exit(EXIT_FAILURE);
        }
        vectors.push_back(p);
    }

    // Add a batch of vectors.
    void add_vectors(const std::vector<std::vector<NumType>>& pts) {
        for (const auto& p : pts) add_vector(p);
    }

    /**
     * Train the IVFFlat index using k-means clustering.
     * 
     * Training process:
     * 1. Run k-means clustering on all input vectors to find num_clusters centroids
     * 2. Assign each vector to its nearest centroid
     * 3. Build inverted lists: for each cluster, store all vectors assigned to it
     * 
     * After training, each inverted list contains pairs of (original_index, vector)
     * to allow retrieval of the original vector and its index during query.
     * 
     */
    void train(int seed = 12345) {
        if (vectors.empty()) {
            std::cerr << "IVFFlat::train: no vectors added" << std::endl;
            std::exit(EXIT_FAILURE);
        }

        // Train via reusable k-means and get final assignment
        std::vector<size_t> final_assignment;
        kmeans::kmeans_train(vectors, num_clusters, vec_dim, kmeans_iters, seed, centroids, final_assignment);

        // Build inverted lists: assign each vector to its cluster's list
        // Store (original_index, vector) so we can return the index during query
        for (auto& lst : inverted_lists) lst.clear();
        for (size_t i = 0; i < vectors.size(); ++i) {
            size_t c = final_assignment[i];
            inverted_lists[c].emplace_back(i, vectors[i]);
        }
    }

    /**
     * Query for k approximate nearest neighbors.
     * 
     * Query process:
     * 1. Find nprobe closest centroids to the query vector
     * 2. Search all vectors in those nprobe clusters exactly (compute true Euclidean distance)
     * 3. Maintain a max-heap of size k to keep the k smallest distances
     * 4. Return neighbors sorted by distance (ascending)
     * 
     * Time complexity: O(num_clusters * vec_dim + nprobe * avg_cluster_size * vec_dim)
     * - Finding closest centroids: O(num_clusters * vec_dim)
     * - Searching selected lists: O(nprobe * avg_cluster_size * vec_dim)
     * 
    
     */
    std::vector<Neighbor> query(const std::vector<NumType>& p, int k, int nprobe = 1) const {
        if (p.size() != vec_dim) {
            std::cerr << "IVFFlat::query: vector dim mismatch" << std::endl;
            std::exit(EXIT_FAILURE);
        }
        if (centroids.empty() || inverted_lists.empty()) {
            std::cerr << "IVFFlat::query: index not trained" << std::endl;
            std::exit(EXIT_FAILURE);
        }
        nprobe = std::max(1, std::min(static_cast<int>(num_clusters), nprobe));

        // Step 1: Find nprobe closest centroids to query vector
        // Compute distance from query to each centroid and select the nprobe smallest
        std::vector<std::pair<double, size_t>> centroid_dists;
        centroid_dists.reserve(num_clusters);
        for (size_t c = 0; c < num_clusters; ++c) {
            double dist = dist_to_centroid(p, centroids[c]);
            centroid_dists.emplace_back(dist, c);
        }
        // Use nth_element to find nprobe smallest without full sort (O(n) vs O(n log n))
        std::nth_element(centroid_dists.begin(), centroid_dists.begin() + (nprobe - 1), centroid_dists.end());
        centroid_dists.resize(nprobe);

        // Step 2: Search selected clusters exactly
        // Use max-heap to maintain top-k (heap with largest on top for easy removal)
        auto cmp = [](const Neighbor& a, const Neighbor& b) { return a.distance < b.distance; };
        std::priority_queue<Neighbor, std::vector<Neighbor>, decltype(cmp)> topK(cmp);

        for (const auto& [_, cid] : centroid_dists) {
            // For each selected cluster, compute exact distance to all vectors in it
            for (const auto& [idx, vec] : inverted_lists[cid]) {
                double dist = euclidean_distance(p, vec);
                topK.emplace(idx, dist);  // Store (index, distance) for each neighbor
                if (static_cast<int>(topK.size()) > k) topK.pop();  // Keep only k smallest
            }
        }

        // Step 3: Extract and return results in ascending distance order
        // Priority queue extracts largest first, so we reverse to get ascending order
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
     * Useful when you want all points within a certain radius rather than a fixed number.
     * 
     *
     */
    std::vector<Neighbor> range_query(const std::vector<NumType>& p, double range, int nprobe = 1) const {
        if (p.size() != vec_dim) {
            std::cerr << "IVFFlat::range_query: vector dim mismatch" << std::endl;
            std::exit(EXIT_FAILURE);
        }
        nprobe = std::max(1, std::min(static_cast<int>(num_clusters), nprobe));

        // Find nprobe closest centroids (same as query method)
        std::vector<std::pair<double, size_t>> centroid_dists;
        centroid_dists.reserve(num_clusters);
        for (size_t c = 0; c < num_clusters; ++c) {
            double dist = dist_to_centroid(p, centroids[c]);
            centroid_dists.emplace_back(dist, c);
        }
        std::nth_element(centroid_dists.begin(), centroid_dists.begin() + (nprobe - 1), centroid_dists.end());
        centroid_dists.resize(nprobe);

        // Collect all vectors within range (no heap needed since we want all matches)
        std::vector<Neighbor> result;
        for (const auto& [_, cid] : centroid_dists) {
            for (const auto& [idx, vec] : inverted_lists[cid]) {
                double dist = euclidean_distance(p, vec);
                if (dist <= range) result.emplace_back(idx, dist);
            }
        }
        // Sort results by distance for consistent output
        std::sort(result.begin(), result.end(), [](const auto& a, const auto& b) { return a.distance < b.distance; });
        return result;
    }

    // Expose read-only accessors
    const std::vector<std::vector<float>>& get_centroids() const { return centroids; }
    const std::vector<std::vector<std::pair<size_t, std::vector<NumType>>>>& get_inverted_lists() const { return inverted_lists; }

private:
    /**
     * Compute Euclidean distance from a vector to a centroid.
     * 
     * Helper function for computing distances between vectors and cluster centroids.
     * Handles type conversion from NumType to float for centroid comparison.
     * 
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
};

#endif


