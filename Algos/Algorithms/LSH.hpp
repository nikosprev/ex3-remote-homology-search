#pragma once
#ifndef LSH_HPP
#define LSH_HPP

#include <vector>
#include <queue>
#include <random>
#include <cstdint>
#include <iostream>
#include <algorithm>
#include <unordered_set>
#include <numeric>
#include <cmath>

#include "neighbor.hpp"
#include "metrics.hpp"
// ----- Utility -----
/*
Create a random Vector Derived from A Normal Distribution
After normalize the vector  
*/
inline std::vector<float> GaussianProjection(size_t size, int* seed) {
    std::default_random_engine generator(*seed);
    std::normal_distribution<float> distribution(0.0f, 1.0f);
    std::vector<float> r(size);
    for (size_t i = 0; i < size; ++i)
        r[i] = distribution(generator);

    // Normalize
    //normalize(r); 

    return r;
}

/*
Mod function that returns always a positive -> built in returns also negatives 
*/
inline int mod(int64_t x, int64_t m) {
    int r = x % m;
    return (r < 0) ? r + m : r;
}

// ----- HashFunction -----
/*
Class of a HashFunction h_i that uses as hash one suitable for the euclidean space 
h_i(p) =  {\floor}{(p*v + t)/w} 
*/
class HashFunction {
    std::vector<float> w;//The random vector 
    float window; //The window 
    float t;
    size_t dim; // The dimension of the vectors 

public:
    HashFunction(size_t dim_, int seed, float  window_)
        : dim(dim_), window(window_) {
        w = GaussianProjection(dim_, &seed);
        std::mt19937 gen(seed);
        std::uniform_real_distribution<float> dist(0.0f, static_cast<float>(window_ ));
        t = dist(gen);
    }

    template <typename NumType>
    int calculate(const std::vector<NumType>& p) const {
        if (p.size() != dim) {
            std::cerr << "Vector size mismatch\n";
            std::exit(EXIT_FAILURE);
        }
        double result = std::inner_product(p.begin(), p.end(), w.begin(), 0.0);
        result = (result + t) / static_cast<double>(window);
        return static_cast<int>(std::floor(result));
    }
};

// ----- AmplifiedHashFunction -----
/*
An amplified HashFunction that works like this: 
It combines multiple HashFunctions .At last it returns the modulo of the result with a large number M
*/
class AmplifiedHashFunction {
    std::vector<HashFunction> hf_table;
    std::vector<int> r_table;

public:
    AmplifiedHashFunction(size_t size, size_t dim, int seed, float window) {
        hf_table.reserve(size);
        for (size_t i = 0; i < size; ++i)
            hf_table.emplace_back(dim, seed + static_cast<int>(i), window);

        r_table.resize(size);
        std::mt19937 gen(seed);
        std::uniform_int_distribution<int> dist(INT32_MIN, INT32_MAX);
        for (size_t i = 0; i < size; ++i)
            r_table[i] = dist(gen);
    }

    template <typename NumType>
    uint64_t calculate_ID(const std::vector<NumType>& p) const {
        const uint64_t M = (1ULL << 32) - 5;
        uint64_t ID = 0;
        for (size_t i = 0; i < hf_table.size(); ++i) {
            ID = mod(mod(static_cast<uint64_t>(hf_table[i].calculate(p)) * r_table[i], M) + ID, M);
        }
        return ID;
    }
};

// ----- LSH -----
/*
    The Class that finds the ANN neighbors of a point q .It 
    creates L hashtables ,with each one has an assigned amplified HashFunction.  
*/
template <typename NumType>
class LSH {
    std::vector<std::vector<std::vector<std::pair<uint64_t, size_t>>>> HashTables;
    std::vector<std::vector<NumType>> vectors;
    std::vector<AmplifiedHashFunction> IDs;

    size_t hashTable_size;
    int num_tables;
    int HashFunction_size;
    float w;
    size_t vec_dim;

public:
    LSH(size_t hashTable_size_, int num_tables_, int HashFunction_size_,
        float w_, size_t vec_dim_, int seed)
        : hashTable_size(hashTable_size_), num_tables(num_tables_),
          HashFunction_size(HashFunction_size_), w(w_), vec_dim(vec_dim_) {

        HashTables.resize(num_tables);
        IDs.reserve(num_tables);

        for (int i = 0; i < num_tables; ++i) {
            HashTables[i].resize(hashTable_size);
            IDs.emplace_back(HashFunction_size, vec_dim, seed + i * 101, w);
        }
    }

    void insert_to_hashTables(const std::vector<NumType>& p) {
        size_t idx = vectors.size();
        vectors.push_back(p);
        for (int t = 0; t < num_tables; ++t) {
            uint64_t id = IDs[t].calculate_ID(p);
            int slot = mod(static_cast<int>(id), hashTable_size);
            HashTables[t][slot].emplace_back(id, idx);
        }
    }

    std::vector<Neighbor> returnANN(const std::vector<NumType>& p, int k, 
                                             bool range_bool = false, float range = 0.0) const {
        std::priority_queue<Neighbor> topK;
        std::unordered_set<size_t> idx_seen;

        for (int t = 0; t < num_tables; ++t) {
            uint64_t id = IDs[t].calculate_ID(p);
            int slot = mod(static_cast<int>(id), hashTable_size);
            for (const auto& entry : HashTables[t][slot]) {
                if (entry.first == id) {
                    size_t idx = entry.second;
                    if (idx_seen.find(idx) == idx_seen.end()) {
                        idx_seen.insert(idx);
                        double dist = euclidean_distance(p, vectors[idx]);
                        if (!range_bool || dist < range) {
                            topK.emplace(idx, dist);
                        }
                    }
                }
            }
            while (topK.size() > static_cast<size_t>(k) && !range_bool) topK.pop();
        }
        std::vector<Neighbor> neighbors;
        while (!topK.empty()) {
            neighbors.push_back(topK.top());
            topK.pop();
        }
        std::reverse(neighbors.begin(), neighbors.end());
        return neighbors;
    }
};

#endif
