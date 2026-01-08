#pragma once 
#ifndef HYPERCUBE_HPP
#define HYPERCUBE_HPP 
#include <vector>
#include <queue>
#include <random>
#include <cstdint>
#include <iostream>
#include <algorithm>
#include <unordered_map>
#include <numeric>
#include <cmath>

#include "neighbor.hpp"
#include "metrics.hpp"
#include "LSH.hpp"




template <typename NumType>
class HypercubeProjection{ 
    HashFunction hf; 
    int t = 0; //default median 
    
    /*
        Note that if the vector is even sized it will return the wrong mathematically median 
        but because we expect the sample to be big enough we will get the correct pretty much every 
        time. 
    */
    int median(std::vector<int>& data) {
        size_t k = data.size() / 2;
        std::nth_element(data.begin(), data.begin() + k, data.end());
        return data[k];
    }

    public: 
        /*
            Constructor of the f function that maps a point to {0,1}
            It requires the points ,the dimension of a point and a float w window for the E2LSH 
            function.
        */
        HypercubeProjection(size_t dim ,float w_ ,int seed ,const std::vector<std::vector<NumType>>& points) :
            hf(dim ,seed , w_) {
                std::vector<int> values; 
                for (auto& point : points ){ 
                    int val = hf.calculate(point); 
                    values.push_back(val); 
                }
                t = median(values); 
            }
        

        int calculate(const std::vector<NumType>& point) const{ 
          return (hf.calculate(point) > t) ? 1 : 0; 
        }
};

template <typename NumType>
class HyperCube{ 
    std::vector<HypercubeProjection<NumType>> hp_table;
    std::unordered_map<uint64_t, std::vector<std::pair<std::vector<NumType>, size_t>>> bit_table; 
    float w; 
    size_t k_proj;    
    size_t vec_dim;  
    int seed; 

    uint64_t find_slot(const std::vector<NumType>& p) const{ 
        uint64_t mask = 0; 
        for (size_t i = 0; i < hp_table.size(); ++i) {
            mask = (mask << 1) | static_cast<std::uint64_t>(hp_table[i].calculate(p));
        }
        return mask; 
    }

    uint64_t next_bit(uint64_t v) const {
        uint64_t t = v | (v - 1);
        return (t + 1) | (((~t & -~t) - 1) >> (__builtin_ctz(v) + 1));  
    }

    void insertPoint(const std::vector<NumType>& point, size_t idx) {
        std::uint64_t slot = find_slot(point);
        bit_table[slot].emplace_back(point, idx); // store vector + index
    }

public:
    HyperCube(const std::vector<std::vector<NumType>>& points, size_t k_proj_, float w_, size_t vec_dim_, int seed_)
        : k_proj(k_proj_), w(w_), vec_dim(vec_dim_), seed(seed_) {

        hp_table.reserve(k_proj_); 
        for (size_t i = 0; i < k_proj_; ++i)
            hp_table.emplace_back(vec_dim, w, seed + static_cast<int>(i), points);
        
        // insert points with their original index
        for (size_t i = 0; i < points.size(); ++i) {
            insertPoint(points[i], i);   //save also the idx of the dataset
        }
    }

    std::vector<Neighbor> returnANN(const std::vector<NumType>& p, int M, int k, int probe = 3,
                                    bool range_bool = false, float range = 0.0) const {

        std::priority_queue<Neighbor> topKNeighbors; 
        uint64_t base_slot = find_slot(p);
        int hammingDist = 0; 
        int exploredCount = 0;        
        int exploredProbesCount = 0;        
        uint64_t border = (1ULL << k_proj); 
        bool explorationLimitReached = false; 
        bool probeLimitReached = false; 

        while (hammingDist < k_proj) {  
            if (hammingDist == 0){ 
                if (bit_table.find(base_slot) != bit_table.end()){
                    for (const auto& entry : bit_table.at(base_slot)){ 
                        double dist = euclidean_distance(p, entry.first);
                        if (!range_bool || dist < range) {          
                            topKNeighbors.emplace(entry.second, dist); // use index
                        }
                        if (++exploredCount > M) { explorationLimitReached = true; break; }
                    }
                }
                while (topKNeighbors.size() > static_cast<size_t>(k) && !range_bool) topKNeighbors.pop();
                if (explorationLimitReached) break;
                hammingDist++; 
                exploredProbesCount++; 
                if (exploredProbesCount == probe){ probeLimitReached = true; break; }
                continue; 
            }

            uint64_t currentSlot; 
            uint64_t hammingMask = 0; 
            for (int i = 0; i < hammingDist; ++i) { 
                hammingMask = (hammingMask << 1) | 1; 
            }

            while (hammingMask <= border) { 
                currentSlot = base_slot ^ hammingMask; 
                if (bit_table.find(currentSlot) != bit_table.end()) { 
                    for (const auto& entry : bit_table.at(currentSlot)){ 
                        double dist = euclidean_distance(p, entry.first);
                        if (!range_bool || dist < range) {
                            topKNeighbors.emplace(entry.second, dist); // use index
                        }
                        if (++exploredCount > M) { explorationLimitReached = true; break; }
                    }
                }
                while (topKNeighbors.size() > static_cast<size_t>(k) && !range_bool) topKNeighbors.pop();
                hammingMask = next_bit(hammingMask); 
                if (explorationLimitReached) break;
                exploredProbesCount++;
                if (exploredProbesCount == probe){ probeLimitReached = true; break; }
            }
            if (explorationLimitReached || probeLimitReached) break;
            hammingDist++;
        }

        std::vector<Neighbor> neighbors;
        while (!topKNeighbors.empty()) {
            neighbors.push_back(topKNeighbors.top());
            topKNeighbors.pop();
        }
        std::reverse(neighbors.begin(), neighbors.end());
        return neighbors;
    }
};







#endif


