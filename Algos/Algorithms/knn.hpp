#pragma once
#ifndef KNN_HPP
#define KNN_HPP

#include <vector>
#include <queue>
#include <random>
#include <cstdint>
#include <iostream>
#include "metrics.hpp"
#include "neighbor.hpp"

template <typename NumType>
std::vector<Neighbor> kNN(const std::vector<std::vector<NumType>> &points ,const std::vector<NumType> &p , int k){ 
    std::priority_queue<Neighbor>  topK;
    size_t idx = 0; 
    for (auto&  q : points){ 
       double dist_qp =  euclidean_distance(q ,p); 
       Neighbor pq(static_cast<uint64_t>(idx), dist_qp); 
       topK.push(pq); 
       if (topK.size() > static_cast<size_t>(k)) { 
        topK.pop() ; 
       }
       idx++; 
    }
    
    std::vector<Neighbor> neighbors;
    while (!topK.empty()) {
        neighbors.push_back(topK.top());
        topK.pop();
    }
    std::reverse(neighbors.begin(), neighbors.end());
    return neighbors; 

}




#endif