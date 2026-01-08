#include <iostream>  
#include <cmath> 
#include "metrics.hpp"


double euclidean_distance(const std::vector<float>  &x  ,const std::vector<float> &y){ 
    if (x.size() != y.size() ){ 
        std::cerr << "Vectors have different size!" << std::endl ;
        exit(1); 
    }
    double dist = 0.0 ; 
    for(int i = 0 ; i < x.size() ; i ++ ){ 
        dist += (x[i] - y[i])*(x[i] - y[i]); 
    }
    return sqrt(dist); 
}

double euclidean_distance(const std::vector<uint8_t>& x, const std::vector<uint8_t>& y) {
    double sum = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        double diff = static_cast<double>(x[i]) - static_cast<double>(y[i]);
        sum += diff * diff;
    }
    return std::sqrt(sum);
}


double norm(const std::vector<float> &x){ 
    double norm = 0; 
    for (size_t i = 0 ; i < x.size(); ++i){ 
        norm += x[i]*x[i]; 
    }
    return sqrt(norm); 
}


void normalize(std::vector<float>& v) {   
    double norm_ = norm(v);
    if (norm_ > 0.0f)
        for (float& x : v)  
            x /= norm_;
}