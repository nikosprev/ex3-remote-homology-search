#pragma once 
#include <vector>
#include <cstdint>

//Overload Euclidean distance to work for both data 
double euclidean_distance(const std::vector<float>  &x  ,const std::vector<float> &y); 
double euclidean_distance(const std::vector<uint8_t>& x, const std::vector<uint8_t>& y);

//Find the norm of a vector
double norm(const std::vector<float> &x); 

//Normalize a vector 
void normalize(std::vector<float>& v); 
