#pragma once
#ifndef NEIGHBOR_HPP
#define NEIGHBOR_HPP

#include <iostream>
#include <cstdint>

struct Neighbor {
    uint64_t idx;      // index of the point
    double distance;   // distance to query

    Neighbor(uint64_t idx_, double d)
        : idx(idx_), distance(d) {}

    bool operator<(const Neighbor& rhs) const {
        return distance < rhs.distance;
    }
};

// Print only the index and distance
inline std::ostream& operator<<(std::ostream& os, const Neighbor& n) {
    os << "Neighbor(idx=" << n.idx << ", distance=" << n.distance << ")\n";
    return os;
}

#endif
