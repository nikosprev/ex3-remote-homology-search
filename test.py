import random
import ann_algos
import math

# ---------------------------
# Helper functions
# ---------------------------
def generate_float_vectors(n, dim, value=None):
    if value is None:
        return [[random.random() for _ in range(dim)] for _ in range(n)]
    else:
        return [[value for _ in range(dim)] for _ in range(n)]


def generate_uint8_vectors(n, dim, value=None):
    if value is None:
        return [[random.randint(0, 255) for _ in range(dim)] for _ in range(n)]
    else:
        return [[value for _ in range(dim)] for _ in range(n)]


def euclidean_distance(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


# ---------------------------
# kNN Tests
# ---------------------------
def test_knn_correctness():
    print("=== Testing Exact kNN correctness ===")

    # Simple dataset: 3 points on 1D line
    points = [[0], [5], [10]]
    query = [2]
    
    res = ann_algos.kNN_float(points, query, 2)

    # Expected closest points: index 0 (distance 2), index 1 (distance 3)
    expected = [(0, 2.0), (1, 3.0)]
    for r, (idx, dist) in zip(res, expected):
        assert r.idx == idx, f"Expected idx {idx}, got {r.idx}"
        assert abs(r.distance - dist) < 1e-6, f"Expected dist {dist}, got {r.distance}"
    
    print("kNN correctness OK\n")


def test_knn_edge_cases():
    print("=== Testing kNN edge cases ===")

    # Empty dataset
    points = []
    query = [1]
    res = ann_algos.kNN_float(points, query, 1)
    assert res == [], "Expected empty result for empty dataset"

    # Query same as points
    points = [[1, 2], [3, 4]]
    query = [1, 2]
    res = ann_algos.kNN_float(points, query, 1)
    assert res[0].idx == 0, "Query identical to first point should return index 0"
    assert res[0].distance == 0.0, "Distance to identical point should be 0"

    print("kNN edge cases OK\n")


# ---------------------------
# IVFFlat Tests
# ---------------------------
def test_ivfflat_simple():
    print("=== Testing IVFFlat simple case ===")
    
    # 4 points on a 1D line
    points = [[0], [1], [2], [3]]
    ivf = ann_algos.IVFFlatFloat(num_clusters=2, vec_dim=1, kmeans_iters=5)
    ivf.add_vectors(points)
    ivf.train(seed=42)

    query = [1.5]
    res = ivf.query(query, k=2, nprobe=1)
    
    # Closest points should be 1 and 2
    expected_indices = {1, 2}
    result_indices = set(n.idx for n in res)
    assert result_indices == expected_indices, f"Expected indices {expected_indices}, got {result_indices}"
    
    print("IVFFlat simple OK\n")


# ---------------------------
# IVFPQ Tests
# ---------------------------
def test_ivfpq_simple():
    print("=== Testing IVFPQ simple case ===")
    
    points = [[0], [5], [10], [15]]
    pq = ann_algos.IVFPQFloat(num_coarse_clusters=2, vec_dim=1, M=1, Ks=256, kmeans_iters=5)
    pq.add_vectors(points)
    pq.train(seed=123)

    query = [6]
    res = pq.query(query, k=2, nprobe=1)
    
    # Expected closest points: 5 and 10
    expected_indices = {1, 2}
    result_indices = set(n.idx for n in res)
    assert result_indices == expected_indices, f"Expected {expected_indices}, got {result_indices}"

    print("IVFPQ simple OK\n")


# ---------------------------
# LSH Tests
# ---------------------------
def test_lsh_simple():
    print("=== Testing LSH simple case ===")
    
    points = [[0], [10], [20]]
    lsh = ann_algos.LSHFloat(hashTable_size=10, num_tables=2, HashFunction_size=1, w=5.0, vec_dim=1, seed=1)

    for p in points:
        lsh.insert_to_hashTables(p)

    query = [12]
    res = lsh.returnANN(query, k=2)
    
    # Closest points should be 10 and 20
    result_indices = set(n.idx for n in res)
    expected_indices = {1, 2}
    assert result_indices == expected_indices, f"Expected {expected_indices}, got {result_indices}"

    print("LSH simple OK\n")


# ---------------------------
# HyperCube Tests
# ---------------------------
def test_hypercube_simple():
    print("=== Testing HyperCube simple case ===")
    
    points = [[0], [5], [10], [15]]
    cube = ann_algos.HyperCubeFloat(points, k_proj=2, w=5.0, vec_dim=1, seed=42)

    query = [6]
    res = cube.returnANN(query, M=2, k=2, probe=2)
    
    # Closest points should be 5 and 10
    result_indices = set(n.idx for n in res)
    expected_indices = {1, 2}
    assert result_indices == expected_indices, f"Expected {expected_indices}, got {result_indices}"

    print("HyperCube simple OK\n")


# ---------------------------
# Uint8 Tests
# ---------------------------
def test_uint8_simple():
    print("=== Testing Uint8 simple case ===")
    
    points = [[0], [128], [255]]
    query = [130]
    res = ann_algos.kNN_uint8(points, query, k=2)

    # Closest points should be 128 and 255
    result_indices = set(n.idx for n in res)
    expected_indices = {1, 2}
    assert result_indices == expected_indices, f"Expected {expected_indices}, got {result_indices}"

    print("Uint8 simple OK\n")


# ---------------------------
# Run all tests
# ---------------------------
if __name__ == "__main__":
    random.seed(0)

    test_knn_correctness()
    test_knn_edge_cases()
    test_ivfflat_simple()
    test_ivfpq_simple()
    test_lsh_simple()
    test_hypercube_simple()
    test_uint8_simple()

    print("ALL TESTS PASSED")
