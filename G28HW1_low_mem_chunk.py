import math

import numpy as np
from pyspark import SparkContext, SparkConf
import sys
import os
import time


def fair_fft_sequential(Ka, Kb, points):
    iterator = iter(points)
    try:
        first_p = next(iterator)
    except StopIteration:
        return []

    # Dynamically determine dimensions from the first point
    D = len(first_p) - 1
    print(f"Dimensions = {D}")
    CHUNK_SIZE = 50000

    coords_chunks = []
    labels_chunks = []

    # Pre-allocate the first dense chunk
    curr_coords = np.empty((CHUNK_SIZE, D), dtype=np.float32)
    curr_labels = np.empty(CHUNK_SIZE, dtype=object)

    idx = 0
    curr_coords[idx] = first_p[:-1]
    curr_labels[idx] = first_p[-1].strip()
    idx += 1

    # Stream the iterator directly into pre-allocated memory
    for p in iterator:
        if idx == CHUNK_SIZE:
            # Chunk is full, save it and allocate a new one
            coords_chunks.append(curr_coords)
            labels_chunks.append(curr_labels)

            curr_coords = np.empty((CHUNK_SIZE, D), dtype=np.float32)
            curr_labels = np.empty(CHUNK_SIZE, dtype=object)
            idx = 0

        curr_coords[idx] = p[:-1]
        curr_labels[idx] = p[-1].strip()
        idx += 1

    # Save the final partially-filled chunk
    if idx > 0:
        coords_chunks.append(curr_coords[:idx])
        labels_chunks.append(curr_labels[:idx])

    # Combine chunks
    coords = np.vstack(coords_chunks)
    labels = np.concatenate(labels_chunks)

    # Free intermediate buffers immediately
    del coords_chunks, labels_chunks, curr_coords, curr_labels

    n = coords.shape[0]
    remKa = Ka
    remKb = Kb
    target_total = Ka + Kb

    centers = []
    is_center = np.zeros(n, dtype=bool)

    idx = int(np.random.uniform(n))
    centers.append(idx)
    is_center[idx] = True

    if labels[idx] == "A":
        remKa -= 1
    else:
        remKb -= 1

    # column-wise computation of distances to avoid intermediate numpy arrays
    min_distances = np.zeros(n, dtype=np.float32)
    for dim in range(coords.shape[1]):
        min_distances += (coords[:, dim] - coords[idx, dim]) ** 2

    # using total num (ka + kb) so we always return the desired num of centers even if for one class we do not have enough
    while len(centers) < target_total and len(centers) < n:

        # mask already selected centers with neg distances
        valid_distances = np.where(is_center, -1.0, min_distances)
        chosen_idx = -1

        if remKa > 0 and remKb > 0:
            chosen_idx = np.argmax(valid_distances)

        elif remKa > 0:
            mask_A = (labels == "A") & (~is_center)
            dists_A = np.where(mask_A, min_distances, -1.0)
            chosen_idx = np.argmax(dists_A)

            if dists_A[chosen_idx] == -1.0:
                chosen_idx = np.argmax(valid_distances)

        elif remKb > 0:
            mask_B = (labels != "A") & (~is_center)
            dists_B = np.where(mask_B, min_distances, -1.0)
            chosen_idx = np.argmax(dists_B)

            if dists_B[chosen_idx] == -1.0:
                chosen_idx = np.argmax(valid_distances)

        label = labels[chosen_idx]
        if label == "A":
            remKa -= 1
        else:
            remKb -= 1

        centers.append(chosen_idx)
        is_center[chosen_idx] = True

        new_center = coords[chosen_idx]

        # column-wise computation of distances to avoid intermediate numpy arrays
        dists_to_new = np.zeros(n, dtype=np.float32)
        for dim in range(coords.shape[1]):
            dists_to_new += (coords[:, dim] - new_center[dim]) ** 2

        min_distances = np.minimum(min_distances, dists_to_new)

    # convert to original tuple format
    return [tuple(coords[i].tolist()) + (labels[i],) for i in centers]


def parse_line(line):
    parts = line.strip().split(",")
    return tuple([float(x) for x in parts[:-1]] + [parts[-1].strip()])


def compute_objective(rdd, centers):
    center_coords = np.array([p[:-1] for p in centers], dtype=np.float32)

    def partition_objective(iterator):
        try:
            first_p = next(iterator)
        except StopIteration:
            yield 0.0
            return

        D = len(first_p) - 1
        CHUNK_SIZE = 50000
        curr_coords = np.empty((CHUNK_SIZE, D), dtype=np.float32)
        idx = 0
        curr_coords[idx] = first_p[:-1]
        idx += 1

        max_obj_in_partition = 0.0

        for p in iterator:
            if idx == CHUNK_SIZE:
                min_sq_dists = np.full(CHUNK_SIZE, np.inf, dtype=np.float32)
                
                for c in center_coords:
                    sq_dist = np.sum((curr_coords - c) ** 2, axis=1)
                    min_sq_dists = np.minimum(min_sq_dists, sq_dist)
                
                chunk_max_obj = np.max(np.sqrt(min_sq_dists))
                max_obj_in_partition = max(max_obj_in_partition, float(chunk_max_obj))

                idx = 0 

            curr_coords[idx] = p[:-1]
            idx += 1

        if idx > 0:
            final_chunk = curr_coords[:idx]
            min_sq_dists = np.full(idx, np.inf, dtype=np.float32)
            
            for c in center_coords:
                sq_dist = np.sum((final_chunk - c) ** 2, axis=1)
                min_sq_dists = np.minimum(min_sq_dists, sq_dist)
                
            chunk_max_obj = np.max(np.sqrt(min_sq_dists))
            max_obj_in_partition = max(max_obj_in_partition, float(chunk_max_obj))

        yield max_obj_in_partition

    return rdd.mapPartitions(partition_objective).max()


def map_reduce_fair_fft(rdd, Ka, Kb, L):
    # R 1: fairfft on local partitions to get coresets
    local_coresets_rdd = rdd.mapPartitions(
        lambda partition_iterator: fair_fft_sequential(Ka, Kb, partition_iterator)
    )

    # R 2: collect all local coresets to the driver
    global_coreset = local_coresets_rdd.collect()

    final_centers = fair_fft_sequential(Ka, Kb, global_coreset)

    return final_centers



def main():
    assert (
        len(sys.argv) >= 5 and len(sys.argv) <= 6
    ), "Usage: python G28HW1.py <file_name> <Ka> <Kb> <L>"
    conf = SparkConf().setAppName("FairFFT")
    conf.set("spark.logLevel", "ERROR")
    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")

    data_path = sys.argv[1]

    Ka = sys.argv[2]
    assert Ka.isdigit(), "Ka must be an integer"
    Ka = int(Ka)

    Kb = sys.argv[3]
    assert Kb.isdigit(), "Kb must be an integer"
    Kb = int(Kb)

    L = sys.argv[4]
    assert L.isdigit(), "L must be an integer"
    L = int(L)


    raw_rdd = sc.textFile(data_path)
    rdd = raw_rdd.map(parse_line).repartition(L)
    
    #removed to prevent memory issues with large datasets
    #rdd.cache() 
    


    N = rdd.count()
    NA = rdd.filter(lambda p: p[-1] == "A").count()
    NB = N - NA

    start_time = time.time()

 
    final_centers = map_reduce_fair_fft(rdd, Ka, Kb, L)

    end_time = time.time()
    running_time_ms = int((end_time - start_time) * 1000)
    objective = compute_objective(rdd, final_centers)

    file_name = os.path.basename(data_path)

    print(f"File path = {file_name}, KA = {Ka}, KB = {Kb}, L = {L}")
    print(f"N = {N}, NA = {NA}, NB = {NB}")

    for center in final_centers:
        coords_str = "[" + ",".join(str(float(val)) for val in center[:-1]) + "]"
        label = center[-1]
        print(f"Center = {coords_str} Label = {label}")

    print(f"Objective function = {objective}")
    print(f"Running time of MRFairFFT = {running_time_ms} ms")


if __name__ == "__main__":
    main()
