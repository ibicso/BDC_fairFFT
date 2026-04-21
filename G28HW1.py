import numpy as np
from pyspark import SparkContext, SparkConf
import sys
import os
import time





def fair_fft_sequential(Ka, Kb, points):
    #if partition is empty, return empty list
    if not points:
        return []
    
    coords = np.array([[float(val) for val in p[:-1]] for p in points])
    labels = np.array([p[-1].strip() for p in points])
    
    n = len(points)
    remKa = Ka
    remKb = Kb
    target_total = Ka + Kb
    
    centers = []
    is_center = np.zeros(n, dtype=bool) 
    min_distances = np.full(n, np.inf)

    idx = int(np.random.uniform(n))
    centers.append(idx)
    is_center[idx] = True

    if labels[idx] == 'A':
        remKa -= 1
    else:
        remKb -= 1

    min_distances = np.sum((coords - coords[idx]) ** 2, axis=1)#fast np implementation of euclidean distance without sqrt

    while len(centers) < target_total and len(centers) < n:
        sorted_indices = np.argsort(min_distances)[::-1]#slow part, maybe wee can do something just getting the max without sorting the entire array
        chosen_idx = None #used to know when to fill up in case we don't have enough points of one class
        
        if remKa > 0 or remKb > 0:
            for candidate_idx in sorted_indices:
                if is_center[candidate_idx]: 
                    continue 
                
                label = labels[candidate_idx]
                if (label == 'A' and remKa > 0) or (label != 'A' and remKb > 0):
                    chosen_idx = candidate_idx
                    if label == 'A':
                        remKa -= 1
                    else:
                        remKb -= 1
                    break
        
        # adding extra points if we didn't have enough points of one class
        if chosen_idx is None:
            for candidate_idx in sorted_indices:
                if not is_center[candidate_idx]:
                    chosen_idx = candidate_idx
                    break
                    
        # Lock in the choice
        centers.append(chosen_idx)
        is_center[chosen_idx] = True
        
        
        new_center = coords[chosen_idx]
        #fast np implementation instead of loop
        dists_to_new = np.sum((coords - new_center) ** 2, axis=1)#fast np implementation of euclidean distance without sqrt
        min_distances = np.minimum(min_distances, dists_to_new)

    centers_tuples = [points[i] for i in centers]
    return centers_tuples



def parse_line(line):
    parts = line.strip().split(",")
    # Convert all features to float, keep the label at the end as a string
    return tuple([float(x) for x in parts[:-1]] + [parts[-1].strip()])


def compute_objective(rdd, centers):
    center_coords = np.array([[float(val) for val in p[:-1]] for p in centers])
    
    def min_dist_to_centers(point):
        p_coords = np.array([float(val) for val in point[:-1]])
        distances = np.sqrt(np.sum((center_coords - p_coords) ** 2, axis=1))
        return np.min(distances)
        
    return rdd.map(min_dist_to_centers).max()

def map_reduce_fair_fft(rdd, Ka, Kb, L):

    initial_partition_sizes = rdd.mapPartitions(lambda it: [sum(1 for _ in it)]).collect()
    print("\n" + "="*50)
    print(f"[ROUND 1 INPUT] Points per partition: {initial_partition_sizes}")
    print(f"[ROUND 1 INPUT] Total points distributed: {sum(initial_partition_sizes)}")
    print("="*50 + "\n")

    #ROUND 1
    # mapPartitions hands an iterator of the partition's data directly to your function.
    # We cast that iterator to a list, extract the coreset, and return it.
    # Spark flattens the returned lists back into a unified RDD of tuples.
    local_coresets_rdd = rdd.mapPartitions(
        lambda partition_iterator: fair_fft_sequential(Ka, Kb, list(partition_iterator))
    )

    coreset_partition_sizes = local_coresets_rdd.mapPartitions(lambda it: [sum(1 for _ in it)]).collect()
    print("\n" + "="*50)
    print(f"[ROUND 1 OUTPUT] Coreset points returned per partition: {coreset_partition_sizes}")
    print(f"[ROUND 1 OUTPUT] Total points entering Reducer: {sum(coreset_partition_sizes)}")
    print("="*50 + "\n")
    #------------
    #ROUND 2
    print("[ROUND 2] Starting global aggregation on single node...")
    # coalesce(1) forces all local coresets onto a single worker node.
    # glom() packages them into a single list.
    # map() runs the sequential algorithm one final time.
    # collect()[0] pulls the final answer out of the RDD and back to the driver node.
    final_centers = local_coresets_rdd.coalesce(1).glom().map(
            lambda global_list: fair_fft_sequential(Ka, Kb, global_list)
        ).collect()[0]
    
    print("\n" + "="*50)
    print(f"[FINAL OUTPUT] Extracted exactly {len(final_centers)} global centers.")
    print("="*50 + "\n")

    return final_centers


def main():
    # CHECKING NUMBER OF CMD LINE PARAMTERS
    assert len(sys.argv) == 5, "Usage: python G28HW1.py <file_name> <Ka> <Kb> <L>"
    # SPARK SETUP
    conf = SparkConf().setAppName('FairFFT')
    # Suppress excessive Spark logging so your print statements are actually readable
    conf.set("spark.logLevel", "ERROR") 
    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")

    # INPUT READING
    data_path = sys.argv[1]
    assert os.path.isfile(data_path), "File or folder not found"

    Ka = sys.argv[2]
    assert Ka.isdigit(), "Ka must be an integer"
    Ka = int(Ka)

    Kb = sys.argv[3]
    assert Kb.isdigit(), "Kb must be an integer"
    Kb = int(Kb)

    L = sys.argv[4]
    assert L.isdigit(), "L must be an integer"
    L = int(L)
    
    # Read the data
    raw_rdd = sc.textFile(data_path)
    rdd = raw_rdd.map(parse_line).repartition(L)
    rdd.cache()
    # dataset stat
    N = rdd.count()
    NA = rdd.filter(lambda p: p[-1] == 'A').count()
    NB = N - NA

    start_time = time.time()



    final_centers = map_reduce_fair_fft(rdd, Ka, Kb, L)

    end_time = time.time()
    running_time_ms = int((end_time - start_time) * 1000)
    objective = compute_objective(rdd, final_centers)
    
    
    print(f"File path = {data_path}, KA = {Ka}, KB = {Kb}, L = {L}")
    print(f"N = {N}, NA = {NA}, NB = {NB}")
    
    for center in final_centers:
        coords = [float(val) for val in center[:-1]]
        label = center[-1]
        print(f"Center = {coords} Label = {label}")
        
    print(f"Objective function = {objective}")
    print(f"Running time of MRFairFFT = {running_time_ms} ms")


if __name__ == "__main__":
    main()