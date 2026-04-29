import numpy as np
from pyspark import SparkContext, SparkConf
import sys
import os
import time





def fair_fft_sequential(Ka, Kb, points):
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

    idx = int(np.random.uniform(n))
    centers.append(idx)
    is_center[idx] = True

    if labels[idx] == 'A':
        remKa -= 1
    else:
        remKb -= 1

    min_distances = np.sum((coords - coords[idx]) ** 2, axis=1)#fast with np and no need to sqrt since we only compare distances
    #using total num (ka + kb) so we always return the desired num of centers even if for one class we do not have enough
    while len(centers) < target_total and len(centers) < n:
        
        #mask already selected centers with neg distances 
        valid_distances = np.where(is_center, -1.0, min_distances)
        chosen_idx = -1
        
        if remKa > 0 and remKb > 0:
            chosen_idx = np.argmax(valid_distances)
            
        elif remKa > 0:
            mask_A = (labels == 'A') & (~is_center)
            dists_A = np.where(mask_A, min_distances, -1.0)
            chosen_idx = np.argmax(dists_A)
            
            if dists_A[chosen_idx] == -1.0:
                chosen_idx = np.argmax(valid_distances)
                
        elif remKb > 0:
            mask_B = (labels != 'A') & (~is_center)
            dists_B = np.where(mask_B, min_distances, -1.0)
            chosen_idx = np.argmax(dists_B)
            

            if dists_B[chosen_idx] == -1.0:
                chosen_idx = np.argmax(valid_distances)
        
        label = labels[chosen_idx]
        if label == 'A':
            remKa -= 1
        else:
            remKb -= 1
            
        centers.append(chosen_idx)
        is_center[chosen_idx] = True
        
        new_center = coords[chosen_idx]
        dists_to_new = np.sum((coords - new_center) ** 2, axis=1)
        min_distances = np.minimum(min_distances, dists_to_new)

    return [points[i] for i in centers]



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
    # R 1: fairfft on local partitions to get coresets
    local_coresets_rdd = rdd.mapPartitions(
        lambda partition_iterator: fair_fft_sequential(Ka, Kb, list(partition_iterator))
    )

    # R 2: collect all local coresets to the driver
    global_coreset = local_coresets_rdd.collect()
    
    final_centers = fair_fft_sequential(Ka, Kb, global_coreset)

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