

import random
import sys
from pyspark import SparkContext, SparkConf
from pyspark.streaming import StreamingContext
from pyspark import StorageLevel
import threading
import math


THRESHOLD = -1
C = 55
P = 8191


def getTrueFreqItems():
    global histogram
    return [key for key in histogram if histogram[key] >= phi*THRESHOLD]

def stickySampling(element, phi, epsilon, delta):
    global S, r, p
    if(element in S):
        S[element] += 1
    else:
        if random.random() <= p:
            S[element] = 1


def getStickyFreqItems(phi, epsilon):
    global S
    return [key for key in S if S[key] >= (phi-epsilon)*THRESHOLD]

def HashCountFunc(x, a, b):
    return ((a*x + b) % P) % C
    

def countMinSketch(element, phi):
    global CMS
    element = int(element)
    current_min = float('inf')
    for i in range(len(CMS)):
        a, b = hash_params[i]
        hash = HashCountFunc(element, a, b)
        CMS[i][hash] += 1

        if CMS[i][hash] < current_min:
            current_min = CMS[i][hash]
    if current_min >= phi*THRESHOLD:
        F_CMS.add(element)

    


def process_batch(time, batch):
    # We are working on the batch at time `time`.
    global streamLength, histogram
    batch_size = batch.count()
    # If we already have enough points (> THRESHOLD), skip this batch.
    if streamLength[0]>=THRESHOLD:
        return
    
    if(streamLength[0] + batch_size > THRESHOLD):
        batch_size = THRESHOLD - streamLength[0]
    
    batch_items = batch.take(batch_size)
    streamLength[0] += batch_size
    
    # Update the streaming state
    for item in batch_items:
        histogram[item] = histogram.get(item, 0) + 1
        stickySampling(item, phi, epsilon, delta)
        countMinSketch(item, phi)

            
    # If we wanted, here we could run some additional code on the global histogram
    if batch_size > 0:
        print("Batch size at time [{0}] is: {1}".format(time, batch_size))

    if streamLength[0] >= THRESHOLD:
        stopping_condition.set()







if __name__ == '__main__':
    assert len(sys.argv) == 8, "USAGE: n, phi, epsilon, delta, d, w, portExp"
    n = int(sys.argv[1])
    THRESHOLD = n
    phi = float(sys.argv[2])
    epsilon = float(sys.argv[3])
    delta = float(sys.argv[4])
    d = int(sys.argv[5])
    w = int(sys.argv[6])
    C = w
    
    portExp = int(sys.argv[7])



    conf = SparkConf().setMaster("local[*]").setAppName("TrueFrequentItems")
    sc = SparkContext(conf=conf)
    ssc = StreamingContext(sc, 0.1)  # Batch duration of 0.1 sec = 100 ms
    ssc.sparkContext.setLogLevel("ERROR")
    
    stopping_condition = threading.Event()

    streamLength = [0]
    histogram = {} # Hash Table for the distinct elements
    S = {} #hash table for  sticky sampling
    r = math.log(1.0 / (delta * phi)) / epsilon
    p = r/THRESHOLD
    CMS = [[0 for _ in range(w)] for _ in range(d)] # count min sketch matrix
    F_CMS = set()#set of freq items for countmin sketch
    hash_params = [(random.randint(1, P-1), random.randint(0, P-1)) for _ in range(d)]

    # CODE TO PROCESS AN UNBOUNDED STREAM OF DATA IN BATCHES
    stream = ssc.socketTextStream("algo.dei.unipd.it", portExp, StorageLevel.MEMORY_AND_DISK)
    # For each batch, to the following.
    # BEWARE: the `foreachRDD` method has "at least once semantics", meaning
    # that the same data might be processed multiple times in case of failure.
    stream.foreachRDD(lambda time, batch: process_batch(time, batch))
    
    # MANAGING STREAMING SPARK CONTEXT
    print("Starting streaming engine")
    ssc.start()
    print("Waiting for shutdown condition")
    stopping_condition.wait()
    print("Stopping the streaming engine")
    
    # The following command stops the execution of the stream. The first boolean, if true, also
    # stops the SparkContext, while the second boolean, if true, stops gracefully by waiting for
    # the processing of all received data to be completed. You might get some error messages when the
    # program ends, but they will not affect the correctness.
    
    ssc.stop(False, False)
    print("Streaming engine stopped")

    # COMPUTE AND PRINT FINAL STATISTICS
    true_freq_items = getTrueFreqItems()
    sticky_freq_items = getStickyFreqItems(phi, epsilon)
    print("INPUT PARAMETERS:")
    print(f"n: {n}, phi: {phi}, epsilon: {epsilon}, delta: {delta}, d: {d}, w: {w}, portExp: {portExp}")

    print("\nTRUE FREQUENT ITEMS:")
    # Cast to integer for sorting, lookup true frequency in histogram using the string key
    for item in sorted(int(x) for x in true_freq_items):
        print(f"{item} {histogram[str(item)]}")

    print("\nF_SS:")
    for item in sorted(int(x) for x in sticky_freq_items):
        print(f"{item} {histogram[str(item)]}")

    print("\nF_CM:")
    # F_CMS already contains integers due to your cast inside countMinSketch
    for item in sorted(int(x) for x in F_CMS):
        print(f"{item} {histogram[str(item)]}")