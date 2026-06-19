import subprocess
import re
import time
import sys
import os

# --- CONFIGURATION ---
SPARK_SCRIPT = "G28HW2.py"
RUNS_PER_TEST = 3

# Base parameters
N = 1000000
PHI = 0.07
DELTA = 0.05
PORT = 8888

# Table 1: Sticky Sampling (varying epsilon)
SS_TESTS = [
    {"eps": 0.01, "d": 5, "w": 15},
    {"eps": 0.02, "d": 5, "w": 15},
    {"eps": 0.04, "d": 5, "w": 15}
]

# Table 2: Count-Min Sketch (varying w)
CM_TESTS = [
    {"eps": 0.04, "d": 5, "w": 15},
    {"eps": 0.04, "d": 5, "w": 30},
    {"eps": 0.04, "d": 5, "w": 60}
]

def run_spark_job(n, phi, epsilon, delta, d, w, port):
    """Executes the PySpark job and captures stdout."""
    cmd = [
        "spark-submit", 
        SPARK_SCRIPT, 
        str(n), str(phi), str(epsilon), str(delta), str(d), str(w), str(port)
    ]
    print(f"    Running: {' '.join(cmd)}")
    
    process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return process.stdout

def parse_output(output_text, n, phi, epsilon, cm_eps=0.04):
    """Parses the PySpark output to count items matching the intervals."""
    freq_thresh = n * phi
    ss_almost_thresh = n * (phi - epsilon)
    cm_almost_thresh = n * (phi - cm_eps)

    stats = {
        "ss": {"freq": 0, "almost": 0, "rare": 0, "dict_size": 0},
        "cm": {"freq": 0, "almost": 0, "rare": 0, "total": 0}
    }

    current_section = None

    for line in output_text.split('\n'):
        line = line.strip()
        
        if line == "STICKY SAMPLING":
            current_section = "SS"
            continue
        elif line == "COUNT-MIN SKETCH":
            current_section = "CM"
            continue
            
        if line.startswith("Size of dictionary ="):
            stats["ss"]["dict_size"] = int(line.split("=")[1].strip())
        elif line.startswith("Size of F_CM ="):
            stats["cm"]["total"] = int(line.split("=")[1].strip())
            
        elif line.startswith("Item ="):
            match = re.search(r"True Freq = (\d+)", line)
            if match:
                true_freq = int(match.group(1))
                
                if current_section == "SS":
                    if true_freq >= freq_thresh:
                        stats["ss"]["freq"] += 1
                    elif true_freq >= ss_almost_thresh:
                        stats["ss"]["almost"] += 1
                    else:
                        stats["ss"]["rare"] += 1
                        
                elif current_section == "CM":
                    if true_freq >= freq_thresh:
                        stats["cm"]["freq"] += 1
                    elif true_freq >= cm_almost_thresh:
                        stats["cm"]["almost"] += 1
                    else:
                        stats["cm"]["rare"] += 1

    return stats

def run_experiment_batch(tests, target_algo, log_file):
    results_avg = []
    
    for test in tests:
        param_val = test['eps'] if target_algo == "SS" else test['w']
        print(f"\n[+] Starting configuration: eps={test['eps']}, d={test['d']}, w={test['w']}")
        
        agg_freq, agg_almost, agg_rare, agg_extra = 0, 0, 0, 0
        
        for run in range(1, RUNS_PER_TEST + 1):
            print(f"  -> Run {run}/{RUNS_PER_TEST}")
            output = run_spark_job(N, PHI, test['eps'], DELTA, test['d'], test['w'], PORT)
            
            # --- DUMP RAW STDOUT TO TEXT FILE ---
            with open(log_file, "a") as f:
                f.write(f"\n{'='*60}\n")
                f.write(f"ALGORITHM: {target_algo} | PARAM: {param_val} | RUN: {run}\n")
                f.write(f"COMMAND ARGS: eps={test['eps']}, w={test['w']}, d={test['d']}\n")
                f.write(f"{'='*60}\n")
                f.write(output)
                f.write("\n")

            # Parse and aggregate
            parsed = parse_output(output, N, PHI, test['eps'])
            
            if target_algo == "SS":
                agg_freq += parsed["ss"]["freq"]
                agg_almost += parsed["ss"]["almost"]
                agg_rare += parsed["ss"]["rare"]
                agg_extra += parsed["ss"]["dict_size"]
            else:
                agg_freq += parsed["cm"]["freq"]
                agg_almost += parsed["cm"]["almost"]
                agg_rare += parsed["cm"]["rare"]
                agg_extra += parsed["cm"]["total"]
            
            time.sleep(2) # Give sockets time to unbind
            
        # Save average data
        results_avg.append({
            "param": param_val,
            "freq": agg_freq / RUNS_PER_TEST,
            "almost": agg_almost / RUNS_PER_TEST,
            "rare": agg_rare / RUNS_PER_TEST,
            "extra": agg_extra / RUNS_PER_TEST
        })
        
    return results_avg

if __name__ == "__main__":
    
    RAW_LOG_FILE = "HW2_Full_Console_Logs.txt"
    
    # Clear the log file if it exists from a previous run
    if os.path.exists(RAW_LOG_FILE):
        os.remove(RAW_LOG_FILE)

    print("========================================")
    print("STARTING TABLE 1: STICKY SAMPLING TESTS")
    print("========================================")
    ss_avg = run_experiment_batch(SS_TESTS, "SS", RAW_LOG_FILE)

    print("\n========================================")
    print("STARTING TABLE 2: COUNT-MIN SKETCH TESTS")
    print("========================================")
    cm_avg = run_experiment_batch(CM_TESTS, "CM", RAW_LOG_FILE)

    # Write AVERAGES to CSV
    with open("HW2_Experiment_Results.csv", "w") as f:
        f.write("STICKY SAMPLING: n=1000000, phi=0.07, delta=0.05, port=8888\n")
        f.write("epsilon,Frequent returned,Almost frequent returned,Rare returned,Dict Size\n")
        for res in ss_avg:
            f.write(f"{res['param']},{res['freq']:.2f},{res['almost']:.2f},{res['rare']:.2f},{res['extra']:.2f}\n")
        
        f.write("\nCOUNT MIN: n=1000000, phi=0.07, d=5, port=8888\n")
        f.write("w,Frequent returned,Almost frequent (eps=0.04),Rare (eps=0.04),Total items returned\n")
        for res in cm_avg:
            f.write(f"{res['param']},{res['freq']:.2f},{res['almost']:.2f},{res['rare']:.2f},{res['extra']:.2f}\n")

    print("\n[✔] All experiments completed.")
    print(" -> Averages saved to 'HW2_Experiment_Results.csv'")
    print(f" -> Raw terminal outputs dumped to '{RAW_LOG_FILE}'")