import glob
import re
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# --- GLOBAL CONFIGURATION ---
# Set to 'avg' to plot the average runtime, or 'min' to plot the minimum runtime
AGGREGATION_METHOD = "avg"


def parse_logs():
    """
    Parses all 'combined_logs_[version].txt' files.
    Returns a Pandas DataFrame containing all parsed data.
    """
    records = []
    file_pattern = "combined_logs_*.txt"
    log_files = glob.glob(file_pattern)

    if not log_files:
        print(f"No files matching '{file_pattern}' found in the current directory.")
        return pd.DataFrame()

    for file in log_files:
        version_match = re.search(r"combined_logs_(.*?)\.txt", file)
        version = version_match.group(1) if version_match else "unknown"

        current_workers = None
        current_dataset = None
        current_l = None
        current_ka = None
        current_kb = None
        current_obj = None

        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                # 1. Look for the injected filename header
                file_match = re.search(r"===\s*FILE:\s*.*?_(\d+)\.log\s*===", line)
                if file_match:
                    current_workers = int(file_match.group(1))
                    continue

                # 2. Look for the configuration line
                config_match = re.search(
                    r"File path = (.*?), KA = (\d+), KB = (\d+), L = (\d+)", line
                )
                if config_match:
                    raw_filename = config_match.group(1)
                    current_ka, current_kb, current_l = (
                        config_match.group(2),
                        config_match.group(3),
                        config_match.group(4),
                    )

                    ds_match = re.search(r"Artificial(.*?)\.csv", raw_filename)
                    current_dataset = ds_match.group(1) if ds_match else raw_filename
                    continue

                # 3. Look for the Objective Function line
                obj_match = re.search(r"Objective function = ([\d\.]+)", line)
                if obj_match:
                    current_obj = float(obj_match.group(1))
                    continue

                # 4. Look for the runtime line
                time_match = re.search(r"Running time of MRFairFFT = (\d+) ms", line)
                if (
                    time_match
                    and current_dataset
                    and current_workers is not None
                    and current_obj is not None
                ):
                    runtime = int(time_match.group(1))
                    records.append(
                        {
                            "Version": version,
                            "Dataset": current_dataset,
                            "Workers": current_workers,
                            "L": int(current_l),
                            "Runtime": runtime,
                            "kA": current_ka,
                            "kB": current_kb,
                            "Objective": current_obj,
                        }
                    )

                    # Reset state variables
                    current_dataset = None
                    current_obj = None

    return pd.DataFrame(records)


def print_objective_table(df):
    """
    Prints a clean table of the objective scores (Mean & Standard Deviation).
    """
    if df.empty:
        return

    print("\n" + "=" * 75)
    print(" 📊 OBJECTIVE FUNCTION SCORES (Mean & Standard Deviation)")
    print("=" * 75)

    # Group by config and calculate both mean and standard deviation (std)
    obj_table = (
        df.groupby(["Version", "Dataset", "kA", "kB", "L"])["Objective"]
        .agg(["mean", "std"])
        .reset_index()
    )

    # Fill NaN values in 'std' with 0.0 (this happens if there is only 1 run for a config)
    obj_table["std"] = obj_table["std"].fillna(0.0)

    # Rename for cleaner console output
    obj_table.rename(
        columns={"mean": "Avg Objective", "std": "Std Deviation"}, inplace=True
    )

    # Format the floats so the console output aligns nicely
    obj_table["Avg Objective"] = obj_table["Avg Objective"].apply(lambda x: f"{x:.4f}")
    obj_table["Std Deviation"] = obj_table["Std Deviation"].apply(lambda x: f"{x:.4f}")

    print(obj_table.to_string(index=False))
    print("=" * 75 + "\n")


def plot_results(df):
    if df.empty:
        print("No valid data found to plot.")
        return

    sns.set_theme(style="whitegrid")

    # 1. Aggregate duplicates for the plot (multiple runs of the same configuration)
    agg_cols = ["Version", "Dataset", "Workers", "L", "kA", "kB"]
    if AGGREGATION_METHOD == "avg":
        df_agg = df.groupby(agg_cols)["Runtime"].mean().reset_index()
        y_label = "Average Runtime (ms)"
    else:
        df_agg = df.groupby(agg_cols)["Runtime"].min().reset_index()
        y_label = "Minimum Runtime (ms)"

    # 2. Generate ONE plot window per version, subplots per dataset
    versions = df_agg["Version"].unique()

    for version in versions:
        v_data = df_agg[df_agg["Version"] == version]
        datasets = v_data["Dataset"].unique()
        num_datasets = len(datasets)

        # Create subplots based on number of datasets
        fig, axes = plt.subplots(
            1, num_datasets, figsize=(6 * num_datasets, 5), squeeze=False
        )
        fig.suptitle(
            f"Scalability - Version: {version} ({y_label})", fontsize=16, y=1.05
        )
        axes = axes.flatten()

        for ax, dataset in zip(axes, datasets):
            ds_data = v_data[v_data["Dataset"] == dataset]

            # Extract kA and kB for the chart description
            ka = ds_data["kA"].iloc[0]
            kb = ds_data["kB"].iloc[0]

            # Plot the line graph
            sns.lineplot(
                data=ds_data,
                x="Workers",
                y="Runtime",
                hue="L",
                marker="o",
                palette="tab10",
                ax=ax,
                linewidth=2,
                markersize=8,
            )

            # Formatting
            ax.set_title(f"{dataset}\n(kA={ka}, kB={kb})", fontsize=14)
            ax.set_ylabel(y_label, fontsize=12)
            ax.set_xlabel("Number of Workers", fontsize=12)

            # Force X-axis to only show the exact worker intervals
            ax.set_xticks(sorted(ds_data["Workers"].unique()))
            ax.legend(title="Partitions (L)")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    df = parse_logs()
    print_objective_table(df)
    plot_results(df)
