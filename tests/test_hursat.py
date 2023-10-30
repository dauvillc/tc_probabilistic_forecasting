"""
Test the hursat data preparation pipeline.
"""
import sys
sys.path.append("./")
import matplotlib.pyplot as plt
from data_processing.datasets import load_ibtracs_data, load_hursat_b1


if __name__ == "__main__":
    # Load the IBTrACS data
    ibtracs_dataset = load_ibtracs_data()
    # Load the HURSAT-B1 data
    found_storms, hursat_b1_dataset = load_hursat_b1(ibtracs_dataset, verbose=True)

    # Select eight random pairs (SID, ISO_TIME) from found_storms,
    # and plot the HURSAT-B1 data for each of them
    # Make sure there is enough space between the subplots
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for i, (sid, time) in enumerate(found_storms.sample(8, random_state=0).itertuples(index=False)):
        # Select the data for the given (SID, ISO_TIME) pair
        hursat_b1_data = hursat_b1_dataset.sel(sid_time=(sid, time))
        # Plot the data
        hursat_b1_data.plot(ax=axes[i // 4, i % 4])
        axes[i // 4, i % 4].set_title(f"{sid} {time}")
    plt.tight_layout()

    plt.savefig("figures/examples/hursat_b1.png")
