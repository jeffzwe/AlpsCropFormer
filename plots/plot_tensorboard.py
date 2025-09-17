import os
import glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import seaborn as sns

# Set seaborn style for publication-ready plots
sns.set_style("whitegrid")
sns.set_palette("viridis")

# Hardcode your tensorboard directory here
TENSORBOARD_DIRECTORY = "/Users/jeffreyzweidler/Desktop/Semester_Project/AlpsCropFormer/models/model_logs/Sentinel/TSViT_fold1.temp_subsample"  # <-- set this
# Only plot up to this step
MAX_STEPS = 75000

def plot_accuracy_from_tensorboard(directory_path, save_path=None):
    """
    Extract and plot accuracy data from TensorBoard files in folders starting with 'Accuracy_'.
    Saves to save_path (or CWD if None).
    """
    # Find all subfolders that start with 'Accuracy_'
    subfolders = sorted(
        f for f in os.listdir(directory_path)
        if os.path.isdir(os.path.join(directory_path, f)) and f.startswith('Accuracy_')
    )

    if not subfolders:
        print("No folders starting with 'Accuracy_' found!")
        return

    print(f"Found {len(subfolders)} accuracy experiment folders")

    # Create figure
    plt.figure(figsize=(12, 8))

    # Use currently active seaborn palette (viridis) for n lines
    colors = sns.color_palette(None, n_colors=len(subfolders))

    plotted_count = 0

    for i, subfolder in enumerate(subfolders):
        subfolder_path = os.path.join(directory_path, subfolder)

        # Find events files
        events_files = glob.glob(os.path.join(subfolder_path, "events.out.*"))
        if not events_files:
            print(f"No events file found in {subfolder}")
            continue

        events_file = events_files[0]

        try:
            # Read the events
            ea = EventAccumulator(events_file)
            ea.Reload()

            # Get scalar tags and find accuracy tag
            scalar_tags = ea.Tags().get('scalars', [])
            if not scalar_tags:
                print(f"No scalar data found in {subfolder}")
                continue

            accuracy_tags = [tag for tag in scalar_tags if 'accuracy' in tag.lower()]
            if not accuracy_tags:
                print(f"No accuracy data found in {subfolder}")
                continue

            # Extract accuracy data
            scalar_events = ea.Scalars(accuracy_tags[0])
            if not scalar_events:
                print(f"Empty accuracy series in {subfolder}")
                continue

            steps = [event.step for event in scalar_events]
            values = [event.value for event in scalar_events]

            # Keep only points up to MAX_STEPS
            filtered = [(s, v) for s, v in zip(steps, values) if s <= MAX_STEPS]
            if not filtered:
                continue
            steps, values = map(list, zip(*filtered))

            # Plot (no legend)
            sns.lineplot(x=steps, y=values, linewidth=2, color=colors[i], legend=False)
            plotted_count += 1

        except Exception as e:
            print(f"Error processing {subfolder}: {e}")
            continue

    if plotted_count == 0:
        print("No data could be plotted!")
        plt.close()
        return

    # Customize plot
    plt.xlabel('Training Steps')
    plt.ylabel('Accuracy')
    plt.title('Instable training using weighted cross entropy')
    sns.despine()
    plt.tight_layout()

    # Save plot
    if save_path is None:
        save_path = os.path.join(os.getcwd(), "plots/accuracy_comparison.png")

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.close()

if __name__ == "__main__":
    tensorboard_directory = TENSORBOARD_DIRECTORY

    if not os.path.exists(tensorboard_directory):
        print(f"Directory {tensorboard_directory} does not exist!")
        raise SystemExit(1)

    print("Extracting TensorBoard data and creating plot...")
    plot_accuracy_from_tensorboard(tensorboard_directory)
    print("Process complete!")