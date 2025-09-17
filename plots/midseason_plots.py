import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd

sns.set_style("whitegrid")
sns.set_palette("viridis")

class MidseasonPlotter:
    def __init__(self, april_dir, may_dir, june_dir):
        self.directories = {
            'April': april_dir,
            'May': may_dir, 
            'June': june_dir
        }
        self.crop_types = list(range(12))  # 0-11
        self.metrics = ['Accuracy_eval', 'IOU_eval']
        self.crop_names = {
            0: 'Background',
            1: 'Meadow / Pasture',
            2: 'Soft Winter Wheat',
            3: 'Corn (Maize)',
            4: 'Winter Barley',
            5: 'Winter Rapeseed',
            6: 'Vines',
            7: 'Beet',
            8: 'Potatoes',
            9: 'Orchard',
            10: 'Berries',
            11: 'Mixed Cereal'
        }

    def read_tensorboard_data(self, log_dir, metric_name, crop_id):
        """Read scalar data from tensorboard events file"""
        subfolder = f"{metric_name}_{crop_id}"
        event_path = os.path.join(log_dir, subfolder)
        
        if not os.path.exists(event_path):
            print(f"Warning: {event_path} not found")
            return None
            
        # Find events file
        events_file = None
        for file in os.listdir(event_path):
            if file.startswith('events.out'):
                events_file = os.path.join(event_path, file)
                break
                
        if not events_file:
            print(f"Warning: No events file found in {event_path}")
            return None
            
        # Read events
        ea = EventAccumulator(events_file)
        ea.Reload()
        
        # Get scalar tags
        tags = ea.Tags()['scalars']
        if not tags:
            print(f"Warning: No scalar tags found in {events_file}")
            return None
            
        # Use first available tag
        tag = tags[0]
        scalar_events = ea.Scalars(tag)
        
        return [(s.step, s.value) for s in scalar_events]
    
    def extract_values_at_steps(self, log_steps):
        """Extract metric values at specified steps for each month/crop combination"""
        results = {}
        
        for month, log_dir in self.directories.items():
            results[month] = {}
            
            for metric in self.metrics:
                results[month][metric] = {}
                
                for crop_id in self.crop_types:
                    data = self.read_tensorboard_data(log_dir, metric, crop_id)
                    
                    if data and month in log_steps and crop_id in log_steps[month]:
                        target_step = log_steps[month][crop_id]
                        
                        # Find closest step
                        steps, values = zip(*data)
                        closest_idx = np.argmin([abs(s - target_step) for s in steps])
                        results[month][metric][crop_id] = values[closest_idx]
                    else:
                        results[month][metric][crop_id] = None
                        
        return results
    
    def create_comparison_plots(self, results, save_path=None):
        """Create comparison plots for accuracy and IOU across months"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        months = ['April', 'May', 'June']
        colors = sns.color_palette(n_colors=len(months))
        
        # Accuracy plot
        for i, month in enumerate(months):
            acc_values = [results[month]['Accuracy_eval'].get(crop_id) 
                         for crop_id in self.crop_types]
            valid_crops = [crop_id for crop_id in self.crop_types 
                          if results[month]['Accuracy_eval'].get(crop_id) is not None]
            valid_values = [v for v in acc_values if v is not None]
            
            ax1.plot(valid_crops, valid_values, 'o-', 
                    color=colors[i], label=month, linewidth=2, markersize=6)
        
        ax1.set_xlabel('Crop Type')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Accuracy Comparison Across Months')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(self.crop_types)
        
        # IOU plot
        for i, month in enumerate(months):
            iou_values = [results[month]['IOU_eval'].get(crop_id) 
                         for crop_id in self.crop_types]
            valid_crops = [crop_id for crop_id in self.crop_types 
                          if results[month]['IOU_eval'].get(crop_id) is not None]
            valid_values = [v for v in iou_values if v is not None]
            
            ax2.plot(valid_crops, valid_values, 'o-', 
                    color=colors[i], label=month, linewidth=2, markersize=6)
        
        ax2.set_xlabel('Crop Type')
        ax2.set_ylabel('IOU')
        ax2.set_title('IOU Comparison Across Months')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(self.crop_types)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_heatmaps(self, results, save_path=None):
        """Create heatmaps showing metric values across crops and months"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        months = ['April', 'May', 'June']
        
        # Prepare data for heatmaps
        acc_data = []
        iou_data = []
        
        for month in months:
            acc_row = [results[month]['Accuracy_eval'].get(crop_id, np.nan) 
                      for crop_id in self.crop_types]
            iou_row = [results[month]['IOU_eval'].get(crop_id, np.nan) 
                      for crop_id in self.crop_types]
            acc_data.append(acc_row)
            iou_data.append(iou_row)
        
        # Create heatmaps
        sns.heatmap(acc_data, annot=True, fmt='.3f', cmap='YlOrRd',
                   xticklabels=self.crop_types, yticklabels=months,
                   ax=ax1, cbar_kws={'label': 'Accuracy'})
        ax1.set_title('Accuracy Heatmap')
        ax1.set_xlabel('Crop Type')
        
        sns.heatmap(iou_data, annot=True, fmt='.3f', cmap='YlGnBu',
                   xticklabels=self.crop_types, yticklabels=months,
                   ax=ax2, cbar_kws={'label': 'IOU'})
        ax2.set_title('IOU Heatmap')
        ax2.set_xlabel('Crop Type')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_bar_plots(self, results, save_path=None):
        """Create bar plots comparing metrics across months for each crop type"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        
        months = ['April', 'May', 'June']
        colors = sns.color_palette(n_colors=len(months))
        
        # Set up bar positions
        x = np.arange(len(self.crop_types))
        width = 0.25
        
        # Accuracy bars
        for i, month in enumerate(months):
            acc_values = [results[month]['Accuracy_eval'].get(crop_id, 0) 
                         for crop_id in self.crop_types]
            ax1.bar(x + i*width, acc_values, width, label=month, color=colors[i], alpha=0.8)
        
        ax1.set_xlabel('Crop Type')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Accuracy Comparison by Crop Type and Month')
        ax1.set_xticks(x + width)
        ax1.set_xticklabels([self.crop_names[i] for i in self.crop_types], rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # IOU bars
        for i, month in enumerate(months):
            iou_values = [results[month]['IOU_eval'].get(crop_id, 0) 
                         for crop_id in self.crop_types]
            ax2.bar(x + i*width, iou_values, width, label=month, color=colors[i], alpha=0.8)
        
        ax2.set_xlabel('Crop Type')
        ax2.set_ylabel('IOU')
        ax2.set_title('IOU Comparison by Crop Type and Month')
        ax2.set_xticks(x + width)
        ax2.set_xticklabels([self.crop_names[i] for i in self.crop_types], rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

def main():
    # Example usage
    april_dir = "/Users/jeffreyzweidler/Desktop/Semester_Project/AlpsCropFormer/models/model_logs/Sentinel/focal_loss/TSViT_fold1.sliding_window.masked_focal.midseason_aggregation_april"
    may_dir = "/Users/jeffreyzweidler/Desktop/Semester_Project/AlpsCropFormer/models/model_logs/Sentinel/focal_loss/TSViT_fold1.sliding_window.masked_focal.midseason_aggregation_may"
    june_dir = "/Users/jeffreyzweidler/Desktop/Semester_Project/AlpsCropFormer/models/model_logs/Sentinel/focal_loss/TSViT_fold1.sliding_window.masked_focal.midseason_aggregation_june"
    
    # Define which log steps to use for each month and crop type
    # You'll need to adjust these based on when each run achieved its maximum
    log_steps = {
        'April': {crop_id: 110000 for crop_id in range(12)},  # Example: all use step 1000
        'May': {crop_id: 89375 for crop_id in range(12)},    # Example: all use step 1500
        'June': {crop_id: 123750 for crop_id in range(12)}     # Example: all use step 800
    }
    
    plotter = MidseasonPlotter(april_dir, may_dir, june_dir)
    results = plotter.extract_values_at_steps(log_steps)
    
    # Create plots
    plotter.create_bar_plots(results, 'plots/midseason_barplots.png')
    # plotter.create_comparison_plots(results, 'plots/midseason_comparison.png')
    # plotter.create_heatmaps(results, 'plots/midseason_heatmaps.png')
    
    # Print summary statistics
    print("\nSummary Statistics:")
    for month in ['April', 'May', 'June']:
        acc_values = [v for v in results[month]['Accuracy_eval'].values() if v is not None]
        iou_values = [v for v in results[month]['IOU_eval'].values() if v is not None]
        
        print(f"\n{month}:")
        print(f"  Accuracy - Mean: {np.mean(acc_values):.3f}, Std: {np.std(acc_values):.3f}")
        print(f"  IOU - Mean: {np.mean(iou_values):.3f}, Std: {np.std(iou_values):.3f}")

if __name__ == "__main__":
    main()
