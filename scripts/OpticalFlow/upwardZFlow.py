import os
import sys
import json
import torch
import numpy as np
import matplotlib
# Set backend before importing pyplot to avoid display issues
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from tqdm import tqdm
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from OpticalFlow.helpers import flowLoader, helpers

def main():
    results_dir = sys.argv[1]
    frame_avg = bool(int(sys.argv[2])) if len(sys.argv) > 2 else False

        # Validate inputs
    if not os.path.exists(results_dir):
        print(f"Error: Results directory '{results_dir}' does not exist.")
        sys.exit(1)

    frame_dirs = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]

    if frame_avg:
        # find frame directories that contain a avg_flow_frame_*.npy file
        frame_dirs = [d for d in frame_dirs if any(f.startswith('avg_flow_frame_') for f in os.listdir(os.path.join(results_dir, d)))]

    # remove non-numeric directories
    frame_dirs = [d for d in frame_dirs if d.isdigit()]

    print(f"Found {len(frame_dirs)} frames in directory: {results_dir}")

    # Find the channel index for cells
    parent_dir = os.path.dirname(results_dir)
    cells_channel = helpers.getCellChannelFromJSON(os.path.join(parent_dir, 'parameters.json'))

    # Create log file
    log_file = os.path.join(results_dir, 'upward_Z_flow.log')
    with open(log_file, 'w') as log:
        log.write(f"Processing frames in directory: {results_dir}\n")
        log.write(f"Using frame averaged flow: {frame_avg}\n")
        log.write(f"Number of frames: {len(frame_dirs)}\n")

        # Generate upward flow in Z for each frame
        upward_flow_over_time = np.zeros((len(frame_dirs), 2), dtype=np.float32)
        downward_flow_over_time = np.zeros((len(frame_dirs), 2), dtype=np.float32)

        for i, frame_dir in tqdm(enumerate(frame_dirs)):

            # determine frame index from directory name
            frame_index = int(frame_dir)
            log.write(f"Processing frame: {frame_index}\n")

            # load flow data for the current frame
            if frame_avg:
                flow_data = flowLoader.load_average_flow_frame(results_dir, frame_index)
                flow_data_Z = flow_data.get('vz', None)
                if flow_data_Z is None:
                    log.write(f"Warning: Flow data in Z for frame {frame_index} is None.\n")
                    continue
            else:
                flow_data_Z = flowLoader.load_Z_flow_frame(results_dir, frame_index)
                if flow_data_Z is None:
                    log.write(f"Warning: Flow data in Z for frame {frame_index} is None.\n")
                    continue               

            # Load raw data for the current frame
            raw_frame = flowLoader.load_raw_data(parent_dir, frame_index, cells_channel, log_file=log)
            if raw_frame is None:
                log.write(f"Warning: Could not load raw data for frame {frame_index}.\n")
                continue
            
            # Convert flow data to pytorch tensors to speed up processing
            flow_data_Z = np.ascontiguousarray(flow_data_Z, dtype=np.float32)
            raw_frame = np.ascontiguousarray(raw_frame, dtype=np.float32)

            if torch.cuda.is_available():
                device = torch.device('cuda')
                flow_data_Z = torch.from_numpy(flow_data_Z).to(device)
                raw_frame = torch.from_numpy(raw_frame).to(device)

                # Calculate the total upward flow in Z and normalize it against raw signal
                upward_flow_z = torch.sum(flow_data_Z > 0)
                downward_flow_z = torch.sum(flow_data_Z < 0)
                raw_signal = torch.sum(raw_frame)

                # Clear CUDA memory
                del flow_data_Z
                del raw_frame
                torch.cuda.empty_cache()
                
            else:
                print("CUDA is not available. Using CPU.")
                upward_flow_z = np.sum(flow_data_Z > 0)
                downward_flow_z = np.sum(flow_data_Z < 0)
                raw_signal = np.sum(raw_frame)
            
            norm_upward_flow_z = upward_flow_z / raw_signal if raw_signal > 0 else 0
            norm_downward_flow_z = abs(downward_flow_z / raw_signal) if raw_signal > 0 else 0
            upward_flow_over_time[i, 0] = frame_index
            upward_flow_over_time[i, 1] = norm_upward_flow_z
            downward_flow_over_time[i, 0] = frame_index
            downward_flow_over_time[i, 1] = norm_downward_flow_z
        
        # Sort the upward flow over time by frame index
        upward_flow_over_time = upward_flow_over_time[upward_flow_over_time[:, 0].argsort()]
        downward_flow_over_time = downward_flow_over_time[downward_flow_over_time[:, 0].argsort()]

        # Save the upward flow over time to a file
        if frame_avg:
            upward_flow_output_file = os.path.join(results_dir, 'upward_Z_flow_over_time_frame_avg.csv')
            downward_flow_output_file = os.path.join(results_dir, 'downward_Z_flow_over_time_frame_avg.csv')
        else:
            upward_flow_output_file = os.path.join(results_dir, 'upward_Z_flow_over_time.csv')
            downward_flow_output_file = os.path.join(results_dir, 'downward_Z_flow_over_time.csv')
        np.savetxt(upward_flow_output_file, upward_flow_over_time, delimiter=',')
        np.savetxt(downward_flow_output_file, downward_flow_over_time, delimiter=',')

        log.write(f"Upward flow over time saved to: {upward_flow_output_file}\n")
        log.write(f"Downward flow over time saved to: {downward_flow_output_file}\n")

        # Plot the results
        plt.figure(figsize=(10, 5))
        plt.plot(upward_flow_over_time[:,0], upward_flow_over_time[:,1], marker='o', linestyle='-', color='b')
        plt.title('Upward Flow in Z Over Time')
        plt.xlabel('Frame Index')
        plt.ylabel('Normalized Upward Flow (Z)')
        if frame_avg:
            plt.title('Upward Flow in Z Over Time (Frame Averaged)')
            plt.savefig(os.path.join(results_dir, 'upward_Z_flow_over_time_frame_avg.png'))
            log.write("Plot saved as upward_Z_flow_over_time_frame_avg.png\n")
        else:
            plt.title('Upward Flow in Z Over Time')
            plt.savefig(os.path.join(results_dir, 'upward_Z_flow_over_time.png'))
            log.write("Plot saved as upward_Z_flow_over_time.png\n")

        plt.figure(figsize=(10, 5))
        plt.plot(downward_flow_over_time[:,0], downward_flow_over_time[:,1], marker='o', linestyle='-', color='r')
        plt.title('Downward Flow in Z Over Time')
        plt.xlabel('Frame Index')
        plt.ylabel('Normalized Downward Flow (Z)')
        if frame_avg:
            plt.title('Downward Flow in Z Over Time (Frame Averaged)')
            plt.savefig(os.path.join(results_dir, 'downward_Z_flow_over_time_frame_avg.png'))
            log.write("Plot saved as downward_Z_flow_over_time_frame_avg.png\n")
        else:
            plt.title('Downward Flow in Z Over Time')
            plt.savefig(os.path.join(results_dir, 'downward_Z_flow_over_time.png'))
            log.write("Plot saved as downward_Z_flow_over_time.png\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python upwardZFlow.py <results_dir> [<frame_avg>]")
        sys.exit(1)

    main()
