import os
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from OpticalFlow.helpers import flowLoader

def getCellChannelFromJSON(jsonFile):
    with open(jsonFile) as f:
        channelSpecs = json.load(f)["channels"]
    cells_found = False
    for i, channelInfo in enumerate(channelSpecs):
        if channelInfo["name"].startswith("cells"):
            cells = i
            if cells_found:
                print(f"Warning: Multiple channels starting with 'cells' found. Multiple cell channels is not supported. Using channel {i}.")
            print(f"Found cell channel: {i}")
            cells_found = True
    if not cells_found:
        print("Error: No channel starting with 'cells' found in parameters.json.")
        return None
    return cells

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

    print(f"Found {len(frame_dirs)} frames in directory: {results_dir}")

    # Find the channel index for cells
    parent_dir = os.path.dirname(results_dir)
    cells_channel = getCellChannelFromJSON(os.path.join(parent_dir, 'parameters.json'))

    # Create log file
    log_file = os.path.join(results_dir, 'upward_Z_flow.log')
    with open(log_file, 'w') as log:
        log.write(f"Processing frames in directory: {results_dir}\n")
        log.write(f"Using frame averaged flow: {frame_avg}\n")
        log.write(f"Number of frames: {len(frame_dirs)}\n")

        # Generate upward flow in Z for each frame
        upward_flow_over_time = np.zeros(len(frame_dirs))

        for frame_dir in tqdm(frame_dirs):
            # determine frame index from directory name
            frame_index = int(frame_dir)
            log.write(f"Processing frame: {frame_index}\n")

            # load flow data for the current frame
            if frame_avg:
                flow_data = flowLoader.load_avg_flow_frame(results_dir, frame_index)
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
                raw_signal = torch.sum(raw_frame)

                # Clear CUDA memory
                del flow_data_Z
                del raw_frame
                torch.cuda.empty_cache()
                
            else:
                print("CUDA is not available. Using CPU.")
                flow_data_Z = np.sum(flow_data_Z > 0)
                raw_signal = np.sum(raw_frame)
            
            norm_upward_flow_z = upward_flow_z / raw_signal if raw_signal > 0 else 0

            upward_flow_over_time[frame_index] = norm_upward_flow_z

        # Save the upward flow over time to a file
        output_file = os.path.join(results_dir, 'upward_Z_flow_over_time.csv')
        np.savetxt(output_file, upward_flow_over_time, delimiter=',')

        log.write(f"Upward flow over time saved to: {output_file}\n")

        # Plot the results
        plt.figure(figsize=(10, 5))
        plt.plot(upward_flow_over_time, marker='o', linestyle='-', color='b')
        plt.title('Upward Flow in Z Over Time')
        plt.xlabel('Frame Index')
        plt.ylabel('Normalized Upward Flow (Z)')
        plt.savefig(os.path.join(results_dir, 'upward_Z_flow_over_time.png'))
        plt.show()

        log.write("Plot saved as upward_Z_flow_over_time.png\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python upwardZFlow.py <results_dir> [<frame_avg>]")
        sys.exit(1)

    main()
