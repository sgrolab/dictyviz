import os
import sys
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from OpticalFlow.helpers import flowLoader

def main():

    results_dir = sys.argv[1]

    nb_frames = int(sys.argv[2])

    # Validate inputs
    if not os.path.exists(results_dir):
        print(f"Error: Results directory '{results_dir}' does not exist.")
        sys.exit(1)

    frame_dirs = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]

    # generate log file
    log_file = os.path.join(results_dir, "average_frames.log")
    with open(log_file, 'w') as log:
        try:
            # calculate the average flow for the first nb_frames
            flow_data = flowLoader.load_first_frames(results_dir, nb_frames, log_file=log)

            avg_flow = np.mean(flow_data, axis=1)  # average flow across slices

            mid_frame_index = nb_frames // 2
            output_dir = os.path.join(results_dir, str(mid_frame_index))
            if not os.path.exists(output_dir):
                raise FileNotFoundError(f"Output directory {output_dir} does not exist.")

            np.save(os.path.join(output_dir, f"avg_flow_frame_{nb_frames//2}.npy"), avg_flow)

            for first_frame_index in range(len(frame_dirs)-nb_frames):
                flowLoader.load_next_frame(flow_data, results_dir, first_frame_index + nb_frames, log_file=log)
                avg_flow = np.mean(flow_data, axis=1)  # average flow across slices

                # save the average flow for this frame
                mid_frame_index += 1
                output_dir = os.path.join(results_dir, str(mid_frame_index))
                if not os.path.exists(output_dir):
                    raise FileNotFoundError(f"Output directory {output_dir} does not exist.")
                np.save(os.path.join(output_dir, f"avg_flow_frame_{mid_frame_index}.npy"), avg_flow)

            print(f"Average flow calculated and saved in {results_dir}")

        except Exception as e:
            print(f"An error occurred while calculating average flow: {e}")
            sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python averageFrames.py <results_dir> <nb_frames>")
        sys.exit(1)

    main()

            

        

