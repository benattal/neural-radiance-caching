import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate2d, savgol_filter
import os
import h5py
import tqdm
import cv2

# Function to read H5 file (already defined in your code)
def read_h5(path):
    with h5py.File(path, 'r') as f:
        frames = np.array(f['data'])
    return frames

# Function to save video (already defined in your code)
def save_video_from_h5(h5_path, output_filename='output_video.mp4', fps=30, bkgd=True, 
                      scale_fac_1=4.0, scale_fac_2=2.0):
    # Either load the data from path or use directly provided data
    if isinstance(h5_path, str):
        data = read_h5(h5_path)
    else:
        data = h5_path
        
    # Extract dimensions
    H, W, T, C = data.shape
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, fps, (W, H))
    
    # Write each frame to the video
    bkg = data[..., :3].sum(-2)
    bkg = (bkg/scale_fac_1)**(1/2.2)
    
    for t in tqdm.tqdm(range(T)):
        frame = (data[:, :, t, :3] / scale_fac_2)**(1/2.2)
        if bkgd:
            frame = 0.9*frame + 0.1*bkg
        frame = np.clip(frame, 0, 1)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write((frame*255).astype('uint8'))
    
    # Release the video writer
    out.release()
    print(f'Video saved at {output_filename}')

# Function to apply the Gaussian pulse response (correlate with pulse)
def apply_pulse_response(transient_data, pulse):
    H, W, T, C = transient_data.shape
    # Create output with same shape as input
    result = np.zeros_like(transient_data)
    
    # Apply correlation for each pixel
    print("Applying pulse response...")
    for h in tqdm.tqdm(range(H)):
        for w in range(W):
            for c in range(C):
                # Correlate the time dimension with the pulse
                result[h, w, :, c] = np.correlate(transient_data[h, w, :, c], pulse, mode='same')
    
    return result

# Function to create CW-ToF response images
def apply_cw_tof_response(transient_data, phase_shifts, frequency=6):
    H, W, T, C = transient_data.shape
    num_phase_shifts = len(phase_shifts)
    result = np.zeros((H, W, num_phase_shifts, C))
    
    # Create time array
    time_axis = np.arange(T) / T
    
    print("Applying CW-ToF responses...")
    for p_idx, phase in enumerate(tqdm.tqdm(phase_shifts)):
        # Create the sinusoidal modulation with the given phase shift
        modulation = np.sin(2 * np.pi * frequency * time_axis + phase)
        
        # Apply modulation to each pixel
        for c in range(C):
            # Multiply transient by modulation and sum over time
            result[:, :, p_idx, c] = np.sum(transient_data[:, :, :, c] * modulation[None, None, :], axis=2)
    
    return result

# Function to apply constant camera response
def apply_constant_response(transient_data):
    # Simply sum over the time dimension
    return np.sum(transient_data, axis=2)

# Main function to process the transient data
def process_transient_data(transient_path, pulse_path, output_dir='./output'):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the transient data
    print(f"Loading transient data from {transient_path}...")
    transient_data = read_h5(transient_path)
    
    # Load the pulse data
    print(f"Loading pulse data from {pulse_path}...")
    pulse_data = np.load(pulse_path)
    
    # Smooth the pulse data 
    pulse_data = savgol_filter(pulse_data, window_length=21, polyorder=3)
    
    # Apply Gaussian pulse response and save video
    print("Processing Gaussian pulse response...")
    pulse_response = apply_pulse_response(transient_data, pulse_data)
    save_video_from_h5(pulse_response, os.path.join(output_dir, "pulse_response.mp4"))
    
    # Apply CW-ToF responses and save images
    print("Processing CW-ToF responses...")
    phase_shifts = [0, np.pi/2, np.pi, 3*np.pi/2]
    cw_tof_responses = apply_cw_tof_response(transient_data, phase_shifts)
    
    # Save each CW-ToF response as an image
    print("Saving CW-ToF response images...")
    for i, phase in enumerate(phase_shifts):
        phase_name = ['0', 'pi_2', 'pi', '3pi_2'][i]
        response_img = np.abs(cw_tof_responses[:, :, i, :3])
        
        # Normalize and apply gamma correction
        response_img = np.clip(response_img / response_img.max(), 0, 1) ** (1/2.2)
        
        # Save as image
        plt.figure(figsize=(10, 10))
        plt.imshow(response_img)
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, f"cw_tof_phase_{phase_name}.png"), 
                    bbox_inches='tight', pad_inches=0, facecolor='black')
        plt.close()
    
    # Apply constant camera response and save image
    print("Processing constant camera response...")
    constant_response = apply_constant_response(transient_data)
    
    # Normalize and apply gamma correction
    constant_response = np.abs(constant_response[:, :, :3])
    constant_response = np.clip(constant_response / constant_response.max(), 0, 1) ** (1/2.2)
    
    # Save as image
    plt.figure(figsize=(10, 10))
    plt.imshow(constant_response)
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, "constant_response.png"), 
                bbox_inches='tight', pad_inches=0, facecolor='black')
    plt.close()
    
    print("All processing complete!")

# Run the main function
if __name__ == "__main__":
    # Set paths
    transient_path = "/scratch/ondemand28/battal/active-yobo/checkpoints/yobo_results/synthetic/peppers_cache_eval/save/transients/0029.h5"
    pulse_path = "/scratch/ondemand28/battal/active-yobo/data/yobo/pulse.npy"
    output_dir = "./sensor_responses"
    
    # Process the data
    process_transient_data(transient_path, pulse_path, output_dir)