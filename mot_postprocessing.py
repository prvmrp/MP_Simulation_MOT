import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import shutil
import glob
import subprocess
import sys

# Try to import imageio for GIF creation, but allow the script to run without it
try:
    import imageio.v2 as iio
    IMAGEIO_AVAILABLE = True
except ImportError:
    IMAGEIO_AVAILABLE = False

# --- CONFIGURATION ---
BASE_DATA_FILENAME = 'mot_data_full.npz' 
GIF_FILENAME = 'mot_evolution.gif'
OUTPUT_DIR = 'mot_frames' # This variable is no longer strictly used for saving images

# >>> USER-DEFINED CLUSTER SETTINGS (Simplified) <<<
CLUSTER_USER = "your_username"      # e.g., "leona"
CLUSTER_HOST = "cluster.server.edu" # e.g., "login.cluster.uni"
REMOTE_BASE_PATH = "/path/to/simulation/results" 


# --- UTILITY: FOLDER NAME GENERATION ---

def generate_folder_name(N, B_prime, T_res_uK, R_res):
    """Reconstructs the folder name exactly as saved by the simulation script."""
    # Use scientific notation matching the simulation output: res_N=1.0e+02_B=1.0e-02T_T=2.0e+02uK_R=1.0e-03m
    N_str = f"{N:.1e}"
    B_str = f"{B_prime:.1e}"
    T_res_str = f"{T_res_uK:.1e}"
    R_res_str = f"{R_res:.1e}"
    
    # Construct the folder name using the simulation's structure
    folder_name = f"res_N={N_str}_B={B_str}T_T={T_res_str}uK_R={R_res_str}m"
    return folder_name

# --- SSH/SCP Utility (Minor Update for Clarity) ---

def download_data_via_scp(user, host, remote_folder, local_filename=BASE_DATA_FILENAME):
    """
    Downloads the simulation data file from a remote cluster based on the generated folder name.
    """
    remote_file = os.path.join(remote_folder, BASE_DATA_FILENAME)
    remote_source = f"{user}@{host}:{remote_file}"
    # The target directory is the folder name itself (e.g., res_N=100...)
    local_target_dir = os.path.basename(remote_folder) 
    local_target_path = os.path.join(local_target_dir, local_filename) 
    
    # Ensure the local directory exists
    os.makedirs(local_target_dir, exist_ok=True)
    
    print(f"\nAttempting to download data from: {remote_source}")
    
    # Construct the SCP command
    scp_command = ['scp', remote_source, local_target_path]
    
    try:
        result = subprocess.run(scp_command, check=True, capture_output=True, text=True)
        print(f"Download successful. Data saved as '{local_target_path}'.")
        return local_target_path # Return the full local file path
        
    except subprocess.CalledProcessError as e:
        print("\n--- SCP DOWNLOAD FAILED ---")
        print(f"Error executing command: {' '.join(scp_command)}")
        print(f"Check configuration and remote path: {remote_file}")
        print(f"Reason: {e.stderr.strip()}")
        print("---------------------------\n")
        return None
    except FileNotFoundError:
        print("\n--- SCP DOWNLOAD FAILED ---")
        print("Error: The 'scp' command was not found. Ensure SSH client tools are installed.")
        print("---------------------------\n")
        return None

# --- UTILITY AND PLOTTING FUNCTIONS (MODIFIED) ---

def load_simulation_data(filename):
    """Loads all data saved by the simulation."""
    try:
        data = np.load(filename, allow_pickle=True)
        print(f"Successfully loaded data from {filename}")
        return data
    except FileNotFoundError:
        print(f"Error: Data file '{filename}' not found locally. Please download it first.")
        return None

def plot_scalar_history_improved(time, r_rms, v_rms, e_k, output_folder):
    """
    Figure 1: Plots RMS radius, RMS velocity, and Total Kinetic Energy history.
    Saves output to the specified folder.
    """
    plt.style.use('ggplot')
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    time_ms = time * 1e3

    # ... (Plotting code remains unchanged)
    axs[0].plot(time_ms, r_rms * 1e3, color='darkblue', alpha=0.8)
    axs[0].set_title("Cloud Confinement (RMS Radius)")
    axs[0].set_xlabel("Time (ms)")
    axs[0].set_ylabel("R_RMS (mm)")
    axs[0].grid(True, linestyle='--')
    axs[0].ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    axs[1].plot(time_ms, v_rms, color='red', alpha=0.8)
    axs[1].set_title("Cloud Cooling (RMS Velocity)")
    axs[1].set_xlabel("Time (ms)")
    axs[1].set_ylabel("V_RMS (m/s)")
    axs[1].grid(True, linestyle='--')

    axs[2].plot(time_ms, e_k * 1e12, color='green', alpha=0.8)
    axs[2].set_title("Total Kinetic Energy")
    axs[2].set_xlabel("Time (ms)")
    axs[2].set_ylabel("E_k (pJ)")
    axs[2].grid(True, linestyle='--')
    axs[2].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    fig.suptitle("Figure 1: Time Evolution (Cooling, Confinement, and Stability)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    save_path = os.path.join(output_folder, 'figure1_scalar_history.png')
    plt.savefig(save_path)
    plt.close(fig) 
    print(f"Saved 'figure1_scalar_history.png' to {output_folder}")


def plot_spatial_confinement_analysis(final_positions, output_folder):
    """
    Figure 2: Plots spatial distribution, density profiles, and dimensional RMS.
    Saves output to the specified folder.
    """
    plt.style.use('default')
    fig, axs = plt.subplots(2, 2, figsize=(14, 12))
    axs = axs.flatten()

    pos_mm = final_positions * 1e3 
    
    # ... (Plotting code remains unchanged)
    rms_x = np.sqrt(np.mean(final_positions[:, 0]**2)) * 1e3
    rms_y = np.sqrt(np.mean(final_positions[:, 1]**2)) * 1e3
    rms_z = np.sqrt(np.mean(final_positions[:, 2]**2)) * 1e3
    max_extent = np.max(np.abs(final_positions)) * 1e3 * 1.05

    axs[0].scatter(pos_mm[:, 0], pos_mm[:, 2], s=5, alpha=0.5, color='darkorange')
    axs[0].set_title("Final Spatial Distribution ($x$ vs. $z$ cross-section)")
    axs[0].set_xlabel("Position $x$ (mm)")
    axs[0].set_ylabel("Position $z$ (mm)")
    axs[0].set_xlim(-max_extent, max_extent)
    axs[0].set_ylim(-max_extent, max_extent)
    axs[0].set_aspect('equal', adjustable='box') 
    axs[0].grid(True)

    rms_values = [rms_x, rms_y, rms_z]
    axs[1].bar(['$R_x$', '$R_y$', '$R_z$'], rms_values, color=['gray', 'gray', 'red'])
    axs[1].set_title("Dimensional RMS Radius (Confinement Check)")
    axs[1].set_ylabel("RMS Radius (mm)")
    axs[1].grid(True, axis='y', linestyle='--')
    
    axs[2].hist(pos_mm[:, 2], bins=40, density=True, color='red', alpha=0.7)
    axs[2].set_title("Density Profile (Z-Axis)")
    axs[2].set_xlabel("Z Position (mm)")
    axs[2].set_ylabel("Normalized Density")
    axs[2].grid(True, axis='y')

    axs[3].hist(pos_mm[:, 0], bins=40, density=True, color='green', alpha=0.7)
    axs[3].set_title("Density Profile (X-Axis)")
    axs[3].set_xlabel("X Position (mm)")
    axs[3].set_ylabel("Normalized Density")
    axs[3].grid(True, axis='y')
    
    fig.suptitle("Figure 2: Spatial and Confinement Analysis", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    save_path = os.path.join(output_folder, 'figure2_spatial_confinement.png')
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved 'figure2_spatial_confinement.png' to {output_folder}")


def plot_phase_space_analysis(final_positions, final_velocities, output_folder):
    """
    Figure 3: Plots phase-space profiles for kinetic verification.
    Saves output to the specified folder.
    """
    plt.style.use('default')
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    pos_mm = final_positions * 1e3 # mm

    # ... (Plotting code remains unchanged)
    axs[0].scatter(pos_mm[:, 0], final_velocities[:, 0], s=5, alpha=0.5, color='teal')
    axs[0].set_title("Phase-Space Profile ($x$ vs. $v_x$) - Damping Check")
    axs[0].set_xlabel("Position $x$ (mm)")
    axs[0].set_ylabel("Velocity $v_x$ (m/s)")
    axs[0].grid(True)
    
    axs[1].hist(final_velocities[:, 0], bins=40, density=True, color='purple', alpha=0.7)
    axs[1].set_title("Final Velocity Distribution ($v_x$)")
    axs[1].set_xlabel("Velocity $v_x$ (m/s)")
    axs[1].set_ylabel("Normalized Probability Density")
    axs[1].grid(True, axis='y')
    
    fig.suptitle("Figure 3: Phase-Space and Kinetic Analysis", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    save_path = os.path.join(output_folder, 'figure3_phase_space.png')
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved 'figure3_phase_space.png' to {output_folder}")


def create_animation_matplotlib(pos_history, time_interval, output_folder):
    """
    Generates the GIF using Matplotlib's FuncAnimation.
    Saves output to the specified folder.
    """
    print("\nStarting GIF generation in memory via Matplotlib...")
    
    # 1. Setup figure and determine limits (UNCHANGED)
    fig, ax = plt.subplots(figsize=(7, 7))
    all_positions_m = np.concatenate(pos_history, axis=0)
    max_pos_m = np.max(np.abs(all_positions_m)) * 1.05 
    max_pos_mm = max_pos_m * 1e3
    pos_frame_0_mm = pos_history[0] * 1e3
    scat = ax.scatter(pos_frame_0_mm[:, 0], pos_frame_0_mm[:, 2], s=5, alpha=0.6, color='darkblue')

    ax.set_xlim(-max_pos_mm, max_pos_mm)
    ax.set_ylim(-max_pos_mm, max_pos_mm)
    ax.set_xlabel("X Position (mm)")
    ax.set_ylabel("Z Position (mm)")
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True)
    time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=12, verticalalignment='top')
    
    # 2. Animation update function (UNCHANGED)
    def update_frame(frame_index):
        positions_mm = pos_history[frame_index] * 1e3
        scat.set_offsets(positions_mm[:, [0, 2]])
        current_time = frame_index * time_interval
        time_text.set_text(f"t = {current_time*1e3:.2f} ms")
        ax.set_title(f"Atom Cloud Evolution")
        return scat, time_text

    # 3. Create the animation object (UNCHANGED)
    anim = animation.FuncAnimation(
        fig, update_frame, frames=len(pos_history), 
        interval=100, blit=False, repeat=False
    )
    
    # 4. Save the animation (MODIFIED)
    try:
        save_path = os.path.join(output_folder, GIF_FILENAME)
        print(f"\nSaving GIF '{GIF_FILENAME}' to {output_folder}...")
        anim.save(save_path, writer='pillow', fps=10) 
        print(f"Successfully created {save_path}.")
    except Exception as e:
        print(f"\nGIF creation failed using Matplotlib/Pillow writer.")
        print("Ensure the 'Pillow' library is installed: pip install pillow")
        print(f"Error details: {e}")

    plt.close(fig)


# --- Main Execution (MODIFIED) ---
if __name__ == '__main__':
    
    # --- STEP 0: PARSE ARGUMENTS AND DETERMINE FILE PATH ---
    
    if len(sys.argv) < 5 or (len(sys.argv) == 5 and sys.argv[1].lower() == 'download'):
        # ... (error message remains unchanged)
        print("\nUSAGE ERROR: Missing simulation parameters.")
        print("Required Arguments: N B_prime T_res_uK R_res [download]")
        print("Example: python post_processor.py 100 0.01 200.0 0.001 download")
        sys.exit(1)
    
    # Parse parameters (N, B', T_res, R_res)
    try:
        N_atoms = int(sys.argv[1])
        B_prime = float(sys.argv[2])
        T_res_uK = float(sys.argv[3])
        R_res = float(sys.argv[4])
    except ValueError:
        print("\nERROR: N, B_prime, T_res_uK, and R_res must be numbers.")
        sys.exit(1)

    # 1. Construct the folder name
    folder_name = generate_folder_name(N_atoms, B_prime, T_res_uK, R_res)
    local_data_path = os.path.join(folder_name, BASE_DATA_FILENAME)
    
    # This variable holds the folder path where all results will be saved.
    OUTPUT_FOLDER_PATH = folder_name
    
    # 2. Check for local data, attempt download if requested
    if not os.path.exists(local_data_path):
        print(f"Required data file '{local_data_path}' not found locally.")
        
        if len(sys.argv) == 6 and sys.argv[5].lower() == 'download':
            # ... (config check remains unchanged)
            if CLUSTER_USER == "your_username" or CLUSTER_HOST == "cluster.server.edu":
                print("\nERROR: Please update CLUSTER_USER and CLUSTER_HOST variables at the top of the script first!")
                sys.exit(1)
                
            remote_folder = os.path.join(REMOTE_BASE_PATH, folder_name)
            
            # Download the data - the function handles folder creation.
            downloaded_path = download_data_via_scp(CLUSTER_USER, CLUSTER_HOST, remote_folder, BASE_DATA_FILENAME)
            
            if downloaded_path is None:
                sys.exit(1) 
        else:
            print(f"To download data for these parameters, rerun with the 'download' flag:")
            print(f">>> python post_processor.py {N_atoms} {B_prime} {T_res_uK} {R_res} download")
            sys.exit(1) 
            
    
    # --- STEP 2: LOAD DATA ---

    sim_data = load_simulation_data(local_data_path) # Use the constructed path
    
    if sim_data is not None:
        # Load data histories (UNCHANGED)
        time_hist = sim_data['time']
        r_rms_hist = sim_data['r_rms']
        v_rms_hist = sim_data['v_rms']
        e_k_hist = sim_data['e_k'] 
        pos_frames = sim_data['positions']
        vel_frames = sim_data['velocities']
        
        dt = sim_data['dt'].item() 
        save_interval = sim_data['save_interval'].item()
        time_interval_frame = dt * save_interval

        # 3. Figure 1: Scalar Time History
        plot_scalar_history_improved(time_hist, r_rms_hist, v_rms_hist, e_k_hist, OUTPUT_FOLDER_PATH)
        
        # 4. Figure 2: Spatial Confinement Analysis
        plot_spatial_confinement_analysis(pos_frames[-1], OUTPUT_FOLDER_PATH)
        
        # 5. Figure 3: Phase Space and Kinetic Analysis
        plot_phase_space_analysis(pos_frames[-1], vel_frames[-1], OUTPUT_FOLDER_PATH)
        
        # 6. Generate GIF
        create_animation_matplotlib(pos_frames, time_interval_frame, OUTPUT_FOLDER_PATH)