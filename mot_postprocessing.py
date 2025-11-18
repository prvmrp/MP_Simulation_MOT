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
DATA_FILENAME = 'mot_data_full.npz' 
GIF_FILENAME = 'mot_evolution.gif'
OUTPUT_DIR = 'mot_frames' 

# >>> USER-DEFINED CLUSTER SETTINGS <<<
# IMPORTANT: Update these three lines with your actual cluster details.
CLUSTER_USER = "your_username"  # e.g., "leona"
CLUSTER_HOST = "cluster.server.edu" # e.g., "login.cluster.uni"
# The full path to the saved data file *on the remote cluster*
REMOTE_FILE_PATH = "/path/to/simulation/results/mot_data_full.npz" 

# --- SSH/SCP Utility ---

def download_data_via_scp(user, host, remote_file, local_filename=DATA_FILENAME):
    """
    Downloads the simulation data file from a remote cluster using SCP based on
    predefined user, host, and remote file path.
    
    NOTE: This requires your local machine to have SSH keys configured for password-less access
    or it will prompt you for your password in the terminal.
    """
    remote_source = f"{user}@{host}:{remote_file}"
    print(f"\nAttempting to download data from: {remote_source}")
    
    # Construct the SCP command
    scp_command = ['scp', remote_source, local_filename]
    
    try:
        # Run the command and capture output
        result = subprocess.run(scp_command, check=True, capture_output=True, text=True)
        
        print(f"Download successful. Data saved as '{local_filename}'.")
        return True
        
    except subprocess.CalledProcessError as e:
        print("\n--- SCP DOWNLOAD FAILED ---")
        print(f"Error executing command: {' '.join(scp_command)}")
        print(f"Check CLUSTER_USER, CLUSTER_HOST, and REMOTE_FILE_PATH definitions in the script.")
        print(f"Reason: {e.stderr.strip()}")
        print("---------------------------\n")
        return False
    except FileNotFoundError:
        print("\n--- SCP DOWNLOAD FAILED ---")
        print("Error: The 'scp' command was not found.")
        print("Ensure SSH/SCP client tools are installed and in your system's PATH.")
        print("---------------------------\n")
        return False

# --- UTILITY AND PLOTTING FUNCTIONS (UNCHANGED) ---

def load_simulation_data(filename):
    """Loads all data saved by the simulation."""
    try:
        data = np.load(filename, allow_pickle=True)
        print(f"Successfully loaded data from {filename}")
        return data
    except FileNotFoundError:
        print(f"Error: Data file '{filename}' not found locally. Use download_data_via_scp() first.")
        return None

def plot_scalar_history_improved(time, r_rms, v_rms, e_k):
    """
    Figure 1: Plots RMS radius, RMS velocity, and Total Kinetic Energy history.
    """
    plt.style.use('ggplot')
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    time_ms = time * 1e3

    # 1. RMS Radius vs. Time (Confinement Check)
    axs[0].plot(time_ms, r_rms * 1e3, color='darkblue', alpha=0.8)
    axs[0].set_title("Cloud Confinement (RMS Radius)")
    axs[0].set_xlabel("Time (ms)")
    axs[0].set_ylabel("R_RMS (mm)")
    axs[0].grid(True, linestyle='--')
    axs[0].ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    # 2. RMS Velocity vs. Time (Cooling Check)
    axs[1].plot(time_ms, v_rms, color='red', alpha=0.8)
    axs[1].set_title("Cloud Cooling (RMS Velocity)")
    axs[1].set_xlabel("Time (ms)")
    axs[1].set_ylabel("V_RMS (m/s)")
    axs[1].grid(True, linestyle='--')

    # 3. Total Kinetic Energy vs. Time (Stability Check)
    axs[2].plot(time_ms, e_k * 1e12, color='green', alpha=0.8)
    axs[2].set_title("Total Kinetic Energy")
    axs[2].set_xlabel("Time (ms)")
    axs[2].set_ylabel("E_k (pJ)")
    axs[2].grid(True, linestyle='--')
    axs[2].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    fig.suptitle("Figure 1: Time Evolution (Cooling, Confinement, and Stability)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('figure1_scalar_history.png')
    plt.close(fig) 
    print("Saved 'figure1_scalar_history.png'")


def plot_spatial_confinement_analysis(final_positions):
    """
    Figure 2: Plots spatial distribution, density profiles, and dimensional RMS.
    """
    plt.style.use('default')
    fig, axs = plt.subplots(2, 2, figsize=(14, 12))
    axs = axs.flatten()

    pos_mm = final_positions * 1e3 
    
    # Calculate dimensional RMS values
    rms_x = np.sqrt(np.mean(final_positions[:, 0]**2)) * 1e3
    rms_y = np.sqrt(np.mean(final_positions[:, 1]**2)) * 1e3
    rms_z = np.sqrt(np.mean(final_positions[:, 2]**2)) * 1e3
    
    max_extent = np.max(np.abs(final_positions)) * 1e3 * 1.05

    # 1. Spatial Scatter Plot (X vs. Z) - Shape Visualization
    axs[0].scatter(pos_mm[:, 0], pos_mm[:, 2], s=5, alpha=0.5, color='darkorange')
    axs[0].set_title("Final Spatial Distribution ($x$ vs. $z$ cross-section)")
    axs[0].set_xlabel("Position $x$ (mm)")
    axs[0].set_ylabel("Position $z$ (mm)")
    
    axs[0].set_xlim(-max_extent, max_extent)
    axs[0].set_ylim(-max_extent, max_extent)
    axs[0].set_aspect('equal', adjustable='box') 
    axs[0].grid(True)

    # 2. Dimensional RMS Bar Chart (Quantifying Asymmetry)
    rms_values = [rms_x, rms_y, rms_z]
    axs[1].bar(['$R_x$', '$R_y$', '$R_z$'], rms_values, color=['gray', 'gray', 'red'])
    axs[1].set_title("Dimensional RMS Radius (Confinement Check)")
    axs[1].set_ylabel("RMS Radius (mm)")
    axs[1].grid(True, axis='y', linestyle='--')
    
    # 3. Density Profile (Z-axis)
    axs[2].hist(pos_mm[:, 2], bins=40, density=True, color='red', alpha=0.7)
    axs[2].set_title("Density Profile (Z-Axis)")
    axs[2].set_xlabel("Z Position (mm)")
    axs[2].set_ylabel("Normalized Density")
    axs[2].grid(True, axis='y')

    # 4. Density Profile (X-axis)
    axs[3].hist(pos_mm[:, 0], bins=40, density=True, color='green', alpha=0.7)
    axs[3].set_title("Density Profile (X-Axis)")
    axs[3].set_xlabel("X Position (mm)")
    axs[3].set_ylabel("Normalized Density")
    axs[3].grid(True, axis='y')
    
    fig.suptitle("Figure 2: Spatial and Confinement Analysis", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('figure2_spatial_confinement.png')
    plt.close(fig)
    print("Saved 'figure2_spatial_confinement.png'")


def plot_phase_space_analysis(final_positions, final_velocities):
    """
    Figure 3: Plots phase-space profiles for kinetic verification.
    """
    plt.style.use('default')
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    pos_mm = final_positions * 1e3 # mm

    # 1. Phase-Space Plot (x vs. vx) - Checks for damping
    axs[0].scatter(pos_mm[:, 0], final_velocities[:, 0], s=5, alpha=0.5, color='teal')
    axs[0].set_title("Phase-Space Profile ($x$ vs. $v_x$) - Damping Check")
    axs[0].set_xlabel("Position $x$ (mm)")
    axs[0].set_ylabel("Velocity $v_x$ (m/s)")
    axs[0].grid(True)
    
    # 2. Velocity Distribution Histogram (vx) - Checks for cooled state
    axs[1].hist(final_velocities[:, 0], bins=40, density=True, color='purple', alpha=0.7)
    axs[1].set_title("Final Velocity Distribution ($v_x$)")
    axs[1].set_xlabel("Velocity $v_x$ (m/s)")
    axs[1].set_ylabel("Normalized Probability Density")
    axs[1].grid(True, axis='y')
    
    fig.suptitle("Figure 3: Phase-Space and Kinetic Analysis", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('figure3_phase_space.png')
    plt.close(fig)
    print("Saved 'figure3_phase_space.png'")


def create_animation_matplotlib(pos_history, time_interval):
    """
    Generates the GIF using Matplotlib's FuncAnimation, bypassing file I/O errors.
    """
    print("\nStarting GIF generation in memory via Matplotlib...")
    
    # 1. Setup figure and determine limits
    fig, ax = plt.subplots(figsize=(7, 7))
    
    all_positions_m = np.concatenate(pos_history, axis=0)
    max_pos_m = np.max(np.abs(all_positions_m)) * 1.05 
    max_pos_mm = max_pos_m * 1e3
    
    # Initialize the scatter plot artist using the first frame
    pos_frame_0_mm = pos_history[0] * 1e3
    scat = ax.scatter(pos_frame_0_mm[:, 0], pos_frame_0_mm[:, 2], s=5, alpha=0.6, color='darkblue')

    ax.set_xlim(-max_pos_mm, max_pos_mm)
    ax.set_ylim(-max_pos_mm, max_pos_mm)
    ax.set_xlabel("X Position (mm)")
    ax.set_ylabel("Z Position (mm)")
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True)

    # Initialize dynamic title text
    time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=12, verticalalignment='top')
    
    # 2. Animation update function
    def update_frame(frame_index):
        """Updates the scatter plot positions for each frame."""
        positions_mm = pos_history[frame_index] * 1e3
        
        # Update the position data of the scatter plot artist
        scat.set_offsets(positions_mm[:, [0, 2]])
        
        current_time = frame_index * time_interval
        time_text.set_text(f"t = {current_time*1e3:.2f} ms")
        ax.set_title(f"Atom Cloud Evolution")
        
        # Return the artists that were updated
        return scat, time_text

    # 3. Create the animation object
    anim = animation.FuncAnimation(
        fig, 
        update_frame, 
        frames=len(pos_history), 
        interval=100, 
        blit=False, 
        repeat=False
    )
    
    # 4. Save the animation
    try:
        print(f"\nSaving GIF '{GIF_FILENAME}'...")
        anim.save(GIF_FILENAME, writer='pillow', fps=10) 
        print(f"Successfully created {GIF_FILENAME}.")
    except Exception as e:
        print(f"\nGIF creation failed using Matplotlib/Pillow writer.")
        print("Ensure the 'Pillow' library is installed: pip install pillow")
        print(f"Error details: {e}")

    plt.close(fig)


# --- Main Execution ---
if __name__ == '__main__':
    
    # --- STEP 0: CHECK FOR DATA FILE AND OFFER DOWNLOAD ---
    
    if not os.path.exists(DATA_FILENAME):
        print(f"The required data file '{DATA_FILENAME}' was not found locally.")
        
        # Check if the user is trying to download
        if len(sys.argv) > 1 and sys.argv[1].lower() == 'download':
            
            # Use the defined constants in the configuration block
            if CLUSTER_USER == "your_username" or CLUSTER_HOST == "cluster.server.edu":
                print("\nERROR: Please update CLUSTER_USER and CLUSTER_HOST variables at the top of the script first!")
                sys.exit(1)
            
            # Call the SCP function using the defined constants
            if not download_data_via_scp(CLUSTER_USER, CLUSTER_HOST, REMOTE_FILE_PATH, DATA_FILENAME):
                sys.exit(1) # Exit if download fails
        else:
            print("\nTo download data from your cluster, please first update the configuration variables")
            print("at the top of this script, then run:")
            print(">>> python mot_post_processor_improved.py download")
            sys.exit(1) # Exit if data is missing and download command isn't used
    
    # --- STEP 1: LOAD DATA ---

    sim_data = load_simulation_data(DATA_FILENAME)
    
    if sim_data is not None:
        # Load data histories
        time_hist = sim_data['time']
        r_rms_hist = sim_data['r_rms']
        v_rms_hist = sim_data['v_rms']
        e_k_hist = sim_data['e_k'] 
        pos_frames = sim_data['positions']
        vel_frames = sim_data['velocities']
        
        dt = sim_data['dt'].item() 
        save_interval = sim_data['save_interval'].item()
        time_interval_frame = dt * save_interval

        # 2. Figure 1: Scalar Time History
        plot_scalar_history_improved(time_hist, r_rms_hist, v_rms_hist, e_k_hist)
        
        # 3. Figure 2: Spatial Confinement Analysis
        plot_spatial_confinement_analysis(pos_frames[-1])
        
        # 4. Figure 3: Phase Space and Kinetic Analysis
        plot_phase_space_analysis(pos_frames[-1], vel_frames[-1])
        
        # 5. Generate GIF
        create_animation_matplotlib(pos_frames, time_interval_frame)