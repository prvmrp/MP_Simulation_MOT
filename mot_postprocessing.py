import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import shutil
import glob
import subprocess
import sys
import re
from typing import Optional, Union
from simulation import *

# Try to import imageio for GIF creation, but allow the script to run without it
try:
    import imageio.v2 as iio
    IMAGEIO_AVAILABLE = True
except ImportError:
    IMAGEIO_AVAILABLE = False

# --- CONFIGURATION & CONSTANTS ---
BASE_DATA_FILENAME = 'mot_data_full.npz' 
GIF_FILENAME = 'mot_evolution.gif'

# Default Constants (used if parameters file cannot be read)
DEFAULT_MASS = 1.443160e-25  # kg (e.g., Rb-87)
DEFAULT_KB = 1.380649e-23  # J/K
JOULE_TO_EV = 6.242e+18
# Variables to be set dynamically later
MASS_ATOM = DEFAULT_MASS
K_BOLTZMANN = DEFAULT_KB


# >>> USER-DEFINED CLUSTER SETTINGS (Simplified) <<<
CLUSTER_USER = "your_username"        # e.g., "leona"
CLUSTER_HOST = "cluster.server.edu" # e.g., "login.cluster.uni"
REMOTE_BASE_PATH = "/path/to/simulation/results" 


# --- PARAMETER READING UTILITIES (UNCHANGED) ---

def read_parameter_from_string(param_content: str, key_name: str) -> Optional[float]:
    """Reads a single numeric parameter from the parameters file content."""
    # Pattern to match the key name, colon, optional whitespace, and capture the float value
    pattern = re.compile(
        fr"^{re.escape(key_name)}:\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)\s*", 
        re.MULTILINE
    )
    
    match = pattern.search(param_content)
    
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            print(f"Warning: Could not convert extracted value '{match.group(1)}' to float for key '{key_name}'.")
            return None
    return None

def load_sim_parameters(folder_path: str):
    """Loads physical constants (mass, kB) from the local parameters file."""
    global MASS_ATOM, K_BOLTZMANN
    
    param_file_path = os.path.join(folder_path, 'parameters.txt')
    
    if os.path.exists(param_file_path):
        try:
            with open(param_file_path, 'r') as f:
                content = f.read()
            
            # Read Atomic Mass
            m_read = read_parameter_from_string(content, "Atom mass (m_atom)")
            if m_read is not None:
                MASS_ATOM = m_read
                print(f"Dynamically set Atom Mass: {MASS_ATOM:.2e} kg")
            
            # Read Boltzmann Constant
            kb_read = read_parameter_from_string(content, "Boltzmann constant (k_B)")
            if kb_read is not None:
                K_BOLTZMANN = kb_read
                print(f"Dynamically set Boltzmann Constant: {K_BOLTZMANN:.2e} J/K")
                
        except Exception as e:
            print(f"Warning: Failed to read constants from '{param_file_path}'. Using defaults. Error: {e}")
    else:
        print(f"Info: Parameters file not found at '{param_file_path}'. Using default constants.")

# --- UTILITY: FOLDER NAME GENERATION (UNCHANGED) ---

def generate_folder_name(N, B_prime, T_res_uK, R_res):
    """Reconstructs the folder name exactly as saved by the simulation script."""
    N_str = f"{N:.1e}"
    B_str = f"{B_prime:.1e}"
    T_res_str = f"{T_res_uK:.1e}"
    R_res_str = f"{R_res:.1e}"
    
    folder_name = f"Results/res_N={N_str}_B={B_str}T_T={T_res_str}uK_R={R_res_str}m"
    return folder_name

# --- SSH/SCP UTILITY (UNCHANGED) ---

def download_data_via_scp(user, host, remote_folder, local_filename=BASE_DATA_FILENAME):
    """Downloads the simulation data file from a remote cluster."""
    remote_file = os.path.join(remote_folder, BASE_DATA_FILENAME)
    remote_source = f"{user}@{host}:{remote_file}"
    local_target_dir = os.path.basename(remote_folder) 
    local_target_path = os.path.join(local_target_dir, local_filename) 
    
    os.makedirs(local_target_dir, exist_ok=True)
    
    print(f"\nAttempting to download data from: {remote_source}")
    
    scp_command = ['scp', remote_source, local_target_path]
    
    try:
        result = subprocess.run(scp_command, check=True, capture_output=True, text=True)
        print(f"Download successful. Data saved as '{local_target_path}'.")
        return local_target_path 
    except subprocess.CalledProcessError as e:
        print("\n--- SCP DOWNLOAD FAILED ---")
        print(f"Error executing command: {' '.join(scp_command)}")
        print(f"Reason: {e.stderr.strip()}")
        return None
    except FileNotFoundError:
        print("\n--- SCP DOWNLOAD FAILED ---")
        print("Error: The 'scp' command was not found. Ensure SSH client tools are installed.")
        return None

# --- UTILITY AND PLOTTING FUNCTIONS (FIXED) ---

def load_simulation_data(filename):
    """Loads all data saved by the simulation."""
    try:
        data = np.load(filename, allow_pickle=True)
        print(f"Successfully loaded data from {filename}")
        return data
    except FileNotFoundError:
        print(f"Error: Data file '{filename}' not found locally. Please download it first.")
        return None

# NEW/FIXED RMS CALCULATION UTILITIES
def compute_rms_history(pos_frames: np.ndarray) -> np.ndarray:
    """Computes R_RMS for each frame in the position history."""
    # pos_frames shape is (n_frames, n_atoms, 3)
    # R^2 = x^2 + y^2 + z^2
    r_sq = np.sum(pos_frames**2, axis=2) # Sum x^2+y^2+z^2 for each atom
    # R_RMS = sqrt(mean(R^2)) over all atoms
    r_rms = np.sqrt(np.mean(r_sq, axis=1)) 
    return r_rms

def compute_vrms_history(vel_frames: np.ndarray) -> np.ndarray:
    """Computes V_RMS for each frame in the velocity history."""
    # V^2 = vx^2 + vy^2 + vz^2
    v_sq = np.sum(vel_frames**2, axis=2) # Sum vx^2+vy^2+vz^2 for each atom
    # V_RMS = sqrt(mean(V^2)) over all atoms
    v_rms = np.sqrt(np.mean(v_sq, axis=1)) 
    return v_rms

def compute_ek_history(v_rms_hist: np.ndarray, n_atoms: int, m_atom: float) -> np.ndarray:
    """Computes Total Kinetic Energy (E_k) history."""
    # E_k = 0.5 * N_atoms * m_atom * V_rms^2
    ek_hist = 0.5 * n_atoms * m_atom * v_rms_hist**2
    return ek_hist

def analyze_virial_theorem(path_input, start_fraction=0.5):
    """
    Analyzes simulation data to test the Virial Theorem.
    
    Args:
        path_input (str): Path to the .npz data file OR the result folder.
        start_fraction (float): Fraction of data to skip (0.0 to 1.0) to 
                                ensure we only analyze the equilibrium state.
    """
    # --- HANDLE FILE PATH VS FOLDER PATH ---
    if os.path.isdir(path_input):
        # User passed a folder, append default filename
        npz_file_path = os.path.join(path_input, "mot_data_full.npz")
    else:
        # User likely passed the full file path
        npz_file_path = path_input

    if not os.path.exists(npz_file_path):
        raise FileNotFoundError(f"Could not find data file at: {npz_file_path}")

    print(f"Loading data from {npz_file_path}...")
    data = np.load(npz_file_path)
    
    # Extract arrays
    time = data['time']
    positions_hist = data['positions']   # Shape: (n_frames, N_atoms, 3)
    velocities_hist = data['velocities'] # Shape: (n_frames, N_atoms, 3)
    
    n_frames = len(time)
    start_frame = int(n_frames * start_fraction)
    
    print(f"Analyzing frames {start_frame} to {n_frames} (Steady State Analysis)...")
    
    # Storage for time-series data
    T_history = []  # Total Kinetic Energy
    W_history = []  # Virial Clausius Term
    ratios = []
    analyzed_time = []

    # Iterate through saved frames
    for i in range(start_frame, n_frames):
        pos = positions_hist[i]
        vel = velocities_hist[i]
        t = time[i]
        
        # 1. Calculate Total Kinetic Energy (T)
        # Sum of 1/2 * m * v^2 for all atoms
        # Note: m_atom is the Superparticle mass
        v_sq = np.sum(vel**2, axis=1)
        total_T = 0.5 * m_atom * np.sum(v_sq)
        
        # 2. Re-calculate Conservative Trapping Forces (F)
        # We MUST NOT include F_diffusion here, as Virial checks mechanical equilibrium.
        # We need to call your physics engine:
        
        # A. Get B-field
        B_mag, B_factor = calculate_magnetic_field_vectorized(pos)
        
        # B. Get Attenuation (if your model uses it)
        I_att_map, p_map = calculate_attenuation_map(pos, vel, B_mag, B_factor)
        
        # C. Get Trapping Force
        F_trap, _ = calculate_trapping_force_full(pos, vel, B_mag, B_factor, I_att_map, p_map)
        
        # 3. Calculate Virial Term (W)
        # W = Sum( r_i . F_i )
        # Dot product of position and force vectors for each atom
        dot_products = np.sum(pos * F_trap, axis=1)
        total_W = np.sum(dot_products)
        
        # 4. Calculate Virial Ratio
        # Theorem: 2*T + W = 0  => Ratio = -W / (2T) should be ~1.0
        if total_T > 0:
            ratio = -total_W / (2 * total_T)
        else:
            ratio = 0
            
        T_history.append(total_T)
        W_history.append(total_W)
        ratios.append(ratio)
        analyzed_time.append(t)

    # Convert to arrays for plotting
    T_arr = np.array(T_history)
    W_arr = np.array(W_history)
    R_arr = np.array(ratios)
    t_arr = np.array(analyzed_time)
    
    # Calculate statistics
    avg_ratio = np.mean(R_arr)
    std_ratio = np.std(R_arr)

    print(f"\n--- Results ---")
    print(f"Mean Virial Ratio: {avg_ratio:.4f} (Ideal = 1.0)")
    print(f"Stability (Std Dev): {std_ratio:.4f}")

    # --- PLOTTING ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Plot 1: Energy Balance
    # We plot T and -0.5*W. If they overlap, Virial holds.
    ax1.plot(t_arr*1e3, T_arr * JOULE_TO_EV, 'b-', label='Kinetic Energy (T)', linewidth=1.5)
    ax1.plot(t_arr*1e3, -0.5 * W_arr * JOULE_TO_EV, 'r--', label='Virial Term (-0.5 * W)', linewidth=1.5)
    ax1.set_ylabel("Energy (eV)")
    ax1.set_title(f"Virial Theorem Test (N={positions_hist.shape[1]})")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: The Ratio
    ax2.plot(t_arr*1e3, R_arr, 'k-', label='Ratio Q = -W / 2T')
    ax2.axhline(1.0, color='g', linestyle='--', linewidth=2, label='Ideal (1.0)')
    ax2.fill_between(t_arr*1e3, 
                     avg_ratio - std_ratio, 
                     avg_ratio + std_ratio, 
                     color='gray', alpha=0.2, label='1-sigma fluctuation')
    
    ax2.set_ylabel("Virial Ratio")
    ax2.set_xlabel("Time (ms)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 2.0) # Focus on the relevant range
    
    plt.tight_layout()
    plt.savefig(path_input + '/Virial.png')

# --- Example Usage ---
# analyze_virial_theorem("Results/res_N=1.0e+04_B=1.0e-01T_T=1.0e+03uK_R=5.0e-04m")

# Replaced direct array loading with calculated history arrays
def plot_scalar_history_improved(time, r_rms, v_rms, e_k, output_folder):
    """
    Figure 1: Plots RMS radius, RMS velocity, and Total Kinetic Energy history.
    """
    plt.style.use('ggplot')
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    time_ms = time * 1e3

    # Plot 1: RMS Radius
    axs[0].plot(time_ms, r_rms * 1e3, color='darkblue', alpha=0.8)
    axs[0].set_title("Cloud Confinement (RMS Radius)")
    axs[0].set_xlabel("Time (ms)")
    axs[0].set_ylabel(r"$R_{RMS}$ (mm)")
    axs[0].grid(True, linestyle='--')
    axs[0].ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    # Plot 2: RMS Velocity
    axs[1].plot(time_ms, v_rms, color='red', alpha=0.8)
    axs[1].set_title("Cloud Cooling (RMS Velocity)")
    axs[1].set_xlabel("Time (ms)")
    axs[1].set_ylabel(r"$V_{RMS}$ (m/s)")
    axs[1].grid(True, linestyle='--')
    
    fig.suptitle("Figure 1: Time Evolution (Cooling, Confinement, and Stability)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    save_path = os.path.join(output_folder, 'figure1_scalar_history.png')
    plt.savefig(save_path)
    plt.close(fig) 
    print(f"Saved 'figure1_scalar_history.png' to {output_folder}")


def plot_spatial_confinement_analysis(final_positions, output_folder):
    """
    Figure 2: Plots spatial distribution, density profiles, and dimensional RMS.
    """
    plt.style.use('default')
    fig, axs = plt.subplots(2, 2, figsize=(14, 12))
    axs = axs.flatten()

    pos_mm = final_positions * 1e3 
    
    # Calculate dimensional RMS from final positions
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
    """
    plt.style.use('default')
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    pos_mm = final_positions * 1e3 # mm

    axs[0].scatter(pos_mm[:, 0], final_velocities[:, 0], s=5, alpha=0.5, color='teal')
    axs[0].set_title("Phase-Space Profile ($x$ vs. $v_x$) - Damping Check")
    axs[0].set_xlabel("Position $x$ (mm)")
    axs[0].set_ylabel("Velocity $v_x$ (m/s)")
    axs[0].grid(True)
    
    axs[1].hist(final_velocities[:, 0]*1e2, bins=40, density=True, color='purple', alpha=0.7)
    axs[1].set_title("Final Velocity Distribution ($v_x$)")
    axs[1].set_xlabel("Velocity $v_x$ (cm/s)")
    axs[1].set_ylabel("Normalized Probability Density")
    axs[1].grid(True, axis='y')
    
    fig.suptitle("Figure 3: Phase-Space and Kinetic Analysis", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    save_path = os.path.join(output_folder, 'figure3_phase_space.png')
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved 'figure3_phase_space.png' to {output_folder}")


# FIXED: Relies on global MASS_ATOM/K_BOLTZMANN
def create_animation_matplotlib(pos_history, v_rms_history, N_atoms, time_interval, output_folder):
    """
    Generates the GIF using Matplotlib's FuncAnimation, showing Z-X and Z-Y plots,
    along with instantaneous V_rms and Temperature (calculated from V_rms).
    """
    print("\nStarting GIF generation in memory via Matplotlib...")
    
    # 1. Setup figure and determine limits 
    plt.style.use('default')
    fig, axs = plt.subplots(1, 2, figsize=(14, 7))
    
    all_positions_m = np.concatenate(pos_history, axis=0)
    max_pos_m = np.max(np.abs(all_positions_m)) * 1.05 
    max_pos_mm = max_pos_m * 1e3
    pos_frame_0_mm = pos_history[0] * 1e3
    
    # --- Left Plot (Z-X) ---
    scat_zx = axs[0].scatter(pos_frame_0_mm[:, 0], pos_frame_0_mm[:, 2], s=5, alpha=0.6, color='darkblue')
    axs[0].set_xlim(-max_pos_mm, max_pos_mm)
    axs[0].set_ylim(-max_pos_mm, max_pos_mm)
    axs[0].set_xlabel("X Position (mm)")
    axs[0].set_ylabel("Z Position (mm)")
    axs[0].set_aspect('equal', adjustable='box')
    axs[0].set_title("Atom Cloud Evolution (Z-X Plane)")
    axs[0].grid(True)
    
    # --- Right Plot (Z-Y) ---
    scat_zy = axs[1].scatter(pos_frame_0_mm[:, 1], pos_frame_0_mm[:, 2], s=5, alpha=0.6, color='darkred')
    axs[1].set_xlim(-max_pos_mm, max_pos_mm)
    axs[1].set_ylim(-max_pos_mm, max_pos_mm)
    axs[1].set_xlabel("Y Position (mm)")
    axs[1].set_ylabel("Z Position (mm)")
    axs[1].set_aspect('equal', adjustable='box')
    axs[1].set_title("Atom Cloud Evolution (Z-Y Plane)")
    axs[1].grid(True)
    
    # Shared time display
    fig.suptitle("Atom Cloud Evolution: Two Cross-Sections", fontsize=16)
    time_text = fig.text(0.5, 0.95, '', transform=fig.transFigure, ha='center', fontsize=14, verticalalignment='top')
    
    # --- V_RMS and Temperature Text Objects (Upper Right Corner) ---
    vrms_text = fig.text(0.98, 0.90, '', transform=fig.transFigure, 
                         ha='right', fontsize=12, color='red', 
                         bbox=dict(boxstyle="square,pad=0.3", fc="white", alpha=0.6, ec="red"))
                         
    temp_text = fig.text(0.98, 0.85, '', transform=fig.transFigure, 
                         ha='right', fontsize=12, color='darkgreen', 
                         bbox=dict(boxstyle="square,pad=0.3", fc="white", alpha=0.6, ec="darkgreen"))
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.9])
    
    
    # 2. Animation update function (Now correct and efficient)
    def update_frame(frame_index):
        # Access constants via global scope
        global MASS_ATOM, K_BOLTZMANN
        
        positions_mm = pos_history[frame_index] * 1e3
        current_vrms = v_rms_history[frame_index]
        
        # CORRECT TEMP CALCULATION: T = m * V_rms^2 / (3 * k_B)
        current_temp = (DEFAULT_MASS * current_vrms**2) / (3 * K_BOLTZMANN)
        
        # Update plot data
        scat_zx.set_offsets(positions_mm[:, [0, 2]])
        scat_zy.set_offsets(positions_mm[:, [1, 2]])
        
        # Update Time
        current_time = frame_index * time_interval
        time_text.set_text(f"t = {current_time*1e3:.2f} ms")
        
        # Update V_RMS and Temperature (displayed in millikelvin)
        vrms_text.set_text(f"$V_{{\\text{{rms}}}}$: {current_vrms*1e2:.2f} cm/s")
        temp_text.set_text(f"$T$: {current_temp*1e6:.2f} uK") 
        
        # Return all artists that were modified
        return scat_zx, scat_zy, time_text, vrms_text, temp_text 

    # 3. Create the animation object
    anim = animation.FuncAnimation(
        fig, update_frame, frames=len(pos_history), 
        interval=100, blit=False, repeat=False
    )
    
    # 4. Save the animation
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

# --- Main Execution (FULLY CORRECTED) ---
if __name__ == '__main__':
    
    # --- STEP 0: PARSE ARGUMENTS AND DETERMINE FILE PATH ---
    
    if len(sys.argv) < 5 or (len(sys.argv) == 5 and sys.argv[1].lower() == 'download'):
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

    # 1. Construct the folder name and paths
    folder_name = generate_folder_name(N_atoms, B_prime, T_res_uK, R_res)
    local_data_path = os.path.join(folder_name, BASE_DATA_FILENAME)
    OUTPUT_FOLDER_PATH = folder_name
    
    # 2. Check for local data, attempt download if requested
    if not os.path.exists(local_data_path):
        print(f"Required data file '{local_data_path}' not found locally.")
        
        if len(sys.argv) == 6 and sys.argv[5].lower() == 'download':
            if CLUSTER_USER == "your_username" or CLUSTER_HOST == "cluster.server.edu":
                print("\nERROR: Please update CLUSTER_USER and CLUSTER_HOST variables at the top of the script first!")
                sys.exit(1)
                
            remote_folder = os.path.join(REMOTE_BASE_PATH, folder_name)
            downloaded_path = download_data_via_scp(CLUSTER_USER, CLUSTER_HOST, remote_folder, BASE_DATA_FILENAME)
            
            if downloaded_path is None:
                sys.exit(1) 
        else:
            print(f"To download data for these parameters, rerun with the 'download' flag:")
            print(f">>> python post_processor.py {N_atoms} {B_prime} {T_res_uK} {R_res} download")
            sys.exit(1) 
            
    
    # --- STEP 2: LOAD CONSTANTS and SIMULATION DATA ---

    # Load physical constants from the parameters file (MASS_ATOM, K_BOLTZMANN)
    load_sim_parameters(OUTPUT_FOLDER_PATH)

    sim_data = load_simulation_data(local_data_path) 
    
    if sim_data is not None:
        # Load raw history arrays
        time_hist = sim_data['time']
        pos_frames = sim_data['positions']
        vel_frames = sim_data['velocities']
        
        # FIXED: Calculate R_RMS, V_RMS, and E_K histories from raw data
        # This prevents the 'KeyError' if the history arrays aren't in the NPZ file
        r_rms_hist = compute_rms_history(pos_frames)
        v_rms_hist = compute_vrms_history(vel_frames)
        e_k_hist = compute_ek_history(v_rms_hist, N_atoms, MASS_ATOM)
        
        dt = sim_data['dt'].item() 
        save_interval = sim_data['save_interval'].item()
        time_interval_frame = dt * save_interval

        # 3. Figure 1: Scalar Time History (Now using calculated history arrays)
        plot_scalar_history_improved(time_hist, r_rms_hist, v_rms_hist, e_k_hist, OUTPUT_FOLDER_PATH)
        
        # 4. Figure 2: Spatial Confinement Analysis
        plot_spatial_confinement_analysis(pos_frames[-1], OUTPUT_FOLDER_PATH)
        
        # 5. Figure 3: Phase Space and Kinetic Analysis
        plot_phase_space_analysis(pos_frames[-1], vel_frames[-1], OUTPUT_FOLDER_PATH)
        
        # 6. Generate GIF
        create_animation_matplotlib(pos_frames, v_rms_hist, N_atoms, time_interval_frame, OUTPUT_FOLDER_PATH)

        analyze_virial_theorem(OUTPUT_FOLDER_PATH)