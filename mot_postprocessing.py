import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import sys
import re
from typing import Optional

# --- CONFIGURATION & CONSTANTS ---
BASE_DATA_FILENAME = 'mot_data_full.npz' 
GIF_FILENAME = 'mot_evolution.gif'

# Default Constants (overwritten by load_sim_parameters)
DEFAULT_MASS = 1.443160e-25  # kg (Rb-87)
DEFAULT_KB = 1.380649e-23  # J/K
JOULE_TO_EV = 6.242e+18

# Global placeholders
MASS_ATOM = DEFAULT_MASS
K_BOLTZMANN = DEFAULT_KB

# --- UTILITY: FOLDER NAME GENERATION ---
def generate_folder_name(N, B_prime, T_res_uK, R_res):
    N_str = f"{N:.1e}"
    B_str = f"{B_prime:.1e}"
    T_res_str = f"{T_res_uK:.1e}"
    R_res_str = f"{R_res:.1e}"
    return f"Results/res_N={N_str}_B={B_str}T_T={T_res_str}uK_R={R_res_str}m"

# --- UTILITY: PARAMETER READING ---
def read_parameter_from_string(param_content: str, key_name: str) -> Optional[float]:
    pattern = re.compile(fr"^{re.escape(key_name)}:\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)\s*", re.MULTILINE)
    match = pattern.search(param_content)
    return float(match.group(1)) if match else None

def load_sim_parameters(folder_path: str):
    global MASS_ATOM, K_BOLTZMANN
    param_file_path = os.path.join(folder_path, 'parameters.txt')
    if os.path.exists(param_file_path):
        try:
            with open(param_file_path, 'r') as f:
                content = f.read()
            m_read = read_parameter_from_string(content, "Atom mass (m_atom)")
            if m_read: MASS_ATOM = m_read
            kb_read = read_parameter_from_string(content, "Boltzmann constant (k_B)")
            if kb_read: K_BOLTZMANN = kb_read
            print(f"Dynamically set Atom Mass: {MASS_ATOM:.2e} kg")
            print(f"Dynamically set Boltzmann Constant: {K_BOLTZMANN:.2e} J/K")
        except Exception:
            print("Warning: Failed to read parameters file. Using defaults.")
    else:
        print("Info: Parameters file not found. Using defaults.")

# --- UTILITY: LOAD DATA ---
def load_simulation_data(filename):
    try:
        data = np.load(filename, allow_pickle=True)
        print(f"Successfully loaded data from {filename}")
        return data
    except FileNotFoundError:
        print(f"Error: Data file '{filename}' not found.")
        return None

# --- CALCULATION UTILITIES ---
def compute_rms_history(pos_frames: np.ndarray) -> np.ndarray:
    r_sq = np.sum(pos_frames**2, axis=2)
    return np.sqrt(np.mean(r_sq, axis=1))

def compute_vrms_history(vel_frames: np.ndarray) -> np.ndarray:
    v_sq = np.sum(vel_frames**2, axis=2)
    return np.sqrt(np.mean(v_sq, axis=1))

def compute_ek_history(v_rms_hist: np.ndarray, n_atoms: int, m_atom: float) -> np.ndarray:
    return 0.5 * n_atoms * m_atom * v_rms_hist**2

# --- PLOTTING FUNCTIONS ---
def plot_scalar_history_improved(time, r_rms, v_rms, e_k, output_folder):
    plt.style.use('ggplot')
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    time_ms = time * 1e3

    axs[0].plot(time_ms, r_rms * 1e3, color='darkblue', alpha=0.8)
    axs[0].set_title("Cloud Confinement (RMS Radius)")
    axs[0].set_xlabel("Time (ms)")
    axs[0].set_ylabel(r"$R_{RMS}$ (mm)")
    axs[0].grid(True, linestyle='--')
    axs[0].ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    axs[1].plot(time_ms, v_rms, color='red', alpha=0.8)
    axs[1].set_title("Cloud Cooling (RMS Velocity)")
    axs[1].set_xlabel("Time (ms)")
    axs[1].set_ylabel(r"$V_{RMS}$ (m/s)")
    axs[1].grid(True, linestyle='--')
    
    fig.suptitle("Figure 1: Time Evolution", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_folder, 'figure1_scalar_history.png'))
    plt.close(fig)
    print(f"Saved 'figure1_scalar_history.png' to {output_folder}")

def plot_spatial_confinement_analysis(final_positions, output_folder):
    plt.style.use('default')
    fig, axs = plt.subplots(2, 2, figsize=(14, 12))
    axs = axs.flatten()
    pos_mm = final_positions * 1e3 
    
    max_extent = np.max(np.abs(final_positions)) * 1e3 * 1.05
    axs[0].scatter(pos_mm[:, 0], pos_mm[:, 2], s=5, alpha=0.5, color='darkorange')
    axs[0].set_title("Spatial Distribution (Z-X)")
    axs[0].set_xlabel("X (mm)"); axs[0].set_ylabel("Z (mm)")
    axs[0].set_xlim(-max_extent, max_extent); axs[0].set_ylim(-max_extent, max_extent)
    axs[0].set_aspect('equal', adjustable='box')

    rms_x = np.sqrt(np.mean(final_positions[:, 0]**2)) * 1e3
    rms_y = np.sqrt(np.mean(final_positions[:, 1]**2)) * 1e3
    rms_z = np.sqrt(np.mean(final_positions[:, 2]**2)) * 1e3
    axs[1].bar(['$R_x$', '$R_y$', '$R_z$'], [rms_x, rms_y, rms_z], color=['gray', 'gray', 'red'])
    axs[1].set_title("Dimensional RMS Radius")
    axs[1].set_ylabel("RMS (mm)")

    axs[2].hist(pos_mm[:, 2], bins=40, density=True, color='red', alpha=0.7)
    axs[2].set_title("Density Profile (Z)")
    axs[3].hist(pos_mm[:, 0], bins=40, density=True, color='green', alpha=0.7)
    axs[3].set_title("Density Profile (X)")
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_folder, 'figure2_spatial_confinement.png'))
    plt.close(fig)
    print(f"Saved 'figure2_spatial_confinement.png' to {output_folder}")

def plot_phase_space_analysis(final_positions, final_velocities, output_folder):
    plt.style.use('default')
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    pos_mm = final_positions * 1e3

    axs[0].scatter(pos_mm[:, 0], final_velocities[:, 0], s=5, alpha=0.5, color='teal')
    axs[0].set_title("Phase-Space (X vs Vx)")
    axs[0].set_xlabel("X (mm)"); axs[0].set_ylabel("Vx (m/s)")
    
    axs[1].hist(final_velocities[:, 0]*1e2, bins=40, density=True, color='purple', alpha=0.7)
    axs[1].set_title("Velocity Distribution (Vx)")
    axs[1].set_xlabel("Vx (cm/s)")
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_folder, 'figure3_phase_space.png'))
    plt.close(fig)
    print(f"Saved 'figure3_phase_space.png' to {output_folder}")

# --- GIF GENERATION (FIXED FOR CLUSTER) ---
def create_animation_matplotlib(pos_history, v_rms_history, N_atoms, time_interval, output_folder):
    """
    Generates GIF. Fixed for Python 3.6/Matplotlib 3.3 'list index out of range' error.
    """
    print("\nStarting GIF generation in memory via Matplotlib...")
    
    # 1. Determine common frame count
    n_frames = int(min(len(pos_history), len(v_rms_history)))
    if n_frames < 10:
        print("Skipping GIF: Not enough frames.")
        return

    # 2. Setup Figure
    plt.style.use('default')
    fig, axs = plt.subplots(1, 2, figsize=(14, 7))
    
    # Determine plot limits
    all_positions_m = np.concatenate(pos_history[:n_frames], axis=0)
    max_pos_mm = np.max(np.abs(all_positions_m)) * 1.05 * 1e3
    
    # Initial Scatter
    pos_0 = pos_history[0] * 1e3
    scat_zx = axs[0].scatter(pos_0[:, 0], pos_0[:, 2], s=5, alpha=0.6, color='darkblue')
    axs[0].set_xlim(-max_pos_mm, max_pos_mm); axs[0].set_ylim(-max_pos_mm, max_pos_mm)
    axs[0].set_title("Z-X Plane"); axs[0].set_xlabel("X (mm)"); axs[0].set_ylabel("Z (mm)")
    
    scat_zy = axs[1].scatter(pos_0[:, 1], pos_0[:, 2], s=5, alpha=0.6, color='darkred')
    axs[1].set_xlim(-max_pos_mm, max_pos_mm); axs[1].set_ylim(-max_pos_mm, max_pos_mm)
    axs[1].set_title("Z-Y Plane"); axs[1].set_xlabel("Y (mm)"); axs[1].set_ylabel("Z (mm)")

    # Text overlays
    time_text = fig.text(0.5, 0.95, '', ha='center', fontsize=14)
    vrms_text = fig.text(0.98, 0.90, '', ha='right', fontsize=12, color='red')
    temp_text = fig.text(0.98, 0.85, '', ha='right', fontsize=12, color='darkgreen')
    
    # 3. Update Function (NO RETURN STATEMENT to fix older MPL bug)
    def update_frame(frame_index):
        if frame_index >= n_frames: return
        
        # Data for this frame
        pos_mm = pos_history[frame_index] * 1e3
        v_rms = v_rms_history[frame_index]
        temp = (MASS_ATOM * v_rms**2) / (3 * K_BOLTZMANN)

        # Update Artists
        scat_zx.set_offsets(pos_mm[:, [0, 2]])
        scat_zy.set_offsets(pos_mm[:, [1, 2]])
        time_text.set_text(f"t = {frame_index * time_interval * 1e3:.1f} ms")
        vrms_text.set_text(f"Vrms: {v_rms*1e2:.2f} cm/s")
        temp_text.set_text(f"T: {temp*1e6:.1f} uK")
        
        # IMPORTANT: Do NOT return anything when blit=False in older Matplotlib

    # 4. Create & Save
    anim = animation.FuncAnimation(fig, update_frame, frames=range(n_frames), interval=100, blit=False)
    
    try:
        save_path = os.path.join(output_folder, GIF_FILENAME)
        print(f"Saving GIF to {output_folder}...")
        anim.save(save_path, writer='pillow', fps=10)
        print("GIF creation successful.")
    except Exception as e:
        print(f"GIF creation failed: {e}")
    finally:
        plt.close(fig)

# --- VIRIAL ANALYSIS (FIXED) ---
def analyze_virial_theorem(path_input, start_fraction=0.5):
    """
    Analyzes simulation data to test the Virial Theorem.
    """
    from mot_simulation import calculate_magnetic_field_vectorized, calculate_attenuation_map, calculate_trapping_force_full
    
    if os.path.isdir(path_input):
        npz_file_path = os.path.join(path_input, "mot_data_full.npz")
    else:
        npz_file_path = path_input

    if not os.path.exists(npz_file_path):
        return

    print(f"Loading data for Virial Analysis...")
    data = np.load(npz_file_path)
    time = data['time']
    positions_hist = data['positions']
    velocities_hist = data['velocities']
    
    n_frames = len(time)
    start_frame = int(n_frames * start_fraction)
    print(f"Analyzing frames {start_frame} to {n_frames} (Steady State)...")
    
    T_hist, W_hist, R_hist, t_hist = [], [], [], []

    try:
        # Retrieve B_prime from folder name if possible, or pass it. 
        # Here we attempt to infer B_prime from folder string if needed, 
        # but better to assume standard physics function signature.
        # Assuming calculate_magnetic_field_vectorized(pos) handles the field.
        # If it needs B_prime passed explicitly, we need to extract it.
        # However, typically 'mot_simulation' has B_prime as a global or default.
        # If your physics engine relies on a global B_prime, ensure it's set in mot_simulation.py
        
        for i in range(start_frame, n_frames):
            pos = positions_hist[i]
            vel = velocities_hist[i]
            
            # 1. Kinetic Energy (Using global MASS_ATOM)
            total_T = 0.5 * MASS_ATOM * np.sum(np.sum(vel**2, axis=1))
            
            # 2. Virial Term (W)
            # Calculate forces (Assumes these functions are imported from mot_simulation)
            B_mag, B_factor = calculate_magnetic_field_vectorized(pos)
            I_att_map, p_map = calculate_attenuation_map(pos, vel, B_mag, B_factor)
            F_trap, _ = calculate_trapping_force_full(pos, vel, B_mag, B_factor, I_att_map, p_map)
            
            total_W = np.sum(np.sum(pos * F_trap, axis=1))
            
            ratio = -total_W / (2 * total_T) if total_T > 0 else 0
            
            T_hist.append(total_T)
            W_hist.append(total_W)
            R_hist.append(ratio)
            t_hist.append(time[i])

        # Plotting
        avg_ratio = np.mean(R_hist)
        print(f"\n--- Virial Results ---\nMean Ratio: {avg_ratio:.4f} (Ideal=1.0)")
        
        plt.style.use('default')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        ax1.plot(np.array(t_hist)*1e3, np.array(T_hist)*JOULE_TO_EV, 'b-', label='T (Kinetic)')
        ax1.plot(np.array(t_hist)*1e3, -0.5*np.array(W_hist)*JOULE_TO_EV, 'r--', label='-0.5*W (Virial)')
        ax1.legend(); ax1.set_ylabel("Energy (eV)"); ax1.set_title("Virial Theorem")
        
        ax2.plot(np.array(t_hist)*1e3, R_hist, 'k-')
        ax2.axhline(1.0, color='g', linestyle='--')
        ax2.set_ylabel("Ratio"); ax2.set_xlabel("Time (ms)"); ax2.set_ylim(0, 2.0)
        
        plt.savefig(os.path.join(output_folder, 'Virial.png'))
        plt.close(fig)
        
    except Exception as e:
        print(f"Virial analysis skipped/failed: {e}")

# --- MAIN EXECUTION ---
if __name__ == '__main__':
    if len(sys.argv) < 5:
        print("Usage: python post_processor.py N B_prime T_res R_res [download]")
        sys.exit(1)

    try:
        N_atoms = int(sys.argv[1])
        B_prime = float(sys.argv[2])
        T_res_uK = float(sys.argv[3])
        R_res = float(sys.argv[4])
    except ValueError:
        print("Error: Invalid arguments."); sys.exit(1)

    folder_name = generate_folder_name(N_atoms, B_prime, T_res_uK, R_res)
    local_data_path = os.path.join(folder_name, BASE_DATA_FILENAME)

    if not os.path.exists(local_data_path):
        print(f"Data not found at {local_data_path}. Run with 'download' to fetch.")
        sys.exit(1)

    # Load globals
    load_sim_parameters(folder_name)
    
    # Load Data
    sim_data = load_simulation_data(local_data_path)
    
    if sim_data:
        # Extract
        time_hist = sim_data['time']
        pos_frames = sim_data['positions']
        vel_frames = sim_data['velocities']
        dt = sim_data['dt'].item()
        save_int = sim_data['save_interval'].item()
        
        # Compute derived stats
        r_rms = compute_rms_history(pos_frames)
        v_rms = compute_vrms_history(vel_frames)
        e_k = compute_ek_history(v_rms, N_atoms, MASS_ATOM)
        
        # Run Visualization
        plot_scalar_history_improved(time_hist, r_rms, v_rms, e_k, folder_name)
        plot_spatial_confinement_analysis(pos_frames[-1], folder_name)
        plot_phase_space_analysis(pos_frames[-1], vel_frames[-1], folder_name)
        
        # GIF
        create_animation_matplotlib(pos_frames, v_rms, N_atoms, dt*save_int, folder_name)
        
        # Virial
        # Update global B_prime in mot_simulation if needed (assuming logic handles it)
        import mot_simulation
        if hasattr(mot_simulation, 'B_PRIME_GRADIENT'):
             mot_simulation.B_PRIME_GRADIENT = B_prime
             
        analyze_virial_theorem(folder_name)
