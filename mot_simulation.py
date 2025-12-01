import numpy as np
from sys import argv
import os

# --- 1. PHYSICAL CONSTANTS AND MOT PARAMETERS ---
hbar = 1.054e-34  # Reduced Planck constant (J*s)
c = 3.0e8         # Speed of light (m/s)
lambda_L = 780e-9 # Example laser wavelength (m)
kL = 2 * np.pi / lambda_L # Laser wave number (1/m)
Gamma = 2 * np.pi * 6.0e6 # Natural linewidth (rad/s)
sigma0_single = 6 * np.pi / kL**2 # Resonant scattering cross section (m^2)
m_atom_single = 1.41e-25 # Mass of a Rubidium atom (kg)
k_B = 1.380649e-23

# Define quantitites for a Superparticle
N_super = 7e3 # N-atoms for each superparticle
sigma0 = sigma0_single * N_super
m_atom = m_atom_single * N_super

# MOT Parameters
B_prime = float(argv[2])     # Magnetic field gradient (T/m)
mu = 1.399e10     # Gyromagnetic ratio (Hz/T, placeholder for simplicity)
delta = -2.5 * Gamma # Laser detuning (rad/s)
Isat = 16.7       # Saturation intensity (W/m^2)
I_infinity = 0.5 * Isat # Beam intensity before entering the cloud (W/m^2)

# Simulation parameters (RENAMED to res_etc. where applicable)
N_atoms = int(argv[1]) # Number of atoms
dt = 3.0e-6
n_steps = 12000 
n_save = 200

SAVE_INTERVAL = n_steps // n_save
T_res_uK = float(argv[3]) # T_init in mK -> T_res_uK
T_res = T_res_uK * 1e-6 # Convert to K for physics
V_res = np.sqrt(k_B * T_res / m_atom_single) # V_init -> V_res
R_res = float(argv[4]) # R_init -> R_res

# Mapping for transitions (q) and beams (alpha)
Q_TRANSITIONS = {-1: 'sigma_minus', 0: 'pi', 1: 'sigma_plus'}
ALPHA_AXES = {0: 'x', 1: 'y', 2: 'z'}
BEAM_DIRECTIONS = {
    ('x', '+'): (0, 1), ('x', '-'): (0, -1),
    ('y', '+'): (1, 1), ('y', '-'): (1, -1),
    ('z', '+'): (2, 1), ('z', '-'): (2, -1),
}

# --- 2. CORE KINETIC MODEL FUNCTIONS (UNCHANGED) ---

def calculate_magnetic_field_vectorized(r_positions):
    """
    Computes B(r) magnitude and relevant B-field factors for all atoms.
    """
    x, y, z = r_positions.T
    
    # B(r) magnitude (Eq. 4)
    B_mag = B_prime * np.sqrt(x**2 + 0.25 * (y**2 + z**2))
    
    # Magnetic factor for polarization fractions (alpha' * B' / 2*B(r))
    B_factor = np.zeros_like(r_positions)
    B_factor[:, 0] = -2 * x / (2 * B_mag)    # x-axis factor (alpha'=2x)
    B_factor[:, 1] = y / (2 * B_mag)    # y-axis factor (alpha'=y)
    B_factor[:, 2] = z / (2 * B_mag) # z-axis factor (alpha'=z)
    
    # Handle division by zero at origin (use a tiny value)
    B_mag[B_mag == 0] = 1e-12 
    B_factor[B_mag == 1e-12] = 0
    B_factor *= B_prime
    
    if B_mag.shape[0] != N_atoms:
        raise ValueError(f"B_mag shape ({B_mag.shape[0]}) does not match N_atoms ({N_atoms}).")
    
    return B_mag, B_factor # (N_atoms,), (N_atoms, 3)

def calculate_polarization_fractions(B_factor):
    """
    Computes the 18 polarization fraction arrays P_alpha_pm_q (Eq. 4).
    """
    p_map = {}
    
    for (axis, sign), (idx, sgn_dir) in BEAM_DIRECTIONS.items():
        # B_alpha_factor = alpha' * B' / 2*B(r)
        B_alpha_factor = B_factor[:, idx]
        
        # Core terms for sigma+ and sigma- (Eq. 4)
        term_pos = 0.25 * (1 + sgn_dir * B_alpha_factor)**2
        term_neg = 0.25 * (1 - sgn_dir * B_alpha_factor)**2
        
        # Assign q=+ and q=- based on MOT helicity convention:
        if axis == 'x':
            # X-axis is right-handed: + is sigma+, - is sigma-
            p_map[(axis, sign, 1)] = term_pos   # sigma+ (q=1)
            p_map[(axis, sign, -1)] = term_neg  # sigma- (q=-1)
        else:
            # Y/Z axes are left-handed (helicity is reversed):
            p_map[(axis, sign, 1)] = term_neg   # sigma+ (q=1)
            p_map[(axis, sign, -1)] = term_pos  # sigma- (q=-1)
            
        # Pi transition (q=0) (Eq. 4)
        p_map[(axis, sign, 0)] = 1.0 - (p_map[(axis, sign, 1)] + p_map[(axis, sign, -1)])
        
    return p_map # Keys: (axis, sign, q) -> (N_atoms,) array

def calculate_scattering_cross_sections(positions, velocities, B_mag, B_factor, I_att_map):
    """
    Computes all 18 scattering cross sections sigma_alpha_pm_q(r, v) (Eq. 5) 
    and the total intensity per transition I_tot_q (Eq. 6).
    """
    p_map = calculate_polarization_fractions(B_factor)
    sigma_map = {}
    I_tot_q_map = {q: np.zeros(N_atoms) for q in Q_TRANSITIONS}
    
    for (axis, sign), (idx, sgn_dir) in BEAM_DIRECTIONS.items():
        I_alpha_pm = I_att_map[(axis, sign)] # Attenuated Intensity (N_atoms,)
        
        v_alpha = velocities[:, idx]
        kLv_alpha = kL * v_alpha
        Doppler_shift = -sgn_dir * kLv_alpha # +/- kL * v_alpha
        
        for q in Q_TRANSITIONS:
            # Zeeman shift: mu * B(r) * q
            Zeeman_shift = mu * B_mag * q 
            
            # Total detuning (rad/s)
            detuning_q = delta + Doppler_shift + Zeeman_shift
            
            # Non-iterative cross section approximation (I_tot_q in denominator removed)
            # sigma_0 / (1 + 4 * detuning_q^2 / Gamma^2)
            sigma_alpha_pm_q = sigma0 / (1 + (4 * detuning_q**2) / Gamma**2)
            
            sigma_map[(axis, sign, q)] = sigma_alpha_pm_q
            
            # Calculate I_tot_q (used for diffusion and final saturation checks) (Eq. 6)
            p_alpha_pm_q = p_map[(axis, sign, q)]
            I_tot_q_map[q] += p_alpha_pm_q * I_alpha_pm
            
    return sigma_map, I_tot_q_map, p_map

import numpy as np

def calculate_attenuation_map(positions, velocities, B_mag, B_factor, grid_res=1e-4):
    """
    Computes attenuation using a 2D Grid approach with PURE NUMPY vectorization.
    """
    I_att_map = {}
    bin_area = grid_res ** 2
    N_atoms = len(positions)
    
    # 1. Initial Guess (I = I_infinity)
    I_guess_map = {k: np.full(N_atoms, I_infinity) for k in BEAM_DIRECTIONS}
    
    # Calculate scattering cross-sections based on un-attenuated light
    sigma_map_guess, _, p_map = calculate_scattering_cross_sections(
        positions, velocities, B_mag, B_factor, I_guess_map
    )

    for (axis, sign), (idx, sgn_dir) in BEAM_DIRECTIONS.items():
        # --- A. Setup Coordinates ---
        # Identify transverse axes (e.g., if Beam is Z(2), transverse are X(0), Y(1))
        transverse_indices = [i for i in [0, 1, 2] if i != idx]
        u_coords = positions[:, transverse_indices[0]]
        v_coords = positions[:, transverse_indices[1]]
        w_coords = positions[:, idx] # Beam axis
        
        # --- B. Compute Total Sigma ---
        sigma_eff_total = np.zeros(N_atoms)
        for q in Q_TRANSITIONS:
            sigma_eff_total += p_map[(axis, sign, q)] * sigma_map_guess[(axis, sign, q)]
            
        # --- C. Binning and Sorting ---
        # 1. Discretize transverse positions into integer bins
        u_bins = np.floor(u_coords / grid_res).astype(np.int64)
        v_bins = np.floor(v_coords / grid_res).astype(np.int64)
        
        # 2. Create a unique hash for each bin to group them
        # Multiplier ensures uniqueness (assuming <1M bins in one direction)
        bin_hashes = u_bins * 1_000_000 + v_bins 
        
        # 3. Define Sort Order:
        # Primary Key: Bin Hash (Group atoms in the same tube)
        # Secondary Key: Beam Axis W (Order atoms upstream -> downstream)
        # Note: If sgn_dir is -1, we negate w_coords so "upstream" is always smaller index
        beam_sort_key = w_coords if sgn_dir == 1 else -w_coords
        
        # np.lexsort sorts by the last key passed, then the second to last...
        # So we pass (Secondary, Primary)
        sort_order = np.lexsort((beam_sort_key, bin_hashes))
        
        # Apply sort
        sigma_sorted = sigma_eff_total[sort_order]
        bins_sorted = bin_hashes[sort_order]
        
        # --- D. Vectorized Grouped CumSum (The Magic Step) ---
        
        # 1. Standard Cumulative Sum over the whole array
        global_cumsum = np.cumsum(sigma_sorted)
        
        # 2. Find indices where the bin changes
        # np.diff != 0 finds transitions. We append [0] to handle the first group.
        # 'starts' will be indices: [0, start_of_bin2, start_of_bin3...]
        change_points = np.flatnonzero(np.diff(bins_sorted)) + 1
        group_starts = np.concatenate(([0], change_points))
        
        # 3. Calculate "Offset" (The cumsum value just before a group starts)
        # For the first group (index 0), the offset is 0.
        # For group starting at 'i', offset is global_cumsum[i-1]
        offsets = np.zeros(len(group_starts))
        offsets[1:] = global_cumsum[group_starts[1:] - 1]
        
        # 4. Expand Offsets to match the full array size
        # We repeat the offset value N times, where N is the number of atoms in that bin
        group_lengths = np.diff(np.concatenate((group_starts, [N_atoms])))
        values_to_subtract = np.repeat(offsets, group_lengths)
        
        # 5. Calculate Local CumSum
        # (Global Sum) - (Sum of all previous groups)
        local_cumsum = global_cumsum - values_to_subtract
        
        # --- E. Calculate Optical Depth and Map Back ---
        
        # Subtract self-scattering (atoms don't shadow themselves)
        O_sorted = (local_cumsum - sigma_sorted) / bin_area
        
        # Unsort: Map values back to original particle indices
        O_depth_final = np.zeros(N_atoms)
        O_depth_final[sort_order] = O_sorted
        
        # Attenuation Eq
        I_att_map[(axis, sign)] = I_infinity * np.exp(-O_depth_final)

    return I_att_map, p_map

def calculate_trapping_force_full(positions, velocities, B_mag, B_factor, I_att_map, p_map):
    """
    Computes the total Trapping Force F_tr (Eq. 1-3).
    """
    
    # Re-calculate cross-sections using the final attenuated intensities I_att_map
    sigma_map, I_tot_q_map, _ = calculate_scattering_cross_sections(
        positions, velocities, B_mag, B_factor, I_att_map
    )
    
    F_tr = np.zeros((N_atoms, 3))
    
    for (axis, sign), (idx, sgn_dir) in BEAM_DIRECTIONS.items():
        I_alpha_pm = I_att_map[(axis, sign)]
        
        F_alpha_pm = np.zeros(N_atoms) # Force magnitude for this beam
        
        for q in Q_TRANSITIONS:
            # F_alpha_pm_q = +/- p_alpha_pm_q * I_alpha_pm * sigma_alpha_pm_q / c (Eq. 3)
            p_alpha_pm_q = p_map[(axis, sign, q)]
            sigma_alpha_pm_q = sigma_map[(axis, sign, q)]
            
            # The force direction (sgn_dir) is 1 for + beams, -1 for - beams.
            force_magnitude = sgn_dir * p_alpha_pm_q * I_alpha_pm * sigma_alpha_pm_q / c
            
            F_alpha_pm += force_magnitude
            
        F_tr[:, idx] += F_alpha_pm
        
    return F_tr, I_tot_q_map

def calculate_diffusion_force_full(I_tot_q_map):
    """
    Computes the Diffusion Force F_diff (Eq. 7-12).
    """
    
    # 1. Total Saturation Parameter s_tot (Eq. 10)
    s_tot_q = {q: I_tot / Isat for q, I_tot in I_tot_q_map.items()}
    s_tot = sum(s_tot_q.values()) # (N_atoms,) array (Eq. 10)
    
    # 2. Diffusion coefficients D_vac and D_las (Eq. 8, 9)
    s_denom = 1 + s_tot
    s_denom_sq = s_denom**2
    s_denom_cu = s_denom_sq * s_denom
    
    D_vac = (hbar * kL)**2 / 4 * s_tot / s_denom # (Eq. 8)
    
    # Simplified D_las (Approx. Eq. 9)
    D_las_numerator = (1 + 1/2 * (delta/Gamma)**2 * s_tot + 2 * s_tot * s_tot / s_denom)
    D_las = (hbar * kL)**2 / 4 * s_tot / s_denom_cu * D_las_numerator 
    
    D_tot = D_vac + D_las # (N_atoms,) array (Eq. 7)
    
    # 3. Diffusion Force (F_diff)
    random_noise = np.random.randn(D_tot.shape[0], 3)
    F_diff = np.sqrt(2 * D_tot / dt)[:, np.newaxis] * random_noise
    
    return F_diff

from scipy.fft import rfftn, irfftn
from scipy.interpolate import RegularGridInterpolator

def calculate_rescattering_force_grid(positions, I_tot_q_map, grid_size=32, padding=2.0):
    """
    Computes Rescattering Force using the Particle-Mesh (FFT) method.
    Math: Solves Poisson Eq: Del^2(Phi) = -source using Green's function G(k) = 1/k^2
    """
    N = len(positions)
    
    # --- 1. Calculate Source Power (Physics) ---
    # Total Intensity I_tot = Sum of intensities in all polarizations
    # (Approximation: treat scalar intensity for saturation)
    I_total_magnitude = sum(I_tot_q_map.values())
    
    # Saturation parameter s = I/Isat
    s_tot = I_total_magnitude / Isat
    
    # Excited State Fraction rho_ee (Steady state 2-level approx)
    detuning_term = 4 * (delta / Gamma)**2
    rho_ee = 0.5 * s_tot / (1 + s_tot + detuning_term)
    
    # Power P = (Photon Energy) * (Scattering Rate) * (N_super)
    photon_energy = hbar * c * kL
    P_source_vec = photon_energy * (Gamma * rho_ee) * N_super
    
    # Rescattering Cross Section sigma_R for the target atoms
    # Wieman approximation: sigma_R is a multiple of laser cross-section
    sigma_L_approx = sigma0_single / (1 + detuning_term)
    sigma_R = 1.5 * sigma_L_approx 
    
    # --- 2. Setup the Grid Geometry ---
    min_pos = np.min(positions, axis=0)
    max_pos = np.max(positions, axis=0)
    center = (max_pos + min_pos) / 2.0
    
    # Determine box size with padding to prevent periodic aliases
    cloud_width = np.max(max_pos - min_pos)
    if cloud_width == 0: cloud_width = 1e-3
    L_box = cloud_width * padding
    
    # Define grid coordinates (Cell centers)
    # Using 'ij' indexing for meshgrid later
    x_axis = np.linspace(center[0] - L_box/2, center[0] + L_box/2, grid_size)
    y_axis = np.linspace(center[1] - L_box/2, center[1] + L_box/2, grid_size)
    z_axis = np.linspace(center[2] - L_box/2, center[2] + L_box/2, grid_size)
    
    dx = L_box / (grid_size - 1)
    
    # --- 3. Mass Assignment (Binning) ---
    # Bin atoms into the 3D grid, summing their Power
    rho_grid, edges = np.histogramdd(
        positions, 
        bins=(grid_size, grid_size, grid_size), 
        range=[(x_axis[0], x_axis[-1]), (y_axis[0], y_axis[-1]), (z_axis[0], z_axis[-1])],
        weights=P_source_vec
    )
    
    # Note: rho_grid is technically "Power per cell". 
    # To get density, we would divide by dx^3, but we can handle constants at the end.

    # --- 4. Poisson Solve in Fourier Space ---
    # Forward FFT
    rho_k = rfftn(rho_grid)
    
    # Construct k-vectors (Frequency domain)
    # fftfreq gives standard order, we multiply by 2pi/L to get physical wavevectors
    kx = np.fft.fftfreq(grid_size, d=dx) * 2 * np.pi
    ky = np.fft.fftfreq(grid_size, d=dx) * 2 * np.pi
    kz = np.fft.rfftfreq(grid_size, d=dx) * 2 * np.pi # Real FFT uses half the z-spectrum
    
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    
    # The Green's Function 1/k^2
    K_sq = KX**2 + KY**2 + KZ**2
    
    # Avoid division by zero at DC component (k=0)
    # This represents the average constant potential, which is arbitrary in physics.
    K_sq[0, 0, 0] = 1.0 
    
    # Solve for Potential in k-space: Phi(k) = rho(k) / k^2
    # Note: We skipped the 4*pi*epsilon_0 constants, will add later.
    phi_k = rho_grid.size * (rho_k / K_sq) # rho_grid.size is scaling factor for FFT normalization
    phi_k[0, 0, 0] = 0.0
    
    # --- 5. Calculate Field (Gradient) in Fourier Space ---
    # E = -Gradient(Phi)  =>  E(k) = -i * k * Phi(k)
    # We compute the 3 components of the field directly in k-space
    Ex_k = -1j * KX * phi_k
    Ey_k = -1j * KY * phi_k
    Ez_k = -1j * KZ * phi_k
    
    # Inverse FFT to get real-space fields
    # We take the real part (imaginary part should be numerical noise)
    Ex_grid = irfftn(Ex_k, s=(grid_size, grid_size, grid_size)).real
    Ey_grid = irfftn(Ey_k, s=(grid_size, grid_size, grid_size)).real
    Ez_grid = irfftn(Ez_k, s=(grid_size, grid_size, grid_size)).real
    
    # --- 6. Interpolation (Force Assignment) ---
    # We map the field at grid points back to the atom positions
    
    # Create interpolators for each component
    # Use bounds_error=False to handle atoms slightly floating off due to float precision
    interp_x = RegularGridInterpolator((x_axis, y_axis, z_axis), Ex_grid, bounds_error=False, fill_value=0)
    interp_y = RegularGridInterpolator((x_axis, y_axis, z_axis), Ey_grid, bounds_error=False, fill_value=0)
    interp_z = RegularGridInterpolator((x_axis, y_axis, z_axis), Ez_grid, bounds_error=False, fill_value=0)
    
    Fx = interp_x(positions)
    Fy = interp_y(positions)
    Fz = interp_z(positions)
    
    F_field = np.stack((Fx, Fy, Fz), axis=1)
    
    # --- 7. Apply Physical Constants ---
    F_rescattering = F_field * (sigma_R / (4 * np.pi * c))
    
    # Important: The DFT assumes values are densities. 
    # We need to divide by dx^3 to get real density?
    # Actually, usually simpler to calibrate K using a 2-particle test case.
    # For this implementation, we assume standard FFT scaling.
    
    return F_rescattering

### 4. How to integrate this efficiently
# Global variable to store the last calculated rescattering force
CACHED_F_RESC = None
RESC_UPDATE_FREQ = 10  # Update rescattering every 10 steps

def get_forces_full(positions, velocities, step_number=0):
    global CACHED_F_RESC
    
    # 1. Standard Forces (Trapping + Attenuation)
    B_mag, B_factor = calculate_magnetic_field_vectorized(positions)
    I_att_map, p_map = calculate_attenuation_map(positions, velocities, B_mag, B_factor)
    F_tr, I_tot_q_map = calculate_trapping_force_full(positions, velocities, B_mag, B_factor, I_att_map, p_map)
    F_diff = calculate_diffusion_force_full(I_tot_q_map)
    
    # 2. Rescattering Force (The Heavy Calculation)
    # Only recalculate every N steps to save time
    if step_number % RESC_UPDATE_FREQ == 0 or CACHED_F_RESC is None:
        CACHED_F_RESC = calculate_rescattering_force_grid(
            positions, I_tot_q_map
        )
        
    return F_tr + F_diff + CACHED_F_RESC

# --- 3. VELOCITY VERLET INTEGRATOR FUNCTION (UNCHANGED) ---

def integrate_motion_verlet(positions, velocities, dt, m_atom, get_forces_func):
    """Performs one step of Velocity Verlet integration."""
    
    total_forces_n = get_forces_func(positions, velocities)
    accelerations_n = total_forces_n / m_atom

    velocities_half = velocities + accelerations_n * (dt / 2.0)

    positions_n1 = positions + velocities_half * dt
    
    total_forces_n1 = get_forces_func(positions_n1, velocities_half) 
    accelerations_n1 = total_forces_n1 / m_atom

    velocities_n1 = velocities_half + accelerations_n1 * (dt / 2.0)
    
    return positions_n1, velocities_n1

# -------------------------------------------------------------
## 💾 Data Saving Function (UPDATED)
# -------------------------------------------------------------

def save_simulation_data(time_history, pos_history, vel_history):
    """
    Creates a dedicated folder based on simulation parameters, writes a 
    parameters.txt file, and saves the history arrays to an NPZ file.
    """
    
    # Format parameters for a safe folder name: N_Bprime_T_res_R_res
    N_str = f"{N_atoms:.1e}"
    B_str = f"{B_prime:.1e}"
    T_res_str = f"{T_res_uK:.1e}"
    R_res_str = f"{R_res:.1e}"
    
    # Create the directory path (RENAMED to T_res and R_res in folder name)
    folder_name = f"Results/res_N={N_str}_B={B_str}T_T={T_res_str}uK_R={R_res_str}m"
    
    try:
        os.makedirs(folder_name, exist_ok=True)
    except Exception as e:
        print(f"Error creating directory {folder_name}: {e}")
        return
        
    # Define file paths
    npz_path = os.path.join(folder_name, 'mot_data_full.npz')
    params_path = os.path.join(folder_name, 'parameters.txt')

    # 1. Save constants and parameters to parameters.txt (UPDATED references)
    with open(params_path, 'w') as f:
        f.write("--- PHYSICAL CONSTANTS ---\n")
        f.write(f"Reduced Planck constant (hbar): {hbar:.2e} J*s\n")
        f.write(f"Speed of light (c): {c:.1e} m/s\n")
        f.write(f"Laser wavelength (lambda_L): {lambda_L:.2e} m\n")
        f.write(f"Natural linewidth (Gamma): {Gamma:.2e} rad/s\n")
        f.write(f"Atom mass (m_atom): {m_atom:.2e} kg\n")
        f.write(f"Boltzmann constant (k_B): {k_B:.2e} J/K\n")
        f.write("\n--- MOT PARAMETERS ---\n")
        f.write(f"Magnetic field gradient (B_prime): {B_prime} T/m\n")
        f.write(f"Gyromagnetic ratio (mu): {mu:.2e} Hz/T\n")
        f.write(f"Laser detuning (delta): {delta:.2e} rad/s\n")
        f.write(f"Saturation intensity (Isat): {Isat:.1f} W/m^2\n")
        f.write(f"Beam intensity (I_infinity): {I_infinity:.1f} W/m^2\n")
        f.write("\n--- SIMULATION PARAMETERS ---\n")
        f.write(f"Number of atoms (N_atoms): {N_atoms}\n")
        f.write(f"Number of atoms in each Superparticle (N_super): {N_super}\n")
        f.write(f"Time step (dt): {dt:.2e} s\n")
        f.write(f"Number of steps (n_steps): {n_steps}\n")
        f.write(f"Initial/Resonance Temperature (T_res): {T_res_uK} mK\n")
        f.write(f"Initial/Resonance Velocity (V_res_RMS): {V_res:.2e} m/s\n")
        f.write(f"Initial/Resonance Radius (R_res_RMS): {R_res:.2e} m\n")
        f.write(f"Save Interval: {SAVE_INTERVAL} steps\n")
        
    # 2. Save history data to NPZ file (UNCHANGED)
    np.savez(
        npz_path, 
        time=np.array(time_history),
        positions=np.array(pos_history),
        velocities=np.array(vel_history),
        dt=dt,
        save_interval=SAVE_INTERVAL
    )

    print(f"\nFinal data and parameters saved in the '{folder_name}' directory.")

# -------------------------------------------------------------
# --- 4. SIMULATION EXECUTION (REVISED) ---
# -------------------------------------------------------------

if __name__ == "__main__":

    # Initial state
    positions = np.random.normal(0.0, R_res, (N_atoms, 3))
    velocities = np.random.normal(0.0, V_res, (N_atoms, 3))

    # Data collection lists (UNCHANGED)
    time_history = []
    pos_history = []
    vel_history = []

    print(f"Starting LOW-DENSITY (N={N_atoms}) MOT Simulation...")
    print(f"Initial Conditions (T_res): {T_res_uK} mK ({V_res:.2e} m/s RMS) and (R_res): {R_res:.2e} m RMS.")

    for step in range(n_steps):
        
        # Integrate one time step (UNCHANGED)
        positions, velocities = integrate_motion_verlet(
            positions, 
            velocities, 
            dt, 
            m_atom, 
            get_forces_full
        )
        
        # Record history (UNCHANGED)
        current_time = step * dt
        r_rms = np.sqrt(np.mean(np.sum(positions**2, axis=1)))
        v_rms = np.sqrt(np.mean(np.sum(velocities**2, axis=1)))
        T = m_atom_single * v_rms**2 / 2 / k_B

        total_kinetic_energy = 0.5 * m_atom_single * np.sum(velocities**2)
        
        if step % SAVE_INTERVAL == 0:
            time_history.append(current_time)
            pos_history.append(positions.copy())
            vel_history.append(velocities.copy())
        
        if step % 1000 == 0: 
            print(f"Time {current_time:.2e} s: R_RMS = {r_rms:.2e} m, V_RMS = {v_rms:.2e} m/s, T = {T*1e6:.2f} uK")

    print("\nSimulation Finished.")
    
    # --- FINAL DATA SAVING CALL ---
    save_simulation_data(
        time_history, pos_history, vel_history
    )
