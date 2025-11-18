import numpy as np

# --- 1. PHYSICAL CONSTANTS AND MOT PARAMETERS ---
hbar = 1.054e-34  # Reduced Planck constant (J*s)
c = 3.0e8         # Speed of light (m/s)
lambda_L = 780e-9 # Example laser wavelength (m)
kL = 2 * np.pi / lambda_L # Laser wave number (1/m)
Gamma = 2 * np.pi * 6.0e6 # Natural linewidth (rad/s)
sigma0 = 6 * np.pi / kL**2 # Resonant scattering cross section (m^2)
m_atom = 1.41e-25 # Mass of a Rubidium atom (kg)

# MOT Parameters
B_prime = 0.1     # Magnetic field gradient (T/m)
mu = 1.399e10     # Gyromagnetic ratio (Hz/T, placeholder for simplicity)
delta = -2.5 * Gamma # Laser detuning (rad/s)
Isat = 16.7       # Saturation intensity (W/m^2)
I_infinity = 0.5 * Isat # Beam intensity before entering the cloud (W/m^2)

# Simulation parameters
N_atoms = 10000 # Number of atoms (Kept at 1,000)
dt = 3.0e-6
n_steps = 6000 
n_save = 50
SAVE_INTERVAL = n_steps // n_save
V_init = 10.0
R_init = 2e-3

# Mapping for transitions (q) and beams (alpha)
Q_TRANSITIONS = {-1: 'sigma_minus', 0: 'pi', 1: 'sigma_plus'}
ALPHA_AXES = {0: 'x', 1: 'y', 2: 'z'}
BEAM_DIRECTIONS = {
    ('x', '+'): (0, 1), ('x', '-'): (0, -1),
    ('y', '+'): (1, 1), ('y', '-'): (1, -1),
    ('z', '+'): (2, 1), ('z', '-'): (2, -1),
}

# --- 2. CORE KINETIC MODEL FUNCTIONS (FULL IMPLEMENTATION) ---

def calculate_magnetic_field_vectorized(r_positions):
    """
    Computes B(r) magnitude and relevant B-field factors for all atoms.
    B_factor corresponds to the (alpha' * B' / 2*B(r)) term from Eq. 4.
    """
    x, y, z = r_positions.T
    
    # B(r) magnitude (Eq. 4)
    B_mag = B_prime * np.sqrt(x**2 + 0.25 * (y**2 + z**2))
    
    # Magnetic factor for polarization fractions (alpha' * B' / 2*B(r))
    B_factor = np.zeros_like(r_positions)
    B_factor[:, 0] = 2 * x / (2 * B_mag)     # x-axis factor (alpha'=x)
    B_factor[:, 1] = y / (2 * B_mag)     # y-axis factor (alpha'=y)
    B_factor[:, 2] = z / (2 * B_mag) # z-axis factor (alpha'=2z)
    
    # Handle division by zero at origin (use a tiny value)
    B_mag[B_mag == 0] = 1e-12 
    B_factor[B_mag == 1e-12] = 0
    B_factor *= B_prime
    
    # N_atoms update (Crucial: B_factor and B_mag must match N_atoms)
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
        term_pos = 0.5 * (1 + sgn_dir * B_alpha_factor)**2
        term_neg = 0.5 * (1 - sgn_dir * B_alpha_factor)**2
        
        # Assign q=+ and q=- based on MOT helicity convention:
        if axis == 'z':
            # Z-axis is right-handed: + is sigma+, - is sigma-
            p_map[(axis, sign, 1)] = term_pos   # sigma+ (q=1)
            p_map[(axis, sign, -1)] = term_neg  # sigma- (q=-1)
        else:
            # X/Y axes are left-handed (helicity is reversed):
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

def calculate_attenuation_map(positions, velocities, B_mag, B_factor):
    """
    Calculates the 6 Attenuated Intensity arrays I_alpha_pm(r).
    This implements the attenuation (Eq. 13) using the cumulative sum approach (Eq. 14).
    """
    I_att_map = {}
    
    # 1. Initial Guess: Assume zero attenuation (O=0 -> I=I_infinity)
    I_guess_map = {k: np.full(N_atoms, I_infinity) for k in BEAM_DIRECTIONS}
    
    # 2. Calculate initial cross-sections based on guess I
    sigma_map_guess, _, p_map = calculate_scattering_cross_sections(
        positions, velocities, B_mag, B_factor, I_guess_map
    )
    
    # 3. Calculate Optical Depth (O_alpha_pm)
    for (axis, sign), (idx, sgn_dir) in BEAM_DIRECTIONS.items():
        coords = positions[:, idx]
        
        # Combine all q transitions for the total scattering cross section O (Eq. 14)
        sigma_eff_total = np.zeros(N_atoms)
        for q in Q_TRANSITIONS:
            sigma_eff_total += p_map[(axis, sign, q)] * sigma_map_guess[(axis, sign, q)]
            
        # O(N log N) Cumulative Sum Method
        # The scale factor 1e-15 is needed to convert the unitless sum of cross-sections 
        # (intended to be integrated over density and distance) into a realistic OD ~ 1.
        sort_indices = np.argsort(coords) if sgn_dir == 1 else np.argsort(-coords)
        sigma_sorted = sigma_eff_total[sort_indices]
        
        O_sorted = np.cumsum(sigma_sorted) * 1e-15
        O_sorted -= sigma_sorted # Subtract the atom's own scattering
        
        O_depth = np.zeros(N_atoms)
        O_depth[sort_indices] = O_sorted
        
        # Attenuated intensity (Eq. 13)
        I_alpha_pm = I_infinity * np.exp(-O_depth)
        I_att_map[(axis, sign)] = I_alpha_pm
        
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

# --- 4. SIMULATION EXECUTION ---

if __name__ == "__main__":
    
    # Helper function to consolidate full force calculations for the integrator
    def get_forces_full(positions, velocities):
        # 1. Get Magnetic Field Vectors
        B_mag, B_factor = calculate_magnetic_field_vectorized(positions)
        
        # 2. Calculate Attenuated Intensities and Polarization Fractions (Coupled Step)
        I_att_map, p_map = calculate_attenuation_map(positions, velocities, B_mag, B_factor)
        
        # 3. Calculate Trapping Force (F_tr)
        F_tr, I_tot_q_map = calculate_trapping_force_full(positions, velocities, B_mag, B_factor, I_att_map, p_map)
        
        # 4. Calculate Diffusion Force (F_diff)
        F_diff = calculate_diffusion_force_full(I_tot_q_map)
        
        return F_tr + F_diff

    # Initial state (MODIFIED for low density (larger R) and high velocity (hotter V))
    # Density reduction factor ~10: R increase by ~2.15
    positions = np.zeros((N_atoms, 3)) + np.random.randn(N_atoms, 3) * R_init  
    # Velocity increase factor 10: 5e-1 * 10 = 5.0
    velocities = np.zeros((N_atoms, 3)) + np.random.randn(N_atoms, 3) * V_init

    # Data collection lists
    time_history = []
    r_rms_history = []
    v_rms_history = []
    e_k_history = [] 
    
    pos_history = []
    vel_history = []

    print(f"Starting LOW-DENSITY (N={N_atoms}) MOT Simulation...")
    print("Initial Conditions: Very Hot (5 m/s RMS) and Large (1.1 mm RMS).")

    for step in range(n_steps):
        
        # Integrate one time step
        positions, velocities = integrate_motion_verlet(
            positions, 
            velocities, 
            dt, 
            m_atom, 
            get_forces_full
        )
        
        # Record history
        current_time = step * dt
        r_rms = np.sqrt(np.mean(np.sum(positions**2, axis=1)))
        v_rms = np.sqrt(np.mean(np.sum(velocities**2, axis=1)))
        
        # Calculate Total Kinetic Energy (1/2 * m * SUM(v^2))
        total_kinetic_energy = 0.5 * m_atom * np.sum(velocities**2)
        
        time_history.append(current_time)
        r_rms_history.append(r_rms)
        v_rms_history.append(v_rms)
        e_k_history.append(total_kinetic_energy)
        
        if step % SAVE_INTERVAL == 0:
            pos_history.append(positions.copy())
            vel_history.append(velocities.copy())
        
        if step % 1000 == 0: 
            print(f"Time {current_time:.2e} s: R_RMS = {r_rms:.2e} m, V_RMS = {v_rms:.2e} m/s")

    # --- FINAL DATA SAVING ---
    np.savez(
        'mot_data_full.npz', 
        time=np.array(time_history),
        r_rms=np.array(r_rms_history),
        v_rms=np.array(v_rms_history),
        e_k=np.array(e_k_history),
        positions=np.array(pos_history),
        velocities=np.array(vel_history),
        dt=dt,
        save_interval=SAVE_INTERVAL
    )

    print("\nSimulation Finished.")
    print(f"Final data saved to 'mot_data_full.npz'.")