#!/usr/bin/env python3
"""
Efficient GW Siren Analysis Runner
Processes single parameter combination per job for maximum parallelization
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
from scipy import integrate
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.integrate import quad
from astropy.cosmology import Planck18 as cosmo
import time
import pickle

# Functions that deal with probability distributions and sampling

def rejection_sampling(N, p, x_low, x_high, y_max, p_args=[]):
    """Vectorized rejection sampling for better performance."""
    sampled_points = []
    remaining = N
    
    while remaining > 0:
        # Sample in batches for efficiency
        batch_size = min(remaining * 10, 10000)  # Oversample by 10x
        x_candidates = np.random.uniform(x_low, x_high, batch_size)
        y_candidates = np.random.uniform(0, y_max, batch_size)
        p_x = p(x_candidates, *p_args)
        
        # Accept points where y <= p(x)
        accepted = x_candidates[y_candidates <= p_x]
        n_accepted = min(len(accepted), remaining)
        sampled_points.extend(accepted[:n_accepted])
        remaining -= n_accepted
    
    return np.array(sampled_points)

# functions that deal with the things that are required to build up the galaxy population distribution. 

def SFR(z):
    """
    Madau-Dickinson star formation rate (SFR) as a function of redshift z.

    Parameters:
        z (float or array): Redshift(s).

    Returns:
        float or np.ndarray: SFR in units of M_sun yr^-1 Mpc^-3.
    """
    return 0.015 * (1+z)**(2.7) / (1 + ((1+z)/2.9)**(5.6)) * 1e9

def P_t_d(t_d, kappa=1, t_min=100e6, norm=True):
    """
    Delay time distribution: P(t_d) ∝ t_d^(-kappa) for t_d >= t_min, 0 otherwise.

    Parameters:
        t_d (float or array): Delay time(s) in years.
        kappa (float): Power-law index.
        t_min (float): Minimum delay time (years).
        norm (bool): If True, normalize the distribution.

    Returns:
        np.ndarray: Probability density values.
    """
    f_x = 0
    H_0_inv = 13.98e9
    if norm:
        # Normalized power-law with cutoff at t_min
        f_x = np.piecewise(t_d, [t_d < t_min, t_d >= t_min], [0, lambda t_d: 1/(np.log(H_0_inv/t_min)) * t_d**(-kappa)])
    else:
        f_x = np.piecewise(t_d, [t_d < t_min, t_d >= t_min], [0, lambda t_d: t_d**(-kappa)])
    return f_x

def R_integrand(z, t_merg, t_min=100e6, kappa=1):
    """
    Integrand for merger rate calculation as a function of redshift.

    Parameters:
        z (float): Redshift.
        t_merg (float): Merger time in years.
        t_min (float): Minimum delay time (years).
        kappa (float): Power-law index for delay time distribution.

    Returns:
        float: Value of the integrand.
    """
    t_d = cosmo.lookback_time(z).to_value('year') - t_merg
    return P_t_d(t_d, kappa=kappa, t_min=t_min) * cosmo.lookback_time_integrand(z) * SFR(z)

def merger_rate(z, t_min=500e6, kappa=1):
    """
    Compute the merger rate at a given redshift by integrating the delay time distribution.

    Parameters:
        z (float): Redshift.
        t_min (float): Minimum delay time (years).
        kappa (float): Power-law index.

    Returns:
        float: Merger rate.
    """
    return integrate.quad(R_integrand, z, np.inf, args=(cosmo.lookback_time(z).to_value('year'), t_min, kappa), limit=300)[0]

# Functions used to compute the probability distributions of siren population in galaxies. 

def P_g_full(g, power_p, power_n, Break, low_g=None, high_g=None):
    """
    Compute a broken power-law probability distribution for a galaxy property (e.g., mass, SFR, metallicity).

    The distribution follows a power law with exponent 'power_p' below the break value, and 'power_n' above it.
    Optionally, lower and upper cut-offs can be applied.

    Parameters:
        g (array-like): Property values (e.g., mass, SFR, metallicity).
        power_p (float): Power-law exponent below the break.
        power_n (float): Power-law exponent above the break.
        Break (float): Break value separating the two regimes.
        low_g (float, optional): Lower cut-off for g.
        high_g (float, optional): Upper cut-off for g.

    Returns:
        np.ndarray: Normalized probability distribution for g.
    """
    g = np.array(g)
    prob = np.zeros(len(g))
    for i in range(len(g)):
        # Below the break, use power_p exponent
        if g[i] < Break:
            prob[i] = g[i] ** power_p
        # Above the break, use power_n exponent and normalize at the break
        if g[i] >= Break:
            prob[i] = g[i] ** power_n / Break ** (power_n - power_p)
    # Apply lower and upper cut-offs if specified
    if low_g is not None:
        prob = prob * np.array([0 if g_val < low_g else 1 for g_val in g])
    if high_g is not None:
        prob = prob * np.array([0 if g_val > high_g else 1 for g_val in g])
    # Normalize the probability distribution
    prob_sum = np.sum(prob)
    if prob_sum > 0:
        return prob / prob_sum
    else:
        return prob


def P_MZR(M, delta_l, delta_h, M_c, low_M=None, high_M=None):
    """
    Compute the probability distribution for the Mass-Metallicity Relation (MZR).

    This is modeled as a broken power-law in stellar mass, with a break at M_c.

    Parameters:
        M (array-like): Stellar mass.
        delta_l (float): Power-law slope below the break (low mass end).
        delta_h (float): Power-law slope above the break (high mass end).
        M_c (float): Break mass.
        low_M (float, optional): Lower cut-off for mass.
        high_M (float, optional): Upper cut-off for mass.

    Returns:
        np.ndarray: Probability distribution for MZR.
    """
    # Use P_g_full with appropriate exponents for the broken power-law
    return P_g_full(M, 1/delta_l, -1/delta_h, M_c, low_M, high_M)

def g_selection_full(df, N, p, p_args, SFR_Z_type):
    """
    Select a subsample of galaxies based on a probability distribution function (PDF) and given parameters.

    Parameters:
        df (DataFrame): Input data containing galaxy properties.
        N (int): Number of galaxies to select.
        p (callable): PDF function for selection.
        p_args (list): Additional arguments for the PDF function.
        SFR_Z_type (str): Type of model to use for selection ('old_powerlaw_model', 'old_powerlaw_observed', 'MZR', 'FMR', 'MZSFR').

    Returns:
        DataFrame: Subsampled data.
    """
    # Check if we have enough galaxies to sample from
    if len(df) == 0:
        raise ValueError("Empty DataFrame provided for galaxy selection")
    
    if N > len(df):
        print(f"Warning: Requested {N} galaxies but only {len(df)} available. Using all available galaxies.")
        N = len(df)
    
    if SFR_Z_type=='old_powerlaw_model':
        sfr=SFR(np.array(df['z']))
        M=np.array(df['stellar_mass'])
        metal=None
        prob=p(M,sfr,metal,*p_args)
        prob=prob/np.sum(prob)
    elif SFR_Z_type=='old_powerlaw_observed':
        sfr=np.array(df['sfr'])
        metal=np.array(df['metallicity'])
        M=np.array(df['stellar_mass'])
        prob=p(M,sfr,metal,*p_args)
        prob=prob/np.sum(prob)
    elif SFR_Z_type=='MZR':
        M=np.array(df['stellar_mass'])
        prob=p(M,*p_args)
        prob_sum = np.sum(prob)
        if prob_sum > 0:
            prob=prob/prob_sum
        else:
            # If all probabilities are zero, use uniform distribution
            prob = np.ones(len(M)) / len(M)
    elif SFR_Z_type=='FMR':
        sfr=np.array(df['sfr'])
        M=np.array(df['stellar_mass'])
        prob=p(M,sfr,*p_args)
        prob=prob/np.sum(prob)
    elif SFR_Z_type=='MZSFR':
        sfr=np.array(df['sfr'])
        metal=np.array(df['metallicity'])
        M=np.array(df['stellar_mass'])
        prob=p(M,sfr,metal,*p_args)
        prob=prob/np.sum(prob)
    else:
        raise ValueError(f"Unknown SFR_Z_type: {SFR_Z_type}")
        
    index=np.array(df.index)
    new_index=np.random.choice(index, size=(N), replace=False, p=prob)    
    new_df=df.loc[new_index]
    
    return new_df

# Functions that deal with arrays.

def find_max(f, x_low, x_high, args):
    """
    Find the maximum value of a function f in the interval [x_low, x_high].

    Parameters:
        f (callable): Function to maximize.
        x_low (float): Lower bound.
        x_high (float): Upper bound.
        args (list): Arguments for the function f.

    Returns:
        float: Maximum value of f in the interval.
    """
    x = np.linspace(x_low, x_high, 1000)  # Increased resolution for better accuracy
    f_x = f(x, *args)
    return max(f_x)

# Functions that deal with binary mass modules

def m_kroupa(m, m_min, m_max, m_power):
    """
    Compute the normalized Kroupa initial mass function (IMF) probability for given masses.

    Parameters:
        m (float or array-like): Mass or array of masses.
        m_min (float): Minimum mass.
        m_max (float): Maximum mass.
        m_power (float): Power-law index.

    Returns:
        np.ndarray: Probability values for each mass.
    """
    # Handle special case where m_power = 1 (logarithmic integral)
    if abs(m_power - 1.0) < 1e-10:
        norm = np.log(m_max / m_min)
    else:
        # Normalization constant for the power-law IMF
        norm = (m_max ** (-m_power + 1) - m_min ** (-m_power + 1)) / (-m_power + 1)
    
    p_m = np.array([])
    if np.isscalar(m):
        m = [m]
    # Loop over all masses and compute probability
    for i in range(len(m)):
        if m[i] < m_min or m[i] > m_max:
            p_m = np.append(p_m, 0)
        else:
            if abs(m_power - 1.0) < 1e-10:
                p_m = np.append(p_m, 1.0 / (m[i] * norm))
            else:
                p_m = np.append(p_m, m[i] ** -m_power / norm)
    return p_m

def sec_mass_sample_beta(mass_sample, m_min, beta):
    """
    Sample secondary masses for binaries using a beta distribution in mass ratio.

    Parameters:
        mass_sample (float or array-like): Primary mass or array of primary masses.
        m_min (float): Minimum mass for secondary.
        beta (float): Power-law index for mass ratio distribution.

    Returns:
        np.ndarray: Array of secondary masses.
    """
    if np.isscalar(mass_sample):
        mass_sample = [mass_sample]
    sec_mass = np.zeros(len(mass_sample))
    for i in range(len(mass_sample)):
        if mass_sample[i] <= m_min:
            sec_mass[i] = m_min
        else:
            p_max = find_max(m_kroupa, m_min, mass_sample[i], [m_min, mass_sample[i], -beta])
            sec_mass[i] = rejection_sampling(1, m_kroupa, m_min, mass_sample[i], p_max, [m_min, mass_sample[i], -beta])[0]
    return sec_mass

def chirp_mass(m1, m2):
    """
    Compute the chirp mass for a binary system.

    Parameters:
        m1 (float): Mass of the primary.
        m2 (float): Mass of the secondary.

    Returns:
        float: Chirp mass.
    """
    return (m1 * m2) ** (3/5) / (m1 + m2) ** (1/5)

# Function to sample binary mass systems.

def binary_masses_sample(df, p_m1_type, p_m1, p_m1_args, p_m2, p_m2_args, z_min, z_max, n_z, t_min=100e6, kappa=1):
    """
    Assign binary component masses (m1, m2) to each galaxy in the input DataFrame.

    This function supports several physical and phenomenological models for the primary mass (m1)
    and uses either analytic or empirical relations for the secondary mass (m2).
    The assignment depends on galaxy properties (mass, SFR, metallicity, redshift) and model type.

    Parameters:
        df (DataFrame): Galaxy sample with properties (mass, SFR, metallicity, redshift).
        p_m1_type (str): Model for primary mass ('M_given', 'MSFR_given', 'Z_given', 'phys', 'phen').
        p_m1 (callable): PDF/function for primary mass sampling.
        p_m1_args (list): Arguments for p_m1.
        p_m2 (callable): Function for secondary mass given m1.
        p_m2_args (list): Arguments for p_m2.
        z_min (float): Minimum redshift for window function grid (used in 'phys' model).
        z_max (float): Maximum redshift for window function grid (used in 'phys' model).
        n_z (int): Number of redshift grid points for window function (used in 'phys' model).
        t_min (float): Minimum delay time (default 100 Myr).
        kappa (float): Power-law index for delay time distribution.

    Returns:
        None. Modifies df in-place by adding 'm1' and 'm2' columns.
    """
    # Check if required columns exist
    if p_m1_type in ['M_given', 'MSFR_given'] and 'stellar_mass' not in df.columns:
        raise ValueError("stellar_mass column required for this model")
    if p_m1_type in ['MSFR_given'] and 'sfr' not in df.columns:
        raise ValueError("sfr column required for MSFR_given model")
    if p_m1_type in ['Z_given'] and 'metallicity' not in df.columns:
        raise ValueError("metallicity column required for Z_given model")
    
    # Model: Primary mass depends only on galaxy mass
    if p_m1_type == 'M_given':
        # p_m1_args: [m_min, m_max, m_power, Z_star, M_Z_star, alpha]
        m_min, m_max, m_power, Z_star, M_Z_star, alpha = p_m1_args
        num = df.shape[0]
        m1 = np.zeros(num)
        m2 = np.zeros(num)
        M = np.array(df['stellar_mass'])
        solar_OH = 10 ** (8.83 - 12)
        solar_metal = 0.017
        for i in range(num):
            # Compute metallicity from mass using empirical relation
            OH = 10 ** (8.96 + 0.31 * np.log10(M[i]) - 0.23 * (np.log10(M[i])) ** 2
                        - 0.017 * (np.log10(M[i])) ** 3 + 0.04 * (np.log10(M[i])) ** 4 - 12)
            metal = 10 ** (np.log10(solar_metal) + np.log10(OH) - np.log10(solar_OH))
            # Compute PISN cutoff mass for this metallicity
            m_PISN = M_Z_star - alpha * np.log10(metal / Z_star)
            # Ensure m_PISN is within reasonable bounds
            m_PISN = max(m_min, min(m_PISN, m_max))
            p_m1_args_new = [m_min, m_PISN, m_power]
            p_max = find_max(p_m1, m_min, m_PISN, p_m1_args_new)
            # Sample primary and secondary masses - CORRECTED BOUNDS
            m1[i] = rejection_sampling(1, p_m1, m_min, m_PISN, p_max, p_m1_args_new)[0]
            m2[i] = p_m2(m1[i], *p_m2_args)[0]
        df.insert(df.shape[1], 'm1', m1)
        df.insert(df.shape[1], 'm2', m2)

    # Model: Primary mass depends on galaxy mass and SFR
    elif p_m1_type == 'MSFR_given':
        # p_m1_args: [m_min, m_max, m_power, Z_star, M_Z_star, alpha]
        m_min, m_max, m_power, Z_star, M_Z_star, alpha = p_m1_args
        num = df.shape[0]
        m1 = np.zeros(num)
        m2 = np.zeros(num)
        M = np.array(df['stellar_mass'])
        sfr = np.array(df['sfr'])
        solar_OH = 10 ** (8.83 - 12)
        solar_metal = 0.017
        for i in range(num):
            # Compute metallicity from mass and SFR using empirical relation
            OH = 10 ** (8.96 + 0.37 * np.log10(M[i]) - 0.14 * np.log10(sfr[i])
                        - 0.19 * (np.log10(M[i])) ** 2 + 0.12 * (np.log10(M[i]) * np.log10(sfr[i]))
                        - 0.054 * (np.log10(sfr[i])) ** 2 - 12)
            metal = 10 ** (np.log10(solar_metal) + np.log10(OH) - np.log10(solar_OH))
            m_PISN = M_Z_star - alpha * np.log10(metal / Z_star)
            # Ensure m_PISN is within reasonable bounds
            m_PISN = max(m_min, min(m_PISN, m_max))
            p_m1_args_new = [m_min, m_PISN, m_power]
            p_max = find_max(p_m1, m_min, m_PISN, p_m1_args_new)
            # CORRECTED BOUNDS
            m1[i] = rejection_sampling(1, p_m1, m_min, m_PISN, p_max, p_m1_args_new)[0]
            m2[i] = p_m2(m1[i], *p_m2_args)[0]
        df.insert(df.shape[1], 'm1', m1)
        df.insert(df.shape[1], 'm2', m2)

    # Model: Primary mass depends only on metallicity
    elif p_m1_type == 'Z_given':
        # p_m1_args: [m_min, m_max, m_power, Z_star, M_Z_star, alpha]
        m_min, m_max, m_power, Z_star, M_Z_star, alpha = p_m1_args
        num = df.shape[0]
        m1 = np.zeros(num)
        m2 = np.zeros(num)
        metal = np.array(df['metallicity'])
        for i in range(num):
            # Handle zero or negative metallicity
            if metal[i] <= 0:
                print(f"Warning: Non-positive metallicity {metal[i]} at index {i}, using minimum value")
                metal[i] = 1e-6
            
            m_PISN = M_Z_star - alpha * np.log10(metal[i] / Z_star)
            # Ensure m_PISN is within reasonable bounds
            m_PISN = max(m_min, min(m_PISN, m_max))
            p_m1_args_new = [m_min, m_PISN, m_power]
            p_max = find_max(p_m1, m_min, m_PISN, p_m1_args_new)
            # CORRECTED BOUNDS
            m1[i] = rejection_sampling(1, p_m1, m_min, m_PISN, p_max, p_m1_args_new)[0]
            m2[i] = p_m2(m1[i], *p_m2_args)[0]
        df.insert(df.shape[1], 'm1', m1)
        df.insert(df.shape[1], 'm2', m2)

    # Model: Physical model with redshift-dependent window function
    elif p_m1_type == 'phys':
        raise NotImplementedError("Physical model requires additional functions (obs_win_normal, etc.) that are not defined")

    # Model: Phenomenological model, primary mass sampled from analytic PDF
    elif p_m1_type == 'phen':
        # p_m1_args: [m_min, m_max, ...]
        m_min = p_m1_args[0]
        m_max = p_m1_args[1]
        num = df.shape[0]
        m1 = np.zeros(num)
        m2 = np.zeros(num)
        p_max = find_max(p_m1, m_min, m_max, p_m1_args)
        for i in range(num):
            m1[i] = rejection_sampling(1, p_m1, m_min, m_max, p_max, p_m1_args)[0]
            m2[i] = p_m2(m1[i], *p_m2_args)[0]
        df.insert(df.shape[1], 'm1', m1)
        df.insert(df.shape[1], 'm2', m2)
    
    else:
        raise ValueError(f"Unknown p_m1_type: {p_m1_type}")
        
# --- Siren Catalog Construction Functions ---
# These functions are designed for single-redshift (single snapshot) usage.
# They sample gravitational wave sirens from a galaxy catalog and assign binary masses.

def sample_sirens_single(gal_cat, N_GW, p, p_args, SFR_Z_type, ampl=1):
    '''
    Sample gravitational wave sirens from a galaxy catalog at a single redshift.

    Parameters:
        gal_cat (DataFrame): Galaxy catalog for a single redshift.
        N_GW (float): Expected number of GW events.
        p (function): Probability function for sampling.
        p_args (tuple): Arguments for the probability function.
        SFR_Z_type (str): Star formation rate and metallicity type.
        ampl (float): Amplitude factor for sampling (default 1).

    Returns:
        DataFrame: Sampled galaxy catalog.
    '''
    N_sample = max(1, int(N_GW * ampl))  # Ensure at least 1 sample
    return g_selection_full(gal_cat, N_sample, p, p_args, SFR_Z_type)

box_size = 205.0

def siren_catalog_single(df_filtered, z, sky_frac, p, p_args, SFR_Z_type,
                        p_m1_type, p_m1, p_m1_args, p_m2, p_m2_args,
                        ampl=1, obs_time=1, t_min=500e6, kappa=1, grid_points=1999, Range=0.01):
    '''
    Generate a catalog of gravitational wave sirens for a single redshift.

    Parameters:
        df_filtered (DataFrame): Filtered galaxy catalog at one redshift.
        z (float): Redshift of the snapshot.
        sky_frac (float): Fraction of the sky covered.
        p, p_args, SFR_Z_type: Sampling parameters.
        p_m1_type, p_m1, p_m1_args, p_m2, p_m2_args: Mass sampling parameters.
        ampl, obs_time, t_min, kappa, grid_points: Other parameters.
        Range (float): Small range around z for interpolation.

    Returns:
        tuple: (original catalog, sampled catalog)
    '''
    # Compute expected number of GW events for this redshift
    def N_GW_snapshot(z, box_size_Mpc, h, obs_time=1, t_min=500e6, kappa=1):
        # Convert box size to comoving Mpc (remove /h)
        box_size = box_size_Mpc / h  # now in Mpc
        # Convert volume to Gpc^3
        V_box = (box_size / 1000.0) ** 3  # Gpc^3

        # Compute merger rate density at snapshot redshift
        R_GW = merger_rate(z, t_min=t_min, kappa=kappa)  # Gpc^-3 yr^-1

        # Number of events
        N_GW = obs_time * V_box * R_GW / (1 + z)
        return N_GW
    
    N_GW = N_GW_snapshot(z, box_size, 0.6774)
    print(f"Expected number of GW events at z={z:.3f}: {N_GW:.2f}")
    
    # Sample sirens from the galaxy catalog
    df_sample = sample_sirens_single(df_filtered, N_GW, p, p_args, SFR_Z_type, ampl=ampl)
    
    # Assign binary masses to the sampled sirens
    binary_masses_sample(df_sample, p_m1_type, p_m1, p_m1_args, p_m2, p_m2_args, z, z, 1, t_min, kappa)
    
    return df_filtered, df_sample
    
def load_snapshot_data(snapshot, data_dir):
    """Load and preprocess galaxy data for a specific snapshot."""
    file_path = os.path.join(data_dir, f'subhalos_snap_{snapshot}.pkl')
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Snapshot file not found: {file_path}")
    
    print(f"Loading snapshot {snapshot} from {file_path}")
    df = pd.read_pickle(file_path)
    
    # Apply mass cut
    mass_cut = 0.1
    galaxies = df[df['SubhaloMassStars'] > mass_cut]
    
    print(f"Snapshot {snapshot}: {len(df)} -> {len(galaxies)} galaxies after mass cut")
    
    # Select and rename columns
    required_cols = ['SubhaloFlag', 'SubhaloMassStars', 'SubhaloSFR', 
                    'SubhaloPos_0', 'SubhaloPos_1', 'SubhaloPos_2', 
                    'Redshift', 'SubhaloGasMetallicity']
    
    galaxies = galaxies[required_cols].copy()
    
    # Rename columns
    galaxies = galaxies.rename(columns={
        'SubhaloMassStars': 'stellar_mass',
        'SubhaloSFR': 'sfr', 
        'SubhaloGasMetallicity': 'metallicity',
        'Redshift': 'z'
    })
    
    # Handle invalid metallicity
    n_zero_metal = np.sum(galaxies['metallicity'] <= 0)
    if n_zero_metal > 0:
        print(f"Warning: {n_zero_metal} galaxies have zero/negative metallicity")
        galaxies.loc[galaxies['metallicity'] <= 0, 'metallicity'] = 1e-6
    
    return galaxies

def get_mc_values():
    """Get the 15 logspaced M_c values."""
    return np.logspace(-1, 2, 15)  # 0.1 to 100 solar masses

def run_single_analysis(snapshot, delta_l, delta_h, mc_idx, output_dir, data_dir):
    """Run analysis for a single parameter combination."""
    
    start_time = time.time()
    
    # Load data
    galaxies = load_snapshot_data(snapshot, data_dir)
    snapshot_redshift = galaxies['z'].iloc[0]
    
    # Get M_c value
    M_c_values = get_mc_values()
    M_c = M_c_values[mc_idx]
    
    print(f"Running analysis:")
    print(f"  Snapshot: {snapshot} (z={snapshot_redshift:.3f})")
    print(f"  Parameters: delta_l={delta_l}, delta_h={delta_h}, M_c={M_c:.3f}")
    
    # Set up parameters
    SFR_Z_type = 'MZR'
    p = P_MZR
    p_args = [delta_l, delta_h, M_c, 0, 150]  # [delta_l, delta_h, M_c, low_M, high_M]
    
    # Binary mass parameters
    p_m1_type = 'Z_given'
    p_m1 = m_kroupa
    p_m1_args = [1, 531, 2.3, 1e-4, 45, 1.5]  # [m_min, m_max, m_power, Z_star, M_Z_star, alpha]
    p_m2 = sec_mass_sample_beta
    p_m2_args = [5, 1]
    
    # Analysis parameters
    ampl = 1e9
    obs_time = 1
    t_min = 500e6
    kappa = 1
    grid_points = 256
    Range = 0.01
    sky_frac = 1.0
    
    # Run the analysis
    try:
        _, siren_cat = siren_catalog_single(
            galaxies, snapshot_redshift, sky_frac,
            p, p_args, SFR_Z_type,
            p_m1_type, p_m1, p_m1_args, p_m2, p_m2_args,
            ampl=ampl, obs_time=obs_time, t_min=t_min, kappa=kappa, 
            grid_points=grid_points, Range=Range
        )
        
        # Process results
        siren_cat = siren_cat.rename(columns={
            'SubhaloPos_0': 'x',
            'SubhaloPos_1': 'y',
            'SubhaloPos_2': 'z_pos'
        })
        
        # Compute chirp mass
        siren_cat['chirp_mass'] = chirp_mass(np.array(siren_cat['m1']), np.array(siren_cat['m2']))
        
        # Add parameter information
        siren_cat['delta_l'] = delta_l
        siren_cat['delta_h'] = delta_h
        siren_cat['M_c'] = M_c
        siren_cat['snapshot'] = snapshot
        
        # Save results
        output_filename = f'siren_cat_snap_{snapshot}_dl_{delta_l:.2f}_dh_{delta_h:.2f}_mc_{M_c:.3f}.pkl'
        output_path = os.path.join(output_dir, output_filename)
        
        siren_cat.to_pickle(output_path)
        
        elapsed_time = time.time() - start_time
        print(f"Analysis completed in {elapsed_time:.2f} seconds")
        print(f"Generated {len(siren_cat)} sirens")
        print(f"Results saved to: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"Error in analysis: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Run GW Siren Analysis')
    parser.add_argument('--snapshot', type=int, required=True, help='Snapshot number')
    parser.add_argument('--delta_l', type=float, required=True, help='Delta_l parameter')
    parser.add_argument('--delta_h', type=float, required=True, help='Delta_h parameter')
    parser.add_argument('--mc_idx', type=int, required=True, help='M_c index (0-14)')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--data_dir', type=str, required=True, help='Data directory')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run analysis
    success = run_single_analysis(
        args.snapshot, args.delta_l, args.delta_h, args.mc_idx,
        args.output_dir, args.data_dir
    )
    
    if success:
        print("Job completed successfully!")
        sys.exit(0)
    else:
        print("Job failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
