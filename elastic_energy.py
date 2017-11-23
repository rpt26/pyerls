from __future__ import division, print_function, with_statement
import numpy as np
import os
import math
import numpy.random
import time
import build
import utils
from scipy import optimize
from scipy import constants
from numpy import float128 as myfloat

try:
    from opt_einsum import contract as einsum
except ImportError:
    einsum = np.einsum

simple_orthorhombic = build.simple_orthorhombic

def calculate_disloc_energy_components(disloc_params, simulation_params):
    slip_system = simulation_params[0]
    alpha = simulation_params[1]
    size = simulation_params[2]
    b = slip_system['latt_params'][0]
    d = slip_system['latt_params'][1]
    c = slip_system['latt_params'][2] # repeat distance paralell to line length
    
    # If we have information about basis systems and need to tranform the stiffness tensor:
    if 'stiffness_basis' in slip_system:
        elastic_tensor = transform_elastic_tensor(slip_system['elastic_tensor'], slip_system=slip_system)
    
    if simulation_params[2][2] != 0:
        line_length = 2 * size[2] * c * constants.angstrom
    elif simulation_params[2][2] == 0:
        line_length = c * constants.angstrom
    
    # create the atomic configuration from the dislocation parameters
    final_atoms, initial_atoms = create_atoms(disloc_params=disloc_params, simulation_params=simulation_params)
    
    y0 = initial_atoms[:,1]
    
    # identify lists of bonds
    x_bonds, y_bonds = identify_bonds(final_atoms, initial_atoms, simulation_params)
    adjacent_y_bonds = identify_adjacent_y_bonds(final_atoms, initial_atoms, y_bonds, simulation_params)
    # define lists of ideal bonds (i.e. unstrained at least locally)
    # these are unit vectors in the ideal direction.
    ideal_x_bonds = find_ideal_x_bonds(x_bonds, initial_atoms, simulation_params)

    # replace missing bonds with their ideal unstrained condition
    mask = einsum('ni,ni->n', x_bonds, x_bonds) == 0
    x_bonds[mask] = ideal_x_bonds[mask] * b
    
    # find the bonds across the slip plane
    misaligned_bonds, local_slip_plane_orientation = identify_misaligned_bonds(initial_atoms,
                                                                               final_atoms,
                                                                               x_bonds,
                                                                               simulation_params)

    # repeat for y
    ideal_y_bonds = np.zeros_like(ideal_x_bonds, dtype=myfloat)
    ideal_y_bonds[:,0] = - ideal_x_bonds[:,1]
    ideal_y_bonds[:,1] = ideal_x_bonds[:,0]
    
    # 
    mask = einsum('ni,ni->n', y_bonds, y_bonds) == 0
    y_bonds[mask] = ideal_y_bonds[mask] * d
    mask = einsum('ni,ni->n', adjacent_y_bonds, adjacent_y_bonds) == 0
    adjacent_y_bonds[mask] = ideal_y_bonds[mask] * d

    average_y_bonds = (y_bonds + adjacent_y_bonds) / 2
    
    # mask the positive x extreme, i.e. cells for which two y_bonds do not exist.
    mask = einsum('ni,ni->n', x_bonds, x_bonds) == 0
    average_y_bonds[mask] = ideal_y_bonds[mask]
    
    #print('Initial atoms:\n', initial_atoms)
    #print('average_y_bonds: \n', average_y_bonds)
    #print('ideal y bonds:\n', ideal_y_bonds)
    
    
    # calculate strains (voigt form) in a list:
    strains = calc_strains(x_bonds, average_y_bonds, ideal_x_bonds, ideal_y_bonds, slip_system=slip_system)
    
    
    # show the distribution of strains in x
    
    core_size = 200
    
    mask = np.fabs(initial_atoms[:,0]) > core_size
    mask = np.logical_and(mask, np.fabs(initial_atoms[:,1]) > core_size)


    # calculate energy from them:
    strain_energy = calc_cumulative_strain_energy(strains, initial_atoms, slip_system=slip_system)
    #the strain energy in bonds tht aren't across the slip plane.

    

    misalignment_energy = calculate_misalignment_energy(misaligned_bonds,
                                                        local_slip_plane_orientation,
                                                        initial_atoms,
                                                        slip_system=slip_system)
    
    slip_plane_strain_energy = calc_strain_energy_across_slip_plane(misaligned_bonds,
                                                                    local_slip_plane_orientation,
                                                                    initial_atoms,
                                                                    slip_system=slip_system)
    # combine and turn into a line energy:
    strain_energy += slip_plane_strain_energy
    total_energy = strain_energy + misalignment_energy
    
    strain_energy /= line_length
    misalignment_energy /= line_length
    total_energy /= line_length
    return total_energy, misalignment_energy, strain_energy


def calculate_disloc_energy(disloc_params, simulation_params):
    energy = calculate_disloc_energy_components(disloc_params, simulation_params)[0]
    return energy

def minimisable_energy(disloc_params, simulation_params, ref_disloc_params=np.array([1.5, 0.4, 0.4, -0.1])):

    ref_energy = 2e-8

    energy = calculate_disloc_energy(disloc_params=disloc_params,
                                     simulation_params=simulation_params)
    relative_energy = energy / ref_energy
    print(relative_energy)
    return relative_energy

def calc_strains(x_bonds, y_bonds, ideal_x_bonds, ideal_y_bonds, slip_system=simple_orthorhombic):
    b = slip_system['latt_params'][0]
    d = slip_system['latt_params'][1]
    c = slip_system['latt_params'][2]
    
    strains = np.zeros((len(x_bonds), 6), dtype=myfloat)
    


    # the component of the x_bond along the ideal x direction as a fraction of b
    relative_x_lengths = einsum('ni,ni->n', x_bonds, ideal_x_bonds) / b
    mask = relative_x_lengths == 0
    relative_x_lengths[mask] = 1
    # turn to strain (note the correction of sign)
    
    del_ux_by_del_x = relative_x_lengths - np.copysign(1, relative_x_lengths)
    
    strains[:,0] = del_ux_by_del_x
    
                                                                   
    # del_vy_by_del_y
    # similarly in y
    relative_y_lengths = einsum('ni,ni->n', y_bonds, ideal_y_bonds) / d
    mask = relative_y_lengths == 0
    relative_y_lengths[mask] = 1    
    del_v_by_del_y  = relative_y_lengths - np.copysign(1, relative_y_lengths)
    strains[:,1] = del_v_by_del_y
    
    # del_vy_by_del_x
    del_v_by_del_x = einsum('ni,ni->n', y_bonds, ideal_x_bonds) / d 
    # del_ux_by_del_y
    del_u_by_del_y = einsum('ni,ni->n', x_bonds, ideal_y_bonds) / b
    
    strains[:,5] = del_v_by_del_x + del_u_by_del_y
    return strains

def create_atoms(disloc_params = np.array([1.2, 0.2, 0.2, -0.1], dtype=myfloat),
                 simulation_params = [simple_orthorhombic, 0.21, [5,5,5]]):

    disloc_params = np.array(disloc_params)
    
    if disloc_params.shape == ():
        full_disloc_params = np.array([disloc_params, 0, 0, 0])
    else:
        full_disloc_params = np.zeros(4, dtype=myfloat)
        full_disloc_params[:len(disloc_params)] = disloc_params
    
    shape = simulation_params[2]
    alpha = simulation_params[1]
    slip_system = simulation_params[0]
    
    
    if 'core_offset' in simulation_params[0]:
        final_atoms, initial_atoms = build.dislocation_long_output(shape=shape,
                                                                   alpha=alpha,
                                                                   disloc_params=full_disloc_params, 
                                                                   slip_system=slip_system,
                                                                   core_offset=simulation_params[0]['core_offset'])
    else:
        final_atoms, initial_atoms = build.dislocation_long_output(shape=shape,
                                                                   alpha=alpha,
                                                                   disloc_params=full_disloc_params, 
                                                                   slip_system=slip_system, 
                                                                   core_offset=[.0, .5, .0])
    return final_atoms, initial_atoms

def find_transform(old_basis, new_basis):
    transform = np.empty((3,3), dtype=myfloat)

    for i in range(3):
        for j in range(3):
            transform[i,j] = np.dot(old_basis[i], new_basis[j])
    return transform


def transform_elastic_tensor(elastic_tensor, slip_system=simple_orthorhombic):
    old_basis = slip_system['stiffness_basis']
    new_basis = slip_system['dislocation_basis']
    transform = find_transform(old_basis, new_basis)
    
    C_ijkl = np.zeros((3, 3, 3, 3), dtype=myfloat)

    indices = [(0,0), (1,1), (2,2), (1,2), (2,0), (0,1), (2,1), (0,2), (1,0)]
    
    index_matrix = np.array([[1, 6, 5],
                             [6, 2, 4],
                             [5, 4, 3]])
    
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    m = index_matrix[i, j] - 1
                    n = index_matrix[k, l] - 1
                    C_ijkl[i,j,k,l] = elastic_tensor[m,n]
    
    new_C_ijkl = einsum('pi,qj,rk,sl,pqrs->ijkl', transform, transform, transform, transform, C_ijkl)
    new_elastic_tensor = np.zeros((6,6), dtype=myfloat)
    for m in range(elastic_tensor.shape[0]):
        for n in range(elastic_tensor.shape[1]):
            ij = indices[m]
            kl = indices[n]
            new_elastic_tensor[m, n] = new_C_ijkl[ij + kl]
    
    return new_elastic_tensor

def find_x_bonds(initial_atoms, final_atoms, simulation_params):
    slip_system = simulation_params[0]
    
    motif = slip_system['motif']
    
    if motif.ndim == 1:
        motif_length = 1
    else:
        motif_length = len(motif)
    
    
    size = simulation_params[2]
    initial_positions = initial_atoms[:,:3]
    final_positions = final_atoms[:,:3]
    
    x_bonds = np.zeros_like(initial_positions, dtype=myfloat)
    b = slip_system['latt_params'][0]
    
    
    rolled_initial = np.roll(initial_positions, -motif_length, axis=0)
    rolled_final = np.roll(final_positions, -motif_length, axis=0)
    mask = ((rolled_initial - initial_positions) - np.array([b, 0, 0]))
    mask = einsum('ni,ni->n', mask, mask)
    mask = mask < 1e-16
    
    # number of bonds expected to be left hanging at the positive x extreme:
    if slip_system['motif'].shape == (4,):
        motif_length = 1
    else:
        motif_length = len(slip_system['motif'])
    
    
    if size[2] == 0:
        no_of_missing_bonds = 2 * size[1] * motif_length
    else:
        no_of_missing_bonds = 2 * size[1] * 2 * size[2] * motif_length
    
    if np.count_nonzero(mask) == (len(mask) - no_of_missing_bonds):
        x_bonds[mask] = rolled_final[mask] - final_positions[mask]
    else:
        # A slow backup if we don't find any bonds assume we've messed up,
        # should work for an arbitrary ordered array
        for i in range(initial_atoms.shape[0]):
            rolled_initial = np.roll(initial_positions, -(i+1), axis=0)
            rolled_final = np.roll(final_positions, -(i+1), axis=0)
            mask = ((rolled_initial - initial_positions) - np.array([b, 0, 0]))
            mask = einsum('ni,ni->n', mask, mask)
            mask = mask < 1e-16
            x_bonds[mask] = rolled_final[mask] - final_positions[mask]

    return x_bonds

def find_y_bonds(initial_atoms, final_atoms, simulation_params):
    #print('finding y bonds')
    
    initial_positions = initial_atoms[:,:3]
    final_positions = final_atoms[:,:3]
    
    slip_system = simulation_params[0]
    d = slip_system['latt_params'][1]
    
    if 'layer_spacings' in slip_system:
        d0 = slip_system['layer_spacings'][0]
        d1 = slip_system['layer_spacings'][1]
    else:
        d0 = None
        d1 = None
    
    
    size = simulation_params[2]
    x_range = size[0]

    # we know some atoms will not have a neighbour in the y direcion:
    # missing at the top
    if slip_system['motif'].ndim == 1:
        motif_length = 1
    else:
        motif_length = len(slip_system['motif'])
    
    
    if size[2] == 0:
        no_of_missing_bonds = 2 * size[0]
    else:
        no_of_missing_bonds = 2 * size[0] * 2 * size[2]
    # missing at the slip plane where bonds don't line up
    if size[2] == 0:
        no_of_missing_bonds += (2 * size[0] - 1)
    else:
        no_of_missing_bonds += ((2 * size[0] + 1) * 2 * size[2])    
    
    y_bonds = np.zeros_like(initial_positions, dtype=myfloat)

    
    if motif_length == 1:
        # above the slip plane this rotation of the array rows gets the right alignemnt
        roll = -2*x_range*motif_length
        mask = mask_bonded_in_y(initial_atoms, final_atoms, simulation_params, roll)
        rotated_initial_positions = np.roll(initial_positions, roll, axis=0)
        rotated_final_positions = np.roll(final_positions, roll, axis=0)
        y_bonds[mask] = rotated_final_positions[mask] - final_positions[mask]
        
        
        # below the slip lane this rotation gets the right alignment
        roll = (-2*x_range + 1) * motif_length
        mask = mask_bonded_in_y(initial_atoms, final_atoms, simulation_params, roll)
        rotated_initial_positions = np.roll(initial_positions, roll, axis=0)
        rotated_final_positions = np.roll(final_positions, roll, axis=0)
        y_bonds[mask] = rotated_final_positions[mask] - final_positions[mask]
        
    else:
        # if there is a motif longer than one it's likely to be two layers
        # which can be handled in a very quick fashion thus:
        roll = -1 # to get the bond inside each unit cell
        mask = mask_bonded_in_y(initial_atoms, final_atoms, simulation_params, roll)
        rotated_initial_positions = np.roll(initial_positions, roll, axis=0)
        rotated_final_positions = np.roll(final_positions, roll, axis=0)
        y_bonds[mask] = rotated_final_positions[mask] - final_positions[mask]
        
        #now to get the bond between unit cells
        # above and below slip planes having slightly different results
        roll = - (2 * x_range * motif_length - 1)
        mask = mask_bonded_in_y(initial_atoms, final_atoms, simulation_params, roll)
        rotated_initial_positions = np.roll(initial_positions, roll, axis=0)
        rotated_final_positions = np.roll(final_positions, roll, axis=0)
        y_bonds[mask] = rotated_final_positions[mask] - final_positions[mask]
    
        roll = -( 2 * (x_range - 1) * motif_length + 1)
        mask = mask_bonded_in_y(initial_atoms, final_atoms, simulation_params, roll)
        rotated_initial_positions = np.roll(initial_positions, roll, axis=0)
        rotated_final_positions = np.roll(final_positions, roll, axis=0)
        y_bonds[mask] = rotated_final_positions[mask] - final_positions[mask]
  
    
    
    # number of hanging bonds at positive y extreme and across the slip plane
    # test to see if we've got it right
    # if not brute force the search
    if np.count_nonzero(einsum('ni,ni->n', y_bonds, y_bonds)) + no_of_missing_bonds != len(y_bonds):
        print('Brute forcing the y_bond search')
        # A slow backup if we don't find any bonds assume we've messed up:
        for i in range(1, initial_atoms.shape[0]):
            roll = i
            rotated_initial_positions = np.roll(initial_positions, roll, axis=0)
            rotated_final_positions = np.roll(final_positions, roll, axis=0)
            mask = mask_bonded_in_y(initial_atoms, final_atoms, simulation_params, roll)
            y_bonds[mask] = rotated_final_positions[mask] - final_positions[mask]
    
    if np.count_nonzero(einsum('ni,ni->n', y_bonds, y_bonds)) + no_of_missing_bonds != len(y_bonds):
        print('Number of identified y bonds does not match the expected')
    
    return y_bonds


def mask_bonded_in_y(initial_atoms, final_atoms, simulation_params, roll):
    
    initial_positions = initial_atoms[:,:3]
    final_positions = final_atoms[:,:3]
    
    slip_system = simulation_params[0]
    d = slip_system['latt_params'][1]
    
    if slip_system['motif'].ndim == 1:
        motif_length = 1
    else:
        motif_length = len(slip_system['motif'])
    
    
    if 'layer_spacings' in slip_system:
        d0 = slip_system['layer_spacings'][0]
        d1 = slip_system['layer_spacings'][1]
    else:
        d0 = None
        d1 = None
    
    
    
    rotated_initial_positions = np.roll(initial_positions, roll, axis=0)
    rotated_final_positions = np.roll(final_positions, roll, axis=0)
    
    if motif_length == 1:
        mask = (rotated_initial_positions - initial_positions - np.array([0, d, 0]))
        mask = einsum('ni,ni->n', mask, mask)
        mask = mask < 1e-16
        
    else:
        mask = np.zeros(len(initial_atoms), dtype=np.bool)
        for d_i in slip_system['layer_spacings']:
            separations_i = (rotated_initial_positions - initial_positions - np.array([0, d_i, 0]))
            separations_i = einsum('ni,ni->n', separations_i, separations_i)
            mask_i = separations_i < 1e-16
            mask = np.logical_or(mask, mask_i)
            
    return mask



def identify_bonds(final_atoms, initial_atoms, simulation_params):
    """Create a list of bonds that correspond to the atoms passed to the function"""
    slip_system = simulation_params[0]
    b = slip_system['latt_params'][0]
    d = slip_system['latt_params'][1]
    n = 0
    # test
    x_bonds = find_x_bonds(initial_atoms, final_atoms, simulation_params)
    y_bonds = find_y_bonds(initial_atoms, final_atoms, simulation_params)
            
    if len(initial_atoms) != len(final_atoms):
        print('Number of initial and final atoms does not match!')
    if len(final_atoms) != len(x_bonds):
        print('Wrong number of x bonds')
    if len(initial_atoms) != len(y_bonds):
        print('Wrong number of y bonds')

    return x_bonds, y_bonds


def identify_adjacent_y_bonds(final_atoms, initial_atoms, y_bonds, simulation_params):
    #print('Finding the adjacent y bonds')
    
    slip_system = simulation_params[0]
    b = slip_system['latt_params'][0]
    size = simulation_params[2]
    n_x = size[0]
    n_y = size[1]
    n_z = size[2]
    
    d = slip_system['latt_params'][1]
    
    if slip_system['motif'].ndim == 1:
        motif_length = 1
    else:
        motif_length = len(slip_system['motif'])
    
    
    if 'layer_spacings' in slip_system:
        d0 = slip_system['layer_spacings'][0]
        d1 = slip_system['layer_spacings'][1]
    else:
        d0 = None
        d1 = None
    
    
    rolled_y_bonds = np.roll(y_bonds, -motif_length, axis=0)
    rolled_final_atoms = np.roll(final_atoms, -motif_length, axis=0)
    rolled_init_atoms = np.roll(initial_atoms, -motif_length, axis=0)
    # find difference between the offsets and the expected bonds:
    diffs = (rolled_init_atoms - initial_atoms) - [b, 0, 0, 0]
    #mask to find those that aren't adjacent atoms:
    mask = einsum('ni, ni->n', diffs, diffs) > 1e-16
    
    
    if slip_system['motif'].shape == (4,):
        motif_length = 1
    else:
        motif_length = len(slip_system['motif'])
        
    if n_z == 0:
        expected_non_adjacent = 2 * n_y * motif_length
    else:
        expected_non_adjacent = 4 * n_y * n_z * motif_length
    
    test = np.count_nonzero(mask) == expected_non_adjacent
    if test:
        adjacent_y_bonds = rolled_y_bonds
    else:
        print('Bruteforcing the finding adjacent y bonds')
        adjacent_y_bonds = np.zeros((len(initial_atoms), 3), dtype=myfloat)
        for n in range(len(initial_atoms)):
            rolled_init_atoms = np.roll(initial_atoms, n, axis=0)
            rolled_y_bonds = np.roll(y_bonds, n, axis=0)
            # find difference between the offsets and the expected bonds:
            diffs = (rolled_init_atoms - initial_atoms) - [b, 0, 0, 0]
            #mask to find those that aren adjacent atoms:
            mask = einsum('ni, ni->n', diffs, diffs) > 1e-16
            adjacent_y_bonds[mask] = rolled_y_bonds[mask]
    
    # Check that the final bruteforced answer is correct:
    # find difference between the offsets and the expected bonds:
    diffs = (rolled_init_atoms - initial_atoms) - [b, 0, 0, 0]
    #mask to find those that aren't adjacent atoms:
    mask = einsum('ni, ni->n', diffs, diffs) > 1e-16
    test = np.count_nonzero(mask) == expected_non_adjacent
    if not test:
        print('Something has gone wrong with finding the adjacent y-bonds')

    # set those bonds that don't relate to adjacent atoms to zero
    adjacent_y_bonds[mask] = 0
        
    
    return adjacent_y_bonds

def find_ideal_x_bonds(x_bonds, initial_atoms, simulation_params):
    #print('Identifying the local lattice orientation')
    slip_system = simulation_params[0]
    size = simulation_params[2]

    b = slip_system['latt_params'][0]

    slip_system = simulation_params[0]
    d = slip_system['latt_params'][1]
    
    if slip_system['motif'].ndim == 1:
        motif_length = 1
    else:
        motif_length = len(slip_system['motif'])
    
    if 'layer_spacings' in slip_system:
        d0 = slip_system['layer_spacings'][0]
        d1 = slip_system['layer_spacings'][1]
    else:
        d0 = None
        d1 = None
    
    ideal_x_bonds = np.zeros_like(x_bonds, dtype=myfloat)

    bonds_rolled_up = np.roll(x_bonds, motif_length, axis=0)
    bonds_rolled_down = np.roll(x_bonds, -motif_length, axis=0)

    atoms_rolled_up = np.roll(initial_atoms, motif_length, axis=0)
    atoms_rolled_down = np.roll(initial_atoms, -motif_length, axis=0)

    # Check that we have in fact found neighbours on wither side# if this is false then the 
    # atom is at an extreme edge of the simulation and does not have a neighbouring bond on both sides
    mask = initial_atoms[:,1] == atoms_rolled_up[:,1]
    mask = np.logical_and(mask, initial_atoms[:,1] == atoms_rolled_down[:,1])
    mask = np.logical_and(mask, initial_atoms[:,2] == atoms_rolled_up[:,2])
    mask = np.logical_and(mask, initial_atoms[:,2] == atoms_rolled_down[:,2])

    # check that the atoms are indeed initially bonded:
    disps_up = initial_atoms - atoms_rolled_up
    disps_up = (einsum('ni,ni->n', disps_up, disps_up) ** 0.5) - b
    disps_down = initial_atoms - atoms_rolled_down
    disps_down = (einsum('ni,ni->n', disps_down, disps_down) ** 0.5) - b
    mask = np.logical_and(mask, disps_up < 1e-8)
    mask = np.logical_and(mask, disps_down < 1e-8)

    # check that bonds exist either side of the atom/bond in question
    up_bond_lengths = einsum('ni,ni->n', bonds_rolled_up, bonds_rolled_up) ** 0.5
    down_bond_lengths = einsum('ni,ni->n', bonds_rolled_down, bonds_rolled_down) ** 0.5
    mask = np.logical_and(mask, up_bond_lengths != 0)
    mask = np.logical_and(mask, down_bond_lengths != 0)

    ideal_x_bonds[mask] = (bonds_rolled_down[mask] + bonds_rolled_up[mask]) / 2

    with np.errstate(divide='ignore', invalid='ignore'):
        ideal_x_bonds /= (einsum('ni,ni->n', ideal_x_bonds, ideal_x_bonds) ** 0.5)[:,None]
        ideal_x_bonds = np.nan_to_num(ideal_x_bonds)

    if size[2] == 0:
        # there are 2 * n_y rows each having 3 bonds that will be disqualified
        expected_no_of_falses = 6 * size[1] * motif_length
    else:
        expected_no_of_falses = 6 * size[1] * size[2] * motif_length


    # Fall back on a slower method if it hasn't worked
    if len(x_bonds) != (np.count_nonzero(mask) + expected_no_of_falses):
        print('Brute forcing the finding of ideal x bonds')
        for i in range(len(x_bonds)):
            atoms_rolled_up = np.roll(initial_atoms, i, axis=0)
            bonds_rolled_up = np.roll(x_bonds, i, axis=0)
            for j in range(len(x_bonds)):
                atoms_rolled_down = np.roll(initial_atoms, -j, axis=0)
                bonds_rolled_down = np.roll(x_bonds, -j, axis=0)
                
                # Check that we have in fact found neighbours on wither side# if this is false then the 
                # atom is at an extreme edge of the simulation and does not have a neighbouring bond on both sides
                mask = initial_atoms[:,1] == atoms_rolled_up[:,1]
                mask = np.logical_and(mask, initial_atoms[:,1] == atoms_rolled_down[:,1])
                mask = np.logical_and(mask, initial_atoms[:,2] == atoms_rolled_up[:,2])
                mask = np.logical_and(mask, initial_atoms[:,2] == atoms_rolled_down[:,2])

                # check that the atoms are indeed initially bonded:
                disps_up = initial_atoms - atoms_rolled_up
                disps_up = (einsum('ni,ni->n', disps_up, disps_up) ** 0.5) - b
                disps_down = initial_atoms - atoms_rolled_down
                disps_down = (einsum('ni,ni->n', disps_down, disps_down) ** 0.5) - b
                mask = np.logical_and(mask, disps_up < 1e-8)
                mask = np.logical_and(mask, disps_down < 1e-8)

                # check that bonds exist either side of the atom/bond in question
                up_bond_lengths = einsum('ni,ni->n', bonds_rolled_up, bonds_rolled_up) ** 0.5
                down_bond_lengths = einsum('ni,ni->n', bonds_rolled_down, bonds_rolled_down) ** 0.5
                mask = np.logical_and(mask, up_bond_lengths != 0)
                mask = np.logical_and(mask, down_bond_lengths != 0)
                
                
                ideal_x_bonds[mask] = (bonds_rolled_down[mask] + bonds_rolled_up[mask]) / 2
        if np.count_nonzero(einsum('ni,ni->n', ideal_x_bonds, ideal_x_bonds)) + expected_no_of_falses != len(x_bonds):
            print("Couldn't identify ideal bonds successfully")       
    return ideal_x_bonds

def identify_misaligned_bonds(initial_atoms, final_atoms, x_bonds, simulation_params):

    slip_system = simulation_params[0]
    size = simulation_params[2]
    x_range = size[0]
    b = slip_system['latt_params'][0]
    
    slip_system = simulation_params[0]
    d = slip_system['latt_params'][1]
    if 'layer_spacings' in slip_system:
        d0 = slip_system['layer_spacings'][0]
        d1 = slip_system['layer_spacings'][1]
    else:
        d0 = None
        d1 = None
    
    if slip_system['motif'].ndim == 1:
        motif_length = 1
    else:
        motif_length = len(slip_system['motif'])
    
    
    x0 = initial_atoms[:,0]
    y0 = initial_atoms[:,1]

    final_positions = final_atoms[:,:3]
    init_positions = initial_atoms[:,:3]

    # y0_A is the distance to the first layer of atoms above the slip plane:
    pos_y0 = y0[y0 > 0]
    y0_A = np.amin(pos_y0)
    
    neg_y0 = y0[y0 < 0]
    y0_B = np.amax(neg_y0)

    init_upper_slip_plane = init_positions[np.fabs(y0 - y0_A) < 1e-8]
    init_lower_slip_plane = init_positions[np.fabs(y0 + y0_B) < 1e-8]
    final_upper_slip_plane = final_positions[np.fabs(y0 - y0_A) < 1e-8]
    final_lower_slip_plane = final_positions[np.fabs(y0 + y0_B) < 1e-8]    

    init_forward_mis_bonds = init_positions[np.fabs(y0 - y0_A) < 1e-8][1:] - init_positions[np.fabs(y0 - y0_B) < 1e-8]
    test = (init_forward_mis_bonds[:,0] > 0).all() and (init_forward_mis_bonds[:,0] <= b).all()
    test = test and ((np.fabs(init_forward_mis_bonds[:,1]) - (y0_A - y0_B)) < 1e-8).all

    if test:
        forward_misaligned_bonds = final_positions[np.fabs(y0 - y0_A) < 1e-10][1:] - final_positions[np.fabs(y0 - y0_B) < 1e-8]
    else:
        print('Could not identify forward leaning misaligned bonds!')

    init_backward_mis_bonds = init_positions[np.fabs(y0 - y0_A) < 1e-8][:-1] - init_positions[np.fabs(y0 - y0_B) < 1e-8]
    test = (init_backward_mis_bonds[:,0] <= 0).all() and (init_backward_mis_bonds[:,0] > -b).all()
    test = test and (init_backward_mis_bonds[:,1] == d).all
    if test:
        backward_misaligned_bonds = final_positions[np.fabs(y0 - y0_A) < 1e-8][:-1] - final_positions[np.fabs(y0 - y0_B) < 1e-8]
    else:
        print('Could not identify backward leaning misaligned bonds!')

    misaligned_bonds = np.vstack((forward_misaligned_bonds, backward_misaligned_bonds))

    slip_plane_orientation = np.zeros_like(x_bonds, dtype=myfloat)
    
    rolled_atoms = np.roll(initial_atoms, motif_length, axis=0)
    rolled_x_bonds = np.roll(x_bonds, motif_length, axis=0)
    mask_1 = np.fabs(y0 - y0_B) < 1e-8

    # mask_1 is the positions of those atoms that are immediately below the slip plane
    # from amongst all of the atoms

    mask_2 = np.fabs((initial_atoms - rolled_atoms)[:,0] - b) < 1e-8
    # those atoms separated in x by a burgers vector
    # next check that all the bonds are real
    mask_2 = np.logical_and(mask_2, einsum('ni,ni->n', x_bonds, x_bonds) != 0)
    mask_2 = np.logical_and(mask_2, einsum('ni,ni->n', rolled_x_bonds, rolled_x_bonds) != 0)

    # mask two is the position of those initial_atoms and rolled_atoms that are immediate neighbours
    # from amongst the complete set and that the corresponding bonds exist

    # use mask_2 to find local plane orientations across the whole crystal:
    slip_plane_orientation[mask_2] = ((x_bonds[mask_2] + rolled_x_bonds[mask_2]) / 2)
    # use mask_1 to cut this down to just the atoms immediately below the slip plane slip plane
    slip_plane_orientation = slip_plane_orientation[mask_1]
    # this will leave zeros
    slip_plane_orientation = np.vstack((slip_plane_orientation, slip_plane_orientation))
    np.count_nonzero(einsum('ni,ni->n', slip_plane_orientation, slip_plane_orientation))

    test = np.count_nonzero(einsum('ni,ni->n',
                                   slip_plane_orientation,
                                   slip_plane_orientation)) == len(slip_plane_orientation) - 4
    if not test:
        print("Couldn't identify the slip plane orientation, you'll have to write a brute force route, clever dick")
    
    # normalise slip plane orientation
    # catching the divide by zero

    with np.errstate(divide='ignore', invalid='ignore'):
        slip_plane_orientation /= (np.einsum('ni,ni->n', slip_plane_orientation, slip_plane_orientation) ** 0.5)[:,None]
        slip_plane_orientation = np.nan_to_num(slip_plane_orientation)
    return misaligned_bonds, slip_plane_orientation   

def frenkel_approx(phi, initial_atoms, slip_system=simple_orthorhombic):
    b = slip_system['latt_params'][0]
    d = slip_system['latt_params'][1]
    l = slip_system['latt_params'][2]  
    
    elastic_tensor = slip_system['elastic_tensor']
    
    if 'stiffness_basis' in slip_system:
        transform = find_transform(slip_system['stiffness_basis'], slip_system['dislocation_basis'])
        elastic_tensor = transform_elastic_tensor(elastic_tensor, slip_system=slip_system)
    if 'layer_stiffnesses' not in slip_system:
        prefactor = (elastic_tensor[5,5] * b * d * l / (4 * (np.pi**2)))
        periodic_term = 1 - np.cos(2 * np.pi * phi / b) 
        units = 1e9 * constants.angstrom**3  ##
        energy = np.sum(units * prefactor * periodic_term) / 2 
    elif 'layer_stiffnesses' in slip_system:
        elastic_tensor_0 = slip_system['layer_stiffnesses'][0]
        elastic_tensor_1 = slip_system['layer_stiffnesses'][1]
        
        if 'stiffness_basis' in slip_system:
            transform = find_transform(slip_system['stiffness_basis'], slip_system['dislocation_basis'])
            elastic_tensor_0 = transform_elastic_tensor(elastic_tensor_0, slip_system=slip_system)
            elastic_tensor_1 = transform_elastic_tensor(elastic_tensor_1, slip_system=slip_system)
        
        
        
        y0 = initial_atoms[:,1]
        # the position of the firt layer of atoms below the slip plane:
        y0_B = np.amax(y0[y0 < 0])
        
        mask_lower_slip_plane = np.fabs(y0 - y0_B) < 1e-10
        
        lower_slip_plane = initial_atoms[mask_lower_slip_plane]
        
        mask_0 = np.fabs(lower_slip_plane[:,3] - 0.) < 1e-10
        mask_1 = np.fabs(lower_slip_plane[:,3] - 1.) < 1e-10
        
        mask_0 = np.hstack((mask_0, mask_0))
        mask_1 = np.hstack((mask_1, mask_1))
        
        units = 1e9 * constants.angstrom**3  ##
        
        d_0 = slip_system['layer_spacings'][0]
        d_1 = slip_system['layer_spacings'][1]
        
        prefactor_0 = (elastic_tensor_0[5,5] * b * d_0 * l / (4 * (np.pi**2)))
        periodic_term_0 = 1 - np.cos(2 * np.pi * phi[mask_0] / b) 
        energy_0 = np.sum(units * prefactor_0 * periodic_term_0) / 2 
        
        prefactor_1 = (elastic_tensor_1[5,5] * b * d_1 * l / (4 * (np.pi**2)))
        periodic_term_1 = 1 - np.cos(2 * np.pi * phi[mask_1] / b) 
        energy_1 = np.sum(units * prefactor_1 * periodic_term_1) / 2 
        
        energy = energy_0 + energy_1
        
        
    # each atom has 2 bonds so divide by 2
    return energy

def calculate_misalignment_energy(misaligned_bonds, 
                                  local_slip_plane_oriention,
                                  initial_atoms,
                                  slip_system=simple_orthorhombic):
    d = slip_system['latt_params'][1]
    
    misalignment_energy = 0
    phi = einsum('ni,ni->n', misaligned_bonds, local_slip_plane_oriention)
    
    
    x0 = initial_atoms[:,0]
    y0 = initial_atoms[:,1]
    
    x0_slip_plane = x0[y0 == -d/2]
    x0_slip_plane = np.vstack((x0_slip_plane, x0_slip_plane))

    # frenkel approximation includes the factor of 0.5 to avoid double counting
    misalignment_energy += frenkel_approx(phi, initial_atoms, slip_system=slip_system)
    return misalignment_energy

def calc_strain_energy_across_slip_plane(misaligned_bonds, local_slip_plane_orientation, initial_atoms, slip_system=simple_orthorhombic):
    b = slip_system['latt_params'][0] #  all in angstroms
    d = slip_system['latt_params'][1]
    c = slip_system['latt_params'][2]
    
    y0 = initial_atoms[:,1]
    # the position of the firt layer of atoms below the slip plane:
    y0_B = np.amax(y0[y0 < 0])

    mask_lower_slip_plane = np.fabs(y0 - y0_B) < 1e-10

    lower_slip_plane = initial_atoms[mask_lower_slip_plane]
    
    if 'layered_stiffnesses' in slip_system:
        d_0 = slip_system['layer_spacings'][0]
        d_1 = slip_system['layer_spacings'][1]

        volume_0 = b * d_0 * c * (constants.angstrom**3)
        volume_1 = b * d_1 * c * (constants.angstrom**3)
        
        mask_0 = np.fabs(lower_slip_plane[:,3] - 0) < 1e-10
        mask_0 = np.hstack((mask_0, mask_0))

        mask_1 = np.fabs(lower_slip_plane[:,3] - 1) < 1e-10
        mask_1 = np.hstack((mask_1, mask_1))

        elastic_tensor_0 = slip_system['layer_stiffnesses'][0]
        elastic_tensor_1 = slip_system['layer_stiffnesses'][1]

        local_slip_plane_normal = np.zeros_like(local_slip_plane_orientation, dtype=myfloat)
        local_slip_plane_normal[:,0] = - local_slip_plane_orientation[:,1]
        local_slip_plane_normal[:,1] = local_slip_plane_orientation[:,0]

        delta_0 = einsum('ni,ni->n', misaligned_bonds[mask_0], local_slip_plane_normal[mask_0]) - d0
        delta_1 = einsum('ni,ni->n', misaligned_bonds[mask_1], local_slip_plane_normal[mask_1]) - d1
        
        strains_0 = np.zeros((len(delta_0), 6), dtype=myfloat)
        strains_0[:,1] = delta_0 / d0

        strains_1 = np.zeros((len(delta_1), 6), dtype=myfloat)
        strains_1[:,1] = delta_1 / d1


        strain_energy_0 = 0.5 * einsum('ij,nj,ni->', elastic_tensor_0, strains_0, strains_0)
        strain_energy_1 = 0.5 * einsum('ij,nj,ni->', elastic_tensor_1, strains_1, strains_0)
        strain_energy_0 *=  constants.giga * volume_0
        strain_energy_1 *=  constants.giga * volume_1
        strain_energy =  strain_energy_0 + strain_energy_1
    else:
        local_slip_plane_normal = np.zeros_like(local_slip_plane_orientation, dtype=myfloat)
        local_slip_plane_normal[:,0] = - local_slip_plane_orientation[:,1]
        local_slip_plane_normal[:,1] = local_slip_plane_orientation[:,0]

        delta = einsum('ni,ni->n', misaligned_bonds, local_slip_plane_normal) - d

        strains = np.zeros((len(delta), 6), dtype=myfloat)
        strains[:,1] = delta / d
        
        elastic_tensor = slip_system['elastic_tensor']
        strain_energy = 0.5 * einsum('ij,nj,ni->', elastic_tensor, strains, strains)
        
        volume = b * d * c * (constants.angstrom**3)
        strain_energy *=  constants.giga * volume 
        
    # over 2 because there's two bonds for each atom across the slip plane
    slip_plane_strain_energy = strain_energy / 2
    return slip_plane_strain_energy

def calc_cumulative_strain_energy(strains, initial_atoms, slip_system=simple_orthorhombic):
    '''given a 2D array each row of which is a strain in voigt notation and a slip system
       with details of a stiffness tensor (6x6) in GPa and lattice parameters in Angstroms
       calculate the total energy in joules.'''
    b = slip_system['latt_params'][0] #  all in angstroms
    d = slip_system['latt_params'][1]
    c = slip_system['latt_params'][2]
    
    elastic_tensor = slip_system['elastic_tensor']
    
    if 'stiffness_basis' in slip_system:
        transform = find_transform(slip_system['stiffness_basis'], slip_system['dislocation_basis'])
        elastic_tensor = transform_elastic_tensor(elastic_tensor, slip_system=slip_system)
    
    volume = b * d * c * (constants.angstrom**3)
    
    # if we don't have different bond lengths and elastic moduli in different layers:
    if 'layer_stiffnesses' not in slip_system:
        strain_energy = 0.5 * einsum('ij,nj,ni->', elastic_tensor, strains, strains)
        strain_energy *=  constants.giga * volume 
    # if it does
    elif 'layer_stiffnesses' in slip_system:
        elastic_tensor_0 = slip_system['layer_stiffnesses'][0]
        elastic_tensor_1 = slip_system['layer_stiffnesses'][1]
        
        if 'stiffness_basis' in slip_system:
            transform = find_transform(slip_system['stiffness_basis'], slip_system['dislocation_basis'])
            elastic_tensor_0 = transform_elastic_tensor(elastic_tensor_0, slip_system=slip_system)
            elastic_tensor_1 = transform_elastic_tensor(elastic_tensor_1, slip_system=slip_system)
                
        mask_0 = initial_atoms[:,3] == 0.
        mask_1 = initial_atoms[:,3] == 1.
        
        
        d_0 = slip_system['layer_spacings'][0]
        d_1 = slip_system['layer_spacings'][1]
        
        volume_0 = b * d_0 * c * (constants.angstrom**3)
        volume_1 = b * d_1 * c * (constants.angstrom**3)
        
        strain_energy_0 = 0.5 * einsum('ij,nj,ni->', elastic_tensor_0, strains[mask_0], strains[mask_0])
        strain_energy_1 = 0.5 * einsum('ij,nj,ni->', elastic_tensor_1, strains[mask_1], strains[mask_1])
        
        strain_energy_0 *=  constants.giga * volume_0
        strain_energy_1 *=  constants.giga * volume_1
        
        strain_energy =  strain_energy_0 + strain_energy_1
        
    return strain_energy #  In joules

