from __future__ import division, print_function, absolute_import, with_statement
import numpy as np
from scipy import constants
import math
import utils
import cProfile

NaCl = dict(latt_params=(5.644, 5.644, 5.644),
            motif = np.array([[0,   0, 0,      1],
                              [0,   0.5, 0.5,  1],
                              [0.5, 0,   0.5,  1],
                              [0.5, 0.5, 0,    1],
                              [0,   0,   0.5, -1],
                              [0,   0.5, 0,   -1],
                              [0.5, 0,   0,   -1],
                              [0.5, 0.5, 0.5, -1]]),
            label='NaCl conventional unit cell',
            r0=2.822,
            madelung_constant = -1.747564594633182190636,
            # Lennard Jones Parameters:
            A_aa=6.439384818844595e-14, #J.A^12
            A_cc=1.2778012887174003e-16,
            A_ca=5.0020615054508415e-15,
            B_aa=6.691270510982213e-18,
            B_cc=5.27171675214672e-19,
            B_ca=2.480149818925444e-18
           )

NaCl_110_001 = dict(latt_params=(5.644/math.sqrt(2),
                                 5.644, 
                                 5.644/math.sqrt(2)), # Angstroms 
                    motif=np.array([[0,   0,   0,   1],    # Na
                                    [0.5, 0.5, 0.5, 1],    # Na
                                    [0.5, 0,   0.5, -1],   # Cl
                                    [0,   0.5, 0,   -1]]), # Cl
                    label='NaCl_110_001',
                    r0=2.822,
                    madelung_constant = -1.747564594633182190636,
                   #Lennard Jones Parameters:
                    A_aa=6.439384818844595e-14, #J.A^12
                    A_cc=1.2778012887174003e-16,
                    A_ca=5.0020615054508415e-15,
                    B_aa=6.691270510982213e-18,
                    B_cc=5.27171675214672e-19,
                    B_ca=2.480149818925444e-18
                   )

NaCl_110_110 = dict(latt_params=(5.644/math.sqrt(2),
                                 (5.644/math.sqrt(2)),
                                 5.644),
                    motif =  np.array([[0,   0,   0,    1],
                                       [0.5, 0.5, 0.5,  1],
                                       [0,   0,   0.5, -1],
                                       [0.5, 0.5, 0,   -1]]),
                    label='NaCl_110_110',
                    r0=2.822,
                    madelung_constant = -1.747564594633182190636,
                   #Lennard Jones Parameters:
                    A_aa=6.439384818844595e-14, #J.A^12
                    A_cc=1.2778012887174003e-16,
                    A_ca=5.0020615054508415e-15,
                    B_aa=6.691270510982213e-18,
                    B_cc=5.27171675214672e-19,
                    B_ca=2.480149818925444e-18
                   )

simple_cubic = dict(latt_params=(3.,3.,3.),
                    motif= np.array([[0,0,0,0]]) 
                   )
primitive_orthorhombic = dict(latt_params=(3.4,3.5,3.6),
                              motif= np.array([[0,0,0,0]]),
                              stiffness_tensor = np.random.rand(3,3,3))
systems = [NaCl_110_001, NaCl_110_110]

simple_orthorhombic = dict(latt_params=(2, 2, 2),
                           motif = np.array([0.,0.,0.,0.], dtype=myfloat),
                           elastic_tensor=np.array([[387.2, 153,   155.4, 0,     0,    0],
                                                    [153,   342.7, 158,   0,     0,    0],
                                                    [155.4, 158,   308.5, 0,     0,    0],
                                                    [0,     0,     0,     133.1, 0,    0],
                                                    [0,     0,     0,     0,     16.8, 0],
                                                    [0,     0,     0,     0,     0,    132]], dtype=myfloat),
                           homogenous=True
                          )

def perfect_xtl(system=NaCl,
                shape=[2, 2, 2]):
    (b,d,c) = system['latt_params']
    
    x_range = int(shape[0])
    y_range = int(shape[1])
    z_range = int(shape[2])
    
    
    num_of_unit_cells = 8 * x_range * y_range * z_range
    num_of_atoms = num_of_unit_cells * len(system['motif'])

    atoms = np.zeros((num_of_atoms,4))

    
    n = 0
    for z in range(-z_range, z_range):
        for y in range(-y_range, y_range):
            for x in range(-x_range, x_range):
                motif = np.copy(system['motif'])
                motif[:,:3] += [x,y,z]
                motif[:,:3] *= system['latt_params']
                atoms[n*len(system['motif']):(n+1)*len(system['motif'])] = motif
                n += 1 
    # In analogy with the defective case we will build the model the same way
    # principly this will mean we include the negative extreme
    # but exclude the positive extreme

    y_cutoff = d * y_range
    z_cutoff = c * z_range
    x_cutoff = b * x_range
    # create a mask that returns true for those atoms inside the simulation:
    # y and z directions are easy, just hard boundaries at plus/minus the cutoff
    y_mask = np.logical_and(y_cutoff > atoms[:,1], atoms[:,1] >= -y_cutoff)

    z_mask = np.logical_and(atoms[:,2] < z_cutoff, atoms[:,2] >= -z_cutoff)
    # x direction depends on whether it is above the slip plane or below it
    # +ve y cutoff is x_cutoff
    x_mask_above_slip_plane = np.logical_and(atoms[:,1] >= 0, atoms[:,0] < x_cutoff)
    x_mask_above_slip_plane = np.logical_and(x_mask_above_slip_plane, atoms[:,0] >= -x_cutoff)
    #below the slip plane, -ve y, the cutoff is less by b/4
    x_mask_below_slip_plane = np.logical_and(atoms[:,1] < 0, atoms[:,0] >= -(x_cutoff))
    x_mask_below_slip_plane = np.logical_and(x_mask_below_slip_plane, atoms[:,0] < (x_cutoff))
    x_mask = np.logical_or(x_mask_above_slip_plane, x_mask_below_slip_plane)
    
    mask = np.logical_and(x_mask, y_mask)
    mask = np.logical_and(mask, z_mask)
    atoms = atoms[mask]
    
    return atoms

def dislocation_long_output(shape=[5, 5, 5], 
                            disloc_params=np.array([1.,0.2,0.2,-0.1]),
                            alpha=0,
                            slip_system=NaCl_110_001, 
                            core_offset=[0, 0.5, 0]): 
    
    
    # The width parameter only makes physical sense if it's positive:
    disloc_params[0] = np.fabs(disloc_params[0])

    latt_params = slip_system['latt_params']
    (b,d,c) = latt_params
    motif = slip_system['motif']
    core_offset = np.array(core_offset)
    # We build a simulation cell a bit big and then 
    # cut off the edges where artefacts are formed

    
    if shape[2] == 0: # It makes sense that for some simulations 
        # we only want one layer in the z direction, along which there is symmetry
        
        x_range = int(shape[0])
        y_range = int(shape[1]) # in units of unit cells
        z_range = 2
        
        initial_atoms = create_2D_initial_atoms(x_range, y_range, motif, alpha, latt_params, core_offset)
    else:
        x_range = int(shape[0])
        y_range = int(shape[1])
        z_range = int(shape[2]) # in units of unit cells

        initial_atoms = create_3D_initial_atoms(x_range, y_range, z_range, motif, alpha, latt_params, core_offset)

    atoms = np.copy(initial_atoms)
    atoms[:,0] += calc_x_disp(initial_atoms,
                                  disloc_params, 
                                  latt_params)
    atoms[:,1] += calc_y_disp(initial_atoms,
                                  disloc_params,
                                  latt_params)

    
    y_cutoff = d * y_range
    z_cutoff = c * z_range
    x_cutoff = b * x_range
    # create a mask that returns true for those atoms inside the simulation:
    # y and z directions are easy, just hard boundaries at plus/minus the cutoff
    y_mask = np.logical_and(y_cutoff > initial_atoms[:,1], initial_atoms[:,1] >= -y_cutoff)

    z_mask = np.logical_and(initial_atoms[:,2] < z_cutoff, initial_atoms[:,2] >= -z_cutoff)
    # x direction depends on whether it is above the slip plane or below it
    # +ve y cutoff is x_cutoff
    
    x_mask_above_slip_plane = np.logical_and(initial_atoms[:,1] >= 0, initial_atoms[:,0] < x_cutoff)
    x_mask_above_slip_plane = np.logical_and(x_mask_above_slip_plane, initial_atoms[:,0] >= -x_cutoff)
    #below the slip plane, -ve y, the cutoff is less by b/4
    x_mask_below_slip_plane = np.logical_and(initial_atoms[:,1] < 0, initial_atoms[:,0] >= -(x_cutoff-b/2))
    x_mask_below_slip_plane = np.logical_and(x_mask_below_slip_plane, initial_atoms[:,0] < (x_cutoff-b/2))
    x_mask = np.logical_or(x_mask_above_slip_plane, x_mask_below_slip_plane)
    
    mask = np.logical_and(x_mask, y_mask)
    mask = np.logical_and(mask, z_mask)

    initial_atoms = initial_atoms[mask]
    atoms = atoms[mask]
    
    net_charge = np.sum(atoms[:,3])
    
    if 'energy_calculation' in slip_system:
        calc_type = slip_system['energy_calculation']
    else:
        calc_type = 'Unknown'
    
    
    if net_charge != 0 and calc_type == 'electrostatic':
        print('Net charge is not zero! (alpha = {:g})'.format(alpha))

    if len(atoms) != len(initial_atoms):
        print('Atoms and Initial Atoms do not hae the same length, stuff has gone wrong...')
    assert len(atoms) == len(initial_atoms)
    
    return atoms, initial_atoms

def create_2D_initial_atoms(x_range, y_range, motif, alpha, latt_params, core_offset):
    b, d, c = latt_params
    sim_size = 4 * (x_range + 2) * (y_range + 2)
    # The dimensions are extended by 2 to allow control of the edge effects
    if motif.ndim == 2:
        motif_length = len(motif)
    elif motif.ndim == 1:
        motif_length = 1
    else:
        raise Exception("Motif has ndim > 2 and cannot be interpreted as either an atom or a list of atoms") 

    num_atoms = motif_length * sim_size

    # easiest to create a simulation that's too big and cut it down:
    n_x = x_range + 2
    n_y = y_range + 2
    
    grid = np.zeros((sim_size, 3))
    grid[:,:2] = np.mgrid[-n_x:n_x, -n_y:n_y].T.reshape(-1,2).astype(float)
    # we might need to shift everything to put the core of the dislocation in the right place
    # within the unit cell:
    grid += core_offset
    x0 = grid[:,0]
    y0 = grid[:,1]
    # offset lower half wrt to upper half:
    x0[y0 < 0] += 0.5
    x0 -= alpha
    # This is now a grid with one row for ach unit cell origin, tile it up to 
    # add a row for each atom:
    if motif_length != 0:
        grid = np.repeat(grid, motif_length, axis=0)
    
    initial_atoms = np.zeros((len(grid), 4))
    initial_atoms[:,:3] = grid

    # add the motif
    if motif_length == 1:
        initial_atoms += motif
    else:
        initial_atoms += np.resize(motif, (initial_atoms.shape))
    
    initial_atoms *= (b, d, c, 1)
    return initial_atoms

def create_3D_initial_atoms(x_range, y_range, z_range, motif, alpha, latt_params, core_offset):
    b, d, c = latt_params
    sim_size = 8 * (x_range + 2) * (y_range + 2) * (z_range + 2)
    # The dimensions are extended by 2 to allow control of the edge effects
    if motif.ndim == 2:
        motif_length = len(motif)
    elif motif.ndim == 1:
        motif_length = 1
    else:
        raise Exception("Motif has ndim > 2 and cannot be interpreted as either an atom or a list of atoms") 

    num_atoms = motif_length * sim_size

    # easiest to create a simulation that's too big and cut it down:
    n_x = x_range + 2
    n_y = y_range + 2
    n_z = z_range + 2
    
    grid = np.mgrid[-n_x:n_x, -n_y:n_y, -n_z:n_z].T.reshape(-1,3).astype(float)
    # we might need to shift everything to put the core of the dislocation in the right place
    # within the unit cell:
    grid += core_offset
    x0 = grid[:,0]
    y0 = grid[:,1]
    # offset lower half wrt to upper half:
    x0[y0 < 0] += 0.5
    x0 -= alpha
    # This is now a grid with one row for ach unit cell origin, tile it up to 
    # add a row for each atom:
    if motif_length != 0:
        grid = np.repeat(grid, motif_length, axis=0)

    initial_atoms = np.zeros((len(grid), 4))
    initial_atoms[:,:3] = grid
    
    # add the motif
    if motif_length == 1:
        initial_atoms += motif
    else:
        initial_atoms += np.resize(motif, (initial_atoms.shape))
    
    initial_atoms *= (b, d, c, 1)
    return initial_atoms

def disloc_unit_cell(n_x, 
                     n_y,
                     n_z,
                     latt_params,
                     motif,
                     core_offset):
    (b, d, c) = latt_params
    origin = np.array([n_x, n_y, n_z])
    #the absolute position of the unit cell origin
    origin[:3] += core_offset 
    origin[:3] *= np.array(latt_params)
    
    if motif.ndim > 1:
        positions = motif[:,:3] #fractional
    elif motif.ndim == 1:
        positions = motif[:3] #fractional
    positions *= latt_params # Angstroms
    positions += origin
    return motif

def calc_x_disp(initial_atoms, 
                disloc_params,
                latt_params):
    # dislocation parameters as defined above
    b, d, c = latt_params
    w = disloc_params[0]
    c1 = disloc_params[1]
    x0 = initial_atoms[:,0]
    y0 = initial_atoms[:,1]
    
    x_disp_normal_term = calc_x_arctan_term(x0, y0, w, b)
    x_disp_distortion_term = calc_x_distortion_term(x0, y0, w, c1)
    x_disp = x_disp_normal_term + x_disp_distortion_term
    return x_disp

def calc_x_distortion_term(x0, y0, w, c1):
    denominator = x0**2 + (y0 + w * np.copysign(1, y0))**2
    numerator = c1 * x0 * y0
    x_distortion = numerator / denominator
    return x_distortion
    
def calc_x_arctan_term(x0, y0, w, b):
    
    with np.errstate(divide='ignore', invalid='ignore'):
        arc_tan_argument = (y0 + w * np.copysign(1, y0)) / x0
    
    arc_tan_term = np.arctan(arc_tan_argument)  - (np.pi * 0.5 * np.copysign(1, x0) * np.copysign(1, y0))
    arc_tan_term *= b / (2 * np.pi)
    return arc_tan_term

def calc_y_disp(atoms, 
                disloc_params,
                latt_params):
    # constants as defined above in the documentation
    b, d, c = latt_params
    w = disloc_params[0]
    c2 = disloc_params[2]
    c3 = disloc_params[3]
    x0 = atoms[:,0]
    y0 = atoms[:,1] # + 0.5 * d * np.copy(1, atoms[:,1])

    y_disp_distortion_term = calc_y_distortion_term(x0, y0, w, c2)
    y_disp_log_term = calc_y_log_term(x0, y0, w, c3, b)
    y_disp = y_disp_distortion_term + y_disp_log_term
    return  y_disp

def calc_y_distortion_term(x0, y0, w, c2):
    denominator = x0 **2 + (y0 + w * np.copysign(1, y0))**2 
    numerator = y0 * (y0 + w * np.copysign(1, y0)) 
    y_distortion_term = numerator / denominator
    y_distortion_term *= c2
    return y_distortion_term

def calc_y_log_term(x0, y0, w, c3, b):
    argument = x0**2 + (y0 + w * np.copysign(1, y0))**2
    y_log_term = np.log(argument/b**2)
    y_log_term *= c3
    return y_log_term

def dislocation(shape=[5, 5, 5], 
                disloc_params=np.array([1,0.2,0.2,-0.1]),
                alpha=0,
                slip_system=NaCl_110_001,
                core_offset=[0,0.25,0]): 
    result = dislocation_long_output(shape=shape, 
                                     disloc_params=disloc_params, 
                                     alpha=alpha, 
                                     slip_system=slip_system,
                                     core_offset=core_offset)
    atoms = result[0]
    return atoms
