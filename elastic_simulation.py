from __future__ import division, print_function, absolute_import, with_statement
import build
import numpy as np
from scipy import constants
from scipy import optimize
import math
import utils
import elastic_energy as ec
import time
import inputs
import os
import sys
from numpy import float128 as myfloat


slip_system = dict(latt_params=(2.5562, 2.0871, 1),
         label='copper',
         motif = np.array([0.,0.,0.,0.], dtype=myfloat),
         elastic_tensor=np.array([[171.0, 114.9, 114.9, 0,     0,     0],
                                  [114.9, 171.0, 114.9, 0,     0,     0],
                                  [114.9, 114.9, 171.0, 0,     0,     0],
                                  [0,     0,     0,     61.0, 0,     0],
                                  [0,     0,     0,     0,     61.0, 0],
                                  [0,     0,     0,     0,     0,     61.0]], dtype=myfloat),
         homogenous=True,
         stiffness_basis = [np.array([1,0,0], dtype=myfloat),
                            np.array([0,1,0], dtype=myfloat),
                            np.array([0,0,1], dtype=myfloat)],
         dislocation_basis = [np.array([1,-1,0], dtype=myfloat)/np.sqrt(2),
                              np.array([1,1,1], dtype=myfloat)/np.sqrt(3),
                              np.array([-1, -1, 2], dtype=myfloat)/np.sqrt(6)] # 
         )

# alternatively use a random slip system from the inputs module.
# slip_system = inputs.systems[np.random.randint(len(inputs.systems))]

def callback(disloc_variables):
    
    if disloc_variables.shape == ():
        disloc_variables = np.array([disloc_variables])
        
    
    np.save(label+'_disloc_variables', disloc_variables)
    print('\n_________________________________________________________')
    print('Checkpointed values: ', disloc_variables)
    print('==========================================================\n')
    #print(disloc_variables, '\n', ec.calculate_disloc_energy_components(disloc_variables, sim_params))
    return None


start = time.time()
while time.time() - start < 40000: # i.e. run overnight
        
    label = slip_system['label']
    print('_________________________________________________________________________')
    print(slip_system['label'])
    print('\n')
    alpha = np.random.random() / 2
    
    dimensions = inputs.sizes[np.random.randint(len(inputs.sizes))]
    
    print('\n Alpha = {} \n__________________________________________________________________'.format(alpha))
    print('\nsize: {} x {} x {}\n'.format(*dimensions))
    sim_params = [slip_system, alpha, dimensions]
    try:
        init_disloc_params = np.load(label+'_disloc_variables.npy')
        if init_disloc_params.shape == ():
            init_disloc_params = np.reshape(init_disloc_params, (1,))
    except:
        init_disloc_params = np.array([1.98, 0.25, 0.38, -0.1], dtype=myfloat)
    print(init_disloc_params)
    optimise_result = optimize.fmin_bfgs(ec.minimisable_energy, 
                                         init_disloc_params, 
                                         args=(sim_params,),
                                         gtol=inputs.tolerance,
                                         callback=callback,
                                         disp=1)
    optimal_disloc_vars = optimise_result
    init_disloc_params = optimal_disloc_vars * (1 + (np.random.random() - 0.5) / 20)

    if optimal_disloc_vars.shape == ():
        optimal_disloc_vars = np.array([optimal_disloc_vars])
    np.savetxt(label+'_disloc_variables.csv', optimal_disloc_vars, delimiter=',')

    print('Optimum dislocation parameters: ', optimal_disloc_vars)
    final_atoms, initial_atoms = ec.create_atoms(disloc_params=optimal_disloc_vars, simulation_params=sim_params)
    energies = ec.calculate_disloc_energy_components(optimal_disloc_vars,
                                                     sim_params)
    print(energies)
    alpha = sim_params[1]
    slip_system = sim_params[0]
  
    filename = 'results' + str(sys.argv[1]) + '_' + str(sys.argv[2]) + '.csv'
    
    with open(filename, 'at') as text_file:
        result_string = (label + ',{},'.format(alpha)
                          + '{},{},{},'.format(*dimensions)
                          + '{},{},{},'.format(*slip_system['latt_params'])
                          + '{},{},{},'.format(*energies))
        for i in range(len(optimal_disloc_vars)):
            result_string += '{},'.format(optimal_disloc_vars[i])
        result_string += '\n'
        text_file.write(result_string)

    print('\nOptimisation done:')
    print(slip_system['label'], 'alpha = {}'.format(sim_params[1]))
    print('Optimised dislocation parameters: ')

    print(optimal_disloc_vars)
    print(energies)
    print('=======================================================================================\n')

