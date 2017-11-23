from __future__ import division, print_function, absolute_import, with_statement
import numpy as np

def write_xtl(input_atoms, 
              filename='PyMOD_visualisation',
              system='NaCl',
              cation='Na',
              anion ='Cl',
              size=None):
    atoms = np.copy(input_atoms)

    
    if size == None:
        x_range = np.amax(atoms[:,0]) - np.amin(atoms[:,0])
        y_range = np.amax(atoms[:,1]) - np.amin(atoms[:,1])
        z_range = np.amax(atoms[:,2]) - np.amin(atoms[:,2])
        
        if x_range > y_range and x_range > z_range:
            size = x_range * 1.15
        elif y_range > z_range:
            size = y_range * 1.15
        else:
            size = z_range * 1.15

    
    
    # translate the dislocation, currently
    # at 0, 0, z, to the middle of the new cell.
    atoms[:,0] += size/2
    atoms[:,1] += size/2
    atoms[:,2] += size/2
    
    with open(filename + '.xtl', 'wt') as file:
        (alpha, beta, gamma) = (90,90,90)
        
        # A load of header stuff
        file.write('TITLE '+ filename +'\n')
        file.write('CELL'+'\n')
        file.write('  {:g}   {:g}   {:g}   {:g}   {:g}   {:g}   '
                   .format(size,size,size,alpha,beta,gamma)+'\n')
        file.write('SYMMETRY NUMBER 1'+'\n')
        file.write('SYMMETRY LABEL P1'+'\n')
        file.write('ATOMS'+'\n\n')
        # Now the actual atoms:
        for atom in atoms:
            if atom[3] < 0:
                element = anion
            elif atom[3] > 0:
                element = cation
            else:
                element = 'Fe'
            condition = (0.05*size < atom[0] < 0.95 * size and 
                         0.05*size < atom[1] < 0.95 * size and 
                         0.05*size < atom[2] < 0.95 * size)
            if condition:
                file.write(element+'           {:g}   {:g}   {:g}'
                           .format(atom[0]/size, 
                                   atom[1]/size,
                                   atom[2]/size)+'\n')

        file.write('EOF')

    print(filename+".xtl file saved.")

