import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
import glob
import time
import os
import scipy.optimize as opt
import inputs
from scipy import constants
import math
import pandas
import subprocess
import glob

einsum = np.einsum
simple_orthorhombic = dict(latt_params=(2, 2, 2),
                           motif = np.array([0.,0.,0.,0.]),
                           elastic_tensor=np.array([[387.2, 153,   155.4, 0,     0,    0],
                                                    [153,   342.7, 158,   0,     0,    0],
                                                    [155.4, 158,   308.5, 0,     0,    0],
                                                    [0,     0,     0,     133.1, 0,    0],
                                                    [0,     0,     0,     0,     16.8, 0],
                                                    [0,     0,     0,     0,     0,    132]]),
                           homogenous=True
                          )

def fourier_series(x, *c):
    y = np.zeros_like(x)
    for n in range(len(c)):
        y += c[n] * np.cos(2 * np.pi * n * x)
    return np.array(y)

def f_prime(x, *c):
    dy_by_dx = np.zeros_like(x)
    for n in range(len(c)):
        dy_by_dx += -2 * np.pi * c[n] * np.sin(2 * np.pi * n * x)
    return dy_by_dx

def negative_f_prime(x, *c):
    grad = f_prime(x, *c)
    grad *= -1.0
    return grad

def f_double_prime(x, *c):
    d2y_by_dx2 = np.zeros_like(x)
    for n in range(len(c)):
        d2y_by_dx2 += - 4 * (np.pi**2) * (n**2) * c[n] * np.cos(2 * np.pi * n * x)
    return d2y_by_dx2

def empirical_energy(dimensions, *params):
    x = dimensions[:,0]
    y = dimensions[:,1]
    z = dimensions[:,2]
    
    energy = 0
    energy += params[0] * x * y * z # volume energy
    energy += params[1] * x * y * 2 # xy surface energy
    energy += params[2] * y * z * 2 # yz surface
    energy += params[3] * z * x * 2
    energy += params[4] * x * 4
    energy += params[5] * y * 4
    energy += params[6] * z * 4
    energy += params[7] * 8
    return energy

def find_transform(old_basis, new_basis):
    transform = np.empty((3,3))

    for i in range(3):
        for j in range(3):
            transform[i,j] = np.dot(old_basis[i], new_basis[j])
    return transform

def transform_elastic_tensor(elastic_tensor, slip_system=simple_orthorhombic):
    old_basis = slip_system['stiffness_basis']
    new_basis = slip_system['dislocation_basis']
    transform = find_transform(old_basis, new_basis)
    
    C_ijkl = np.zeros((3, 3, 3, 3))

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
    
    new_C_ijkl = einsum('ip,jq,kr,ls,pqrs->ijkl', transform, transform, transform, transform, C_ijkl)
    new_elastic_tensor = np.zeros((6,6))
    for m in range(elastic_tensor.shape[0]):
        for n in range(elastic_tensor.shape[1]):
            ij = indices[m]
            kl = indices[n]
            new_elastic_tensor[m, n] = new_C_ijkl[ij + kl]
    
    return new_elastic_tensor

def transform_strain(strain, slip_system=simple_orthorhombic):
    old_basis = slip_system['dislocation_basis']
    new_basis = slip_system['stiffness_basis']
    
    ransform = find_transform(old_basis, new_basis)



def plot_tau_P(data, label, n_coeffs=4):
    if not (data['n_x'] == data['n_x'].iloc[0]).all():
        return 0
    else:
        size = data['n_x'].iloc[0]
    
    rel_U_tot = data['U_tot (J/m)'] - data['U_tot (J/m)'][np.argmin(data['alpha'])]
    rel_U_strain = data['U_strain (J/m)'] - data['U_strain (J/m)'][np.argmin(data['alpha'])]
    rel_U_mis = data['U_misalign (J/m)'] - data['U_misalign (J/m)'][np.argmin(data['alpha'])]

    plt.plot(data['alpha'], rel_U_tot, 'go', label='Total')
    plt.plot(data['alpha'], rel_U_strain, 'bs', label='Strain')
    plt.plot(data['alpha'], rel_U_mis, 'r^', label='Misalign')
    plt.xlim((0,1))
    plt.title('Energy changes with alpha for '+label+ ' {}x{}'.format(size, size))
    plt.xlabel('Alpha')
    plt.ylabel('Energy (J/m)')
    plt.savefig(label + ' {}x{} Energies vs a.pdf'.format(size, size))
    plt.clf()
    
    fit_coeffs = tuple(opt.curve_fit(fourier_series, data['alpha'], rel_U_tot, p0=np.ones(n_coeffs,))[0])


    alpha_at_max_grad = opt.minimize_scalar(negative_f_prime, args=fit_coeffs).x
    max_dU_by_dalpha = f_prime(alpha_at_max_grad, *fit_coeffs) #  this is already per line length

    # tau_p = du/d(alpha) / (lb^2)

    for system in inputs.systems:
        if label == system['label']:
            b = system['latt_params'][0]
            d = system['latt_params'][1]
            c = system['latt_params'][2]
            if 'stiffness_basis' in system:
                elastic_tensor = transform_elastic_tensor(system['elastic_tensor'], slip_system=system)
            else:
                elastic_tensor = system['elastic_tensor']

            G = np.sum(elastic_tensor[:,-1])

            tau_p = max_dU_by_dalpha / (b * constants.angstrom)**2
            line_energy = np.amin(data['U_tot (J/m)'])
            print(label, size, 'unit cells from origin')
            print('G = {:g}GPa'.format(G))
            print('Tau_p (MPa) = ', tau_p * 1e-6)
            print('Tau_p / c_44 = ', tau_p / (G*1e9))
            print('Line Energy = {:g} J/m'.format(np.amin(data['U_tot (J/m)'])))
            print('__________________________________________________________________\n')
            summary_results = ('{}, {}, {}, {}, {}, {}, '.format(line_energy,
                                                                 G,
                                                                 tau_p * 1e-6,
                                                                 2 * size * c,
                                                                 size,
                                                                 d/b)
                               + label 
                               + '\n')
            print(summary_results)
            with open('summary.csv', 'at') as summary_file:
                summary_file.write(summary_results)
            
            
            sampled_alphas = np.linspace(0, 1, num=250)
            fitted_energies = fourier_series(sampled_alphas, *fit_coeffs)
            
            empirical_fit = np.zeros((250,2))
            empirical_fit[:,0] = sampled_alphas
            empirical_fit[:,1] = fitted_energies
            
            np.savetxt("{}x{}_U_tot_vs_a_empirical_fit.csv".format(size, size),
                       empirical_fit, 
                       delimiter=',', 
                       header='alpha,energy')
            
            
            plt.clf()
            plt.plot(sampled_alphas, fitted_energies, 'g-', label='Fitted')
            plt.plot(data['alpha'], rel_U_tot, 'go', label='Modelled')
            plt.xlim((0,1))
            plt.title('Energy vs alpha for ' + label + ' {}x{}'.format(size, size))
            plt.ylabel('Relative Line Energy (J/m)')
            plt.xlabel('Alpha')
            plt.figtext(0.6, 0.2, 'Tau_p = {:g} MPa\nT_p/G = {:2e}'.format(tau_p * 1e-6, tau_p / (G*1e9)))

            plt.savefig(label + ' {}x{} U_tot vs a.pdf'.format(size, size))
            plt.clf()
            

                
            
    return tau_p

with open('temp.csv', 'wt') as out_file:
    out_file.write('label,alpha,n_x,n_y,n_z,b,d,c,U_tot (J/m),U_misalign (J/m),U_strain (J/m),width,c1,c2,c3,\n')
    for filename in glob.glob('results*.csv'):
        with open(filename, 'rt') as in_file:
            for line in in_file:
                out_file.write(line)

results = pandas.read_csv('temp.csv')

with open('summary.csv', 'wt') as summary_file:
    summary_file.write('Line Energy J/m, G GPa, Tau_p MPa, Size nm, size unit cells, d_b_ratio, Phase\n')

label = 'copper'
size = 800
results_slice = results[results['label'] == label]
results_slice = results_slice[results_slice['n_x'] == size]
plot_tau_P(results_slice, label)
