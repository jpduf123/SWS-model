# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 15:20:54 2020

@author: jpduf
"""

#%% replicating Tononi model
#IF YOU WANT TO USE BRIAN2TOOLS YOU NEED TO USE PYTHON 3.6
#CAVE I changed the Idk activation function to **-3.5 as it seemed to be an error, it was 3.5 in the papers but that produced an activation function that was activated at hyperpolarized states so clearly something wrong
#design the Chebyshev filter stuff

#CAVE the Iks (slow inacitvating potassium current) only appears in the 2009 paper, and not in 2005 and 2010 paper!

import os
parent_path = 'D:\\JP OneDrive\OneDrive\\Dokumente\\computational\\UP states models\\tononi model' # where I load the functions module
# parent_path = 'C:\\Users\\jpduf\\OneDrive\\Dokumente\\computational\\UP states models\\tononi model'
os.chdir(parent_path)

from brian2 import * #no need to import matplotlib or numpy as PyLab is imported with brian2
from scipy import signal
from functions import * #the functions I wrote (appended to the end of thi script too)
import pickle
from datetime import datetime 
#from brian2tools import *
#clear_cache('cython')

#DO YOU WANT TO REDO THE CONNECTION INDICES OR JUST LOAD THEM FROM A PREVIOUSLY SAVED FILE, doing the indices takes a several minutes especially if doing the whole network
redo_connection_indices = False
connection_indices_path = 'D:\\Computational results\\2021-04-11 1301 0_11\\connections_indices_grid_size_20' # the path to use if you want to reload them 

##path for results and where to store connection indices if you redo them for this run
results_dir = '''D:\\Computational results'''
curr_time = datetime.today().strftime('%Y-%m-%d %H%M')
curr_results = os.path.join(results_dir, curr_time)
os.mkdir(curr_results)

# parameters for sleep or no?
sleep = True

# all the currents every cell has --> this is also required in the functions module so change that there too!
current_list = ['Inal', 'Ikl', 'Ih', 'It', 'Inap', 'Idk', 'Iks', 'Ikca', 'Isyn_ampa', 'Isyn_nmda', 'Isyn_gaba_a', 'Isyn_gaba_b']
int_current_list = ['Inal', 'Ikl', 'Ih', 'It', 'Inap', 'Idk', 'Iks', 'Ikca']
syn_current_list = ['Isyn_ampa', 'Isyn_nmda', 'Isyn_gaba_a', 'Isyn_gaba_b']
    
# plot_voltage_dependency_curve('1/(1 + (0.25*D)**(-3.5))', 'D', lower_bound = 0, upper_bound = 1)
# plot_voltage_dependency_curve('1/(1 + (0.25*D)**-3.5)', 'D')

# need to define this function here because of the globals() call
def get_var_name(var):
    '''
    Parameters
    ----------
    var : any object

    Returns
    -------
    var_name : the actual name of the object instance as string
    '''
    for x, oid in globals().items():
        if oid is var:
            return x
    
#%% base parameters
    
defaultclock.dt = 0.1*ms

#Autapses? I say no here but it isn't specified in the papers. If changed then redo connection indices with keyword changed
autapses = False

Ena = 30*mV
Ek = -90*mV
Eh = -43*mV
Eca = 140*mV # CAVE this is not in the papers I just took the one from Bazhenov

# ------------------------------------------------------- general layout ----------------------------------------------------------------
grid_size = 20 #30x30 grid for each layer

CX_layers = ['L2', 'L4', 'L5']

Neurongroup_names = ['exc_L2', 'exc_L4', 'exc_L5', 'inh_L2', 'inh_L4', 'inh_L5', 'T_exc', 'T_inh', 'NRT']

cells_per_CX_pnt = 3
cells_per_T_pnt = 2
cells_per_NRT_pnt = 1

exc_cells_per_CX_pnt = 2
inh_cells_per_CX_pnt = 1
exc_cells_per_T_pnt = 1
inh_cells_per_T_pnt = 1


cells_per_CX_layer = grid_size**2*cells_per_CX_pnt
exc_cells_per_CX_layer = grid_size**2*exc_cells_per_CX_pnt
inh_cells_per_CX_layer = grid_size**2*inh_cells_per_CX_pnt

cells_per_T_layer = grid_size**2*cells_per_T_pnt
exc_cells_per_T_layer = grid_size**2*exc_cells_per_T_pnt
inh_cells_per_T_layer = grid_size**2*inh_cells_per_T_pnt

cells_per_NRT_layer = grid_size**2*cells_per_NRT_pnt



#indices of cortical layer
cortex_indx = linspace(1, cells_per_CX_layer, cells_per_CX_layer, dtype = int).reshape(cells_per_CX_pnt, grid_size, grid_size) #make an array map with neuron numbers. This makes n 30x30 arrays with neuron numbers 1-900 and 901-1800 etc.... Used for the topographic mapping 
cortex_exc_indx = linspace(1, exc_cells_per_CX_layer, exc_cells_per_CX_layer, dtype = int).reshape(exc_cells_per_CX_pnt, grid_size, grid_size)
cortex_inh_indx = linspace(1, inh_cells_per_CX_layer, inh_cells_per_CX_layer, dtype = int).reshape(inh_cells_per_CX_pnt, grid_size, grid_size)

#total th indices
T_indx = linspace(1, cells_per_T_layer, cells_per_T_layer, dtype = int).reshape(cells_per_T_pnt, grid_size, grid_size)

#indices of matrix and core cells in thalamus
T_exc_indx = linspace(1, exc_cells_per_T_layer, exc_cells_per_T_layer, dtype = int).reshape(exc_cells_per_T_pnt, grid_size, grid_size)
matrix_cells_ratio = 0.2 #how many cells in thalamus are matrix cells

matrix_indx = T_exc_indx.copy()
matrix_indx_numbers = linspace(1, exc_cells_per_T_pnt*grid_size**2, int((exc_cells_per_T_pnt*grid_size**2)*0.2), dtype = int)
for elem in nditer(matrix_indx):
    if elem not in matrix_indx_numbers:
        indx = where(matrix_indx == elem)
        matrix_indx[indx[0][0], indx[1][0], indx[2][0]] = 0        

core_indx = T_exc_indx.copy()
for elem in nditer(core_indx):
    if elem in matrix_indx_numbers:
        indx = where(core_indx == elem)
        core_indx[indx[0][0], indx[1][0], indx[2][0]] = 0 
        
T_inh_indx = linspace(1, inh_cells_per_T_layer, inh_cells_per_T_layer, dtype = int).reshape(inh_cells_per_T_pnt, grid_size, grid_size)

NRT_indx = linspace(1, cells_per_NRT_layer, cells_per_NRT_layer, dtype = int).reshape(cells_per_NRT_pnt, grid_size, grid_size)

#%% Connectivity parameters
exc_connections_CX = ['L2_L2_exc', 'L4_L4_exc', 'L5_L5_exc', 'L2_L5_exc', 'L4_L2_exc', 'L5_L4_exc', 'L5_L2_exc']
inh_connections_CX = ['L2_L2_inh', 'L4_L4_inh', 'L5_L5_inh', 'L2_L2_column_inh', 'L2_L4_inh', 'L2_L5_inh']
cortical_connections = exc_connections_CX + inh_connections_CX

exc_connections_th = ['L5_NRT_exc', 'L5_Tcore_exc', 'L5_Tmatrix_exc', 'Tcore_L4_exc', 'Tcore_L5_exc', 'Tmatrix_L2_exc', 'Tmatrix_L5_exc', 'T_NRT_exc']
inh_connections_th = ['T_T_inh', 'NRT_T_gaba_a_inh', 'NRT_T_gaba_b_inh']
thalamic_connections = exc_connections_th + inh_connections_th

interarea_connections = ['L2_pre_L4_exc', 'L5_post_L2_exc', 'L5_post_L4_exc']

connections_list_total = cortical_connections + thalamic_connections + interarea_connections

exc_connections_total = exc_connections_CX + exc_connections_th + interarea_connections
inh_connections_total = inh_connections_CX + inh_connections_th
inh_connections_gaba_a_total = inh_connections_total
inh_connections_gaba_a_total.remove('NRT_T_gaba_b_inh')

thalamocortical_connections = ['Tcore_L4_exc', 'Tcore_L5_exc', 'Tmatrix_L2_exc', 'Tmatrix_L5_exc'] 
corticothalamic_connections = ['L5_NRT_exc', 'L5_Tcore_exc', 'L5_Tmatrix_exc'] 
intrathalamic_connections = ['T_NRT_exc','T_T_inh', 'NRT_T_gaba_a_inh', 'NRT_T_gaba_b_inh']


# ------------------------------------------------------------------ Connection table --------------------------------------------------------
std_connections = 7.5 # standard deviation of the Gaussian: 7.5 in the original 2005 tononi paper. they never specify it again after so I guess it't the same for the 2009 and 2010 papers

#intra-area connections
#intra-laminar excitatory connections
L2_L2_exc_pmax = 0.1
L2_L2_exc_radius = 12

L4_L4_exc_pmax = 0.05
L4_L4_exc_radius = 7

L5_L5_exc_pmax = 0.1
L5_L5_exc_radius = 12

#intra-laminar inhibitory connections
L2_L2_inh_pmax = 0.25
L2_L2_inh_radius = 7

L4_L4_inh_pmax = 0.25
L4_L4_inh_radius = 7

L5_L5_inh_pmax = 0.25
L5_L5_inh_radius = 7

#inter-laminar excitatory connections --> basically the cortical column
L2_L5_exc_pmax = 1
L2_L5_exc_radius = 2

L4_L2_exc_pmax = 1
L4_L2_exc_radius = 2

L5_L2_exc_pmax = 1
L5_L2_exc_radius = 2

L5_L4_exc_pmax = 1
L5_L4_exc_radius = 2

#inter-laminar inhibitory connections --> they have another L2_L2 connection, not too sure if that's a typo but it's in all the papers
L2_L2_column_inh_pmax = 1
L2_L2_column_inh_radius = 2

L2_L4_inh_pmax = 1
L2_L4_inh_radius = 2

L2_L5_inh_pmax = 1
L2_L5_inh_radius = 2

#corticothalamic connections
L5_NRT_exc_pmax = 0.15
L5_NRT_exc_radius = 12

L5_Tcore_exc_pmax = 0.15
L5_Tcore_exc_radius = 12

L5_Tmatrix_exc_pmax = 1
L5_Tmatrix_exc_radius = 2

#thalamocortical connections
Tcore_L4_exc_pmax = 0.2
Tcore_L4_exc_radius = 4

Tcore_L5_exc_pmax = 0.2
Tcore_L5_exc_radius = 4

Tmatrix_L2_exc_pmax = 0.1
Tmatrix_L2_exc_radius = 12

Tmatrix_L5_exc_pmax = 0.1
Tmatrix_L5_exc_radius = 12

#thalamic connections
T_NRT_exc_pmax = 1
T_NRT_exc_radius = 2

T_T_inh_pmax = 0.3
T_T_inh_radius = 2

NRT_T_gaba_a_inh_pmax = 0.4
NRT_T_gaba_a_inh_radius = 5

NRT_T_gaba_b_inh_pmax = 0.2
NRT_T_gaba_b_inh_radius = 5


#inter-area connections
L2_pre_L4_exc_pmax = 0.2
L2_pre_L4_exc_radius = 12

L5_post_L2_exc_pmax = 0.15
L5_post_L2_exc_radius = 12

L5_post_L5_exc_pmax = 0.15
L5_post_L5_exc_radius = 12

# ----------------------------------------------------- make connection indices ---------------------------------------------------------------

 #if 'connection indices' not in os.getcwd() else None # change to where I save the connection indices to save them/NRTrieve them
connection_indices_dict = {}
if redo_connection_indices:
    # os.chdir(curr_results)
    # connections_dir = f'{curr_results}\\connections_indices_grid_size_{grid_size}'
    # os.mkdir(connections_dir)
    # os.chdir(connections_dir)
    os.chdir(connection_indices_path)
    for connection in connections_list_total:
        print(f'working on {connection}')
        p_max = globals()[connection + '_pmax']
        radius = globals()[connection + '_radius']
        if connection in cortical_connections:
            target = cortex_indx
            if 'exc' in connection:
                source = cortex_exc_indx
            elif 'inh' in connection:
                source = cortex_inh_indx
            # sort target cells into exc and inh cells
            connection_indices_dict[connection + '_con_ind'] = make_connection_indices(source, target, p_max, radius, std_connections, grid_size, layers_for_progress_report = connection)
            # savetxt(connection, connection_indices_dict[connection + '_con_ind'], delimiter = ',')
            connection_indices_dict[connection + '_exc_con_ind'] = connection_indices_dict[connection + '_con_ind'][:, np.where(connection_indices_dict[connection + '_con_ind'][1] <= exc_cells_per_CX_layer)[0]] - 1 # minus one because neurons 1-1800 in the function but indexed 0-1799 in NeuronGroup
            savetxt(connection + '_exc_con_ind', connection_indices_dict[connection + '_exc_con_ind'], delimiter = ',')
            connection_indices_dict[connection + '_inh_con_ind'] = connection_indices_dict[connection + '_con_ind'][:, np.where(connection_indices_dict[connection + '_con_ind'][1] > exc_cells_per_CX_layer)[0]] - 1 # 
            connection_indices_dict[connection + '_inh_con_ind'][1] = connection_indices_dict[connection + '_inh_con_ind'][1] - exc_cells_per_CX_layer # minus the number of excitatory cells to get the indices to range from 0-899
            savetxt(connection + '_inh_con_ind', connection_indices_dict[connection + '_inh_con_ind'], delimiter = ',')
        elif connection in thalamocortical_connections:
            target = cortex_indx
            if 'matrix' in connection:
                source = matrix_indx
            elif 'core' in connection:
                source = core_indx
            # sort target cells into exc and inh cells
            connection_indices_dict[connection + '_con_ind'] = make_connection_indices(source, target, p_max, radius, std_connections, grid_size, layers_for_progress_report = connection)
            # savetxt(connection, connection_indices_dict[connection + '_con_ind'], delimiter = ',')
            connection_indices_dict[connection + '_exc_con_ind'] = connection_indices_dict[connection + '_con_ind'][:, np.where(connection_indices_dict[connection + '_con_ind'][1] <= exc_cells_per_CX_layer)[0]] - 1 # minus one because neurons 1-1800 in the function but indexed 0-1799 in NeuronGroup
            savetxt(connection + '_exc_con_ind', connection_indices_dict[connection + '_exc_con_ind'], delimiter = ',')
            connection_indices_dict[connection + '_inh_con_ind'] = connection_indices_dict[connection + '_con_ind'][:, np.where(connection_indices_dict[connection + '_con_ind'][1] > exc_cells_per_CX_layer)[0]] - 1 # 
            connection_indices_dict[connection + '_inh_con_ind'][1] = connection_indices_dict[connection + '_inh_con_ind'][1] - exc_cells_per_CX_layer # minus the number of excitatory cells to get the indices to range from 0-899
            savetxt(connection + '_inh_con_ind', connection_indices_dict[connection + '_inh_con_ind'], delimiter = ',')
        elif connection in corticothalamic_connections:
            source = cortex_exc_indx # all corticothalamic connections are excitatory
            if 'NRT' in connection:
                target = NRT_indx #only exc_inh connections in this case
                connection_indices_dict[connection + '_inh_con_ind'] = make_connection_indices(source, target, p_max, radius, std_connections, grid_size, layers_for_progress_report = connection) - 1 
                savetxt(connection + '_inh_con_ind', connection_indices_dict[connection + '_inh_con_ind'], delimiter = ',')
            elif 'matrix' in connection:
                target = vstack((matrix_indx, (T_inh_indx + exc_cells_per_T_layer)))
                connection_indices_dict[connection + '_con_ind'] = make_connection_indices(source, target, p_max, radius, std_connections, grid_size, layers_for_progress_report = connection)
                # savetxt(connection, connection_indices_dict[connection + '_con_ind'], delimiter = ',')
                connection_indices_dict[connection + '_exc_con_ind'] = connection_indices_dict[connection + '_con_ind'][:, np.where(connection_indices_dict[connection + '_con_ind'][1] <= exc_cells_per_T_layer)[0]] - 1 # minus one because neurons 1-1800 in the function but indexed 0-1799 in NeuronGroup
                savetxt(connection + '_exc_con_ind', connection_indices_dict[connection + '_exc_con_ind'], delimiter = ',')
                connection_indices_dict[connection + '_inh_con_ind'] = connection_indices_dict[connection + '_con_ind'][:, np.where(connection_indices_dict[connection + '_con_ind'][1] > exc_cells_per_T_layer)[0]] - 1 # 
                connection_indices_dict[connection + '_inh_con_ind'][1] = connection_indices_dict[connection + '_inh_con_ind'][1] - exc_cells_per_T_layer # minus the number of excitatory cells to get the indices to range from 0-899
                savetxt(connection + '_inh_con_ind', connection_indices_dict[connection + '_inh_con_ind'], delimiter = ',')
            elif 'core' in connection:
                target = vstack((core_indx, (T_inh_indx + exc_cells_per_T_layer)))
                connection_indices_dict[connection + '_con_ind'] = make_connection_indices(source, target, p_max, radius, std_connections, grid_size, layers_for_progress_report = connection)
                # savetxt(connection, connection_indices_dict[connection + '_con_ind'], delimiter = ',')
                connection_indices_dict[connection + '_exc_con_ind'] = connection_indices_dict[connection + '_con_ind'][:, np.where(connection_indices_dict[connection + '_con_ind'][1] <= exc_cells_per_T_layer)[0]] - 1 # minus one because neurons 1-1800 in the function but indexed 0-1799 in NeuronGroup
                savetxt(connection + '_exc_con_ind', connection_indices_dict[connection + '_exc_con_ind'], delimiter = ',')
                connection_indices_dict[connection + '_inh_con_ind'] = connection_indices_dict[connection + '_con_ind'][:, np.where(connection_indices_dict[connection + '_con_ind'][1] > exc_cells_per_T_layer)[0]] - 1 # 
                connection_indices_dict[connection + '_inh_con_ind'][1] = connection_indices_dict[connection + '_inh_con_ind'][1] - exc_cells_per_T_layer # minus the number of excitatory cells to get the indices to range from 0-899
                savetxt(connection + '_inh_con_ind', connection_indices_dict[connection + '_inh_con_ind'], delimiter = ',')
        elif connection in intrathalamic_connections:
            if 'T_NRT' in connection: 
                source = T_exc_indx
                target = NRT_indx
                connection_indices_dict[connection + '_inh_con_ind'] = make_connection_indices(source, target, p_max, radius, std_connections, grid_size, layers_for_progress_report = connection) - 1
                savetxt(connection + '_inh_con_ind', connection_indices_dict[connection + '_inh_con_ind'], delimiter = ',')
            elif 'T_T' in connection:
                source = T_inh_indx
                target = T_indx
                connection_indices_dict[connection + '_con_ind'] = make_connection_indices(source, target, p_max, radius, std_connections, grid_size, layers_for_progress_report = connection)
                # savetxt(connection, connection_indices_dict[connection + '_con_ind'], delimiter = ',')
                connection_indices_dict[connection + '_exc_con_ind'] = connection_indices_dict[connection + '_con_ind'][:, np.where(connection_indices_dict[connection + '_con_ind'][1] <= exc_cells_per_T_layer)[0]] - 1 # minus one because neurons 1-1800 in the function but indexed 0-1799 in NeuronGroup
                savetxt(connection + '_exc_con_ind', connection_indices_dict[connection + '_exc_con_ind'], delimiter = ',')
                connection_indices_dict[connection + '_inh_con_ind'] = connection_indices_dict[connection + '_con_ind'][:, np.where(connection_indices_dict[connection + '_con_ind'][1] > exc_cells_per_T_layer)[0]] - 1 # 
                connection_indices_dict[connection + '_inh_con_ind'][1] = connection_indices_dict[connection + '_inh_con_ind'][1] - exc_cells_per_T_layer # minus the number of excitatory cells to get the indices to range from 0-899
                savetxt(connection + '_inh_con_ind', connection_indices_dict[connection + '_inh_con_ind'], delimiter = ',')
            elif 'NRT_T' in connection:
                source = NRT_indx
                target = T_indx
                connection_indices_dict[connection + '_con_ind'] = make_connection_indices(source, target, p_max, radius, std_connections, grid_size, layers_for_progress_report = connection)
                # savetxt(connection, connection_indices_dict[connection + '_con_ind'], delimiter = ',')
                connection_indices_dict[connection + '_exc_con_ind'] = connection_indices_dict[connection + '_con_ind'][:, np.where(connection_indices_dict[connection + '_con_ind'][1] <= exc_cells_per_T_layer)[0]] - 1 # minus one because neurons 1-1800 in the function but indexed 0-1799 in NeuronGroup
                savetxt(connection + '_exc_con_ind', connection_indices_dict[connection + '_exc_con_ind'], delimiter = ',')
                connection_indices_dict[connection + '_inh_con_ind'] = connection_indices_dict[connection + '_con_ind'][:, np.where(connection_indices_dict[connection + '_con_ind'][1] > exc_cells_per_T_layer)[0]] - 1 # 
                connection_indices_dict[connection + '_inh_con_ind'][1] = connection_indices_dict[connection + '_inh_con_ind'][1] - exc_cells_per_T_layer # minus the number of excitatory cells to get the indices to range from 0-899
                savetxt(connection + '_inh_con_ind', connection_indices_dict[connection + '_inh_con_ind'], delimiter = ',')
        elif connection in interarea_connections:
            source = cortex_exc_indx
            target = cortex_indx
            connection_indices_dict[connection + '_con_ind'] = make_connection_indices(source, target, p_max, radius, std_connections, grid_size, layers_for_progress_report = connection)
            # savetxt(connection, connection_indices_dict[connection + '_con_ind'], delimiter = ',')
            connection_indices_dict[connection + '_exc_con_ind'] = connection_indices_dict[connection + '_con_ind'][:, np.where(connection_indices_dict[connection + '_con_ind'][1] <= exc_cells_per_CX_layer)[0]] - 1 # minus one because neurons 1-1800 in the function but indexed 0-1799 in NeuronGroup
            savetxt(connection + '_exc_con_ind', connection_indices_dict[connection + '_exc_con_ind'], delimiter = ',')
            connection_indices_dict[connection + '_inh_con_ind'] = connection_indices_dict[connection + '_con_ind'][:, np.where(connection_indices_dict[connection + '_con_ind'][1] > exc_cells_per_CX_layer)[0]] - 1 # 
            connection_indices_dict[connection + '_inh_con_ind'][1] = connection_indices_dict[connection + '_inh_con_ind'][1] - exc_cells_per_CX_layer # minus the number of excitatory cells to get the indices to range from 0-899
            savetxt(connection + '_inh_con_ind', connection_indices_dict[connection + '_inh_con_ind'], delimiter = ',')
else:
    os.chdir(connection_indices_path)    
    for filename in os.listdir():
        #load the connections in a dict
        connection_indices_dict[filename] = loadtxt(filename, dtype = int, delimiter = ',')
        
os.chdir(parent_path)

#%% synaptic channels
tau_p = 200*ms #time constant for recovery of short-term depression, is 200ms for all synapses

#- ------------------------------------------------------------- AMPA receptors -----------------------------------------------
E_ampa = 0*mV

tau_1_ampa = 0.5*ms
tau_2_ampa = 2.4*ms
t_peak_ampa = (tau_1_ampa*tau_2_ampa/(tau_2_ampa - tau_1_ampa))*np.log(tau_2_ampa/tau_1_ampa)
delta_p_ampa = 0.075

ampa_str = 0.133 #with 0.133 and the individual connection strenghts from the paper it gets VERY bursty and short up states

g_max_ampa_L2_L2_exc = 0.38*ampa_str#0.133
g_max_ampa_L4_L4_exc = 0.38*ampa_str#0.133
g_max_ampa_L5_L5_exc = 0.38*ampa_str#0.133
g_max_ampa_L2_L5_exc = 1.88*ampa_str#0.133
g_max_ampa_L4_L2_exc = 1.88*ampa_str#0.133
g_max_ampa_L5_L2_exc = 0.47*ampa_str#0.133
g_max_ampa_L5_L4_exc = 0.47*ampa_str#0.133
g_max_ampa_L5_NRT_exc = 4.5*ampa_str#0.133
g_max_ampa_L5_Tcore_exc = 1.69*ampa_str#0.133
g_max_ampa_L5_Tmatrix_exc = 2.18*ampa_str#0.133
g_max_ampa_Tcore_L4_exc = 9.0*ampa_str#0.133
g_max_ampa_Tcore_L5_exc = 2.25*ampa_str#0.133
g_max_ampa_Tmatrix_L2_exc = 1.5*ampa_str#0.133
g_max_ampa_Tmatrix_L5_exc = 1.5*ampa_str#0.133
g_max_ampa_T_NRT_exc = 3.75*ampa_str#0.133
g_max_ampa_L2_pre_L4_exc = 0.94*ampa_str
g_max_ampa_L5_post_L2_exc = 0.45*ampa_str 
g_max_ampa_L5_post_L5_exc = 0.45*ampa_str

for con in exc_connections_total:
    globals()['eqs_syn_ampa_' + con] = (f'''
                                        Isyn_ampa_{con}_post = P * g_max_ampa_{con} * ((exp(-(t-t_lastspike)/tau_1_ampa) - exp(-(t-t_lastspike)/tau_2_ampa))/(exp(-t_peak_ampa/tau_1_ampa) - exp(-t_peak_ampa/tau_2_ampa))) * (V_post - E_ampa) : volt (summed)
                                        dP/dt = (1-P)/tau_p  : 1 (event-driven) # short-term depression
                                        t_lastspike : second
                                        ''')

on_pre_eqs_ampa = ('''
                t_lastspike = t  
                P += -delta_p_ampa*int(P>delta_p_ampa) - P*int(P<delta_p_ampa)    # make sure P doesn't go below 0 as that would inverse the synaptic current      
                ''')
#------------------------------------------------------------- NMDA receptons -------------------------------------------------
#NMDA: here I model the voltage-dependency of NMDA using the same as in Bazhenov 2016. Tononi uses a dual exponential function for activation (i.e. activation is not instant) but I couldn't find the exact equation so fuck it
E_nmda = 0*mV

tau_1_nmda = 4*ms
tau_2_nmda = 40*ms
t_peak_nmda = (tau_1_nmda*tau_2_nmda/(tau_2_nmda - tau_1_nmda))*np.log(tau_2_nmda/tau_1_nmda)
delta_p_nmda = 0.075

nmda_str = 0.133 #with 0.133 and the individual connection strenghts from the paper it gets VERY bursty and short up states

g_max_nmda_L2_L2_exc = 0.38*nmda_str#0.133
g_max_nmda_L4_L4_exc = 0.38*nmda_str#0.133
g_max_nmda_L5_L5_exc = 0.38*nmda_str#0.133
g_max_nmda_L2_L5_exc = 1.88*nmda_str#0.133
g_max_nmda_L4_L2_exc = 1.88*nmda_str#0.133
g_max_nmda_L5_L2_exc = 0.47*nmda_str#0.133
g_max_nmda_L5_L4_exc = 0.47*nmda_str#0.133
g_max_nmda_L5_NRT_exc = 4.5*nmda_str#0.133
g_max_nmda_L5_Tcore_exc = 1.69*nmda_str#0.133
g_max_nmda_L5_Tmatrix_exc = 2.18*nmda_str#0.133
g_max_nmda_Tcore_L4_exc = 9.0*nmda_str#0.133
g_max_nmda_Tcore_L5_exc = 2.25*nmda_str#0.133
g_max_nmda_Tmatrix_L2_exc = 1.5*nmda_str#0.133
g_max_nmda_Tmatrix_L5_exc = 1.5*nmda_str#0.133
g_max_nmda_T_NRT_exc = 3.75*nmda_str#0.133
g_max_nmda_L2_pre_L4_exc = 0.94*nmda_str
g_max_nmda_L5_post_L2_exc = 0.45*nmda_str 
g_max_nmda_L5_post_L5_exc = 0.45*nmda_str


#you can't have a run_regularly with a summed variable (in this case the Isyn_post). So I have to actually create the equations one by one with the appropiate postsynaptic summed variable
for con in exc_connections_total:
    globals()['eqs_syn_nmda_' + con] = (f'''
                                        Isyn_nmda_{con}_post = P * g_max_nmda_{con} * V_dependency * ((exp(-(t-t_lastspike)/tau_1_nmda) - exp(-(t-t_lastspike)/tau_2_nmda))/(exp(-t_peak_ampa/tau_1_nmda) - exp(-t_peak_ampa/tau_2_nmda))) * (V_post - E_nmda) : volt (summed)
                                        dP/dt = (1-P)/tau_p  : 1 (event-driven) # short-term depression
                                        V_dependency = 1/(1 + exp(-(V_post - (-25*mV))/(12.5*mV))) : 1     # voltage-dependent term (unblocking of Mg block)
                                        t_lastspike : second
                                        ''')
                
on_pre_eqs_nmda = ('''
                t_lastspike = t  
                P += -delta_p_nmda*int(P>delta_p_nmda) - P*int(P<delta_p_nmda)    # make sure P doesn't go below 0 as that would inverse the synaptic current      
                ''')

        
# ------------------------------------------------------------------ GABA A receptors -------------------------------------------------
E_gaba_a_CX = -70*mV
E_gaba_a_T = -80*mV

tau_1_gaba_a = 1*ms
tau_2_gaba_a = 7*ms
t_peak_gaba_a = (tau_1_gaba_a*tau_2_gaba_a/(tau_2_gaba_a - tau_1_gaba_a))*np.log(tau_2_gaba_a/tau_1_gaba_a)
delta_p_gaba_a = 0.0375

gaba_a_str = 0.4 #0.33 in paper

g_max_gaba_a_L2_L2_inh = 0.95*gaba_a_str
g_max_gaba_a_L4_L4_inh = 0.95*gaba_a_str
g_max_gaba_a_L5_L5_inh = 0.88*gaba_a_str
g_max_gaba_a_L2_L2_column_inh = 1.19*gaba_a_str
g_max_gaba_a_L2_L4_inh = 1.19*gaba_a_str
g_max_gaba_a_L2_L5_inh = 1.19*gaba_a_str
g_max_gaba_a_T_T_inh = 1.59*gaba_a_str
g_max_gaba_a_NRT_T_gaba_a_inh = 0.35*gaba_a_str

for con in inh_connections_gaba_a_total:
    if 'T' not in con: #different E_gaba for th and CX
        globals()['eqs_syn_gaba_a_' + con] = (f'''
                                            Isyn_gaba_a_{con}_post = P * g_max_gaba_a_{con} * ((exp(-(t-t_lastspike)/tau_1_gaba_a) - exp(-(t-t_lastspike)/tau_2_gaba_a))/(exp(-t_peak_ampa/tau_1_gaba_a) - exp(-t_peak_ampa/tau_2_gaba_a))) * (V_post - E_gaba_a_CX) : volt (summed)
                                            dP/dt = (1-P)/tau_p  : 1 (event-driven) # short-term depression
                                            t_lastspike : second
                                            ''')
    
    else:
        globals()['eqs_syn_gaba_a_' + con] = (f'''
                                                Isyn_gaba_a_{con}_post = P * g_max_gaba_a_{con} * ((exp(-(t-t_lastspike)/tau_1_gaba_a) - exp(-(t-t_lastspike)/tau_2_gaba_a))/(exp(-t_peak_ampa/tau_1_gaba_a) - exp(-t_peak_ampa/tau_2_gaba_a))) * (V_post - E_gaba_a_T) : volt (summed)
                                                dP/dt = (1-P)/tau_p  : 1 (event-driven) # short-term depression
                                                t_lastspike : second
                                                ''')
      
on_pre_eqs_gaba_a = ('''
                    t_lastspike = t  
                    P += -delta_p_gaba_a*int(P>delta_p_gaba_a) - P*int(P<delta_p_gaba_a)    # make sure P doesn't go below 0 as that would inverse the synaptic current,so only subtract deltaP if that that doesnt bring P under 0      
                    ''')
                
# -------------------------------------------------------------- GABA B -----------------------------------------------------
E_gaba_b = -90*mV
delta_p_gaba_b = 0.0375

T_syn = 0.3*ms #that's the length of the neurotransmitter pulse in the Bazhenov paper. Tononi and Bazhenov use the same gaba_b model but Tononi does not specify the length of the neurotransmitter pulse

# K1_c = 0.18/ms
# K2_c = 0.0096/ms
# K3_c = 0.19/ms
# K4_c = 0.06/ms
# Kd_c = 17.8 #dissociation constant

K1_t = 0.66/ms
K2_t = 0.02/ms
K3_t = 0.083/ms
K4_t = 0.0079/ms
Kd_t = 100

g_max_gaba_b_NRT_T_gaba_b_inh = 1 # only one gaba_b connection (NRT_T)

# eqs_syn_gaba_b_CX = ('''
#                 Isyn_gaba_b_post = g_max_gaba_b * G**4/(G**4 + Kd_c) * (V_post - E_gaba_b) : volt (summed)
#                 dG/dt = K3_c * R - K4_c * G : 1                     # G is the concentration of activated G protein (what drives the actual synaptic event in the end)
#                 dR/dt = K1_c * S * P * (1-R) - K2_c*R : 1       #R is the fraction of activated receptors, S the strength of each synaptic event (scaled by P in this case)
#                 S = int((t-t_lastspike) < T_syn) : 1            #S is a synaptic event
#                 t_lastspike : second
#                 dP/dt = (1-P)/tau_p  : 1  (event-driven)        # short-term depression                        
#                 ''')
                
eqs_syn_gaba_b_th = ('''
                Isyn_gaba_b_post = g_max_gaba_b_NRT_T_gaba_b_inh * G**4/(G**4 + Kd_t) * (V_post - E_gaba_b) : volt (summed)
                dG/dt = K3_t * R - K4_t * G : 1                 # G is the concentration of activated G protein (what drives the actual synaptic event in the end)
                dR/dt = K1_t * S * P * (1-R) - K2_t*R : 1       #R is the fraction of activated receptors, S the strength of each synaptic event (scaled by P in this case)
                S = int((t-t_lastspike) < T_syn) : 1            #S is a synaptic event
                t_lastspike : second
                dP/dt = (1-P)/tau_p  : 1 (event-driven)         # short-term depression                        
                ''')

on_pre_eqs_gaba_b = ('''
                    t_lastspike = t  
                    P += -delta_p_gaba_b*int(P>delta_p_gaba_b) - P*int(P<delta_p_gaba_b)    # make sure P doesn't go below 0 as that would inverse the synaptic current,so only subtract deltaP if that that doesnt bring P under 0      
                    ''')

#on pre equation is the same for every synapse type --> reset t_lastspike and update P                

#%%Cortex excitatory
#From steady state Vm (approx -85 I think, needs about 62mV of Iapp to spiketrain)
tau_m_CX_exc = 15*ms
thresh_ss_CX_exc = -51*mvolt
tau_thresh_CX_exc = 1*ms
tau_spike_CX_exc = 1.3*ms
time_spike_CX_exc = 1.4*ms

if sleep:
    #conductances in 2010 don't seem to work correctly, when I reduce gkl it at least seems to have some spontaneous firing
    g_kl_CX_exc = 0.55 # changed from 0.55 conductances are unitless here, because no capacitance (no area or volume defined for a cell) but rather a membrane time constant is used
    g_dk_CX_exc = 0.75 #depolarization activated potassium current
    g_ks_CX_exc = 6
    g_nal_CX_exc = 0.05
    # I add the g_nap as neurongroup attribute when I build the model. changed from paper
    g_nap_CX_exc_L2 = 3.3
    g_nap_CX_exc_L4 = 3.5
    g_nap_CX_exc_L5 = 3.5 
    g_spike_CX_exc = 1
    g_h_CX_exc = 2  # changed from paper
    
    
    #conductances from 2005 paper, g_ks_CX_exc and g_nal_CX_exc are not specified
    # g_kl_CX_exc = 1.85
    # g_dk_CX_exc = 1.25 
    # g_ks_CX_exc = 6 #g_ks DOESNT EXIT IN PAPER!!
    # g_nal_CX_exc = 0.05 #doesnt say
    # g_nap_CX_exc = 1.25
    # g_spike_CX_exc = 1
    # g_h_CX_exc = 2
    
else: 
    g_kl_CX_exc = 0.49 #conductances are unitless here, because no capacitance (no area or volume defined for a cell) but rather a membrane time constant is used
    g_dk_CX_exc = 0.75 #depolarization activated potassium current
    g_ks_CX_exc = 6
    g_nal_CX_exc = 0.05
    g_nap_CX_exc = 2
    g_spike_CX_exc = 1
    g_h_CX_exc = 0.4

D_thresh = -10*mV #threshold of the logistic function of D from Idk

poiss_str_CX_exc_L2 = 0.008
poiss_str_CX_exc_L4 = 0.008
poiss_str_CX_exc_L5 = 0.008

# units for currents is volt and not amps because conductance is unitless
for layer in CX_layers:
    globals()['eqs_CX_exc_' + layer] = (f'''
                                        dV/dt = (- Inal - Ikl - Iint - Isyn + Iapp)/tau_m_CX_exc - g_spike_CX_exc * (V - Ek)/tau_spike_CX_exc : volt
                                        Inal = g_nal_CX_exc*(V - Ena) : volt
                                        Ikl = g_kl_CX_exc * (V - Ek) : volt
                                        
                                        Iint = Iks + Inap + Idk + Ih : volt
                                        
                                        Iks = g_ks_CX_exc * m_ks * (V - Ek) : volt                  # slow noninactivating potassium current
                                        dm_ks/dt = (m_ks_ss - m_ks)/tau_m_ks : 1
                                        m_ks_ss = 1/(1 + exp(-(V + 34*mvolt)/(6.5*mvolt))) : 1
                                        tau_m_ks = (8*ms)/(exp(-(V + 55*mvolt)/(30*mvolt)) + exp((V + 55*mvolt)/(30*mvolt))) : second
                                        
                                        Inap = g_nap_CX_exc_{layer} * m_nap ** 3 * (V - Ena) : volt        #they took the same equations as in Compte 2002. Pers Na current activates rapidly near spike threshold and deactivates very slowly
                                        m_nap = 1/(1 + exp(-(V + 55.7*mvolt)/(7.7*mV))) : 1
                                        g_nap_CX_exc_{layer} : 1
                                        
                                        Idk = g_dk_CX_exc * m_dk * (V-Ek) : volt                   #depolarization-activated potassium conductance, replaces Na-dependent K current in Compte --> here the term D combines Ca- and Na dependency by accumulating during depolarization
                                        m_dk = 1/(1 + (0.25*D)**(-3.5)) : 1                    #instantaneous activation, no time constant ever described
                                        dD/dt = D_influx - D*(1-0.001)/(800*ms) : 1
                                        D_influx = 1/(1 + exp(-(V-D_thresh)/(5*mV)))/ms : Hz
                                        
                                        Ih : volt
                                        dm_h/dt = (m_h_ss - m_h)/tau_m_h : 1
                                        m_h_ss = 1/(1 + exp((V + 75*mV)/(5.5*mV))) : 1
                                        tau_m_h = 1*ms/(exp(-14.59 - 0.086*V/mV) + exp(-1.87 + 0.0701*V/mV)) : second
                                        
                                        
                                        dthresh/dt = -(thresh - thresh_ss_CX_exc)/tau_thresh_CX_exc : volt #threshold for spikes
                                        g_spike_CX_exc = int((t - lastspike) < time_spike_CX_exc) : 1
                                        lastspike : second
                                        
                                        Isyn = Isyn_ampa + Isyn_nmda + Isyn_gaba_a + Isyn_gaba_b + Ipoiss: volt
                                        
                                        Isyn_ampa = Isyn_ampa_L2_L2_exc + Isyn_ampa_L4_L4_exc + Isyn_ampa_L5_L5_exc + Isyn_ampa_L2_L5_exc + Isyn_ampa_L4_L2_exc + Isyn_ampa_L5_L2_exc + Isyn_ampa_L5_L4_exc + Isyn_ampa_L5_NRT_exc + Isyn_ampa_L5_Tcore_exc + Isyn_ampa_L5_Tmatrix_exc + Isyn_ampa_Tcore_L4_exc + Isyn_ampa_Tcore_L5_exc + Isyn_ampa_Tmatrix_L2_exc + Isyn_ampa_Tmatrix_L5_exc + Isyn_ampa_T_NRT_exc + Isyn_ampa_L2_pre_L4_exc + Isyn_ampa_L5_post_L2_exc + Isyn_ampa_L5_post_L5_exc: volt
                                        Isyn_ampa_L2_L2_exc: volt
                                        Isyn_ampa_L4_L4_exc : volt
                                        Isyn_ampa_L5_L5_exc : volt
                                        Isyn_ampa_L2_L5_exc : volt
                                        Isyn_ampa_L4_L2_exc : volt
                                        Isyn_ampa_L5_L2_exc : volt
                                        Isyn_ampa_L5_L4_exc : volt 
                                        Isyn_ampa_L5_NRT_exc : volt
                                        Isyn_ampa_L5_Tcore_exc : volt
                                        Isyn_ampa_L5_Tmatrix_exc : volt
                                        Isyn_ampa_Tcore_L4_exc : volt
                                        Isyn_ampa_Tcore_L5_exc : volt
                                        Isyn_ampa_Tmatrix_L2_exc : volt
                                        Isyn_ampa_Tmatrix_L5_exc : volt
                                        Isyn_ampa_T_NRT_exc : volt
                                        Isyn_ampa_L2_pre_L4_exc : volt
                                        Isyn_ampa_L5_post_L2_exc : volt
                                        Isyn_ampa_L5_post_L5_exc : volt
                                        
                                        Isyn_nmda = Isyn_nmda_L2_L2_exc + Isyn_nmda_L4_L4_exc + Isyn_nmda_L5_L5_exc + Isyn_nmda_L2_L5_exc + Isyn_nmda_L4_L2_exc + Isyn_nmda_L5_L2_exc + Isyn_nmda_L5_L4_exc + Isyn_nmda_L5_NRT_exc + Isyn_nmda_L5_Tcore_exc + Isyn_nmda_L5_Tmatrix_exc + Isyn_nmda_Tcore_L4_exc + Isyn_nmda_Tcore_L5_exc + Isyn_nmda_Tmatrix_L2_exc + Isyn_nmda_Tmatrix_L5_exc + Isyn_nmda_T_NRT_exc + Isyn_nmda_L5_post_L2_exc + Isyn_nmda_L5_post_L2_exc + Isyn_nmda_L5_post_L5_exc: volt
                                        Isyn_nmda_L2_L2_exc : volt
                                        Isyn_nmda_L4_L4_exc : volt
                                        Isyn_nmda_L5_L5_exc : volt
                                        Isyn_nmda_L2_L5_exc : volt
                                        Isyn_nmda_L4_L2_exc : volt
                                        Isyn_nmda_L5_L2_exc : volt
                                        Isyn_nmda_L5_L4_exc : volt
                                        Isyn_nmda_L5_NRT_exc : volt
                                        Isyn_nmda_L5_Tcore_exc : volt
                                        Isyn_nmda_L5_Tmatrix_exc : volt
                                        Isyn_nmda_Tcore_L4_exc : volt
                                        Isyn_nmda_Tcore_L5_exc : volt
                                        Isyn_nmda_Tmatrix_L2_exc : volt
                                        Isyn_nmda_Tmatrix_L5_exc : volt
                                        Isyn_nmda_T_NRT_exc : volt
                                        Isyn_nmda_L2_pre_L4_exc : volt
                                        Isyn_nmda_L5_post_L2_exc : volt
                                        Isyn_nmda_L5_post_L5_exc : volt
                        
                                        Isyn_gaba_a = Isyn_gaba_a_L2_L2_inh + Isyn_gaba_a_L4_L4_inh + Isyn_gaba_a_L5_L5_inh + Isyn_gaba_a_L2_L2_column_inh + Isyn_gaba_a_L2_L4_inh + Isyn_gaba_a_L2_L5_inh + Isyn_gaba_a_T_T_inh + Isyn_gaba_a_NRT_T_gaba_a_inh : volt
                                        Isyn_gaba_a_L2_L2_inh : volt
                                        Isyn_gaba_a_L4_L4_inh : volt
                                        Isyn_gaba_a_L5_L5_inh : volt
                                        Isyn_gaba_a_L2_L2_column_inh : volt
                                        Isyn_gaba_a_L2_L4_inh : volt
                                        Isyn_gaba_a_L2_L5_inh : volt
                                        Isyn_gaba_a_T_T_inh : volt
                                        Isyn_gaba_a_NRT_T_gaba_a_inh : volt
                                        
                                        Isyn_gaba_b : volt
                                        
                                        Ipoiss = int(t-t_last_poisson < 5*ms)*poiss_str_CX_exc_{layer}*(V - E_ampa) : volt
                                        t_last_poisson : second
                                        poiss : 1
                                        
                                        Iapp : volt # an external current source you can apply if you want (needs 62mV about to start spiking)
                                      ''')
              
#connect a few cells to check the synaptic connections
# CX_exc = NeuronGroup(3, model = eqs_CX_exc, method = 'rk4', threshold = 'V > thresh',
#                 reset = '''V = Ena
#                             thresh = Ena
#                             lastspike = t''')

# CX_exc.V = -85*mV
# CX_exc.thresh = thresh_ss_CX_exc

# syns_ampa = Synapses(CX_exc, CX_exc, model = eqs_syn_ampa, on_pre = on_pre_eqs, delay = 1*ms, method = 'rk4')
# syns_ampa.connect(i = 0, j = [1,2])

# syns_nmda = Synapses(CX_exc, CX_exc, model = eqs_syn_nmda, on_pre = on_pre_eqs, delay = 1*ms, method = 'rk4')
# syns_nmda.connect(i = 0, j = [1,2])

# syns_gaba_a = Synapses(CX_exc, CX_exc, model = eqs_syn_gaba_a_CX, on_pre = on_pre_eqs, delay = 1*ms, method = 'rk4')
# syns_gaba_a.connect(i = 0, j = [1,2])

# syns_gaba_b = Synapses(CX_exc, CX_exc, model = eqs_syn_gaba_b_CX, on_pre = on_pre_eqs, delay = 1*ms, method = 'rk4')
# syns_gaba_b.connect(i = 0, j = [1,2])
# #you have to explicitely index the synapse variables when assigning them, not like neurongroup variables
# syns_gaba_b.P[:] = 1
# N = NeuronGroup(1, eqs_CX_exc, method = 'rk4', threshold = 'V > thresh', 
#                 reset = '''V = Ena
#                             thresh = Ena
#                             lastspike = t''')

# N.V = -70*mV
# N.D = 0.001
# N.thresh = thresh_ss_CX_exc
# N.run_regularly('Ih = g_h * m_h * (V - Eh)')

# P = PoissonInput(N, 'V', 3, 1*Hz, weight = 1*mV) #unclear how much noise they really have in the 2010 model, they say mean 1Hz 0.5+25mV, but Down states in the figures show more noise than that. in the 2005 model they clearly have a lot more noise

# mon = StateMonitor(N, variables = True, record = True)

# network = Network(N, mon, P)

# network.run(5000*ms, report = 'text')
# plot(mon.t/ms, mon.V[0])
# xlabel('time ms')
# ylabel('volt')


#%% Cortex inhibitory

tau_m_CX_inh = 7*ms
thresh_ss_CX_inh = -53*mvolt
tau_thresh_CX_inh = 1*ms
tau_spike_CX_inh = 0.55*ms
time_spike_CX_inh = 0.75*ms

if sleep:
    #conductances in 2010 don't seem to work correctly, when I reduce gkl it at least seems to have some spontaneous firing
    g_kl_CX_inh = 0.55 # changed from 0.55 conductances are unitless here, because no capacitance (no area or volume defined for a cell) but rather a membrane time constant is used
    g_dk_CX_inh = 0.75 #depolarization activated potassium current
    g_ks_CX_inh = 6
    g_nal_CX_inh = 0.05
    g_nap_CX_inh = 2 #changed from 2. 4 e.g. (with g_kl_CX_inh 0.45 and changed Inap inflection point to -57.7) gives slow wave like behavior in individual cells at ca. 1 Hz. 3.5 gives the cell being just below threshold with Poisson inputs making it fire.
    g_spike_CX_inh = 1

D_thesh = -10*mV #threshold of the logistic function of D from Idk

poiss_str_CX_inh_L2 = 0.007
poiss_str_CX_inh_L4 = 0.007
poiss_str_CX_inh_L5 = 0.007

for layer in CX_layers:
    globals()['eqs_CX_inh_' + layer]  = (f'''
                                        dV/dt = (- Inal - Ikl - Iint - Isyn + Iapp)/tau_m_CX_inh - g_spike_CX_inh * (V - Ek)/tau_spike_CX_inh : volt
                                        Inal = g_nal_CX_inh*(V - Ena) : volt
                                        Ikl = g_kl_CX_inh * (V - Ek) : volt
                        
                                        Iint = Iks + Inap + Idk : volt
                                        
                                        Iks = g_ks_CX_inh * m_ks * (V - Ek) : volt                  # slow noninactivating potassium current
                                        dm_ks/dt = (m_ks_ss - m_ks)/tau_m_ks : 1
                                        m_ks_ss = 1/(1 + exp(-(V + 34*mvolt)/(6.5*mvolt))) : 1
                                        tau_m_ks = (8*ms)/(exp(-(V + 55*mvolt)/(30*mvolt)) + exp((V + 55*mvolt)/(30*mvolt))) : second
                                        
                                        Inap = g_nap_CX_inh * m_nap ** 3 * (V - Ena) : volt        #they took the same equations as in Compte 2002. Pers Na current activates rapidly near spike threshold and deactivates very slowly
                                        m_nap = 1/(1 + exp(-(V + 55.7*mvolt)/(7.7*mV))) : 1
                                        
                                        Idk = g_dk_CX_inh * m_dk * (V-Ek) : volt                    #depolarization-activated potassium conductance, replaces Na-dependent K current in Compte --> here the term D combines Ca- and Na dependency by accumulating during depolarization
                                        m_dk = 1/(1 + (0.25*D)**(-3.5)) : 1
                                        dD/dt = D_influx - D*(1-0.001)/(800*ms) : 1
                                        D_influx = 1/(1 + exp(-(V-D_thresh)/(5*mV)))/ms : Hz
                                        
                                        
                                        dthresh/dt = -(thresh - thresh_ss_CX_inh)/tau_thresh_CX_inh : volt #threshold for spikes, is set to Ena at each crossing and decays back to ss
                                        g_spike_CX_inh = int((t - lastspike) < time_spike_CX_inh) : 1
                                        lastspike : second
                                        
                                        
                                        Isyn = Isyn_ampa + Isyn_nmda + Isyn_gaba_a + Isyn_gaba_b + Ipoiss: volt
                                        
                                        Isyn_ampa = Isyn_ampa_L2_L2_exc + Isyn_ampa_L4_L4_exc + Isyn_ampa_L5_L5_exc + Isyn_ampa_L2_L5_exc + Isyn_ampa_L4_L2_exc + Isyn_ampa_L5_L2_exc + Isyn_ampa_L5_L4_exc + Isyn_ampa_L5_NRT_exc + Isyn_ampa_L5_Tcore_exc + Isyn_ampa_L5_Tmatrix_exc + Isyn_ampa_Tcore_L4_exc + Isyn_ampa_Tcore_L5_exc + Isyn_ampa_Tmatrix_L2_exc + Isyn_ampa_Tmatrix_L5_exc + Isyn_ampa_T_NRT_exc + Isyn_ampa_L2_pre_L4_exc + Isyn_ampa_L5_post_L2_exc + Isyn_ampa_L5_post_L5_exc: volt
                                        Isyn_ampa_L2_L2_exc: volt
                                        Isyn_ampa_L4_L4_exc : volt
                                        Isyn_ampa_L5_L5_exc : volt
                                        Isyn_ampa_L2_L5_exc : volt
                                        Isyn_ampa_L4_L2_exc : volt
                                        Isyn_ampa_L5_L2_exc : volt
                                        Isyn_ampa_L5_L4_exc : volt 
                                        Isyn_ampa_L5_NRT_exc : volt
                                        Isyn_ampa_L5_Tcore_exc : volt
                                        Isyn_ampa_L5_Tmatrix_exc : volt
                                        Isyn_ampa_Tcore_L4_exc : volt
                                        Isyn_ampa_Tcore_L5_exc : volt
                                        Isyn_ampa_Tmatrix_L2_exc : volt
                                        Isyn_ampa_Tmatrix_L5_exc : volt
                                        Isyn_ampa_T_NRT_exc : volt
                                        Isyn_ampa_L2_pre_L4_exc : volt
                                        Isyn_ampa_L5_post_L2_exc : volt
                                        Isyn_ampa_L5_post_L5_exc : volt
                                        
                                        Isyn_nmda = Isyn_nmda_L2_L2_exc + Isyn_nmda_L4_L4_exc + Isyn_nmda_L5_L5_exc + Isyn_nmda_L2_L5_exc + Isyn_nmda_L4_L2_exc + Isyn_nmda_L5_L2_exc + Isyn_nmda_L5_L4_exc + Isyn_nmda_L5_NRT_exc + Isyn_nmda_L5_Tcore_exc + Isyn_nmda_L5_Tmatrix_exc + Isyn_nmda_Tcore_L4_exc + Isyn_nmda_Tcore_L5_exc + Isyn_nmda_Tmatrix_L2_exc + Isyn_nmda_Tmatrix_L5_exc + Isyn_nmda_T_NRT_exc + Isyn_nmda_L5_post_L2_exc + Isyn_nmda_L5_post_L2_exc + Isyn_nmda_L5_post_L5_exc: volt
                                        Isyn_nmda_L2_L2_exc : volt
                                        Isyn_nmda_L4_L4_exc : volt
                                        Isyn_nmda_L5_L5_exc : volt
                                        Isyn_nmda_L2_L5_exc : volt
                                        Isyn_nmda_L4_L2_exc : volt
                                        Isyn_nmda_L5_L2_exc : volt
                                        Isyn_nmda_L5_L4_exc : volt
                                        Isyn_nmda_L5_NRT_exc : volt
                                        Isyn_nmda_L5_Tcore_exc : volt
                                        Isyn_nmda_L5_Tmatrix_exc : volt
                                        Isyn_nmda_Tcore_L4_exc : volt
                                        Isyn_nmda_Tcore_L5_exc : volt
                                        Isyn_nmda_Tmatrix_L2_exc : volt
                                        Isyn_nmda_Tmatrix_L5_exc : volt
                                        Isyn_nmda_T_NRT_exc : volt
                                        Isyn_nmda_L2_pre_L4_exc : volt
                                        Isyn_nmda_L5_post_L2_exc : volt
                                        Isyn_nmda_L5_post_L5_exc : volt
                        
                                        Isyn_gaba_a = Isyn_gaba_a_L2_L2_inh + Isyn_gaba_a_L4_L4_inh + Isyn_gaba_a_L5_L5_inh + Isyn_gaba_a_L2_L2_column_inh + Isyn_gaba_a_L2_L4_inh + Isyn_gaba_a_L2_L5_inh + Isyn_gaba_a_T_T_inh + Isyn_gaba_a_NRT_T_gaba_a_inh : volt
                                        Isyn_gaba_a_L2_L2_inh : volt
                                        Isyn_gaba_a_L4_L4_inh : volt
                                        Isyn_gaba_a_L5_L5_inh : volt
                                        Isyn_gaba_a_L2_L2_column_inh : volt
                                        Isyn_gaba_a_L2_L4_inh : volt
                                        Isyn_gaba_a_L2_L5_inh : volt
                                        Isyn_gaba_a_T_T_inh : volt
                                        Isyn_gaba_a_NRT_T_gaba_a_inh : volt
                                        
                                        Isyn_gaba_b : volt
                                        
                                        Ipoiss = int(t-t_last_poisson < 5*ms)*poiss_str_CX_inh_{layer}*(V - E_ampa) : volt
                                        t_last_poisson : second
                                        poiss : 1
                                        
                                        Iapp : volt
                                        ''')

# N = NeuronGroup(1, eqs_CX_int, method = 'rk4', threshold = 'V > thresh', 
#                 reset = '''V = Ena
#                             thresh = Ena
#                             lastspike = t''')

# N.V = -70*mV
# N.thresh = thresh_ss
# N.D = 0.001

# P = PoissonInput(N, 'V', 1, 1*Hz, weight = 1*mV) #unclear how much noise they really have in the 2010 model, they say mean 1Hz 0.5+25mV, but Down states in the figures show more noise than that. in the 2005 model they clearly have a lot more noise

# mon = StateMonitor(N, variables = True, record = True)

# network = Network(N, mon, P)

# network.run(5000*ms, report = 'text')
# store(name = 'before')
# subplot(121)
# plot(mon.t/ms, mon.V[0])
# xlabel('time ms')
# ylabel('volt')
# subplot(122)
# plot_currents(mon)

#%%Thalamus nucleus cells

tau_m_t_exc = 7*ms
thresh_ss_t_exc = -53*mvolt
tau_thresh_t_exc = 1*ms
tau_spike_t_exc = 0.55*ms
time_spike_t_exc = 0.75*ms

g_kl_t_exc = 0.55 #conductances are unitless here, because no capacitance but rather a time constant is used
g_nal_t_exc = 0.05
g_nap_t_exc = 2
g_h_t_exc = 0.4
g_t_t_exc = 12
g_spike_t_exc = 1

poiss_str_t_exc = 0.17

eqs_T_exc = ('''
            dV/dt = (- Inal - Ikl - Iint - Isyn + Iapp)/tau_m_t_exc - g_spike_t_exc * (V - Ek)/tau_spike_t_exc : volt
            Inal = g_nal_t_exc*(V - Ena) : volt
            Ikl = g_kl_t_exc * (V - Ek) : volt
            
            Iint = Inap + Ih + It : volt
            
            Inap = g_nap_t_exc * m_nap ** 3 * (V - Ena) : volt        #they took the same equations as in Compte 2002. Pers Na current activates rapidly near spike threshold and deactivates very slowly
            m_nap = 1/(1 + exp(-(V + 55.7*mvolt)/(7.7*mV))) : 1
            
            Ih = g_h_t_exc * m_h * (V - Eh) : volt                     #pacemaker current
            dm_h/dt = (m_h_ss - m_h)/tau_m_h : 1
            m_h_ss = 1/(1 + exp((V + 75*mV)/(5.5*mV))) : 1
            tau_m_h = 1*ms/(exp(-14.59 - 0.086*V/mV) + exp(-1.87 + 0.0701*V/mV)) : second
            
            It = g_t_t_exc * m_t * h_t * (V - Eca) : volt                             # low-threshold, fast-activating Ca current
            dm_t/dt = (m_t_ss - m_t)/tau_m_t : 1
            m_t_ss = 1/(1 + exp(-(V + 52*mV)/(7.4*mV))) : 1
            tau_m_t = 0.15*ms/(exp((V + 27*mV)/(10*mV)) + exp(-(V + 102*mV)/(15*mV))) + 0.44*ms : second
            dh_t/dt = (h_t_ss - h_t)/tau_h_t : 1 
            h_t_ss = 1/(1 + exp((V + 80*mV)/(5*mV))) : 1                   
            tau_h_t = 22.7*ms + 0.27*ms/(exp((V + 48*mV)/(4*mV)) + exp(-(V + 407*mV)/(50*mV))) : second
            
            
            dthresh/dt = -(thresh - thresh_ss_t_exc)/tau_thresh_t_exc : volt #threshold for spikes, is set to Ena at each crossing and decays back to ss
            g_spike = int((t - lastspike) < time_spike_t_exc) : 1
            lastspike : second
            
            
            Isyn = Isyn_ampa + Isyn_nmda + Isyn_gaba_a + Isyn_gaba_b + Ipoiss: volt
            
            Isyn_ampa = Isyn_ampa_L2_L2_exc + Isyn_ampa_L4_L4_exc + Isyn_ampa_L5_L5_exc + Isyn_ampa_L2_L5_exc + Isyn_ampa_L4_L2_exc + Isyn_ampa_L5_L2_exc + Isyn_ampa_L5_L4_exc + Isyn_ampa_L5_NRT_exc + Isyn_ampa_L5_Tcore_exc + Isyn_ampa_L5_Tmatrix_exc + Isyn_ampa_Tcore_L4_exc + Isyn_ampa_Tcore_L5_exc + Isyn_ampa_Tmatrix_L2_exc + Isyn_ampa_Tmatrix_L5_exc + Isyn_ampa_T_NRT_exc + Isyn_ampa_L2_pre_L4_exc + Isyn_ampa_L5_post_L2_exc + Isyn_ampa_L5_post_L5_exc: volt
            Isyn_ampa_L2_L2_exc: volt
            Isyn_ampa_L4_L4_exc : volt
            Isyn_ampa_L5_L5_exc : volt
            Isyn_ampa_L2_L5_exc : volt
            Isyn_ampa_L4_L2_exc : volt
            Isyn_ampa_L5_L2_exc : volt
            Isyn_ampa_L5_L4_exc : volt 
            Isyn_ampa_L5_NRT_exc : volt
            Isyn_ampa_L5_Tcore_exc : volt
            Isyn_ampa_L5_Tmatrix_exc : volt
            Isyn_ampa_Tcore_L4_exc : volt
            Isyn_ampa_Tcore_L5_exc : volt
            Isyn_ampa_Tmatrix_L2_exc : volt
            Isyn_ampa_Tmatrix_L5_exc : volt
            Isyn_ampa_T_NRT_exc : volt
            Isyn_ampa_L2_pre_L4_exc : volt
            Isyn_ampa_L5_post_L2_exc : volt
            Isyn_ampa_L5_post_L5_exc : volt
            
            Isyn_nmda = Isyn_nmda_L2_L2_exc + Isyn_nmda_L4_L4_exc + Isyn_nmda_L5_L5_exc + Isyn_nmda_L2_L5_exc + Isyn_nmda_L4_L2_exc + Isyn_nmda_L5_L2_exc + Isyn_nmda_L5_L4_exc + Isyn_nmda_L5_NRT_exc + Isyn_nmda_L5_Tcore_exc + Isyn_nmda_L5_Tmatrix_exc + Isyn_nmda_Tcore_L4_exc + Isyn_nmda_Tcore_L5_exc + Isyn_nmda_Tmatrix_L2_exc + Isyn_nmda_Tmatrix_L5_exc + Isyn_nmda_T_NRT_exc + Isyn_nmda_L5_post_L2_exc + Isyn_nmda_L5_post_L2_exc + Isyn_nmda_L5_post_L5_exc: volt
            Isyn_nmda_L2_L2_exc : volt
            Isyn_nmda_L4_L4_exc : volt
            Isyn_nmda_L5_L5_exc : volt
            Isyn_nmda_L2_L5_exc : volt
            Isyn_nmda_L4_L2_exc : volt
            Isyn_nmda_L5_L2_exc : volt
            Isyn_nmda_L5_L4_exc : volt
            Isyn_nmda_L5_NRT_exc : volt
            Isyn_nmda_L5_Tcore_exc : volt
            Isyn_nmda_L5_Tmatrix_exc : volt
            Isyn_nmda_Tcore_L4_exc : volt
            Isyn_nmda_Tcore_L5_exc : volt
            Isyn_nmda_Tmatrix_L2_exc : volt
            Isyn_nmda_Tmatrix_L5_exc : volt
            Isyn_nmda_T_NRT_exc : volt
            Isyn_nmda_L2_pre_L4_exc : volt
            Isyn_nmda_L5_post_L2_exc : volt
            Isyn_nmda_L5_post_L5_exc : volt
    
            Isyn_gaba_a = Isyn_gaba_a_L2_L2_inh + Isyn_gaba_a_L4_L4_inh + Isyn_gaba_a_L5_L5_inh + Isyn_gaba_a_L2_L2_column_inh + Isyn_gaba_a_L2_L4_inh + Isyn_gaba_a_L2_L5_inh + Isyn_gaba_a_T_T_inh + Isyn_gaba_a_NRT_T_gaba_a_inh : volt
            Isyn_gaba_a_L2_L2_inh : volt
            Isyn_gaba_a_L4_L4_inh : volt
            Isyn_gaba_a_L5_L5_inh : volt
            Isyn_gaba_a_L2_L2_column_inh : volt
            Isyn_gaba_a_L2_L4_inh : volt
            Isyn_gaba_a_L2_L5_inh : volt
            Isyn_gaba_a_T_T_inh : volt
            Isyn_gaba_a_NRT_T_gaba_a_inh : volt
            
            Isyn_gaba_b : volt
            
            Ipoiss = int(t-t_last_poisson < 5*ms)*poiss_str_t_exc*(V - E_ampa) : volt
            t_last_poisson : second
            poiss : 1
            
            Iapp : volt
          ''')


tau_m_t_inh = 7*ms
thresh_ss_t_inh = -53*mvolt
tau_thresh_t_inh = 1*ms
tau_spike_t_inh = 0.55*ms
time_spike_t_inh = 0.75*ms

g_kl_t_inh = 0.55 #conductances are unitless here, because no capacitance but rather a time constant is used
g_nal_t_inh = 0.05
g_ks_t_inh = 6
g_nap_t_inh = 2
g_t_t_inh = 12
g_spike_t_inh = 1

poiss_str_t_inh = 0.005

eqs_T_inh = ('''
            dV/dt = (- Inal - Ikl - Iint - Isyn + Iapp)/tau_m_t_inh - g_spike_t_inh * (V - Ek)/tau_spike_t_inh : volt
            Inal = g_nal_t_inh*(V - Ena) : volt
            Ikl = g_kl_t_inh * (V - Ek) : volt

            Iint = Iks + Inap : volt
            
            Iks = g_ks_t_inh * m_ks * (V - Ek) : volt                  # slow noninactivating potassium current
            dm_ks/dt = (m_ks_ss - m_ks)/tau_m_ks : 1
            m_ks_ss = 1/(1 + exp(-(V + 34*mvolt)/(6.5*mvolt))) : 1
            tau_m_ks = (8*ms)/(exp(-(V + 55*mvolt)/(30*mvolt)) + exp((V + 55*mvolt)/(30*mvolt))) : second
            
            Inap = g_nap_t_inh * m_nap ** 3 * (V - Ena) : volt        
            m_nap = 1/(1 + exp(-(V + 55.7*mvolt)/(7.7*mV))) : 1
            
            
            dthresh/dt = -(thresh - thresh_ss_t_inh)/tau_thresh_t_inh : volt #threshold for spikes, is set to Ena at each crossing and decays back to ss
            g_spike_t_inh = int((t - lastspike) < time_spike_t_inh) : 1
            lastspike : second
            
            
            Isyn = Isyn_ampa + Isyn_nmda + Isyn_gaba_a + Isyn_gaba_b + Ipoiss: volt
            
            Isyn_ampa = Isyn_ampa_L2_L2_exc + Isyn_ampa_L4_L4_exc + Isyn_ampa_L5_L5_exc + Isyn_ampa_L2_L5_exc + Isyn_ampa_L4_L2_exc + Isyn_ampa_L5_L2_exc + Isyn_ampa_L5_L4_exc + Isyn_ampa_L5_NRT_exc + Isyn_ampa_L5_Tcore_exc + Isyn_ampa_L5_Tmatrix_exc + Isyn_ampa_Tcore_L4_exc + Isyn_ampa_Tcore_L5_exc + Isyn_ampa_Tmatrix_L2_exc + Isyn_ampa_Tmatrix_L5_exc + Isyn_ampa_T_NRT_exc + Isyn_ampa_L2_pre_L4_exc + Isyn_ampa_L5_post_L2_exc + Isyn_ampa_L5_post_L5_exc: volt
            Isyn_ampa_L2_L2_exc: volt
            Isyn_ampa_L4_L4_exc : volt
            Isyn_ampa_L5_L5_exc : volt
            Isyn_ampa_L2_L5_exc : volt
            Isyn_ampa_L4_L2_exc : volt
            Isyn_ampa_L5_L2_exc : volt
            Isyn_ampa_L5_L4_exc : volt 
            Isyn_ampa_L5_NRT_exc : volt
            Isyn_ampa_L5_Tcore_exc : volt
            Isyn_ampa_L5_Tmatrix_exc : volt
            Isyn_ampa_Tcore_L4_exc : volt
            Isyn_ampa_Tcore_L5_exc : volt
            Isyn_ampa_Tmatrix_L2_exc : volt
            Isyn_ampa_Tmatrix_L5_exc : volt
            Isyn_ampa_T_NRT_exc : volt
            Isyn_ampa_L2_pre_L4_exc : volt
            Isyn_ampa_L5_post_L2_exc : volt
            Isyn_ampa_L5_post_L5_exc : volt
            
            Isyn_nmda = Isyn_nmda_L2_L2_exc + Isyn_nmda_L4_L4_exc + Isyn_nmda_L5_L5_exc + Isyn_nmda_L2_L5_exc + Isyn_nmda_L4_L2_exc + Isyn_nmda_L5_L2_exc + Isyn_nmda_L5_L4_exc + Isyn_nmda_L5_NRT_exc + Isyn_nmda_L5_Tcore_exc + Isyn_nmda_L5_Tmatrix_exc + Isyn_nmda_Tcore_L4_exc + Isyn_nmda_Tcore_L5_exc + Isyn_nmda_Tmatrix_L2_exc + Isyn_nmda_Tmatrix_L5_exc + Isyn_nmda_T_NRT_exc + Isyn_nmda_L5_post_L2_exc + Isyn_nmda_L5_post_L2_exc + Isyn_nmda_L5_post_L5_exc: volt
            Isyn_nmda_L2_L2_exc : volt
            Isyn_nmda_L4_L4_exc : volt
            Isyn_nmda_L5_L5_exc : volt
            Isyn_nmda_L2_L5_exc : volt
            Isyn_nmda_L4_L2_exc : volt
            Isyn_nmda_L5_L2_exc : volt
            Isyn_nmda_L5_L4_exc : volt
            Isyn_nmda_L5_NRT_exc : volt
            Isyn_nmda_L5_Tcore_exc : volt
            Isyn_nmda_L5_Tmatrix_exc : volt
            Isyn_nmda_Tcore_L4_exc : volt
            Isyn_nmda_Tcore_L5_exc : volt
            Isyn_nmda_Tmatrix_L2_exc : volt
            Isyn_nmda_Tmatrix_L5_exc : volt
            Isyn_nmda_T_NRT_exc : volt
            Isyn_nmda_L2_pre_L4_exc : volt
            Isyn_nmda_L5_post_L2_exc : volt
            Isyn_nmda_L5_post_L5_exc : volt

            Isyn_gaba_a = Isyn_gaba_a_L2_L2_inh + Isyn_gaba_a_L4_L4_inh + Isyn_gaba_a_L5_L5_inh + Isyn_gaba_a_L2_L2_column_inh + Isyn_gaba_a_L2_L4_inh + Isyn_gaba_a_L2_L5_inh + Isyn_gaba_a_T_T_inh + Isyn_gaba_a_NRT_T_gaba_a_inh : volt
            Isyn_gaba_a_L2_L2_inh : volt
            Isyn_gaba_a_L4_L4_inh : volt
            Isyn_gaba_a_L5_L5_inh : volt
            Isyn_gaba_a_L2_L2_column_inh : volt
            Isyn_gaba_a_L2_L4_inh : volt
            Isyn_gaba_a_L2_L5_inh : volt
            Isyn_gaba_a_T_T_inh : volt
            Isyn_gaba_a_NRT_T_gaba_a_inh : volt
            
            Isyn_gaba_b : volt
            
            Ipoiss = int(t-t_last_poisson < 5*ms)*poiss_str_t_inh*(V - E_ampa) : volt
            t_last_poisson : second
            poiss : 1
            
            Iapp : volt
            ''')



#%%Reticular Nucleus

tau_m_NRT = 7*ms

thresh_ss_NRT = -53*mvolt
tau_thresh_NRT = 1*ms
tau_spike_NRT = 0.55*ms
time_spike_NRT = 0.75*ms

g_kl_NRT = 0.4 
g_nal_NRT = 0.05
g_nap_NRT = 2
g_t_NRT = 12
g_kca_NRT = 48
g_spike_NRT = 1

Ca_eq = 0.0024
tau_ca = 160*ms

poiss_str_NRT = 0.005

eqs_NRT = ('''
            dV/dt = (- Inal - Ikl - Iint - Isyn + Iapp)/tau_m_NRT - g_spike_NRT * (V - Ek)/tau_spike_NRT : volt
            Inal = g_nal_NRT*(V - Ena) : volt
            Ikl = g_kl_NRT * (V - Ek) : volt
            
            Iint = Inap + It + Ikca : volt
            
            Inap = g_nap_NRT * m_nap ** 3 * (V - Ena) : volt        
            m_nap = 1/(1 + exp(-(V + 55.7*mvolt)/(7.7*mV))) : 1
         
            It = g_t_NRT * m_t * h_t * (V - Eca) : volt                             # low-threshold, fast-activating Ca current
            dm_t/dt = (m_t_ss - m_t)/tau_m_t : 1
            m_t_ss = 1/(1 + exp(-(V + 52*mV)/(7.4*mV))) : 1
            tau_m_t = 0.15*ms/(exp((V + 27*mV)/(10*mV)) + exp(-(V + 102*mV)/(15*mV))) + 0.44*ms : second
            dh_t/dt = (h_t_ss - h_t)/tau_h_t : 1 
            h_t_ss = 1/(1 + exp((V + 80*mV)/(5*mV))) : 1                   
            tau_h_t = 22.7*ms + 0.27*ms/(exp((V + 48*mV)/(4*mV)) + exp(-(V + 407*mV)/(50*mV))) : second
            
            Ikca = g_kca_NRT * m_kca**2 * (V - Ek) : volt
            dm_kca/dt = (48/ms)*Ca**2*(1 - m_kca) - (0.03/ms)*m_kca : 1
            dCa/dt = -5.18*10**(-6)*(It/volt/ms) + (Ca_eq - Ca)/tau_ca : 1
            
            
            dthresh/dt = -(thresh - thresh_ss_NRT)/tau_thresh_NRT : volt #threshold for spikes, is set to Ena at each crossing and decays back to ss
            g_spike = int((t - lastspike) < time_spike_NRT) : 1
            lastspike : second
            
            
            Isyn = Isyn_ampa + Isyn_nmda + Isyn_gaba_a + Isyn_gaba_b + Ipoiss: volt
            
            Isyn_ampa = Isyn_ampa_L2_L2_exc + Isyn_ampa_L4_L4_exc + Isyn_ampa_L5_L5_exc + Isyn_ampa_L2_L5_exc + Isyn_ampa_L4_L2_exc + Isyn_ampa_L5_L2_exc + Isyn_ampa_L5_L4_exc + Isyn_ampa_L5_NRT_exc + Isyn_ampa_L5_Tcore_exc + Isyn_ampa_L5_Tmatrix_exc + Isyn_ampa_Tcore_L4_exc + Isyn_ampa_Tcore_L5_exc + Isyn_ampa_Tmatrix_L2_exc + Isyn_ampa_Tmatrix_L5_exc + Isyn_ampa_T_NRT_exc + Isyn_ampa_L2_pre_L4_exc + Isyn_ampa_L5_post_L2_exc + Isyn_ampa_L5_post_L5_exc: volt
            Isyn_ampa_L2_L2_exc: volt
            Isyn_ampa_L4_L4_exc : volt
            Isyn_ampa_L5_L5_exc : volt
            Isyn_ampa_L2_L5_exc : volt
            Isyn_ampa_L4_L2_exc : volt
            Isyn_ampa_L5_L2_exc : volt
            Isyn_ampa_L5_L4_exc : volt 
            Isyn_ampa_L5_NRT_exc : volt
            Isyn_ampa_L5_Tcore_exc : volt
            Isyn_ampa_L5_Tmatrix_exc : volt
            Isyn_ampa_Tcore_L4_exc : volt
            Isyn_ampa_Tcore_L5_exc : volt
            Isyn_ampa_Tmatrix_L2_exc : volt
            Isyn_ampa_Tmatrix_L5_exc : volt
            Isyn_ampa_T_NRT_exc : volt
            Isyn_ampa_L2_pre_L4_exc : volt
            Isyn_ampa_L5_post_L2_exc : volt
            Isyn_ampa_L5_post_L5_exc : volt
            
            Isyn_nmda = Isyn_nmda_L2_L2_exc + Isyn_nmda_L4_L4_exc + Isyn_nmda_L5_L5_exc + Isyn_nmda_L2_L5_exc + Isyn_nmda_L4_L2_exc + Isyn_nmda_L5_L2_exc + Isyn_nmda_L5_L4_exc + Isyn_nmda_L5_NRT_exc + Isyn_nmda_L5_Tcore_exc + Isyn_nmda_L5_Tmatrix_exc + Isyn_nmda_Tcore_L4_exc + Isyn_nmda_Tcore_L5_exc + Isyn_nmda_Tmatrix_L2_exc + Isyn_nmda_Tmatrix_L5_exc + Isyn_nmda_T_NRT_exc + Isyn_nmda_L5_post_L2_exc + Isyn_nmda_L5_post_L2_exc + Isyn_nmda_L5_post_L5_exc: volt
            Isyn_nmda_L2_L2_exc : volt
            Isyn_nmda_L4_L4_exc : volt
            Isyn_nmda_L5_L5_exc : volt
            Isyn_nmda_L2_L5_exc : volt
            Isyn_nmda_L4_L2_exc : volt
            Isyn_nmda_L5_L2_exc : volt
            Isyn_nmda_L5_L4_exc : volt
            Isyn_nmda_L5_NRT_exc : volt
            Isyn_nmda_L5_Tcore_exc : volt
            Isyn_nmda_L5_Tmatrix_exc : volt
            Isyn_nmda_Tcore_L4_exc : volt
            Isyn_nmda_Tcore_L5_exc : volt
            Isyn_nmda_Tmatrix_L2_exc : volt
            Isyn_nmda_Tmatrix_L5_exc : volt
            Isyn_nmda_T_NRT_exc : volt
            Isyn_nmda_L2_pre_L4_exc : volt
            Isyn_nmda_L5_post_L2_exc : volt
            Isyn_nmda_L5_post_L5_exc : volt
            
            Isyn_gaba_a = Isyn_gaba_a_L2_L2_inh + Isyn_gaba_a_L4_L4_inh + Isyn_gaba_a_L5_L5_inh + Isyn_gaba_a_L2_L2_column_inh + Isyn_gaba_a_L2_L4_inh + Isyn_gaba_a_L2_L5_inh + Isyn_gaba_a_T_T_inh + Isyn_gaba_a_NRT_T_gaba_a_inh : volt
            Isyn_gaba_a_L2_L2_inh : volt
            Isyn_gaba_a_L4_L4_inh : volt
            Isyn_gaba_a_L5_L5_inh : volt
            Isyn_gaba_a_L2_L2_column_inh : volt
            Isyn_gaba_a_L2_L4_inh : volt
            Isyn_gaba_a_L2_L5_inh : volt
            Isyn_gaba_a_T_T_inh : volt
            Isyn_gaba_a_NRT_T_gaba_a_inh : volt
            
            Isyn_gaba_b : volt
            
            Ipoiss = int(t-t_last_poisson < 5*ms)*poiss_str_NRT*(V - E_ampa) : volt
            t_last_poisson : second
            poiss : 1
            
            Iapp : volt
            ''')

# N = NeuronGroup(1, eqs_NRT, method = 'rk4', threshold = 'V > thresh', 
#                 reset = '''V = Ena
#                             thresh = Ena
#                             lastspike = t''')

# N.V = -70*mV
# N.thresh = thresh_ss_NRT

# P = PoissonInput(N, 'V', 1, 1*Hz, weight = 1*mV) #unclear how much noise they really have in the 2010 model, they say mean 1Hz 0.5+25mV, but Down states in the figures show more noise than that. in the 2005 model they clearly have a lot more noise

# mon = StateMonitor(N, variables = True, record = True)

# network = Network(N, mon, P)

# network.run(5000*ms, report = 'text')
# plot(mon.t/ms, mon.V[0])
# xlabel('time ms')
# ylabel('volt')
# plot_currents(mon)


#%% building the model 
# os.chdir(curr_results)

currents_CX_exc = ['Inal', 'Ikl', 'Ih', 'Inap', 'Idk', 'Iks', 'Isyn', 'Isyn_ampa', 'Isyn_nmda', 'Isyn_gaba_a', 'Isyn_gaba_b', 'Ipoiss']                      
currents_CX_inh = ['Inal', 'Ikl', 'Inap', 'Idk', 'Iks', 'Isyn', 'Isyn_ampa', 'Isyn_nmda', 'Isyn_gaba_a', 'Isyn_gaba_b', 'Ipoiss']                      
currents_T_exc = ['Inal', 'Ikl', 'Ih', 'Inap', 'It', 'Isyn', 'Isyn_ampa', 'Isyn_nmda', 'Isyn_gaba_a', 'Isyn_gaba_b', 'Ipoiss']                      
currents_T_inh = ['Inal', 'Ikl', 'Inap', 'Iks', 'Isyn', 'Isyn_ampa', 'Isyn_nmda', 'Isyn_gaba_a', 'Isyn_gaba_b', 'Ipoiss']                      
currents_NRT = ['Inal', 'Ikl', 'Inap', 'It', 'Ikca', 'Isyn', 'Isyn_ampa', 'Isyn_nmda', 'Isyn_gaba_a', 'Isyn_gaba_b', 'Ipoiss']                      


currents_syn_ampa = ['Isyn_ampa_L2_L2_exc', 'Isyn_ampa_L4_L4_exc', 'Isyn_ampa_L5_L5_exc', 'Isyn_ampa_L2_L5_exc', 'Isyn_ampa_L4_L2_exc', 'Isyn_ampa_L5_L2_exc', 'Isyn_ampa_L5_L4_exc', 'Isyn_ampa_L5_NRT_exc', 'Isyn_ampa_L5_Tcore_exc', 'Isyn_ampa_L5_Tmatrix_exc', 'Isyn_ampa_Tcore_L4_exc', 'Isyn_ampa_Tcore_L5_exc', 'Isyn_ampa_Tmatrix_L2_exc', 'Isyn_ampa_Tmatrix_L5_exc', 'Isyn_ampa_T_NRT_exc']
currents_syn_nmda = ['Isyn_nmda_L2_L2_exc', 'Isyn_nmda_L4_L4_exc', 'Isyn_nmda_L5_L5_exc', 'Isyn_nmda_L2_L5_exc', 'Isyn_nmda_L4_L2_exc', 'Isyn_nmda_L5_L2_exc', 'Isyn_nmda_L5_L4_exc', 'Isyn_nmda_L5_NRT_exc', 'Isyn_nmda_L5_Tcore_exc', 'Isyn_nmda_L5_Tmatrix_exc', 'Isyn_nmda_Tcore_L4_exc', 'Isyn_nmda_Tcore_L5_exc', 'Isyn_nmda_Tmatrix_L2_exc', 'Isyn_nmda_Tmatrix_L5_exc', 'Isyn_nmda_T_NRT_exc']
currents_syn_gaba = ['Isyn_gaba_a_L2_L2_inh', 'Isyn_gaba_a_L4_L4_inh', 'Isyn_gaba_a_L5_L5_inh', 'Isyn_gaba_a_L2_L2_column_inh', 'Isyn_gaba_a_L2_L4_inh', 'Isyn_gaba_a_L2_L5_inh', 'Isyn_gaba_a_T_T_inh', 'Isyn_gaba_a_NRT_T_gaba_a_inh']
currents_syn_total = currents_syn_ampa + currents_syn_nmda + currents_syn_gaba

for nrn in ['L2', 'L4', 'L5', 'T', 'NRT']:
    globals()[f'syn_current_list_{nrn}'] = []
    for curr in currents_syn_total:
        if f'{nrn}_exc' in curr or f'{nrn}_inh' in curr or f'{nrn}_column' in curr:
            globals()[f'syn_current_list_{nrn}'].append(curr)


Neurongroup_list = []

Synapse_list = []

voltage_monitor_list = []

current_monitor_list = []

spikes_monitor_list = []

# -------------------------------------------LAYER 2---------------------------------------------
globals()[f'CX_exc_L2_[area]'] = NeuronGroup(exc_cells_per_CX_layer, model = eqs_CX_exc_L2, method = 'rk4', threshold = 'V > thresh',
                reset = '''V = Ena
                            thresh = Ena
                            lastspike = t''',
                events = {'poisson' : 'poiss < poiss_rate_CX_exc_L2*dt'})

CX_exc_L2.V = '-75*mV + rand()*5*mV'
CX_exc_L2.run_regularly('poiss = rand()')
CX_exc_L2.run_on_event('poisson', 't_last_poisson = t')
CX_exc_L2.g_nap_CX_exc_L2 = g_nap_CX_exc_L2
poiss_rate_CX_exc_L2 = 150*Hz
Neurongroup_list.append(CX_exc_L2)
# P_exc_L5 = PoissonInput(CX_exc_L2, 'V', 15, 1*Hz, weight = 1*mV)

spikes_exc_L2 = SpikeMonitor(CX_exc_L2)
spikes_monitor_list.append(spikes_exc_L2)
voltage_monitor_exc_L2 = StateMonitor(CX_exc_L2, variables = 'V', record = True)
voltage_monitor_list.append(voltage_monitor_exc_L2)
current_monitor_exc_L2 = StateMonitor(CX_exc_L2[100:125], variables = currents_CX_exc + syn_current_list_L2, record = True)
current_monitor_list.append(current_monitor_exc_L2)

CX_inh_L2 = NeuronGroup(inh_cells_per_CX_layer, model = eqs_CX_inh_L2, method = 'rk4', threshold = 'V > thresh',
                reset = '''V = Ena
                            thresh = Ena
                            lastspike = t''',
                events = {'poisson' : 'poiss < poiss_rate_CX_inh_L2*dt'})

CX_inh_L2.V = '-75*mV + rand()*5*mV'
CX_inh_L2.run_regularly('poiss = rand()')
CX_inh_L2.run_on_event('poisson', 't_last_poisson = t')
poiss_rate_CX_inh_L2 = 150*Hz
Neurongroup_list.append(CX_inh_L2)
# P_exc_L5 = PoissonInput(CX_exc_L2, 'V', 15, 1*Hz, weight = 1*mV)

spikes_inh_L2 = SpikeMonitor(CX_inh_L2)
spikes_monitor_list.append(spikes_inh_L2)
voltage_monitor_inh_L2 = StateMonitor(CX_inh_L2, variables = 'V', record = True)
voltage_monitor_list.append(voltage_monitor_inh_L2)
current_monitor_inh_L2 = StateMonitor(CX_inh_L2[100:125], variables = currents_CX_inh + syn_current_list_L2, record = True)
current_monitor_list.append(current_monitor_inh_L2)

L2_L2_exc_exc_con_ampa = Synapses(CX_exc_L2, CX_exc_L2, model = eqs_syn_ampa_L2_L2_exc, on_pre = on_pre_eqs_ampa, method = 'rk4')
L2_L2_exc_exc_con_ampa.connect(i = connection_indices_dict['L2_L2_exc_exc_con_ind'][0], j = connection_indices_dict['L2_L2_exc_exc_con_ind'][1])
L2_L2_exc_exc_con_ampa.delay = 1*ms + clip(0.25*randn(),0,1)*msecond
Synapse_list.append(L2_L2_exc_exc_con_ampa)

L2_L2_exc_exc_con_nmda = Synapses(CX_exc_L2, CX_exc_L2, model = eqs_syn_nmda_L2_L2_exc, on_pre = on_pre_eqs_nmda, method = 'rk4')
L2_L2_exc_exc_con_nmda.connect(i = connection_indices_dict['L2_L2_exc_exc_con_ind'][0], j = connection_indices_dict['L2_L2_exc_exc_con_ind'][1])
L2_L2_exc_exc_con_nmda.delay = 1*ms + clip(0.25*randn(),0,1)*ms
Synapse_list.append(L2_L2_exc_exc_con_nmda)

L2_L2_inh_exc_con_gaba_a = Synapses(CX_inh_L2, CX_exc_L2, model = eqs_syn_gaba_a_L2_L2_inh, on_pre = on_pre_eqs_gaba_a, method = 'rk4')
L2_L2_inh_exc_con_gaba_a.connect(i = connection_indices_dict['L2_L2_inh_exc_con_ind'][0], j = connection_indices_dict['L2_L2_inh_exc_con_ind'][1])
L2_L2_inh_exc_con_gaba_a.delay = 0.75*ms + clip(0.1*randn(),0,1)*ms
Synapse_list.append(L2_L2_inh_exc_con_gaba_a)

L2_L2_inh_inh_con_gaba_a = Synapses(CX_inh_L2, CX_inh_L2, model = eqs_syn_gaba_a_L2_L2_inh, on_pre = on_pre_eqs_gaba_a, method = 'rk4')
L2_L2_inh_inh_con_gaba_a.connect(i = connection_indices_dict['L2_L2_inh_inh_con_ind'][0], j = connection_indices_dict['L2_L2_inh_inh_con_ind'][1])
L2_L2_inh_inh_con_gaba_a.delay = 0.75*ms + clip(0.1*randn(),0,1)*ms
Synapse_list.append(L2_L2_inh_inh_con_gaba_a)

L2_L2_exc_inh_con_ampa = Synapses(CX_exc_L2, CX_inh_L2, model = eqs_syn_ampa_L2_L2_exc, on_pre = on_pre_eqs_ampa, method = 'rk4')
L2_L2_exc_inh_con_ampa.connect(i = connection_indices_dict['L2_L2_exc_inh_con_ind'][0], j = connection_indices_dict['L2_L2_exc_inh_con_ind'][1])
L2_L2_exc_inh_con_ampa.delay = 1*ms + clip(0.25*randn(),0,1)*ms
Synapse_list.append(L2_L2_exc_inh_con_ampa)

L2_L2_exc_inh_con_nmda = Synapses(CX_exc_L2, CX_inh_L2, model = eqs_syn_nmda_L2_L2_exc, on_pre = on_pre_eqs_nmda, method = 'rk4')
L2_L2_exc_inh_con_nmda.connect(i = connection_indices_dict['L2_L2_exc_inh_con_ind'][0], j = connection_indices_dict['L2_L2_exc_inh_con_ind'][1])
L2_L2_exc_inh_con_nmda.delay = 1*ms + clip(0.25*randn(),0,1)*ms
Synapse_list.append(L2_L2_exc_inh_con_nmda)


# -----------------------------------------------------LAYER 4---------------------------------------------

CX_exc_L4 = NeuronGroup(exc_cells_per_CX_layer, model = eqs_CX_exc_L4, method = 'rk4', threshold = 'V > thresh',
                reset = '''V = Ena
                            thresh = Ena
                            lastspike = t''',
                events = {'poisson' : 'poiss < poiss_rate_CX_exc_L4*dt'})

CX_exc_L4.V = '-75*mV + rand()*5*mV'
CX_exc_L4.run_regularly('poiss = rand()')
CX_exc_L4.run_on_event('poisson', 't_last_poisson = t')
CX_exc_L4.g_nap_CX_exc_L4 = g_nap_CX_exc_L4
poiss_rate_CX_exc_L4 = 150*Hz
Neurongroup_list.append(CX_exc_L4)
# P_exc_L5 = PoissonInput(CX_exc_L4, 'V', 15, 1*Hz, weight = 1*mV)

spikes_exc_L4 = SpikeMonitor(CX_exc_L4)
spikes_monitor_list.append(spikes_exc_L4)
voltage_monitor_exc_L4 = StateMonitor(CX_exc_L4, variables = 'V', record = True)
voltage_monitor_list.append(voltage_monitor_exc_L4)
current_monitor_exc_L4 = StateMonitor(CX_exc_L4[100:125], variables = currents_CX_exc + syn_current_list_L4, record = True)
current_monitor_list.append(current_monitor_exc_L4)

CX_inh_L4 = NeuronGroup(inh_cells_per_CX_layer, model = eqs_CX_inh_L4, method = 'rk4', threshold = 'V > thresh',
                reset = '''V = Ena
                            thresh = Ena
                            lastspike = t''',
                events = {'poisson' : 'poiss < poiss_rate_CX_inh_L4*dt'})

CX_inh_L4.V = '-75*mV + rand()*5*mV'
CX_inh_L4.run_regularly('poiss = rand()')
CX_inh_L4.run_on_event('poisson', 't_last_poisson = t')
poiss_rate_CX_inh_L4 = 150*Hz
Neurongroup_list.append(CX_inh_L4)
# P_exc_L5 = PoissonInput(CX_exc_L4, 'V', 15, 1*Hz, weight = 1*mV)

spikes_inh_L4 = SpikeMonitor(CX_inh_L4)
spikes_monitor_list.append(spikes_inh_L4)
voltage_monitor_inh_L4 = StateMonitor(CX_inh_L4, variables = 'V', record = True)
voltage_monitor_list.append(voltage_monitor_inh_L4)
current_monitor_inh_L4 = StateMonitor(CX_inh_L4[100:125], variables = currents_CX_inh + syn_current_list_L4, record = True)
current_monitor_list.append(current_monitor_inh_L4)

L4_L4_exc_exc_con_ampa = Synapses(CX_exc_L4, CX_exc_L4, model = eqs_syn_ampa_L4_L4_exc, on_pre = on_pre_eqs_ampa, method = 'rk4')
L4_L4_exc_exc_con_ampa.connect(i = connection_indices_dict['L4_L4_exc_exc_con_ind'][0], j = connection_indices_dict['L4_L4_exc_exc_con_ind'][1])
L4_L4_exc_exc_con_ampa.delay = 1*ms + clip(0.25*randn(),0,1)*ms
Synapse_list.append(L4_L4_exc_exc_con_ampa)

L4_L4_exc_exc_con_nmda = Synapses(CX_exc_L4, CX_exc_L4, model = eqs_syn_nmda_L4_L4_exc, on_pre = on_pre_eqs_nmda, method = 'rk4')
L4_L4_exc_exc_con_nmda.connect(i = connection_indices_dict['L4_L4_exc_exc_con_ind'][0], j = connection_indices_dict['L4_L4_exc_exc_con_ind'][1])
L4_L4_exc_exc_con_nmda.delay = 1*ms + clip(0.25*randn(),0,1)*ms
Synapse_list.append(L4_L4_exc_exc_con_nmda)


L4_L4_inh_exc_con_gaba_a = Synapses(CX_inh_L4, CX_exc_L4, model = eqs_syn_gaba_a_L4_L4_inh, on_pre = on_pre_eqs_gaba_a, method = 'rk4')
L4_L4_inh_exc_con_gaba_a.connect(i = connection_indices_dict['L4_L4_inh_exc_con_ind'][0], j = connection_indices_dict['L4_L4_inh_exc_con_ind'][1])
L4_L4_inh_exc_con_gaba_a.delay = 0.75*ms + clip(0.1*randn(),0,1)*ms
Synapse_list.append(L4_L4_inh_exc_con_gaba_a)

L4_L4_inh_inh_con_gaba_a = Synapses(CX_inh_L4, CX_inh_L4, model = eqs_syn_gaba_a_L4_L4_inh, on_pre = on_pre_eqs_gaba_a, method = 'rk4')
L4_L4_inh_inh_con_gaba_a.connect(i = connection_indices_dict['L4_L4_inh_inh_con_ind'][0], j = connection_indices_dict['L4_L4_inh_inh_con_ind'][1])
L4_L4_inh_inh_con_gaba_a.delay = 0.75*ms + clip(0.1*randn(),0,1)*ms
Synapse_list.append(L4_L4_inh_inh_con_gaba_a)

L4_L4_exc_inh_con_ampa = Synapses(CX_exc_L4, CX_inh_L4, model = eqs_syn_ampa_L4_L4_exc, on_pre = on_pre_eqs_ampa, method = 'rk4')
L4_L4_exc_inh_con_ampa.connect(i = connection_indices_dict['L4_L4_exc_inh_con_ind'][0], j = connection_indices_dict['L4_L4_exc_inh_con_ind'][1])
L4_L4_exc_inh_con_ampa.delay = 1*ms + clip(0.25*randn(),0,1)*ms
Synapse_list.append(L4_L4_exc_inh_con_ampa)

L4_L4_exc_inh_con_nmda = Synapses(CX_exc_L4, CX_inh_L4, model = eqs_syn_nmda_L4_L4_exc, on_pre = on_pre_eqs_nmda, method = 'rk4')
L4_L4_exc_inh_con_nmda.connect(i = connection_indices_dict['L4_L4_exc_inh_con_ind'][0], j = connection_indices_dict['L4_L4_exc_inh_con_ind'][1])
L4_L4_exc_inh_con_nmda.delay = 1*ms + clip(0.25*randn(),0,1)*ms
Synapse_list.append(L4_L4_exc_inh_con_nmda)

#----------------------------------------------------------- LAYER 5 -----------------------------------------------

CX_exc_L5 = NeuronGroup(exc_cells_per_CX_layer, model = eqs_CX_exc_L5, method = 'rk4', threshold = 'V > thresh',
                reset = '''V = Ena
                            thresh = Ena
                            lastspike = t''',
                events = {'poisson' : 'poiss < poiss_rate_CX_exc_L5*dt'})

CX_exc_L5.V = '-75*mV + rand()*5*mV'
CX_exc_L5.g_nap_CX_exc_L5 = g_nap_CX_exc_L5

IB_cells_indx = np.linspace(0,(exc_cells_per_CX_layer - 1),int(0.3*exc_cells_per_CX_layer), dtype = int)
#need to use for loop because subgroup creation can only use contiguous integers
for n in IB_cells_indx:
    CX_exc_L5[n].run_regularly('Ih = g_h_CX_exc * m_h * (V - Eh)')
    CX_exc_L5[n].g_nap_CX_exc_L5 = 4

CX_exc_L5.run_regularly('poiss = rand()')
CX_exc_L5.run_on_event('poisson', 't_last_poisson = t')
poiss_rate_CX_exc_L5 = 150*Hz
Neurongroup_list.append(CX_exc_L5)
# P_exc_L5 = PoissonInput(CX_exc_L5, 'V', 15, 1*Hz, weight = 1*mV)

spikes_exc_L5 = SpikeMonitor(CX_exc_L5)
spikes_monitor_list.append(spikes_exc_L5)
voltage_monitor_exc_L5 = StateMonitor(CX_exc_L5, variables = 'V', record = True)
voltage_monitor_list.append(voltage_monitor_exc_L5)
current_monitor_exc_L5 = StateMonitor(CX_exc_L5[100:125], variables = currents_CX_exc + syn_current_list_L5, record = True)
current_monitor_list.append(current_monitor_exc_L5)

CX_inh_L5 = NeuronGroup(inh_cells_per_CX_layer, model = eqs_CX_inh_L5, method = 'rk4', threshold = 'V > thresh',
                reset = '''V = Ena
                            thresh = Ena
                            lastspike = t''',
                events = {'poisson' : 'poiss < poiss_rate_CX_inh_L5*dt'})

CX_inh_L5.V = '-75*mV + rand()*5*mV'
                            
CX_inh_L5.run_regularly('poiss = rand()')
CX_inh_L5.run_on_event('poisson', 't_last_poisson = t')
poiss_rate_CX_inh_L5 = 150*Hz
Neurongroup_list.append(CX_inh_L5)
# P_inh_L5 = PoissonInput(CX_inh_L5, 'V', 2, 1*Hz, weight = 1*mV)

spikes_inh_L5 = SpikeMonitor(CX_inh_L5)
spikes_monitor_list.append(spikes_inh_L5)
voltage_monitor_inh_L5 = StateMonitor(CX_inh_L5, variables = 'V', record = True)
voltage_monitor_list.append(voltage_monitor_inh_L5)
current_monitor_inh_L5 = StateMonitor(CX_inh_L5[100:125], variables = currents_CX_inh + syn_current_list_L5, record = True)
current_monitor_list.append(current_monitor_inh_L5)

L5_L5_exc_exc_con_ampa = Synapses(CX_exc_L5, CX_exc_L5, model = eqs_syn_ampa_L5_L5_exc, on_pre = on_pre_eqs_ampa, method = 'rk4')
L5_L5_exc_exc_con_ampa.connect(i = connection_indices_dict['L5_L5_exc_exc_con_ind'][0], j = connection_indices_dict['L5_L5_exc_exc_con_ind'][1])
L5_L5_exc_exc_con_ampa.delay = 1*ms + clip(0.25*randn(),0,1)*ms
Synapse_list.append(L5_L5_exc_exc_con_ampa)

L5_L5_exc_exc_con_nmda = Synapses(CX_exc_L5, CX_exc_L5, model = eqs_syn_nmda_L5_L5_exc, on_pre = on_pre_eqs_nmda, method = 'rk4')
L5_L5_exc_exc_con_nmda.connect(i = connection_indices_dict['L5_L5_exc_exc_con_ind'][0], j = connection_indices_dict['L5_L5_exc_exc_con_ind'][1])
L5_L5_exc_exc_con_nmda.delay = 1*ms + clip(0.25*randn(),0,1)*ms
Synapse_list.append(L5_L5_exc_exc_con_nmda)

L5_L5_inh_exc_con_gaba_a = Synapses(CX_inh_L5, CX_exc_L5, model = eqs_syn_gaba_a_L5_L5_inh, on_pre = on_pre_eqs_gaba_a, method = 'rk4')
L5_L5_inh_exc_con_gaba_a.connect(i = connection_indices_dict['L5_L5_inh_exc_con_ind'][0], j = connection_indices_dict['L5_L5_inh_exc_con_ind'][1])
L5_L5_inh_exc_con_gaba_a.delay = 0.75*ms + clip(0.1*randn(),0,1)*ms
Synapse_list.append(L5_L5_inh_exc_con_gaba_a)

L5_L5_inh_inh_con_gaba_a = Synapses(CX_inh_L5, CX_inh_L5, model = eqs_syn_gaba_a_L5_L5_inh, on_pre = on_pre_eqs_gaba_a, method = 'rk4')
L5_L5_inh_inh_con_gaba_a.connect(i = connection_indices_dict['L5_L5_inh_inh_con_ind'][0], j = connection_indices_dict['L5_L5_inh_inh_con_ind'][1])
L5_L5_inh_inh_con_gaba_a.delay = 0.75*ms + clip(0.1*randn(),0,1)*ms
Synapse_list.append(L5_L5_inh_inh_con_gaba_a)

L5_L5_exc_inh_con_ampa = Synapses(CX_exc_L5, CX_inh_L5, model = eqs_syn_ampa_L5_L5_exc, on_pre = on_pre_eqs_ampa, method = 'rk4')
L5_L5_exc_inh_con_ampa.connect(i = connection_indices_dict['L5_L5_exc_inh_con_ind'][0], j = connection_indices_dict['L5_L5_exc_inh_con_ind'][1])
L5_L5_exc_inh_con_ampa.delay = 1*ms + clip(0.25*randn(),0,1)*ms
Synapse_list.append(L5_L5_exc_inh_con_ampa)

L5_L5_exc_inh_con_nmda = Synapses(CX_exc_L5, CX_inh_L5, model = eqs_syn_nmda_L5_L5_exc, on_pre = on_pre_eqs_nmda, method = 'rk4')
L5_L5_exc_inh_con_nmda.connect(i = connection_indices_dict['L5_L5_exc_inh_con_ind'][0], j = connection_indices_dict['L5_L5_exc_inh_con_ind'][1])
L5_L5_exc_inh_con_nmda.delay = 1*ms + clip(0.25*randn(),0,1)*ms
Synapse_list.append(L5_L5_exc_inh_con_nmda)


#------------------------------------------------- exc inter-laminar connections ------------------------------------

# -------------------------------------------------- L2_L5 -------------------------------------------------------------
L2_L5_exc_exc_con_ampa = Synapses(CX_exc_L2, CX_exc_L5, model = eqs_syn_ampa_L2_L5_exc, on_pre = on_pre_eqs_ampa, method = 'rk4')
L2_L5_exc_exc_con_ampa.connect(i = connection_indices_dict['L2_L5_exc_exc_con_ind'][0], j = connection_indices_dict['L2_L5_exc_exc_con_ind'][1])
L2_L5_exc_exc_con_ampa.delay = 1*ms + clip(0.25*randn(),0,1)*ms
Synapse_list.append(L2_L5_exc_exc_con_ampa)

L2_L5_exc_exc_con_nmda = Synapses(CX_exc_L2, CX_exc_L5, model = eqs_syn_nmda_L2_L5_exc, on_pre = on_pre_eqs_nmda, method = 'rk4')
L2_L5_exc_exc_con_nmda.connect(i = connection_indices_dict['L2_L5_exc_exc_con_ind'][0], j = connection_indices_dict['L2_L5_exc_exc_con_ind'][1])
L2_L5_exc_exc_con_nmda.delay = 1*ms + clip(0.25*randn(),0,1)*ms
Synapse_list.append(L2_L5_exc_exc_con_nmda)

L2_L5_exc_inh_con_ampa = Synapses(CX_exc_L2, CX_inh_L5, model = eqs_syn_ampa_L2_L5_exc, on_pre = on_pre_eqs_ampa, method = 'rk4')
L2_L5_exc_inh_con_ampa.connect(i = connection_indices_dict['L2_L5_exc_inh_con_ind'][0], j = connection_indices_dict['L2_L5_exc_inh_con_ind'][1])
L2_L5_exc_inh_con_ampa.delay = 1*ms + clip(0.25*randn(),0,1)*ms
Synapse_list.append(L2_L5_exc_inh_con_ampa)

L2_L5_exc_inh_con_nmda = Synapses(CX_exc_L2, CX_inh_L5, model = eqs_syn_nmda_L2_L5_exc, on_pre = on_pre_eqs_nmda, method = 'rk4')
L2_L5_exc_inh_con_nmda.connect(i = connection_indices_dict['L2_L5_exc_inh_con_ind'][0], j = connection_indices_dict['L2_L5_exc_inh_con_ind'][1])
L2_L5_exc_inh_con_nmda.delay = 1*ms + clip(0.25*randn(),0,1)*ms
Synapse_list.append(L2_L5_exc_inh_con_nmda)


# -------------------------------------------------- L4_L2 -------------------------------------------------------------
L4_L2_exc_exc_con_ampa = Synapses(CX_exc_L4, CX_exc_L2, model = eqs_syn_ampa_L4_L2_exc, on_pre = on_pre_eqs_ampa, method = 'rk4')
L4_L2_exc_exc_con_ampa.connect(i = connection_indices_dict['L4_L2_exc_exc_con_ind'][0], j = connection_indices_dict['L4_L2_exc_exc_con_ind'][1])
L4_L2_exc_exc_con_ampa.delay = 1*ms + clip(0.25*randn(),0,1)*ms
Synapse_list.append(L4_L2_exc_exc_con_ampa)

L4_L2_exc_exc_con_nmda = Synapses(CX_exc_L4, CX_exc_L2, model = eqs_syn_nmda_L4_L2_exc, on_pre = on_pre_eqs_nmda, method = 'rk4')
L4_L2_exc_exc_con_nmda.connect(i = connection_indices_dict['L4_L2_exc_exc_con_ind'][0], j = connection_indices_dict['L4_L2_exc_exc_con_ind'][1])
L4_L2_exc_exc_con_nmda.delay = 1*ms + clip(0.25*randn(),0,1)*ms
Synapse_list.append(L4_L2_exc_exc_con_nmda)

L4_L2_exc_inh_con_ampa = Synapses(CX_exc_L4, CX_inh_L2, model = eqs_syn_ampa_L4_L2_exc, on_pre = on_pre_eqs_ampa, method = 'rk4')
L4_L2_exc_inh_con_ampa.connect(i = connection_indices_dict['L4_L2_exc_inh_con_ind'][0], j = connection_indices_dict['L4_L2_exc_inh_con_ind'][1])
L4_L2_exc_inh_con_ampa.delay = 1*ms + clip(0.25*randn(),0,1)*ms
Synapse_list.append(L4_L2_exc_inh_con_ampa)

L4_L2_exc_inh_con_nmda = Synapses(CX_exc_L4, CX_inh_L2, model = eqs_syn_nmda_L4_L2_exc, on_pre = on_pre_eqs_nmda, method = 'rk4')
L4_L2_exc_inh_con_nmda.connect(i = connection_indices_dict['L4_L2_exc_inh_con_ind'][0], j = connection_indices_dict['L4_L2_exc_inh_con_ind'][1])
L4_L2_exc_inh_con_nmda.delay = 1*ms + clip(0.25*randn(),0,1)*ms
Synapse_list.append(L4_L2_exc_inh_con_nmda)


# -------------------------------------------------- L5_L2 -------------------------------------------------------------
L5_L2_exc_exc_con_ampa = Synapses(CX_exc_L5, CX_exc_L2, model = eqs_syn_ampa_L5_L2_exc, on_pre = on_pre_eqs_ampa, method = 'rk4')
L5_L2_exc_exc_con_ampa.connect(i = connection_indices_dict['L5_L2_exc_exc_con_ind'][0], j = connection_indices_dict['L5_L2_exc_exc_con_ind'][1])
L5_L2_exc_exc_con_ampa.delay = 1*ms + clip(0.25*randn(),0,1)*ms
Synapse_list.append(L5_L2_exc_exc_con_ampa)

L5_L2_exc_exc_con_nmda = Synapses(CX_exc_L5, CX_exc_L2, model = eqs_syn_nmda_L5_L2_exc, on_pre = on_pre_eqs_nmda, method = 'rk4')
L5_L2_exc_exc_con_nmda.connect(i = connection_indices_dict['L5_L2_exc_exc_con_ind'][0], j = connection_indices_dict['L5_L2_exc_exc_con_ind'][1])
L5_L2_exc_exc_con_nmda.delay = 1*ms + clip(0.25*randn(),0,1)*ms
Synapse_list.append(L5_L2_exc_exc_con_nmda)

L5_L2_exc_inh_con_ampa = Synapses(CX_exc_L5, CX_inh_L2, model = eqs_syn_ampa_L5_L2_exc, on_pre = on_pre_eqs_ampa, method = 'rk4')
L5_L2_exc_inh_con_ampa.connect(i = connection_indices_dict['L5_L2_exc_inh_con_ind'][0], j = connection_indices_dict['L5_L2_exc_inh_con_ind'][1])
L5_L2_exc_inh_con_ampa.delay = 1*ms + clip(0.25*randn(),0,1)*ms
Synapse_list.append(L5_L2_exc_inh_con_ampa)

L5_L2_exc_inh_con_nmda = Synapses(CX_exc_L5, CX_inh_L2, model = eqs_syn_nmda_L5_L2_exc, on_pre = on_pre_eqs_nmda, method = 'rk4')
L5_L2_exc_inh_con_nmda.connect(i = connection_indices_dict['L5_L2_exc_inh_con_ind'][0], j = connection_indices_dict['L5_L2_exc_inh_con_ind'][1])
L5_L2_exc_inh_con_nmda.delay = 1*ms + clip(0.25*randn(),0,1)*ms
Synapse_list.append(L5_L2_exc_inh_con_nmda)


# -------------------------------------------------- L5_L4 -------------------------------------------------------------
L5_L4_exc_exc_con_ampa = Synapses(CX_exc_L5, CX_exc_L4, model = eqs_syn_ampa_L5_L4_exc, on_pre = on_pre_eqs_ampa, method = 'rk4')
L5_L4_exc_exc_con_ampa.connect(i = connection_indices_dict['L5_L4_exc_exc_con_ind'][0], j = connection_indices_dict['L5_L4_exc_exc_con_ind'][1])
L5_L4_exc_exc_con_ampa.delay = 1*ms + clip(0.25*randn(),0,1)*ms
Synapse_list.append(L5_L4_exc_exc_con_ampa)

L5_L4_exc_exc_con_nmda = Synapses(CX_exc_L5, CX_exc_L4, model = eqs_syn_nmda_L5_L4_exc, on_pre = on_pre_eqs_nmda, method = 'rk4')
L5_L4_exc_exc_con_nmda.connect(i = connection_indices_dict['L5_L4_exc_exc_con_ind'][0], j = connection_indices_dict['L5_L4_exc_exc_con_ind'][1])
L5_L4_exc_exc_con_nmda.delay = 1*ms + clip(0.25*randn(),0,1)*ms
Synapse_list.append(L5_L4_exc_exc_con_nmda)

L5_L4_exc_inh_con_ampa = Synapses(CX_exc_L5, CX_inh_L4, model = eqs_syn_ampa_L5_L4_exc, on_pre = on_pre_eqs_ampa, method = 'rk4')
L5_L4_exc_inh_con_ampa.connect(i = connection_indices_dict['L5_L4_exc_inh_con_ind'][0], j = connection_indices_dict['L5_L4_exc_inh_con_ind'][1])
L5_L4_exc_inh_con_ampa.delay = 1*ms + clip(0.25*randn(),0,1)*ms
Synapse_list.append(L5_L4_exc_inh_con_ampa)

L5_L4_exc_inh_con_nmda = Synapses(CX_exc_L5, CX_inh_L4, model = eqs_syn_nmda_L5_L4_exc, on_pre = on_pre_eqs_nmda, method = 'rk4')
L5_L4_exc_inh_con_nmda.connect(i = connection_indices_dict['L5_L4_exc_inh_con_ind'][0], j = connection_indices_dict['L5_L4_exc_inh_con_ind'][1])
L5_L4_exc_inh_con_nmda.delay = 1*ms + clip(0.25*randn(),0,1)*ms
Synapse_list.append(L5_L4_exc_inh_con_nmda)


#---------------------------------------------------- inh inter-laminar connections ------------------------------------

# -------------------------------------------------- L2_L2 -------------------------------------------------------------

L2_L2_column_inh_exc_con_gaba_a = Synapses(CX_inh_L2, CX_exc_L2, model = eqs_syn_gaba_a_L2_L2_column_inh, on_pre = on_pre_eqs_gaba_a, method = 'rk4')
L2_L2_column_inh_exc_con_gaba_a.connect(i = connection_indices_dict['L2_L2_column_inh_exc_con_ind'][0], j = connection_indices_dict['L2_L2_column_inh_exc_con_ind'][1])
L2_L2_column_inh_exc_con_gaba_a.delay = 4*ms + clip(0.25*randn(),0,1)*ms
Synapse_list.append(L2_L2_column_inh_exc_con_gaba_a)

L2_L2_column_inh_inh_con_gaba_a = Synapses(CX_inh_L2, CX_inh_L2, model = eqs_syn_gaba_a_L2_L2_column_inh, on_pre = on_pre_eqs_gaba_a, method = 'rk4')
L2_L2_column_inh_inh_con_gaba_a.connect(i = connection_indices_dict['L2_L2_column_inh_inh_con_ind'][0], j = connection_indices_dict['L2_L2_column_inh_inh_con_ind'][1])
L2_L2_column_inh_inh_con_gaba_a.delay = 4*ms + clip(0.25*randn(),0,1)*ms
Synapse_list.append(L2_L2_column_inh_inh_con_gaba_a)


# -------------------------------------------------- L2_L4 -------------------------------------------------------------

L2_L4_inh_exc_con_gaba_a = Synapses(CX_inh_L2, CX_exc_L4, model = eqs_syn_gaba_a_L2_L4_inh, on_pre = on_pre_eqs_gaba_a, method = 'rk4')
L2_L4_inh_exc_con_gaba_a.connect(i = connection_indices_dict['L2_L4_inh_exc_con_ind'][0], j = connection_indices_dict['L2_L4_inh_exc_con_ind'][1])
L2_L4_inh_exc_con_gaba_a.delay = 4*ms + clip(0.25*randn(),0,1)*ms
Synapse_list.append(L2_L4_inh_exc_con_gaba_a)

L2_L4_inh_inh_con_gaba_a = Synapses(CX_inh_L2, CX_inh_L4, model = eqs_syn_gaba_a_L2_L4_inh, on_pre = on_pre_eqs_gaba_a, method = 'rk4')
L2_L4_inh_inh_con_gaba_a.connect(i = connection_indices_dict['L2_L4_inh_inh_con_ind'][0], j = connection_indices_dict['L2_L4_inh_inh_con_ind'][1])
L2_L4_inh_inh_con_gaba_a.delay = 4*ms + clip(0.25*randn(),0,1)*ms
Synapse_list.append(L2_L4_inh_inh_con_gaba_a)


# -------------------------------------------------- L2_L5 -------------------------------------------------------------

L2_L5_inh_exc_con_gaba_a = Synapses(CX_inh_L2, CX_exc_L5, model = eqs_syn_gaba_a_L2_L5_inh, on_pre = on_pre_eqs_gaba_a, method = 'rk4')
L2_L5_inh_exc_con_gaba_a.connect(i = connection_indices_dict['L2_L5_inh_exc_con_ind'][0], j = connection_indices_dict['L2_L5_inh_exc_con_ind'][1])
L2_L5_inh_exc_con_gaba_a.delay = 4*ms + clip(0.25*randn(),0,1)*ms
Synapse_list.append(L2_L5_inh_exc_con_gaba_a)

L2_L5_inh_inh_con_gaba_a = Synapses(CX_inh_L2, CX_inh_L5, model = eqs_syn_gaba_a_L2_L5_inh, on_pre = on_pre_eqs_gaba_a, method = 'rk4')
L2_L5_inh_inh_con_gaba_a.connect(i = connection_indices_dict['L2_L5_inh_inh_con_ind'][0], j = connection_indices_dict['L2_L5_inh_inh_con_ind'][1])
L2_L5_inh_inh_con_gaba_a.delay = 4*ms + clip(0.25*randn(),0,1)*ms
Synapse_list.append(L2_L5_inh_inh_con_gaba_a)


# --------------------------------------------------------- thalamic nucleus ---------------------------------------------------

T_exc = NeuronGroup(exc_cells_per_T_layer, model = eqs_T_exc, method = 'rk4', threshold = 'V > thresh',
                reset = '''V = Ena
                            thresh = Ena
                            lastspike = t''',
                events = {'poisson' : 'poiss < poiss_rate_T_exc*dt'})

T_exc.V = '-75*mV + rand()*5*mV'

T_exc.run_regularly('poiss = rand()')
T_exc.run_on_event('poisson', 't_last_poisson = t')
poiss_rate_T_exc = 150*Hz
Neurongroup_list.append(T_exc)

spikes_T_exc = SpikeMonitor(T_exc)
spikes_monitor_list.append(spikes_T_exc)
voltage_monitor_T_exc = StateMonitor(T_exc, variables = 'V', record = True)
voltage_monitor_list.append(voltage_monitor_T_exc)
current_monitor_T_exc = StateMonitor(T_exc[100:125], variables = currents_T_exc + syn_current_list_T, record = True)
current_monitor_list.append(current_monitor_T_exc)

T_inh = NeuronGroup(inh_cells_per_T_layer, model = eqs_T_inh, method = 'rk4', threshold = 'V > thresh',
                reset = '''V = Ena
                            thresh = Ena
                            lastspike = t''',
                events = {'poisson' : 'poiss < poiss_rate_T_inh*dt'})

T_inh.V = '-75*mV + rand()*5*mV'

T_inh.run_regularly('poiss = rand()')
T_inh.run_on_event('poisson', 't_last_poisson = t')
poiss_rate_T_inh = 150*Hz
Neurongroup_list.append(T_inh)

spikes_T_inh = SpikeMonitor(T_inh)
spikes_monitor_list.append(spikes_T_inh)
voltage_monitor_T_inh = StateMonitor(T_inh, variables = 'V', record = True)
voltage_monitor_list.append(voltage_monitor_T_inh)
current_monitor_T_inh = StateMonitor(T_inh[100:125], variables = currents_T_inh + syn_current_list_T, record = True)
current_monitor_list.append(current_monitor_T_inh)


# ----------------------------------------------------- reticular nucleus -------------------------------------------------------

NRT = NeuronGroup(cells_per_NRT_layer, model = eqs_NRT, method = 'rk4', threshold = 'V > thresh',
                reset = '''V = Ena
                            thresh = Ena
                            lastspike = t''',
                events = {'poisson' : 'poiss < poiss_rate_NRT*dt'})

NRT.V = '-75*mV + rand()*5*mV'

NRT.run_regularly('poiss = rand()')
NRT.run_on_event('poisson', 't_last_poisson = t')
poiss_rate_NRT = 150*Hz
Neurongroup_list.append(NRT)

spikes_NRT = SpikeMonitor(NRT)
spikes_monitor_list.append(spikes_NRT)
voltage_monitor_NRT = StateMonitor(NRT, variables = 'V', record = True)
voltage_monitor_list.append(voltage_monitor_NRT)
current_monitor_NRT = StateMonitor(NRT[100:125], variables = currents_NRT + syn_current_list_NRT, record = True)
current_monitor_list.append(current_monitor_NRT)

# --------------------------------------------------- corticothalamic connections ------------------------------------------------

# --------------------------------------------------- L5_core -----------------------------------------------------
L5_Tcore_exc_exc_con_ampa = Synapses(CX_exc_L5, T_exc, model = eqs_syn_ampa_L5_Tcore_exc, on_pre = on_pre_eqs_ampa, method = 'rk4')
L5_Tcore_exc_exc_con_ampa.connect(i = connection_indices_dict['L5_Tcore_exc_exc_con_ind'][0], j = connection_indices_dict['L5_Tcore_exc_exc_con_ind'][1])
L5_Tcore_exc_exc_con_ampa.delay = 12*ms + clip(2*randn(),0,5)*ms
Synapse_list.append(L5_Tcore_exc_exc_con_ampa)

L5_Tcore_exc_exc_con_nmda = Synapses(CX_exc_L5, T_exc, model = eqs_syn_nmda_L5_Tcore_exc, on_pre = on_pre_eqs_nmda, method = 'rk4')
L5_Tcore_exc_exc_con_nmda.connect(i = connection_indices_dict['L5_Tcore_exc_exc_con_ind'][0], j = connection_indices_dict['L5_Tcore_exc_exc_con_ind'][1])
L5_Tcore_exc_exc_con_nmda.delay = 12*ms + clip(2*randn(),0,5)*ms
Synapse_list.append(L5_Tcore_exc_exc_con_nmda)

L5_Tcore_exc_inh_con_ampa = Synapses(CX_exc_L5, T_inh, model = eqs_syn_ampa_L5_Tcore_exc, on_pre = on_pre_eqs_ampa, method = 'rk4')
L5_Tcore_exc_inh_con_ampa.connect(i = connection_indices_dict['L5_Tcore_exc_inh_con_ind'][0], j = connection_indices_dict['L5_Tcore_exc_inh_con_ind'][1])
L5_Tcore_exc_inh_con_ampa.delay = 12*ms + clip(2*randn(),0,5)*ms
Synapse_list.append(L5_Tcore_exc_inh_con_ampa)

L5_Tcore_exc_inh_con_nmda = Synapses(CX_exc_L5, T_inh, model = eqs_syn_nmda_L5_Tcore_exc, on_pre = on_pre_eqs_nmda, method = 'rk4')
L5_Tcore_exc_inh_con_nmda.connect(i = connection_indices_dict['L5_Tcore_exc_inh_con_ind'][0], j = connection_indices_dict['L5_Tcore_exc_inh_con_ind'][1])
L5_Tcore_exc_inh_con_nmda.delay = 12*ms + clip(2*randn(),0,5)*ms
Synapse_list.append(L5_Tcore_exc_inh_con_nmda)

# --------------------------------------------------- L5_matrix -----------------------------------------------------
L5_Tmatrix_exc_exc_con_ampa = Synapses(CX_exc_L5, T_exc, model = eqs_syn_ampa_L5_Tmatrix_exc, on_pre = on_pre_eqs_ampa, method = 'rk4')
L5_Tmatrix_exc_exc_con_ampa.connect(i = connection_indices_dict['L5_Tmatrix_exc_exc_con_ind'][0], j = connection_indices_dict['L5_Tmatrix_exc_exc_con_ind'][1])
L5_Tmatrix_exc_exc_con_ampa.delay = 5*ms + clip(1*randn(),0,5)*ms
Synapse_list.append(L5_Tmatrix_exc_exc_con_ampa)

L5_Tmatrix_exc_exc_con_nmda = Synapses(CX_exc_L5, T_exc, model = eqs_syn_nmda_L5_Tmatrix_exc, on_pre = on_pre_eqs_nmda, method = 'rk4')
L5_Tmatrix_exc_exc_con_nmda.connect(i = connection_indices_dict['L5_Tmatrix_exc_exc_con_ind'][0], j = connection_indices_dict['L5_Tmatrix_exc_exc_con_ind'][1])
L5_Tmatrix_exc_exc_con_nmda.delay = 5*ms + clip(1*randn(),0,5)*ms
Synapse_list.append(L5_Tmatrix_exc_exc_con_nmda)

L5_Tmatrix_exc_inh_con_ampa = Synapses(CX_exc_L5, T_inh, model = eqs_syn_ampa_L5_Tmatrix_exc, on_pre = on_pre_eqs_ampa, method = 'rk4')
L5_Tmatrix_exc_inh_con_ampa.connect(i = connection_indices_dict['L5_Tmatrix_exc_inh_con_ind'][0], j = connection_indices_dict['L5_Tmatrix_exc_inh_con_ind'][1])
L5_Tmatrix_exc_inh_con_ampa.delay = 5*ms + clip(1*randn(),0,5)*ms
Synapse_list.append(L5_Tmatrix_exc_inh_con_ampa)

L5_Tmatrix_exc_inh_con_nmda = Synapses(CX_exc_L5, T_inh, model = eqs_syn_nmda_L5_Tmatrix_exc, on_pre = on_pre_eqs_nmda, method = 'rk4')
L5_Tmatrix_exc_inh_con_nmda.connect(i = connection_indices_dict['L5_Tmatrix_exc_inh_con_ind'][0], j = connection_indices_dict['L5_Tmatrix_exc_inh_con_ind'][1])
L5_Tmatrix_exc_inh_con_nmda.delay = 5*ms + clip(1*randn(),0,5)*ms
Synapse_list.append(L5_Tmatrix_exc_inh_con_nmda)

# --------------------------------------------------- L5_NRT -----------------------------------------------------
L5_NRT_exc_inh_con_ampa = Synapses(CX_exc_L5, NRT, model = eqs_syn_ampa_L5_NRT_exc, on_pre = on_pre_eqs_ampa, method = 'rk4')
L5_NRT_exc_inh_con_ampa.connect(i = connection_indices_dict['L5_NRT_exc_inh_con_ind'][0], j = connection_indices_dict['L5_NRT_exc_inh_con_ind'][1])
L5_NRT_exc_inh_con_ampa.delay = 5*ms + clip(1*randn(),0,5)*ms
Synapse_list.append(L5_NRT_exc_inh_con_ampa)

L5_NRT_exc_inh_con_nmda = Synapses(CX_exc_L5, NRT, model = eqs_syn_nmda_L5_NRT_exc, on_pre = on_pre_eqs_nmda, method = 'rk4')
L5_NRT_exc_inh_con_nmda.connect(i = connection_indices_dict['L5_NRT_exc_inh_con_ind'][0], j = connection_indices_dict['L5_NRT_exc_inh_con_ind'][1])
L5_NRT_exc_inh_con_nmda.delay = 5*ms + clip(1*randn(),0,5)*ms
Synapse_list.append(L5_NRT_exc_inh_con_nmda)


# --------------------------------------------------- thalamocortical connections -------------------------------------------------

# --------------------------------------------------- Tcore_L4 -------------------------------------------------------------------

Tcore_L4_exc_exc_con_ampa = Synapses(T_exc, CX_exc_L4, model = eqs_syn_ampa_Tcore_L4_exc, on_pre = on_pre_eqs_ampa, method = 'rk4')
Tcore_L4_exc_exc_con_ampa.connect(i = connection_indices_dict['Tcore_L4_exc_exc_con_ind'][0], j = connection_indices_dict['Tcore_L4_exc_exc_con_ind'][1])
Tcore_L4_exc_exc_con_ampa.delay = 7*ms + clip(0.5*randn(),0,5)*ms
Synapse_list.append(Tcore_L4_exc_exc_con_ampa)

Tcore_L4_exc_exc_con_nmda = Synapses(T_exc, CX_exc_L4, model = eqs_syn_nmda_Tcore_L4_exc, on_pre = on_pre_eqs_nmda, method = 'rk4')
Tcore_L4_exc_exc_con_nmda.connect(i = connection_indices_dict['Tcore_L4_exc_exc_con_ind'][0], j = connection_indices_dict['Tcore_L4_exc_exc_con_ind'][1])
Tcore_L4_exc_exc_con_nmda.delay = 7*ms + clip(0.5*randn(),0,5)*ms
Synapse_list.append(Tcore_L4_exc_exc_con_nmda)

Tcore_L4_exc_inh_con_ampa = Synapses(T_exc, CX_inh_L4, model = eqs_syn_ampa_Tcore_L4_exc, on_pre = on_pre_eqs_ampa, method = 'rk4')
Tcore_L4_exc_inh_con_ampa.connect(i = connection_indices_dict['Tcore_L4_exc_inh_con_ind'][0], j = connection_indices_dict['Tcore_L4_exc_inh_con_ind'][1])
Tcore_L4_exc_inh_con_ampa.delay = 7*ms + clip(0.5*randn(),0,5)*ms
Synapse_list.append(Tcore_L4_exc_inh_con_ampa)

Tcore_L4_exc_inh_con_nmda = Synapses(T_exc, CX_inh_L4, model = eqs_syn_nmda_Tcore_L4_exc, on_pre = on_pre_eqs_nmda, method = 'rk4')
Tcore_L4_exc_inh_con_nmda.connect(i = connection_indices_dict['Tcore_L4_exc_inh_con_ind'][0], j = connection_indices_dict['Tcore_L4_exc_inh_con_ind'][1])
Tcore_L4_exc_inh_con_nmda.delay = 7*ms + clip(0.5*randn(),0,5)*ms
Synapse_list.append(Tcore_L4_exc_inh_con_nmda)

# ---------------------------------------------------- Tcore_L5 --------------------------------------------------------------------

Tcore_L5_exc_exc_con_ampa = Synapses(T_exc, CX_exc_L5, model = eqs_syn_ampa_Tcore_L5_exc, on_pre = on_pre_eqs_ampa, method = 'rk4')
Tcore_L5_exc_exc_con_ampa.connect(i = connection_indices_dict['Tcore_L5_exc_exc_con_ind'][0], j = connection_indices_dict['Tcore_L5_exc_exc_con_ind'][1])
Tcore_L5_exc_exc_con_ampa.delay = 7*ms + clip(0.5*randn(),0,5)*ms
Synapse_list.append(Tcore_L5_exc_exc_con_ampa)

Tcore_L5_exc_exc_con_nmda = Synapses(T_exc, CX_exc_L5, model = eqs_syn_nmda_Tcore_L5_exc, on_pre = on_pre_eqs_nmda, method = 'rk4')
Tcore_L5_exc_exc_con_nmda.connect(i = connection_indices_dict['Tcore_L5_exc_exc_con_ind'][0], j = connection_indices_dict['Tcore_L5_exc_exc_con_ind'][1])
Tcore_L5_exc_exc_con_nmda.delay = 7*ms + clip(0.5*randn(),0,5)*ms
Synapse_list.append(Tcore_L5_exc_exc_con_nmda)

Tcore_L5_exc_inh_con_ampa = Synapses(T_exc, CX_inh_L5, model = eqs_syn_ampa_Tcore_L5_exc, on_pre = on_pre_eqs_ampa, method = 'rk4')
Tcore_L5_exc_inh_con_ampa.connect(i = connection_indices_dict['Tcore_L5_exc_inh_con_ind'][0], j = connection_indices_dict['Tcore_L5_exc_inh_con_ind'][1])
Tcore_L5_exc_inh_con_ampa.delay = 7*ms + clip(0.5*randn(),0,5)*ms
Synapse_list.append(Tcore_L5_exc_inh_con_ampa)

Tcore_L5_exc_inh_con_nmda = Synapses(T_exc, CX_inh_L5, model = eqs_syn_nmda_Tcore_L5_exc, on_pre = on_pre_eqs_nmda, method = 'rk4')
Tcore_L5_exc_inh_con_nmda.connect(i = connection_indices_dict['Tcore_L5_exc_inh_con_ind'][0], j = connection_indices_dict['Tcore_L5_exc_inh_con_ind'][1])
Tcore_L5_exc_inh_con_nmda.delay = 7*ms + clip(0.5*randn(),0,5)*ms
Synapse_list.append(Tcore_L5_exc_inh_con_nmda)

# ---------------------------------------------------Tmatrix_L2 -------------------------------------------------------------------

Tmatrix_L2_exc_exc_con_ampa = Synapses(T_exc, CX_exc_L2, model = eqs_syn_ampa_Tmatrix_L2_exc, on_pre = on_pre_eqs_ampa, method = 'rk4')
Tmatrix_L2_exc_exc_con_ampa.connect(i = connection_indices_dict['Tmatrix_L2_exc_exc_con_ind'][0], j = connection_indices_dict['Tmatrix_L2_exc_exc_con_ind'][1])
Tmatrix_L2_exc_exc_con_ampa.delay = 7*ms + clip(2*randn(),0,5)*ms
Synapse_list.append(Tmatrix_L2_exc_exc_con_ampa)

Tmatrix_L2_exc_exc_con_nmda = Synapses(T_exc, CX_exc_L2, model = eqs_syn_nmda_Tmatrix_L2_exc, on_pre = on_pre_eqs_nmda, method = 'rk4')
Tmatrix_L2_exc_exc_con_nmda.connect(i = connection_indices_dict['Tmatrix_L2_exc_exc_con_ind'][0], j = connection_indices_dict['Tmatrix_L2_exc_exc_con_ind'][1])
Tmatrix_L2_exc_exc_con_nmda.delay = 7*ms + clip(2*randn(),0,5)*ms
Synapse_list.append(Tmatrix_L2_exc_exc_con_nmda)

Tmatrix_L2_exc_inh_con_ampa = Synapses(T_exc, CX_inh_L2, model = eqs_syn_ampa_Tmatrix_L2_exc, on_pre = on_pre_eqs_ampa, method = 'rk4')
Tmatrix_L2_exc_inh_con_ampa.connect(i = connection_indices_dict['Tmatrix_L2_exc_inh_con_ind'][0], j = connection_indices_dict['Tmatrix_L2_exc_inh_con_ind'][1])
Tmatrix_L2_exc_inh_con_ampa.delay = 7*ms + clip(2*randn(),0,5)*ms
Synapse_list.append(Tmatrix_L2_exc_inh_con_ampa)

Tmatrix_L2_exc_inh_con_nmda = Synapses(T_exc, CX_inh_L2, model = eqs_syn_nmda_Tmatrix_L2_exc, on_pre = on_pre_eqs_nmda, method = 'rk4')
Tmatrix_L2_exc_inh_con_nmda.connect(i = connection_indices_dict['Tmatrix_L2_exc_inh_con_ind'][0], j = connection_indices_dict['Tmatrix_L2_exc_inh_con_ind'][1])
Tmatrix_L2_exc_inh_con_nmda.delay = 7*ms + clip(2*randn(),0,5)*ms
Synapse_list.append(Tmatrix_L2_exc_inh_con_nmda)

# ------------------------------------------------------ Tmatrix_L5 --------------------------------------------------------------

Tmatrix_L5_exc_exc_con_ampa = Synapses(T_exc, CX_exc_L5, model = eqs_syn_ampa_Tmatrix_L5_exc, on_pre = on_pre_eqs_ampa, method = 'rk4')
Tmatrix_L5_exc_exc_con_ampa.connect(i = connection_indices_dict['Tmatrix_L5_exc_exc_con_ind'][0], j = connection_indices_dict['Tmatrix_L5_exc_exc_con_ind'][1])
Tmatrix_L5_exc_exc_con_ampa.delay = 7*ms + clip(2*randn(),0,5)*ms
Synapse_list.append(Tmatrix_L5_exc_exc_con_ampa)

Tmatrix_L5_exc_exc_con_nmda = Synapses(T_exc, CX_exc_L5, model = eqs_syn_nmda_Tmatrix_L5_exc, on_pre = on_pre_eqs_nmda, method = 'rk4')
Tmatrix_L5_exc_exc_con_nmda.connect(i = connection_indices_dict['Tmatrix_L5_exc_exc_con_ind'][0], j = connection_indices_dict['Tmatrix_L5_exc_exc_con_ind'][1])
Tmatrix_L5_exc_exc_con_nmda.delay = 7*ms + clip(2*randn(),0,5)*ms
Synapse_list.append(Tmatrix_L5_exc_exc_con_nmda)

Tmatrix_L5_exc_inh_con_ampa = Synapses(T_exc, CX_inh_L5, model = eqs_syn_ampa_Tmatrix_L5_exc, on_pre = on_pre_eqs_ampa, method = 'rk4')
Tmatrix_L5_exc_inh_con_ampa.connect(i = connection_indices_dict['Tmatrix_L5_exc_inh_con_ind'][0], j = connection_indices_dict['Tmatrix_L5_exc_inh_con_ind'][1])
Tmatrix_L5_exc_inh_con_ampa.delay = 7*ms + clip(2*randn(),0,5)*ms
Synapse_list.append(Tmatrix_L5_exc_inh_con_ampa)

Tmatrix_L5_exc_inh_con_nmda = Synapses(T_exc, CX_inh_L5, model = eqs_syn_nmda_Tmatrix_L5_exc, on_pre = on_pre_eqs_nmda, method = 'rk4')
Tmatrix_L5_exc_inh_con_nmda.connect(i = connection_indices_dict['Tmatrix_L5_exc_inh_con_ind'][0], j = connection_indices_dict['Tmatrix_L5_exc_inh_con_ind'][1])
Tmatrix_L5_exc_inh_con_nmda.delay = 7*ms + clip(2*randn(),0,5)*ms
Synapse_list.append(Tmatrix_L5_exc_inh_con_nmda)


# --------------------------------------------------- intrathalamic connections --------------------------------------------------

# ---------------------------------------------------- T_NRT ------------------------------------------------------------

T_NRT_exc_inh_con_ampa = Synapses(T_exc, NRT, model = eqs_syn_ampa_T_NRT_exc, on_pre = on_pre_eqs_ampa, method = 'rk4')
T_NRT_exc_inh_con_ampa.connect(i = connection_indices_dict['T_NRT_exc_inh_con_ind'][0], j = connection_indices_dict['T_NRT_exc_inh_con_ind'][1])
T_NRT_exc_inh_con_ampa.delay = 2*ms + clip(0.25*randn(),0,5)*ms
Synapse_list.append(T_NRT_exc_inh_con_ampa)

T_NRT_exc_inh_con_nmda = Synapses(T_exc, NRT, model = eqs_syn_nmda_T_NRT_exc, on_pre = on_pre_eqs_nmda, method = 'rk4')
T_NRT_exc_inh_con_nmda.connect(i = connection_indices_dict['T_NRT_exc_inh_con_ind'][0], j = connection_indices_dict['T_NRT_exc_inh_con_ind'][1])
T_NRT_exc_inh_con_nmda.delay = 2*ms + clip(0.25*randn(),0,5)*ms
Synapse_list.append(T_NRT_exc_inh_con_nmda)

# ---------------------------------------------------- T_T ----------------------------------------------------------------

T_T_inh_exc_con_gaba_a = Synapses(T_inh, T_exc, model = eqs_syn_gaba_a_T_T_inh, on_pre = on_pre_eqs_gaba_a, method = 'rk4')
T_T_inh_exc_con_gaba_a.connect(i = connection_indices_dict['T_T_inh_exc_con_ind'][0], j = connection_indices_dict['T_T_inh_exc_con_ind'][1])
T_T_inh_exc_con_gaba_a.delay = 1*ms + clip(0.25*randn(),0,1)*ms
Synapse_list.append(T_T_inh_exc_con_gaba_a)

T_T_inh_inh_con_gaba_a = Synapses(T_inh, T_inh, model = eqs_syn_gaba_a_T_T_inh, on_pre = on_pre_eqs_gaba_a, method = 'rk4')
T_T_inh_inh_con_gaba_a.connect(i = connection_indices_dict['T_T_inh_inh_con_ind'][0], j = connection_indices_dict['T_T_inh_inh_con_ind'][1])
T_T_inh_inh_con_gaba_a.delay = 1*ms + clip(0.25*randn(),0,1)*ms
Synapse_list.append(T_T_inh_inh_con_gaba_a)

# ---------------------------------------------------- NRT_T_gaba_a --------------------------------------------------------------------

NRT_T_inh_exc_con_gaba_a = Synapses(NRT, T_exc, model = eqs_syn_gaba_a_NRT_T_gaba_a_inh, on_pre = on_pre_eqs_gaba_a, method = 'rk4')
NRT_T_inh_exc_con_gaba_a.connect(i = connection_indices_dict['NRT_T_gaba_a_inh_exc_con_ind'][0], j = connection_indices_dict['NRT_T_gaba_a_inh_exc_con_ind'][1])
NRT_T_inh_exc_con_gaba_a.delay = 1.5*ms + clip(0.25*randn(),0,1)*ms
Synapse_list.append(NRT_T_inh_exc_con_gaba_a)

NRT_T_inh_inh_con_gaba_a = Synapses(NRT, T_inh, model = eqs_syn_gaba_a_NRT_T_gaba_a_inh, on_pre = on_pre_eqs_gaba_a, method = 'rk4')
NRT_T_inh_inh_con_gaba_a.connect(i = connection_indices_dict['NRT_T_gaba_a_inh_inh_con_ind'][0], j = connection_indices_dict['NRT_T_gaba_a_inh_inh_con_ind'][1])
NRT_T_inh_inh_con_gaba_a.delay = 1.5*ms + clip(0.25*randn(),0,1)*ms
Synapse_list.append(NRT_T_inh_inh_con_gaba_a)

# ---------------------------------------------------- NRNRT_T_gaba_b ----------------------------------------------------------------

NRT_T_inh_exc_con_gaba_b = Synapses(NRT, T_exc, model = eqs_syn_gaba_b_th, on_pre = on_pre_eqs_gaba_b, method = 'rk4')
NRT_T_inh_exc_con_gaba_b.connect(i = connection_indices_dict['NRT_T_gaba_b_inh_exc_con_ind'][0], j = connection_indices_dict['NRT_T_gaba_b_inh_exc_con_ind'][1])
NRT_T_inh_exc_con_gaba_b.delay = 1.5*ms + clip(0.25*randn(),0,1)*ms
Synapse_list.append(NRT_T_inh_exc_con_gaba_b)

NRT_T_inh_inh_con_gaba_b = Synapses(NRT, T_inh, model = eqs_syn_gaba_b_th, on_pre = on_pre_eqs_gaba_b, method = 'rk4')
NRT_T_inh_inh_con_gaba_b.connect(i = connection_indices_dict['NRT_T_gaba_b_inh_inh_con_ind'][0], j = connection_indices_dict['NRT_T_gaba_b_inh_inh_con_ind'][1])
NRT_T_inh_inh_con_gaba_b.delay = 1.5*ms + clip(0.25*randn(),0,1)*ms
Synapse_list.append(NRT_T_inh_inh_con_gaba_b)
    
    
    #--------------------------------------------------- create network -----------------------------------------------------------------------
    
    network = Network(CX_exc_L2, CX_inh_L2, CX_exc_L4, CX_inh_L4, CX_exc_L5, CX_inh_L5,
                      T_exc, T_inh, NRT,
                      Synapse_list, 
                      spikes_exc_L2, spikes_exc_L4, spikes_exc_L5, 
                      spikes_inh_L2, spikes_inh_L4, spikes_inh_L5,
                      spikes_T_exc, spikes_T_inh, spikes_NRT,
                      voltage_monitor_exc_L2, voltage_monitor_exc_L4, voltage_monitor_exc_L5, 
                      voltage_monitor_inh_L2, voltage_monitor_inh_L4, voltage_monitor_inh_L5,
                      voltage_monitor_T_exc, voltage_monitor_T_inh, voltage_monitor_NRT,
                      current_monitor_exc_L2, current_monitor_exc_L4, current_monitor_exc_L5, 
                      current_monitor_inh_L2, current_monitor_inh_L4, current_monitor_inh_L5,
                      current_monitor_T_exc, current_monitor_T_inh, current_monitor_NRT)

print('model built')
# network.store('initialized', filename = 'initialized')


#just get the names of the monitors in a list so I can use it for the filename when I pickle them and I know the original monitors I had in the network
monitor_names = []
for monitor in spikes_monitor_list + voltage_monitor_list + current_monitor_list:
    monitor_names.append(get_var_name(monitor))

#%%
print('running model')
network.run(5000*ms, report = 'text', profile = True)


plot(voltage_monitor_T_exc.t/ms, voltage_monitor_exc_L5.V[3])
# # plot(voltage_monitor_T_exc.t/ms, voltage_monitor__inh.V[3])

# plot(voltage_monitor_T_exc.t/ms, current_monitor_exc_L4.Iks[2])
# plot(voltage_monitor_T_exc.t/ms, current_monitor_exc_L4.Idk[2])
# plot(voltage_monitor_T_exc.t/ms, ampa_con)
# plot(voltage_monitor_T_exc.t/ms, gaba_con)

# # # plot(voltage_monitor_T_exc.t/ms, voltage_monitor_exc_L4.V[3])
# # # plot(voltage_monitor_T_exc.t/ms, voltage_monitor_exc_L5.V[11])

# plot(voltage_monitor_T_exc.t/ms, voltage_monitor_inh_L2.V[30])
# title('inh_L2')
# plot_currents(current_monitor_exc_L4, 'exc_L4', neuron_to_plot = 30, detailed_syn_only = True)
# plot_currents(current_monitor_exc_L5, 'exc_L5', neuron_to_plot = 30)


plot(spikes_exc_L2.t/ms, spikes_exc_L2.i, '.k')
# xlabel('Time (ms)')
# ylabel('Neuron index')

# # # plot(spikes_inh_L2.t/ms, spikes_inh_L2.i, '.k')
# # # xlabel('Time (ms)')
# # # ylabel('Neuron index')

colored_voltage_traces(voltage_monitor_exc_L4, 'V', last_neuron_index = 20)

# cycle_through_traces(voltage_monitor_exc_L2, 'V')

# plot(voltage_monitor_exc.t/ms, voltage_monitor_exc.V[1])


#%% ------------------------------------------------------ dumping on disk and resetting the monitors (save memory!) ---------------------------------------------------
# os.chdir(curr_results)
# # os.chdir(f'{curr_results} 0_125')
# # os.chdir('D:\\Computational results\\2021-04-11 1520')
# for monitor, name in zip(all_monitors, monitor_names): 
#     print(f'saving {name}')
#     data = monitor.get_states()
#     with open(f'{name}.data', 'wb') as f:
#         pickle.dump(data, f)
#     del data

#delete the monitors
# for monitor in monitor_names:
#     try:
#         del globals()[monitor]
#     except KeyError:
#         print(f'couldnt find {monitor}')
#         continue

# # DELETE THE MONITORLISTS BECAUSE THE MONITORS ARE STILL STORED THERE FROM THE ORIGINAL RUN AND TAKE UP A LOT OF MEMORY
# del all_monitors
# del spikes_monitor_list
# del voltage_monitor_list
# del current_monitor_list

# # redo the monitors and reintegrate them in the network
# for cellgroup, name in zip(Neurongroup_list, Neurongroup_names):
#     globals()[f'spikes_{name}'] = SpikeMonitor(cellgroup)
#     network.add(globals()[f'spikes_{name}'])
#     globals()[f'voltage_monitor_{name}'] = StateMonitor(cellgroup, variables = 'V', record = True)
#     network.add(globals()[f'voltage_monitor_{name}'])
#     #do the current monitors individually because they take up a lot of memory so you might want to record less from one cell group
#     current_monitor_exc_L2 = StateMonitor(CX_exc_L2[100:125], variables = currents_CX_exc + syn_current_list_L2, record = True)
#     network.add(current_monitor_exc_L2)
#     current_monitor_inh_L2 = StateMonitor(CX_inh_L2[100:125], variables = currents_CX_inh + syn_current_list_L2, record = True)
#     network.add(current_monitor_inh_L2)
#     current_monitor_exc_L4 = StateMonitor(CX_exc_L4[100:125], variables = currents_CX_exc + syn_current_list_L4, record = True)
#     network.add(current_monitor_exc_L4)
#     current_monitor_inh_L4 = StateMonitor(CX_inh_L4[100:125], variables = currents_CX_inh + syn_current_list_L4, record = True)
#     network.add(current_monitor_inh_L4)
#     current_monitor_exc_L5 = StateMonitor(CX_exc_L5[100:125], variables = currents_CX_exc + syn_current_list_L5, record = True)
#     network.add(current_monitor_exc_L5)
#     current_monitor_inh_L5 = StateMonitor(CX_exc_L5[100:125], variables = currents_CX_inh + syn_current_list_L5, record = True)
#     network.add(current_monitor_inh_L5)
#     current_monitor_T_exc = StateMonitor(T_exc[100:125], variables = currents_T_exc + syn_current_list_T, record = True)
#     network.add(current_monitor_T_exc)
#     current_monitor_T_inh = StateMonitor(T_inh[100:125], variables = currents_T_inh + syn_current_list_T, record = True)
#     network.add(current_monitor_T_inh)
#     current_monitor_NRT = StateMonitor(NRT[100:125], variables = currents_NRT + syn_current_list_NRT, record = True)
#     network.add(current_monitor_NRT)

    
    
    # if 'L2' in name or 'L4' in name or 'L5' in name:
    #     if 'exc' in name:
    #         currents = globals()['currents_CX_exc'] + globals()[f'syn_current_list{name[0:2]}']
    #     elif 'inh' in name:
    #         currents = globals()['currents_CX_inh'] + globals()[f'syn_current_list{name[0:2]}']
    # else:
    #     currents =
    # globals()[f'current_monitor_{cellgroup}'] = StateMonitor(cellgroup[100:125], variables = currents_T_exc + syn_current_list_T, record = True)
    
    # network.remove('spikes_exc_L2')

#%% just fucking around a bit to test the synapse equations
# CX_exc.Iapp[0] = 65*mV

# mon = StateMonitor(CX_exc, variables = True, record = True)
# # monsyn = StateMonitor(syns_gaba_b, variables = ['P','S','G'], record = True) #always better to explicitly declare which variables you want to record to avoid weird bugs

# # network = Network(CX_exc, syns_gaba_b, mon, monsyn)
# network = Network(CX_exc, mon, P)
# network.run(5000*ms, report = 'text')
                
#  plot(mon.t/ms, mon.V[1]41
# #plot(mon.t/ms, mon.V[1])
# xlabel('time ms')
# ylabel('volt')

#%%
# #%%one layer model
# variables_to_record = ['V','Inal', 'Ikl', 'Ih', 'It', 'Inap', 'Idk', 'Ikca', 'Isyn', 'Isyn_ampa', 'Isyn_nmda', 'Isyn_gaba_a', 'Isyn_gaba_b', 'Ipoiss']                      
# CX_exc_L5 = NeuronGroup(1800, model = eqs_CX_exc, method = 'rk4', threshold = 'V > thresh',
#                 reset = '''V = Ena
#                             thresh = Ena
#                             lastspike = t''',
#                 events = {'poisson' : 'poiss < poiss_rate_CX_exc*dt'})
# CX_exc_L5.V = -70*mV

# IB_cells_indx = np.linspace(0,1799,int(0.3*1800), dtype = int)
# #need to use for loop because subgroup creation can only use contiguous integers
# for n in IB_cells_indx:
#     CX_exc_L5[n].run_regularly('Ih = g_h_CX_exc * m_h * (V - Eh)')

# CX_exc_L5.run_regularly('poiss = rand()')
# CX_exc_L5.run_on_event('poisson', 't_last_poisson = t')
# poiss_rate_CX_exc = 15*Hz
# # P_exc_L5 = PoissonInput(CX_exc_L5, 'V', 15, 1*Hz, weight = 1*mV)

# spikes_exc = SpikeMonitor(CX_exc_L5)
# voltage_monitor_exc = StateMonitor(CX_exc_L5[0:25], variables = True, record = True)


# CX_inh_L5 = NeuronGroup(900, model = eqs_CX_int, method = 'rk4', threshold = 'V > thresh',
#                 reset = '''V = Ena
#                             thresh = Ena
#                             lastspike = t''',
#                 events = {'poisson' : 'poiss < poiss_rate_CX_inh*dt'})
# CX_inh_L5.V = -70*mV
                            
# CX_inh_L5.run_regularly('poiss = rand()')
# CX_inh_L5.run_on_event('poisson', 't_last_poisson = t')
# poiss_rate_CX_inh = 6*Hz
# # P_inh_L5 = PoissonInput(CX_inh_L5, 'V', 2, 1*Hz, weight = 1*mV)

# spikes_inh = SpikeMonitor(CX_inh_L5)
# voltage_monitor_inh = StateMonitor(CX_inh_L5[0:25], variables = True , record = True)
                            
# L5_L5_exc_exc_con_ampa = Synapses(CX_exc_L5, CX_exc_L5, model = eqs_syn_ampa, on_pre = on_pre_eqs, method = 'rk4')
# L5_L5_exc_exc_con_ampa.connect(i = connection_indices_dict['L5_L5_exc_exc_con_ind'][0], j = connection_indices_dict['L5_L5_exc_exc_con_ind'][1])
# L5_L5_exc_exc_con_ampa.delay = 1*ms + clip(0.25*randn(),0,1)*ms
# # visualise_connectivity(L5_L5_con_ampa, line_plot = False)
# L5_L5_exc_exc_con_nmda = Synapses(CX_exc_L5, CX_exc_L5, model = eqs_syn_nmda, on_pre = on_pre_eqs, method = 'rk4')
# L5_L5_exc_exc_con_nmda.connect(i = connection_indices_dict['L5_L5_exc_exc_con_ind'][0], j = connection_indices_dict['L5_L5_exc_exc_con_ind'][1])
# L5_L5_exc_exc_con_nmda.delay = 1*ms + clip(0.25*randn(),0,1)*ms

# L5_L5_inh_exc_con_gaba_a = Synapses(CX_inh_L5, CX_exc_L5, model = eqs_syn_gaba_a_CX, on_pre = on_pre_eqs, method = 'rk4')
# L5_L5_inh_exc_con_gaba_a.connect(i = connection_indices_dict['L5_L5_inh_exc_con_ind'][0], j = connection_indices_dict['L5_L5_inh_exc_con_ind'][1])
# L5_L5_inh_exc_con_gaba_a.delay = 0.75*ms + clip(0.1*randn(),0,1)*ms

# L5_L5_inh_inh_con_gaba_a = Synapses(CX_inh_L5, CX_inh_L5, model = eqs_syn_gaba_a_CX, on_pre = on_pre_eqs, me_od = 'rk4')
# L5_L5_inh_inh_con_gaba_a.connect(i = connection_indices_dict['L5_L5_inh_inh_con_ind'][0], j = connection_indices_dict['L5_L5_inh_inh_con_ind'][1])
# L5_L5_inh_inh_con_gaba_a.delay = 0.75*ms + clip(0.1*randn(),0,1)*ms

# L5_L5_exc_inh_con_ampa = Synapses(CX_exc_L5, CX_inh_L5, model = eqs_syn_ampa, on_pre = on_pre_eqs, method = 'rk4')
# L5_L5_exc_inh_con_ampa.connect(i = connection_indices_dict['L5_L5_exc_inh_con_ind'][0], j = connection_indices_dict['L5_L5_exc_inh_con_ind'][1])
# L5_L5_exc_inh_con_ampa.delay = 1*ms + clip(0.25*randn(),0,1)*ms
# # visualise_connectivity(L5_L5_con_ampa, line_plot = False)
# L5_L5_exc_inh_con_nmda = Synapses(CX_exc_L5, CX_inh_L5, model = eqs_syn_nmda, on_pre = on_pre_eqs, method = 'rk4')
# L5_L5_exc_inh_con_nmda.connect(i = connection_indices_dict['L5_L5_exc_inh_con_ind'][0], j = connection_indices_dict['L5_L5_exc_inh_con_ind'][1])
# L5_L5_exc_inh_con_nmda.delay = 1*ms + clip(0.25*randn(),0,1)*ms

# network_onelayer = Network(CX_exc_L5, CX_inh_L5, L5_L5_exc_exc_con_ampa, L5_L5_exc_exc_con_nmda, L5_L5_exc_inh_con_ampa, L5_L5_exc_inh_con_nmda, L5_L5_inh_exc_con_gaba_a, L5_L5_inh_inh_con_gaba_a, spikes_exc, spikes_inh, voltage_monitor_exc, voltage_monitor_inh)
# network_onelayer.store('initialized')
# # P_exc_L5_1 = PoissonInput(CX_exc_L5, 'V', 20, 1*Hz, weight = 0.5*mV)
# # network = Network(CX_exc_L5, CX_inh_L5, L5_L5_exc_exc_con_ampa, L5_L5_exc_exc_con_nmda, L5_L5_exc_inh_con_ampa, L5_L5_exc_inh_con_nmda, L5_L5_inh_exc_con_gaba_a, L5_L5_inh_inh_con_gaba_a, spikes_exc, spikes_inh, voltage_monitor_exc, voltage_monitor_inh, P_exc_L5, P_inh_L5, P_exc_L5_1)


#%% some objects necessary for this model (same as functions module)

# class Struct:
#     """
#     just pass in keys and values to build the struct like Matlab
#     """
#     def __init__(self, **entries):
#         self.__dict__.update(entries)

# def connection_indices(source_cells_per_point, target_cells_per_point, p_max, radius, std_connections, Layers_for_progress_report = None):
#     """
#     function to give the connection matrix 2xN_con (first row = presyn indices, second row = postsyn indices) given the parameters in the paper
#     I can't quite figure out exactly what the connectivity is in Tononi paper 
#     i)strength = adjustment of synaptic conductance I guess??, ii) are the two exc neurons in the cortical columns of the same layer connected by definition? iii) how exactly are the topographic points calculated for the radius?. Bazhenov paper in contrast has quite simple connectivity with one-dimensional cortical layers and simple radii of connections
#     I just say if it has a radius of 12 then it's a square with size 24x24 from that neuron, it might be wrong but doesn't specify in the paper
    
#     input: 
#         source_cells_per_point = how many cells per topographic point at the source? (e.g. 2 in cortex if doing excitatory connections)
#         Layers_for_progress_report = include the input and output layers (e.g. l2-l4) if you want progress report
#     """
#     i = np.array([], dtype=int) # pre and post neuron numbers that are connected
#     j = np.array([], dtype=int)
    
#     # prob_scaling_factor = p_max/(1/(2*pi*std_connections**2)**0.5) #by how much to scale the gaussian curve given the pmax (which is the probability at the center of the gaussian curve) 
    
#     source_cell_number = grid_size**2*source_cells_per_point
#     target_cell_number = grid_size**2*target_cells_per_point
    
#     neuron_pre = np.linspace(1,source_cell_number, source_cell_number, dtype = int).reshape(source_cells_per_point,30,30) #make an array map with neuron numbers. This makes n 30x30 arrays with neuron numbers 1-900 and 901-1800 etc.... Used for the topographic mapping 
#     neuron_post = np.linspace(1,target_cell_number, target_cell_number, dtype = int).reshape(target_cells_per_point,30,30)
    
#     for pre in np.nditer(neuron_pre):
#         if Layers_for_progress_report is not None:
#             print('pre neuron ' + Layers_for_progress_report + ' ' + str(pre) + '/' + str(source_cell_number)) # just progress report so I know it works
#         indx_pre = np.where(neuron_pre == pre)# gives a tuple of arrays with each array containing the index in that dimension. First element is irrelevant for us as it is the index along the "depth" axis (i.e. how many neurons deep per topographic grid point)
#         neurons_post_within_radius = neuron_post[:, clip((indx_pre[1][0]-(radius+1)),0,10000):clip((indx_pre[1][0]+(radius+1)),0,10000), clip((indx_pre[2][0]-(radius+1)),0,10000):clip((indx_pre[2][0]+(radius+1)),0,10000)] # which neurons fall within the connection radius (here a square of size 2xradius (radius+1 bc stop index not incl.)),I clip to 10000 arbitrarily
        
#         ran = False
#         for post in np.nditer(neurons_post_within_radius):
#             #autapses?
#             if autapses is False and not ran:
#                 if pre == post:
#                     ran = True
#                     continue
#             indx_post = np.where(neuron_post == post)
#             distance = sqrt((indx_pre[1][0] - indx_post[1][0])**2 + (indx_pre[2][0] - indx_post[2][0])**2) #what is the distance between the pre and post neuron, use pythagorean theorem
            
#             p = p_max*exp(-distance**2/(2*std_connections**2))
#             # p = prob_scaling_factor*exp(-distance**2/(2*std_connections**2))/(2*pi*std_connections**2)**0.5 # probability of connection according to scaled gaussian
#             if p > np.random.rand():
#                 i = np.hstack((i, pre))
#                 j = np.hstack((j, post))
                              
#     return np.array([i, j])


# def visualise_connectivity(S, line_plot = False, scatter_plot = True):
#     '''
#     to plot connectivity of the synapses. In the Tononi model connectivity is quite heavy so the line plot just gets completely saturated, therefore default False
#     Parameters
#     ----------
#     S : Synapses to plot (one Synapse class instance).
    
#     line_plot : TYPE, optional
#         the line plot with i and j on opposite sides and lines connecting them. The default is False.
#     scatter_plot : TYPE, optional
#         classic scatter plot with i and j on x and y axis. The default is True.

#     Returns
#     -------
#     plots showing connectivity

#     '''
#     Ns = len(S.source)
#     Nt = len(S.target)
#     figure(figsize=(10, 4))
#     if line_plot and scatterplot:
#         subplot(121)
#     if line_plot:
#         plot(zeros(Ns), arange(Ns), 'ok', ms=10)
#         plot(ones(Nt), arange(Nt), 'ok', ms=10)
#         for i, j in zip(S.i, S.j):
#             plot([0, 1], [i, j], '-k')
#         xticks([0, 1], ['Source', 'Target'])
#         ylabel('Neuron index')
#         xlim(-0.1, 1.1)
#         ylim(-1, max(Ns, Nt))
#     if line_plot and scatterplot:
#         subplot(122)
#     if scatter_plot:
#         plot(S.i, S.j, 'ok')
#         xlim(-1, Ns)
#         ylim(-1, Nt)
#         xlabel('Source neuron index')
#         ylabel('Target neuron index')


# def plot_voltage_dependency_curve(equation, variable, lower_bound = -100, upper_bound = 100):
#     '''
#     function to produce a plot of the voltage dependency curve of an equation, from -100 to +100mV.
#     useful for example in plotting HH gating variables, NMDA voltage response curve etc...
#     Equation has to be provided without units
#     variable is the variable you want to plot that's in the equation
#     '''
#     a = []
#     b = np.linspace(lower_bound, upper_bound, 10000)

#     # var = variable #change the string of the variable to a variable name
#     for x in b:
#         vars()[variable] = x
#         c = eval(equation)
#         a.append(c)
    
#     plot(b,a)
#     xlabel('voltage, mV')


# def colored_voltage_traces(Monitor, voltage_variable, first_neuron_index = 0, last_neuron_index = None, min_colored = -0.08, max_colored = -0.05):
#     '''
#     to colormap the voltage trace of many neurons. 
    
#     Parameters
#     ----------
#     Monitor : StateMonitor to plot
#     first_neuron : first neuron to plot as index
#     last_neuron : last neuron to plot
#     min_colored : the voltage value that will have the min color (for U/D states -80)
#     max_colored : the voltage value that will have the max color (for U/D states -50)

#     Returns
#     -------
#     A colormap of the voltage traces.

#     '''
#     run = 0
#     for neuron in np.arange(first_neuron_index, Monitor.n_indices):
#         if run == 0:
#             stacked_traces = getattr(Monitor, voltage_variable)[neuron]
#         else:
#             trace = getattr(Monitor, voltage_variable)[neuron]
#             stacked_traces = vstack((stacked_traces,trace))
#         run +=1
#     fig, ax = subplots()
#     plot = ax.pcolormesh(stacked_traces, cmap = 'RdYlBu_r', vmin = min_colored, vmax = max_colored)
#     fig.colorbar(plot, ax = ax)
#     # figure out which x-axis ticks are the seconds
#     xtick_pos, = np.where(np.mod(Monitor.t/second, 1) == 0) #where the remainder of the division of time vector and 1 is 0 (i.e. integer i.e. a whole second)
#     ax.set_xticks(xtick_pos)
#     xtick_labels = np.arange(int(len(Monitor.t)*(Monitor.clock.dt/second)))
#     ax.set_xticklabels(xtick_labels)
#     xlabel('time in seconds')
#     ylabel('neuron number')
 

# def plot_currents(Monitor, neuron_to_plot = 0, int_currents = True, syn_currents = True, current_list = current_list):
#     '''
#     Parameters
#     ----------
#     Monitor : StateMonitor with the currents
#     neuron_to_plot: index of neuron to plot in StateMonitor
#     int_currents: plot intrinsic currents?
#     syn_currents: plot synaptic currents?
#     Returns
#     -------
#     plot with all the conductances in the cell. Click on legend to toggle currents on and off
#     '''
#     if int_currents == False:
#         current_list = syn_current_list 
#     elif syn_currents == False: 
#         current_list = int_current_list
#     else:
#         current_list = current_list
#     lines = []
#     fig, ax = subplots()
#     for indx, current in enumerate(current_list):
#         try:
#             vars()['line' + str(indx)], = ax.plot(Monitor.t/ms, getattr(Monitor, current)[0], label = current) 
#             lines.append(vars()['line' + str(indx)])
#         except:
#             print(f'no current {current} in this cell')
#     xlabel('time ms')
#     ylabel('conductance')
#     ax.set_title('click on legend line to toggle line on/off')
#     leg = ax.legend(fancybox=True, shadow=True)
    
#     lined = {} #to map legend lines to originial lines
#     for legline, origline in zip(leg.get_lines(), lines):
#         legline.set_picker(True)  # Enable picking on the legend line.
#         lined[legline] = origline
        
#     def on_pick(event):
#         # On the pick event, find the original line corresponding to the legend
#         # proxy line, and toggle its visibility.
#         legline = event.artist
#         origline = lined[legline]
#         visible = not origline.get_visible()
#         origline.set_visible(visible)
#         # Change the alpha on the line in the legend so we can see what lines
#         # have been toggled.
#         legline.set_alpha(1.0 if visible else 0.2)
#         fig.canvas.draw()
    
#     fig.canvas.mpl_connect('pick_event', on_pick) #pick_event is the event id for selecting an "artist" on figure


# def calc_LFP(monitor):
#     '''
#     sum up the postsynaptic currents of all excitatory cells in a layer and inverse the signal to calculate the LFP, as in 2007 Tononi. 
#     They have some constants in the equation but they are the same for every neuron so it just scales the LFP.
#     In 2005 they simply average out the membrane potential to get an LFP-like signal but it's obviously not very interesting
#     '''
#     LFP = np.array([])
#     for j in range(monitor.n_indices):
#         Ij = monitor.Isyn[j]
#         if j == 0:
#             LFP = Ij
#         else:
#             LFP = np.add(Ij, LFP)
#     # filter_cheby = cheby2()
#     fig, ax = subplots()
#     ax.plot(monitor.t/ms, LFP)
#     ax.invert_yaxis()
    
# from matplotlib.widgets import Button
    
# def cycle_through_traces(monitor, variable, n_rows = 4, n_cols = 4): 
#     fig, ax = plt.subplots(n_rows, n_cols)
#     for ind, ax1 in enumerate(ax.flatten()):
#         ax1.plot(monitor.t/ms, monitor.V[ind])
#         ax1.set_xlabel('time ms')
#         ax1.set_title('Neuron ' + str(ind))
        
#     class event_handling:
#         ind = 0
#         def next(self, event):
#             self.ind += ax.size #increases by the number of cells in the plot
#             for ind, ax1 in enumerate(ax.flatten()):
#                 ax1.clear()
#                 try:
#                     ax1.plot(monitor.t/ms, monitor.V[ind + self.ind])
#                     ax1.set_xlabel('time ms')
#                     ax1.set_title('Neuron ' + str(ind + self.ind))
#                 except IndexError:
#                     continue
#             plt.draw()
    
#         def prev(self, event):
#             self.ind -= ax.size #increases by the number of cells in the plot
#             for ind, ax1 in enumerate(ax.flatten()): 
#                 ax1.clear()
#                 ax1.plot(monitor.t/ms, monitor.V[ind + self.ind])
#                 ax1.set_xlabel('time ms')
#                 ax1.set_title('Neuron ' + str(ind + self.ind))
#             plt.draw()
        
#         def plot_enlarge(self, event):
#               if event.inaxes in ax:
#                  newfig, newax = plt.subplots()
#                  plot_to_enlarge = fig.axes.index(event.inaxes)
#                  newax.plot(ax.flatten()[plot_to_enlarge].get_lines()[0].get_xdata(), ax.flatten()[plot_to_enlarge].get_lines()[0].get_ydata())
#                  newfig.canvas.manager.window.showMaximized()
                 
#     callback = event_handling()
#     axprev = plt.axes([0.7, 0.01, 0.075, 0.075])
#     axnext = plt.axes([0.81, 0.01, 0.075, 0.075])
#     bnext = Button(axnext, 'Next')
#     bnext.on_clicked(callback.next)
#     bprev = Button(axprev, 'Previous')
#     bprev.on_clicked(callback.prev)
#     axprev._button = bnext #create dummy reference (don't quite get why but I think it needs an explicit reference as an attribute of the plt.axes because the variable bnext is gone after function is called). Putting globals()['bnext'] = Button(axnext, 'Next') works too
#     axnext._button = bprev
#     fig.canvas.mpl_connect("button_press_event", callback.plot_enlarge)




# with open('voltage_monitor_exc_L2.data','rb') as f:
#     new_dict = pickle.load(f)
#     infile.close()


# min_colored = -0.08
# max_colored = -0.05
# run = 0
# last_neuron = 60
# for neuron in np.arange(1, last_neuron):
#     if run == 0:
#         stacked_traces = a['V'][neuron]
#     else:
#         trace = a['V'][neuron]
#         stacked_traces = np.vstack((stacked_traces,trace))
#     run +=1
# fig, ax = plt.subplots()
# plot = ax.pcolormesh(stacked_traces, cmap = 'RdYlBu_r', vmin = min_colored, vmax = max_colored)
# fig.colorbar(plot, ax = ax)
# figure out which x-axis ticks are the seconds
# xtick_pos, = np.where(np.mod(a['t']/second, 1*second) == 0) #where the remainder of the division of time vector and 1 is 0 (i.e. integer i.e. a whole second)
# ax.set_xticks(xtick_pos)
# xtick_labels = np.arange(int(len(a['t'])*(0.0001*second)))
# ax.set_xticklabels(xtick_labels)
# xlabel('time in seconds')
# ylabel('neuron number')

#%% single L5 cell
# tau_m_CX_exc = 15*ms
# thresh_ss_CX_exc = -51*mvolt
# tau_thresh_CX_exc = 1*ms
# tau_spike_CX_exc = 1.3*ms
# time_spike_CX_exc = 1.4*ms

# g_kl_CX_exc = 0.55 # changed from 0.55 conductances are unitless here, because no capacitance (no area or volume defined for a cell) but rather a membrane time constant is used
# g_dk_CX_exc = 0.75 #depolarization activated potassium current
# g_ks_CX_exc = 6
# g_nal_CX_exc = 0.05
# g_nap_CX_exc = 4 #changed from 2. 4 e.g. (with g_kl_CX_exc 0.45, Ih, and changed Inap inflection point to -57.7) gives slow wave like behavior (1-3 spikes at a time) in individual IB cells at ca. 1 Hz. 3.5 gives the cell being just below threshold with Poisson inputs making it fire.
# g_spike_CX_exc = 1
# g_h_CX_exc = 2

# D_thresh = -10*mV #threshold of the logistic function of D from Idk

# poiss_str_CX_exc_L2 = 0.02
# poiss_str_CX_exc_L4 = 0.02
# poiss_str_CX_exc_L5 = 0.02

# eqs =                               (f'''
#                                     dV/dt = (- Inal - Ikl - Iint + Iapp)/tau_m_CX_exc - g_spike_CX_exc * (V - Ek)/tau_spike_CX_exc : volt
#                                     Inal = g_nal_CX_exc*(V - Ena) : volt
#                                     Ikl = g_kl_CX_exc * (V - Ek) : volt
                                    
#                                     Iint = Iks + Inap + Idk + Ih : volt
                                    
#                                     Iks = g_ks_CX_exc * m_ks * (V - Ek) : volt                  # slow noninactivating potassium current
#                                     dm_ks/dt = (m_ks_ss - m_ks)/tau_m_ks : 1
#                                     m_ks_ss = 1/(1 + exp(-(V + 34*mvolt)/(6.5*mvolt))) : 1      # is wrong in the 2009 paper, checked it in the wang 1999 paper
#                                     tau_m_ks = (8*ms)/(exp(-(V + 55*mvolt)/(30*mvolt)) + exp((V + 55*mvolt)/(30*mvolt))) : second
                                    
#                                     Inap = g_nap_CX_exc * m_nap ** 3 * (V - Ena) : volt        #they took the same equations as in Compte 2002. Pers Na current activates rapidly near spike threshold and deactivates very slowly
#                                     m_nap = 1/(1 + exp(-(V + 55.7*mvolt)/(7.7*mV))) : 1
                                    
#                                     Idk = g_dk_CX_exc * m_dk * (V - Ek) : volt                 #depolarization-activated potassium conductance, replaces Na-dependent K current in Compte --> here the term D combines Ca- and Na dependency by accumulating during depolarization
#                                     m_dk = 1/(1 + (0.25*D)**(-3.5)) : 1                        #instantaneous activation, no time constant ever described
#                                     dD/dt = D_influx - D*(1-0.001)/(800*ms) : 1
#                                     D_influx = 1/(1 + exp(-(V-D_thresh)/(5*mV)))/ms : Hz
                                    
#                                     Ih : volt
#                                     dm_h/dt = (m_h_ss - m_h)/tau_m_h : 1
#                                     m_h_ss = 1/(1 + exp((V + 75*mV)/(5.5*mV))) : 1
#                                     tau_m_h = 1*ms/(exp(-14.59 - 0.086*V/mV) + exp(-1.87 + 0.0701*V/mV)) : second
                                    
                                    
#                                     dthresh/dt = -(thresh - thresh_ss_CX_exc)/tau_thresh_CX_exc : volt #threshold for spikes
#                                     g_spike_CX_exc = int((t - lastspike) < time_spike_CX_exc) : 1
#                                     lastspike : second
                                    
#                                     # Isyn = Isyn_ampa + Isyn_nmda + Isyn_gaba_a + Isyn_gaba_b + Ipoiss: volt
                                    
#                                     # Isyn_ampa = Isyn_ampa_L2_L2_exc + Isyn_ampa_L4_L4_exc + Isyn_ampa_L5_L5_exc + Isyn_ampa_L2_L5_exc + Isyn_ampa_L4_L2_exc + Isyn_ampa_L5_L2_exc + Isyn_ampa_L5_L4_exc + Isyn_ampa_L5_NRT_exc + Isyn_ampa_L5_Tcore_exc + Isyn_ampa_L5_Tmatrix_exc + Isyn_ampa_Tcore_L4_exc + Isyn_ampa_Tcore_L5_exc + Isyn_ampa_Tmatrix_L2_exc + Isyn_ampa_Tmatrix_L5_exc + Isyn_ampa_T_NRT_exc: volt
#                                     # Isyn_ampa_L2_L2_exc: volt
#                                     # Isyn_ampa_L4_L4_exc : volt
#                                     # Isyn_ampa_L5_L5_exc : volt
#                                     # Isyn_ampa_L2_L5_exc : volt
#                                     # Isyn_ampa_L4_L2_exc : volt
#                                     # Isyn_ampa_L5_L2_exc : volt
#                                     # Isyn_ampa_L5_L4_exc : volt 
#                                     # Isyn_ampa_L5_NRT_exc : volt
#                                     # Isyn_ampa_L5_Tcore_exc : volt
#                                     # Isyn_ampa_L5_Tmatrix_exc : volt
#                                     # Isyn_ampa_Tcore_L4_exc : volt
#                                     # Isyn_ampa_Tcore_L5_exc : volt
#                                     # Isyn_ampa_Tmatrix_L2_exc : volt
#                                     # Isyn_ampa_Tmatrix_L5_exc : volt
#                                     # Isyn_ampa_T_NRT_exc : volt
                                    
#                                     # Isyn_nmda = Isyn_nmda_L2_L2_exc + Isyn_nmda_L4_L4_exc + Isyn_nmda_L5_L5_exc + Isyn_nmda_L2_L5_exc + Isyn_nmda_L4_L2_exc + Isyn_nmda_L5_L2_exc + Isyn_nmda_L5_L4_exc + Isyn_nmda_L5_NRT_exc + Isyn_nmda_L5_Tcore_exc + Isyn_nmda_L5_Tmatrix_exc + Isyn_nmda_Tcore_L4_exc + Isyn_nmda_Tcore_L5_exc + Isyn_nmda_Tmatrix_L2_exc + Isyn_nmda_Tmatrix_L5_exc + Isyn_nmda_T_NRT_exc : volt
#                                     # Isyn_nmda_L2_L2_exc : volt
#                                     # Isyn_nmda_L4_L4_exc : volt
#                                     # Isyn_nmda_L5_L5_exc : volt
#                                     # Isyn_nmda_L2_L5_exc : volt
#                                     # Isyn_nmda_L4_L2_exc : volt
#                                     # Isyn_nmda_L5_L2_exc : volt
#                                     # Isyn_nmda_L5_L4_exc : volt
#                                     # Isyn_nmda_L5_NRT_exc : volt
#                                     # Isyn_nmda_L5_Tcore_exc : volt
#                                     # Isyn_nmda_L5_Tmatrix_exc : volt
#                                     # Isyn_nmda_Tcore_L4_exc : volt
#                                     # Isyn_nmda_Tcore_L5_exc : volt
#                                     # Isyn_nmda_Tmatrix_L2_exc : volt
#                                     # Isyn_nmda_Tmatrix_L5_exc : volt
#                                     # Isyn_nmda_T_NRT_exc : volt
                    
#                                     # Isyn_gaba_a = Isyn_gaba_a_L2_L2_inh + Isyn_gaba_a_L4_L4_inh + Isyn_gaba_a_L5_L5_inh + Isyn_gaba_a_L2_L2_column_inh + Isyn_gaba_a_L2_L4_inh + Isyn_gaba_a_L2_L5_inh + Isyn_gaba_a_T_T_inh + Isyn_gaba_a_NRT_T_gaba_a_inh : volt
#                                     # Isyn_gaba_a_L2_L2_inh : volt
#                                     # Isyn_gaba_a_L4_L4_inh : volt
#                                     # Isyn_gaba_a_L5_L5_inh : volt
#                                     # Isyn_gaba_a_L2_L2_column_inh : volt
#                                     # Isyn_gaba_a_L2_L4_inh : volt
#                                     # Isyn_gaba_a_L2_L5_inh : volt
#                                     # Isyn_gaba_a_T_T_inh : volt
#                                     # Isyn_gaba_a_NRT_T_gaba_a_inh : volt
                                    
#                                     # Isyn_gaba_b : volt
                                    
#                                     # Ipoiss = int(t-t_last_poisson < 5*ms)*poiss_str_CX_exc_L5*(V - E_ampa) : volt
#                                     # t_last_poisson : second
#                                     # poiss : 1
                                    
#                                     Iapp : volt # an external current source you can apply if you want (needs 62mV about to start spiking)
#                                   ''')

# N = NeuronGroup(1, eqs, method = 'rk4', threshold = 'V > thresh', 
#                 reset = '''V = Ena
#                             thresh = Ena
#                             lastspike = t''')

# N.V = -70*mV
# N.thresh = thresh_ss_CX_exc
# N.D = 0.001
# N.run_regularly('Ih = g_h_CX_exc * m_h * (V - Eh)')


# P = PoissonInput(N, 'V', 1, 100*Hz, weight = 0.25*mV) #unclear how much noise they really have in the 2010 model, they say mean 1Hz 0.5+25mV, but Down states in the figures show more noise than that. in the 2005 model they clearly have a lot more noise

# mon = StateMonitor(N, variables = True, record = True)

# network = Network(N, mon, P)

# network.run(5000*ms, report = 'text')
# store(name = 'before')
# plot(mon.t/ms, mon.V[0])
# xlabel('time ms')
# ylabel('volt')


# plot_currents(mon, 'L5')
# plot(mon.t/ms, mon.Ih[0], label = 'Ih')
# plot(mon.t/ms, mon.Inap[0], label = 'I')
# plot(mon.t/ms, mon.Iks[0])
# # legend()

