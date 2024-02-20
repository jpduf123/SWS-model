# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 15:20:54 2020

@author: jpduf
"""

# CAVE I changed the Idk activation function to **-3.5 as it seemed to be an error, it was 3.5 in the papers but that produced an activation function that was activated at hyperpolarized states so clearly something wrong
# design the Chebyshev filter stuff for LFP
#CAVE the Iks (slow inacitvating potassium current) only appears in the 2009 paper, and not in 2005 and 2010 paper!

import os
from brian2 import * # no need to import matplotlib or numpy as PyLab is imported with brian2
from scipy import signal 
from functions import * #the functions I wrote (appended to the end of the script too)
import pickle
from datetime import datetime 
from matplotlib import animation
import seaborn as sns
import ffmpeg
import sys
import copy
import elephant
import neo
#from brian2tools import * # need python 3.6
#clear_cache('cython')

# path of the connections to load from if reloading them (if redoing them anyway ignore this, they will be saved in a new folder anyway)
# connection_indices_path = r'G:\Computational results\connection_indices'

# where the results will be stored (depends on computer you run this on)
if os.path.isdir(r'E:\Computational results'):  
    results_dir = r'E:\Computational results' # path for results and connection indices if you redo them for this run
elif os.path.isdir(r'G:\Computational results'):  
    results_dir = r'G:\Computational results' # path for results and connection indices if you redo them for this run
else:
    results_dir = r'C:\computational_Results'
    print('NO COMPUTATIONAL RESULT FILE, CHECK HARD DRIVE IS CONNECTED')
    

# DO YOU WANT TO REDO THE CONNECTION INDICES OR JUST LOAD THEM FROM A PREVIOUSLY SAVED FILE, doing the indices takes a few minutes especially if doing the whole network
redo_connection_indices = True

# all the currents every cell has --> this is also required in the functions module so change that there too!
current_list = ['Inal', 'Ikl', 'Ih', 'It', 'Inap', 'Idk', 'Iks', 'Ikca', 'Isyn_ampa', 'Isyn_nmda', 'Isyn_gaba_a', 'Isyn_gaba_b']
int_current_list = ['Inal', 'Ikl', 'Ih', 'It', 'Inap', 'Idk', 'Iks', 'Ikca'] # intrinsic currents
syn_current_list = ['Isyn_ampa', 'Isyn_nmda', 'Isyn_gaba_a', 'Isyn_gaba_b'] # synaptic currents
    
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
    

#%% BASE PARAMETERS
# saved as a dictionary, the values are then unpacked into global namespace before building the model.

reload_params = False
params_directory = r''
if reload_params:
    os.chdir(params_directory)
    with open('params.data', 'rb') as f:
        p = pickle.load(f)

else:
    # needs to be defined outside dictionary because used in dictionary (is also saved in the dictionary)
    ampa_str = 1 
    nmda_str = 1 
    gaba_a_str = 1 # 0.33 in paper
    
    p = {  
    
    'defaultclock.dt' : 0.1*ms,
    
    'autapses' : False, # I say no here but it isn't specified in the papers.
    
    'Ena' : 30*mV,
    'Ek' : -90*mV,
    'Eh' : -43*mV,
    'Eca' : 140*mV, # CAVE this is not in the papers I just took the one from Bazhenov
        
    
    # ------------------------------------------------------- general layout ----------------------------------------------------------------,
    'grid_size' : 50, 
    
    'CX_layers' : ['L5'],
    
    'Neurongroup_names' : ['CX_exc_L5', 'CX_inh_L5'],
    
    # topographic arrangement of the network: how many cells per distinct point in the network?
    'cells_per_CX_pnt' : 3,
    
    'exc_cells_per_CX_pnt' : 2,
    'inh_cells_per_CX_pnt' : 1,
    
    'IB_cells_proportion' : 0.13,
    
    
    # ------------------------------------------------------------------ recording monitors params --------------------------------------------------------,

    'record_timestep_current' : 1, # in ms, saves a lot of memory
    'record_timestep_voltage' : 1, # in ms, saves a lot of memory
    'record_timestep_Isyn' : 1, # in ms, saves a lot of memory
    
    'cells_to_record_current' : 15, # how many cells to record from for the currents

    # ------------------------------------------------------------------ Connection table --------------------------------------------------------,
    'L5_L5_exc_std_connections' : 5.5, # standard deviation of the Gaussian connection function: 7.5 in the original 2005 tononi paper.
    'L5_L5_inh_std_connections' : 7.5, # standard deviation of the Gaussian connection function: 7.5 in the original 2005 tononi paper.
 
    # ------------------------------ intra-area connections --------------------------------------,
    #intra-laminar excitatory connections,
    'L5_L5_exc_pmax' : 0.1, #changed from 0.1
    'L5_L5_exc_radius' : 12, #changed from 12
    
    #intra-laminar inhibitory connections,
    'L5_L5_inh_pmax' : 0.25, #0.25
    'L5_L5_inh_radius' : 7, # 7
    
    # --------------------------------------------- Synaptic parameters ---------------------------------------------,
    'initial_synaptic_conductance_randomness' : 0.05, # how the synaptic weights are distributed across the model in the initialisation. 1 means std of randmoness = conductance value. 

    'synaptic_conductance_randomness' : 0.05, # how the synaptic weights are updated within each synapse after each event (= randomness of quantal release). 1 means std of randmoness = conductance value.
    
    'tau_p' : 200*ms, # time constant for recovery of short-term depression, is 200ms for all synapses
    
    'poisson_off_threshold' : -55*mV, # turn off poisson input at a certain voltage threshold of the cell? (to not have too much excitatory input)
    
    # --------------------------------------------- Synaptic delays ---------------------------------------------,
    'base_delay_L5_L5_exc' : 2*ms, # base delay that every synapse will have
    'delay_factor_L5_L5_exc' : 0.5*ms, # additional delay based on the topographical distance

    'base_delay_L5_L5_inh' : 2*ms, 
    'delay_factor_L5_L5_inh' : 0.5*ms, 

    # ------------------------------------------------------------- AMPA receptors -----------------------------------------------,
    'E_ampa' : 0*mV,
    
    'tau_1_ampa' : 0.5*ms,
    'tau_2_ampa' : 2.4*ms,
    
    'delta_p_ampa' : 0.075,
    
    'ampa_str' : ampa_str, # with 0.133 and the individual connection strenghts from the paper it gets VERY bursty and short up states
    
    'g_max_ampa_L5_L5_exc' : 0.25*ampa_str, # 0.133 in paper
    
    #------------------------------------------------------------- NMDA receptons -------------------------------------------------,
    # here I model the voltage-dependency of NMDA using the same as in Bazhenov 2016. Tononi uses a dual exponential function for activation (i.e. activation is not instant) but I couldn't find the exact equation
    'E_nmda' : 0*mV,
    
    'tau_1_nmda' : 4*ms,
    'tau_2_nmda' : 40*ms,
    
    'delta_p_nmda' : 0.075,
    
    'nmda_str' : nmda_str, #with 0.133 and the individual connection strengths from the paper it gets VERY bursty and short up states,
    'g_max_nmda_L5_L5_exc' : 0.25*nmda_str, #
    
    # ------------------------------------------------------------------ GABA A receptors -------------------------------------------------,
    'E_gaba_a_CX' : -70*mV,
    
    'tau_1_gaba_a' : 1*ms,
    'tau_2_gaba_a' : 7*ms,
    
    'delta_p_gaba_a' : 0.0375,
    
    'gaba_a_str' : gaba_a_str, 
    'g_max_gaba_a_L5_L5_inh' : 0.35*gaba_a_str, #0.33 in paper
    
    
    # --------------------------------------------- intrinsic currents ---------------------------------------------------
    'conductance_randomness' : 0.01, # 1 means std of randmoness = conductance value.
    'conductance_randomness_gh' : 1, # increase this to have coupled oscillators at more disparate frequencies

    # -------------------------------------------- Cortex excitatory cells ---------------------------------------------,
    # From steady state Vm (approx -85 I think) needs about 62mV of Iapp to spiketrain),
    # conductances in 2010 don't seem to work correctly, when I reduce gkl it at least seems to have some spontaneous firing
    
    'tau_m_CX_exc' : 15*ms,
    
    'thresh_ss_CX_exc' : -51*mvolt,
    'tau_thresh_CX_exc' : 1*ms,
    'tau_spike_CX_exc' : 1.3*ms,
    'time_spike_CX_exc' : 1.4*ms,
    'g_spike_CX_exc' : 1,
    
    'g_kl_CX_exc_L5' : 0.55,
    
    'g_dk_CX_exc_L5' : 2.5, # changed from 0.75
    'dk_time_constant_exc_L5' : 1000*ms,
    
    'g_ks_CX_exc_L5' : 6,
    
    'g_nal_CX_exc_L5' : 0.05,
    
    'g_nap_CX_exc_L5' : 2, # changed from 3.5. 
    
    'g_h_CX_exc_L5' : 3.8,  # changed from 2 before. changed from paper
    
    'D_thresh_CX_exc' : -10*mV, #threshold of the logistic function of D from Idk
    
    'poiss_str_CX_exc_L5' : 0.02,
    
    'poiss_str_2_CX_exc_L5' : 0.2,

    'poiss_rate_CX_exc_L5' : 150*Hz,
    
    'poiss_rate_2_CX_exc_L5' : 0.5*Hz,

    
    # -------------------------------------------- Cortex inhibitory cells ---------------------------------------------
    
    'tau_m_CX_inh' : 7*ms,
    
    'thresh_ss_CX_inh' : -53*mvolt,
    'tau_thresh_CX_inh' : 1*ms,
    'tau_spike_CX_inh' : 0.55*ms,
    'time_spike_CX_inh' : 0.75*ms,
    'g_spike_CX_inh' : 1,
    
    'g_kl_CX_inh_L5' : 0.55,
    
    'g_dk_CX_inh_L5' : 0.75,
    'dk_time_constant_inh_L5' : 1000*ms,
    
    'g_ks_CX_inh_L5' : 6,
    
    'g_nal_CX_inh_L5' : 0.05,
    
    'g_nap_CX_inh_L5' : 2,
    
    'D_thresh_CX_inh' : -10*mV, #threshold of the logistic function of D from Idk
    
    'poiss_str_CX_inh_L5' : 0.007,
    
    'poiss_rate_CX_inh_L5' : 150*Hz,
    
    }

#extract all the params into the global namespace
for key, value in p.items():
    globals()[key] = value

        
        
#%% CELL INDICES

# total cells per layer, need it for network building
cells_per_CX_layer = grid_size**2*cells_per_CX_pnt
exc_cells_per_CX_layer = grid_size**2*exc_cells_per_CX_pnt
inh_cells_per_CX_layer = grid_size**2*inh_cells_per_CX_pnt

# indices of individual neurons within cortical layers
cortex_indx = linspace(1, cells_per_CX_layer, cells_per_CX_layer, dtype = int).reshape(cells_per_CX_pnt, grid_size, grid_size) #make an array map with neuron numbers. This makes n grid_sizexgrid_size arrays with neuron numbers 1-900 and 901-1800 etc.... Used for the topographic mapping 
cortex_exc_indx = linspace(1, exc_cells_per_CX_layer, exc_cells_per_CX_layer, dtype = int).reshape(exc_cells_per_CX_pnt, grid_size, grid_size)
cortex_inh_indx = linspace(1, inh_cells_per_CX_layer, inh_cells_per_CX_layer, dtype = int).reshape(inh_cells_per_CX_pnt, grid_size, grid_size)

# IB_cells_indx = np.linspace(0, (exc_cells_per_CX_layer - 1), int(IB_cells_proportion*exc_cells_per_CX_layer), dtype = int)
IB_cells_indx = np.random.choice(exc_cells_per_CX_layer, size=int(IB_cells_proportion*exc_cells_per_CX_layer), replace=False).astype(int)
plot_grid_indices(IB_cells_indx, 2, grid_size)


#%% CONNECTION NAMES

exc_connections_CX = ['L5_L5_exc']
inh_connections_CX = ['L5_L5_inh']
cortical_connections = exc_connections_CX + inh_connections_CX

connections_list_total = cortical_connections

exc_connections_total = exc_connections_CX
inh_connections_total = inh_connections_CX
inh_connections_gaba_a_total = inh_connections_total.copy()


#%% CONNECTION MATRIX ---------------------------------- make connection indices -----------------------------

redo_connection_indices = True

connection_indices_dict = {}
if redo_connection_indices:    
    for connection in connections_list_total:
        print(f'working on {connection}')
        p_max = globals()[connection + '_pmax']
        radius = globals()[connection + '_radius']
        std_connections = globals()[connection + '_std_connections']
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
            connection_indices_dict[connection + '_inh_con_ind'] = connection_indices_dict[connection + '_con_ind'][:, np.where(connection_indices_dict[connection + '_con_ind'][1] > exc_cells_per_CX_layer)[0]] - 1 # 
            connection_indices_dict[connection + '_inh_con_ind'][1] = connection_indices_dict[connection + '_inh_con_ind'][1] - exc_cells_per_CX_layer # minus the number of excitatory cells to get the indices to range from 0-899
        
else:
    os.chdir(os.path.join(connection_indices_path, f'connections_indices_grid_size_{grid_size}'))
    for filename in [i for i in os.listdir() if 'params.data' not in i]:
        #load the connections in a dict
        connection_indices_dict[filename] = loadtxt(filename, dtype = int, delimiter = ',')
    os.chdir(results_dir)


#%% SYNAPTIC CHANNELS ------------------------------- synaptic channels -----------------------------------------------
# on_pre equations are essentially the same for every synapse type 
# --> reset t_lastspike to 0 and update P (short term depression) with the corresponding delta_P               

# you can't have a run_regularly with a summed variable (in this case the Isyn_post)
# --> I have to create the equations (summing up all the currents together) for each postsynaptic summed variable in the NeuronGroup equations themselves



 # -------------------------- AMPA
t_peak_ampa = (tau_1_ampa*tau_2_ampa/(tau_2_ampa - tau_1_ampa))*np.log(tau_2_ampa/tau_1_ampa)

for con in exc_connections_total:
    globals()['eqs_syn_ampa_' + con] = (f'''
                                            Isyn_ampa_{con}_post = P * g_max_ampa_{con}_rand * ((exp(-(t-t_lastspike)/tau_1_ampa) - exp(-(t-t_lastspike)/tau_2_ampa))/(exp(-t_peak_ampa/tau_1_ampa) - exp(-t_peak_ampa/tau_2_ampa))) * (V_post - E_ampa) : volt (summed)
                                            dP/dt = (1-P)/tau_p : 1 # short-term depression
                                            g_max_ampa_{con}_rand : 1
                                            t_lastspike : second
                                            step_increase : 1
                                            ''')
                                        
    # allow some randomness in the synapses - update g_max every time to allow for randomness in quantal release of vesicles
    globals()['on_pre_eqs_ampa_' + con] = (f'''
                                            t_lastspike = t  
                                            P += -delta_p_ampa*int(P>delta_p_ampa) - P*int(P<delta_p_ampa)    # make sure P doesn't go below 0 as that would inverse the synaptic current (i.e. if P is smaller than delta_p, subtract P not delta_p)
                                            g_max_ampa_{con}_rand = g_max_ampa_{con} + randn() * g_max_ampa_{con} * {synaptic_conductance_randomness} + step_increase
                                            ''')


 # ------------------------ NMDA
t_peak_nmda = (tau_1_nmda*tau_2_nmda/(tau_2_nmda - tau_1_nmda))*np.log(tau_2_nmda/tau_1_nmda)

for con in exc_connections_total:
    globals()['eqs_syn_nmda_' + con] = (f'''
                                            Isyn_nmda_{con}_post = P * g_max_nmda_{con}_rand * V_dependency * ((exp(-(t-t_lastspike)/tau_1_nmda) - exp(-(t-t_lastspike)/tau_2_nmda))/(exp(-t_peak_ampa/tau_1_nmda) - exp(-t_peak_ampa/tau_2_nmda))) * (V_post - E_nmda) : volt (summed)
                                            dP/dt = (1-P)/tau_p  : 1 # short-term depression
                                            g_max_nmda_{con}_rand  : 1
                                            V_dependency = 1/(1 + exp(-(V_post - (-25*mV))/(12.5*mV))) : 1     # voltage-dependent term (unblocking of Mg block)
                                            t_lastspike : second
                                            step_increase : 1
                                            ''')
                        
    globals()['on_pre_eqs_nmda_' + con] = (f'''
                                            t_lastspike = t  
                                            P += -delta_p_nmda*int(P>delta_p_nmda) - P*int(P<delta_p_nmda)    # make sure P doesn't go below 0 as that would inverse the synaptic current     
                                            g_max_nmda_{con}_rand = g_max_nmda_{con} + randn() * g_max_nmda_{con} * {synaptic_conductance_randomness} + step_increase
                                            ''')

        


 # ---------------------- GABA A
t_peak_gaba_a = (tau_1_gaba_a*tau_2_gaba_a/(tau_2_gaba_a - tau_1_gaba_a))*np.log(tau_2_gaba_a/tau_1_gaba_a)

for con in inh_connections_gaba_a_total:
    globals()['eqs_syn_gaba_a_' + con] = (f'''
                                        Isyn_gaba_a_{con}_post = P * g_max_gaba_a_{con}_rand * ((exp(-(t-t_lastspike)/tau_1_gaba_a) - exp(-(t-t_lastspike)/tau_2_gaba_a))/(exp(-t_peak_ampa/tau_1_gaba_a) - exp(-t_peak_ampa/tau_2_gaba_a))) * (V_post - E_gaba_a_CX) : volt (summed)
                                        dP/dt = (1-P)/tau_p  : 1 # short-term depression
                                        g_max_gaba_a_{con}_rand  : 1
                                        t_lastspike : second
                                        step_increase : 1
                                        ''')
    
    globals()['on_pre_eqs_gaba_a_' + con] = (f'''
                                            t_lastspike = t  
                                            P += -delta_p_gaba_a*int(P>delta_p_gaba_a) - P*int(P<delta_p_gaba_a)    # make sure P doesn't go below 0 as that would inverse the synaptic current,so only subtract deltaP if that that doesnt bring P under 0      
                                            g_max_gaba_a_{con}_rand = g_max_gaba_a_{con} + randn() * g_max_gaba_a_{con} * {synaptic_conductance_randomness} + step_increase
                                            ''')



#%% NEURONGROUPS ------------------------------- Cortex excitatory -----------------------------------------------
# conductances are unitless, because no capacitance (no area or volume defined for a cell), rather a membrane time constant is used

# Declaring a variable within the equations (i.e. define its units, like g_h_CX_exc_L5:1)
# --> becomes a variable of the NeuronGroup and you have to explicitly set its value when creating the NeuronGroup (else =0 if not defined in global namespace)
# --> advantage: you can manipulate the variable for each cell individually, e.g. to give some randomness to the model

for layer in CX_layers:
    globals()['eqs_CX_exc_' + layer] = (f'''
                                        dV/dt = (- Inal - Ikl - Iint - Isyn + Iapp)/tau_m_CX_exc - g_spike_CX_exc * (V - Ek)/tau_spike_CX_exc : volt
                                       

                                        # LEAK CURRENTS

                                        Inal = g_nal_CX_exc_{layer} * (V - Ena) : volt
                                        g_nal_CX_exc_{layer} : 1
                                        Ikl = g_kl_CX_exc_{layer} * (V - Ek) : volt
                                        g_kl_CX_exc_{layer} : 1
                                        

                                        # INTRINSIC CURRENTS
                                        
                                        Iint = Iks + Inap + Idk + Ih : volt
                                        
                                        Iks = g_ks_CX_exc_{layer} * m_ks * (V - Ek) : volt                  # slow noninactivating potassium current
                                        dm_ks/dt = (m_ks_ss - m_ks)/tau_m_ks : 1
                                        m_ks_ss = 1/(1 + exp(-(V + 34*mvolt)/(6.5*mvolt))) : 1
                                        tau_m_ks = (8*ms)/(exp(-(V + 55*mvolt)/(30*mvolt)) + exp((V + 55*mvolt)/(30*mvolt))) : second
                                        g_ks_CX_exc_{layer} : 1
                                        
                                        Inap = g_nap_CX_exc_{layer} * m_nap ** 3 * (V - Ena) : volt        #they took the same equations as in Compte 2002. Pers Na current activates rapidly near spike threshold and deactivates very slowly
                                        m_nap = 1/(1 + exp(-(V + 55.7*mvolt)/(7.7*mV))) : 1
                                        g_nap_CX_exc_{layer} : 1
                                        
                                        Idk = g_dk_CX_exc_{layer} * m_dk * (V-Ek) : volt                    #depolarization-activated potassium conductance, replaces Na-dependent K current in Compte --> here the term D combines Ca- and Na dependency by accumulating during depolarization
                                        m_dk = 1/(1 + (0.25*D)**(-3.5)) : 1                                 #instantaneous activation, no time constant ever described
                                        dD/dt = D_influx - D*(1-0.001)/(dk_time_constant_exc_L5) : 1
                                        D_influx = 1/(1 + exp(-(V-D_thresh_CX_exc)/(5*mV)))/ms : Hz
                                        g_dk_CX_exc_{layer} : 1
                                        dk_time_constant_exc_L5 : second
                                        
                                        Ih : volt
                                        dm_h/dt = (m_h_ss - m_h)/tau_m_h : 1
                                        m_h_ss = 1/(1 + exp((V + 75*mV)/(5.5*mV))) : 1
                                        tau_m_h = 1*ms/(exp(-14.59 - 0.086*V/mV) + exp(-1.87 + 0.0701*V/mV)) : second
                                        g_h_CX_exc_L5 : 1
                                        
                                        
                                        # SPIKING CURRENT

                                        dthresh/dt = -(thresh - thresh_ss_CX_exc)/tau_thresh_CX_exc : volt #threshold for spikes
                                        g_spike_CX_exc = int((t - lastspike) < time_spike_CX_exc) : 1
                                        lastspike : second
                                        

                                        # SYNAPTIC CURRENTS
                                        
                                        Isyn = Isyn_ampa + Isyn_nmda + Isyn_gaba_a + Isyn_gaba_b + Ipoiss + Ipoiss_2: volt
                                        
                                        Isyn_ampa = Isyn_ampa_L2_L2_exc + Isyn_ampa_L4_L4_exc + Isyn_ampa_L5_L5_exc + Isyn_ampa_L2_L5_exc + Isyn_ampa_L4_L2_exc + Isyn_ampa_L5_L2_exc + Isyn_ampa_L5_L4_exc : volt
                                        Isyn_ampa_L2_L2_exc: volt
                                        Isyn_ampa_L4_L4_exc : volt
                                        Isyn_ampa_L5_L5_exc : volt
                                        Isyn_ampa_L2_L5_exc : volt
                                        Isyn_ampa_L4_L2_exc : volt
                                        Isyn_ampa_L5_L2_exc : volt
                                        Isyn_ampa_L5_L4_exc : volt 
                                        
                                        Isyn_nmda = Isyn_nmda_L2_L2_exc + Isyn_nmda_L4_L4_exc + Isyn_nmda_L5_L5_exc + Isyn_nmda_L2_L5_exc + Isyn_nmda_L4_L2_exc + Isyn_nmda_L5_L2_exc + Isyn_nmda_L5_L4_exc : volt
                                        Isyn_nmda_L2_L2_exc : volt
                                        Isyn_nmda_L4_L4_exc : volt
                                        Isyn_nmda_L5_L5_exc : volt
                                        Isyn_nmda_L2_L5_exc : volt
                                        Isyn_nmda_L4_L2_exc : volt
                                        Isyn_nmda_L5_L2_exc : volt
                                        Isyn_nmda_L5_L4_exc : volt

                                        Isyn_gaba_a = Isyn_gaba_a_L2_L2_inh + Isyn_gaba_a_L4_L4_inh + Isyn_gaba_a_L5_L5_inh + Isyn_gaba_a_L2_L2_column_inh + Isyn_gaba_a_L2_L4_inh + Isyn_gaba_a_L2_L5_inh : volt
                                        Isyn_gaba_a_L2_L2_inh : volt
                                        Isyn_gaba_a_L4_L4_inh : volt
                                        Isyn_gaba_a_L5_L5_inh : volt
                                        Isyn_gaba_a_L2_L2_column_inh : volt
                                        Isyn_gaba_a_L2_L4_inh : volt
                                        Isyn_gaba_a_L2_L5_inh : volt
                                                                                
                                        Ipoiss = int(t-t_last_poisson < 5*ms)*poiss_str_CX_exc_{layer}*(V - E_ampa) : volt
                                        t_last_poisson : second
                                        poiss : 1
                                        
                                        Ipoiss_2 = int(t-t_last_poisson_2 < 5*ms)*poiss_str_2_CX_exc_{layer}*(V - E_ampa) : volt
                                        t_last_poisson_2 : second
                                        poiss_2 : 1
                                        

                                        # # EXTERNAL CURRENT

                                        Iapp : volt # an external current source you can apply if you want (needs 62mV about to start spiking)
                                        
                                      ''')




#%% -------------------------------------------- Cortex inhibitory -----------------------------------------------

for layer in CX_layers:
    globals()['eqs_CX_inh_' + layer]  = (f'''
                                        dV/dt = (- Inal - Ikl - Iint - Isyn + Iapp)/tau_m_CX_inh - g_spike_CX_inh * (V - Ek)/tau_spike_CX_inh : volt
                                        Inal = g_nal_CX_inh_{layer} * (V - Ena) : volt
                                        g_nal_CX_inh_{layer} : 1
                                        
                                        Ikl = g_kl_CX_inh_{layer} * (V - Ek) : volt
                                        g_kl_CX_inh_{layer} : 1
                                        
                                        Iint = Iks + Inap + Idk : volt
                                        
                                        Iks = g_ks_CX_inh_{layer} * m_ks * (V - Ek) : volt                  # slow noninactivating potassium current
                                        dm_ks/dt = (m_ks_ss - m_ks)/tau_m_ks : 1
                                        m_ks_ss = 1/(1 + exp(-(V + 34*mvolt)/(6.5*mvolt))) : 1
                                        tau_m_ks = (8*ms)/(exp(-(V + 55*mvolt)/(30*mvolt)) + exp((V + 55*mvolt)/(30*mvolt))) : second
                                        g_ks_CX_inh_{layer} : 1
                                        
                                        Inap = g_nap_CX_inh_{layer} * m_nap ** 3 * (V - Ena) : volt        #they took the same equations as in Compte 2002. Pers Na current activates rapidly near spike threshold and deactivates very slowly
                                        m_nap = 1/(1 + exp(-(V + 55.7*mvolt)/(7.7*mV))) : 1
                                        g_nap_CX_inh_{layer} : 1
                                        
                                        Idk = g_dk_CX_inh_{layer} * m_dk * (V-Ek) : volt                    #depolarization-activated potassium conductance, replaces Na-dependent K current in Compte --> here the term D combines Ca- and Na dependency by accumulating during depolarization
                                        m_dk = 1/(1 + (0.25*D)**(-3.5)) : 1
                                        dD/dt = D_influx - D*(1 - 0.001)/(dk_time_constant_inh_L5) : 1
                                        D_influx = 1/(1 + exp(-(V - D_thresh_CX_inh)/(5*mV)))/ms : Hz
                                        g_dk_CX_inh_{layer} : 1
                                        dk_time_constant_inh_L5 : second
                                        
                                        dthresh/dt = -(thresh - thresh_ss_CX_inh)/tau_thresh_CX_inh : volt #threshold for spikes, is set to Ena at each crossing and decays back to ss
                                        g_spike_CX_inh = int((t - lastspike) < time_spike_CX_inh) : 1
                                        lastspike : second
                                        
                                        
                                        Isyn = Isyn_ampa + Isyn_nmda + Isyn_gaba_a + Isyn_gaba_b + Ipoiss: volt
                                        
                                        Isyn_ampa = Isyn_ampa_L2_L2_exc + Isyn_ampa_L4_L4_exc + Isyn_ampa_L5_L5_exc + Isyn_ampa_L2_L5_exc + Isyn_ampa_L4_L2_exc + Isyn_ampa_L5_L2_exc + Isyn_ampa_L5_L4_exc : volt
                                        Isyn_ampa_L2_L2_exc: volt
                                        Isyn_ampa_L4_L4_exc : volt
                                        Isyn_ampa_L5_L5_exc : volt
                                        Isyn_ampa_L2_L5_exc : volt
                                        Isyn_ampa_L4_L2_exc : volt
                                        Isyn_ampa_L5_L2_exc : volt
                                        Isyn_ampa_L5_L4_exc : volt 

                                        Isyn_nmda = Isyn_nmda_L2_L2_exc + Isyn_nmda_L4_L4_exc + Isyn_nmda_L5_L5_exc + Isyn_nmda_L2_L5_exc + Isyn_nmda_L4_L2_exc + Isyn_nmda_L5_L2_exc + Isyn_nmda_L5_L4_exc : volt
                                        Isyn_nmda_L2_L2_exc : volt
                                        Isyn_nmda_L4_L4_exc : volt
                                        Isyn_nmda_L5_L5_exc : volt
                                        Isyn_nmda_L2_L5_exc : volt
                                        Isyn_nmda_L4_L2_exc : volt
                                        Isyn_nmda_L5_L2_exc : volt
                                        Isyn_nmda_L5_L4_exc : volt

                        
                                        Isyn_gaba_a = Isyn_gaba_a_L2_L2_inh + Isyn_gaba_a_L4_L4_inh + Isyn_gaba_a_L5_L5_inh + Isyn_gaba_a_L2_L2_column_inh + Isyn_gaba_a_L2_L4_inh + Isyn_gaba_a_L2_L5_inh : volt
                                        Isyn_gaba_a_L2_L2_inh : volt
                                        Isyn_gaba_a_L4_L4_inh : volt
                                        Isyn_gaba_a_L5_L5_inh : volt
                                        Isyn_gaba_a_L2_L2_column_inh : volt
                                        Isyn_gaba_a_L2_L4_inh : volt
                                        Isyn_gaba_a_L2_L5_inh : volt

                                        
                                        Ipoiss = int(t-t_last_poisson < 5*ms)*poiss_str_CX_inh_{layer}*(V - E_ampa) : volt
                                        t_last_poisson : second
                                        poiss : 1
                                        
                                        Iapp : volt
                                        
                                        ''')
                                        
                                        
                                        
                                        
                                        
#%%
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




#%% --------------------------------------------------------- building the model ----------------------------------------------------------------

cells_to_record_current_indices_exc_CX = np.arange(0, exc_cells_per_CX_layer - 1, np.ceil(exc_cells_per_CX_layer/cells_to_record_current), dtype = int) # indices of cells to record from, spread out evenly across the grid
cells_to_record_current_indices_inh_CX = np.arange(0, inh_cells_per_CX_layer - 1, np.ceil(inh_cells_per_CX_layer/cells_to_record_current), dtype = int) # indices of cells to record from, spread out evenly across the grid

# distance matrix for the synaptic delays
grid_matrix = linspace(1, grid_size**2, grid_size**2, dtype = int).reshape(grid_size, grid_size) - 1
connection_distance_dict = {}
for key, value in connection_indices_dict.items():
    print(key)
    temp_indices = (value)%grid_size**2
    temp_indices_pre = temp_indices[0,:]
    temp_indices_post = temp_indices[1,:]
    temp_pos_pre = np.zeros([2, len(temp_indices_pre)])
    temp_pos_post = np.zeros([2, len(temp_indices_post)])
    for ind, pre in np.ndenumerate(temp_indices_pre):
        temp_pos_pre[:,ind] = np.where(grid_matrix == pre)
    for ind, post in np.ndenumerate(temp_indices_post):
        temp_pos_post[:,ind] = np.where(grid_matrix == post)
    connection_distance_dict[key] = np.linalg.norm(temp_pos_pre - temp_pos_post, axis = 0)
    
    
#%%
currents_CX_exc = ['Inal', 'Ikl', 'Ih', 'Inap', 'Idk', 'Iks', 'Isyn', 'Isyn_ampa', 'Isyn_nmda', 'Isyn_gaba_a', 'Isyn_gaba_b', 'Ipoiss', 'Ipoiss_2']                      
currents_CX_inh = ['Inal', 'Ikl', 'Inap', 'Idk', 'Iks', 'Isyn', 'Isyn_ampa', 'Isyn_nmda', 'Isyn_gaba_a', 'Isyn_gaba_b', 'Ipoiss']                      

currents_syn_ampa = ['Isyn_ampa_L5_L5_exc']
currents_syn_nmda = ['Isyn_nmda_L5_L5_exc']
currents_syn_gaba = ['Isyn_gaba_a_L5_L5_inh']
currents_syn_total = currents_syn_ampa + currents_syn_nmda + currents_syn_gaba

for nrn in ['L5']:
    globals()[f'syn_current_list_{nrn}'] = []
    for curr in currents_syn_total:
        if f'{nrn}_exc' in curr or f'{nrn}_inh' in curr or f'{nrn}_column' in curr:
            globals()[f'syn_current_list_{nrn}'].append(curr)


Neurongroup_list = []
Synapse_list = []
monitor_names = []
voltage_monitor_list = []
current_monitor_list = []
spikes_monitor_list = []


#%% ----------------------------------------------------------- EXC CELLS -----------------------------------------------

CX_exc_L5 = NeuronGroup(exc_cells_per_CX_layer, model = eqs_CX_exc_L5, method = 'rk4', threshold = 'V > thresh',
                reset = '''V = Ena
                            thresh = Ena
                            lastspike = t''',
                events = {'poisson' : 'poiss < poiss_rate_CX_exc_L5*dt * int(V < poisson_off_threshold)',
                          'poisson_2' : 'poiss_2 < poiss_rate_2_CX_exc_L5*dt'})

CX_exc_L5.V = '-75*mV + rand()*10*mV'
CX_exc_L5.D = 0.01
CX_exc_L5.thresh = thresh_ss_CX_exc

CX_exc_L5.g_nal_CX_exc_L5 = g_nal_CX_exc_L5 + (np.random.randn(exc_cells_per_CX_layer) * g_nal_CX_exc_L5 * conductance_randomness)
CX_exc_L5.g_kl_CX_exc_L5 = g_kl_CX_exc_L5 + (np.random.randn(exc_cells_per_CX_layer) * g_kl_CX_exc_L5 * conductance_randomness)
CX_exc_L5.g_ks_CX_exc_L5 = g_ks_CX_exc_L5 + (np.random.randn(exc_cells_per_CX_layer) * g_ks_CX_exc_L5 * conductance_randomness)
CX_exc_L5.g_dk_CX_exc_L5 = g_dk_CX_exc_L5 + (np.random.randn(exc_cells_per_CX_layer) * g_dk_CX_exc_L5 * conductance_randomness)
CX_exc_L5.dk_time_constant_exc_L5 = dk_time_constant_exc_L5
CX_exc_L5.g_nap_CX_exc_L5 = g_nap_CX_exc_L5 + (np.random.randn(exc_cells_per_CX_layer) * g_nap_CX_exc_L5 * conductance_randomness)

# -------------- intrinsically bursting cells:
# indices selected randomly:
# IB_cells_indx = np.random.choice(exc_cells_per_CX_layer, size=int(IB_cells_proportion*exc_cells_per_CX_layer), replace=False).astype(int)
# indices selected linearly:
# IB_cells_indx = np.linspace(0, (exc_cells_per_CX_layer - 1), int(IB_cells_proportion*exc_cells_per_CX_layer), dtype = int)

g_h_CX_exc_L5_array = g_h_CX_exc_L5 + (np.random.rand(len(IB_cells_indx)) * conductance_randomness_gh) - conductance_randomness_gh/2
g_nap_CX_exc_L5_array = g_nap_CX_exc_L5 + (np.random.randn(len(IB_cells_indx)) * g_nap_CX_exc_L5 * conductance_randomness)

#need to use for loop because subgroup creation can only use contiguous integers
for n_ind, n in enumerate(list(IB_cells_indx)):
    CX_exc_L5[n].g_h_CX_exc_L5 = g_h_CX_exc_L5_array[n_ind]
    CX_exc_L5[n].run_regularly('Ih = g_h_CX_exc_L5 * m_h * (V - Eh)')
    CX_exc_L5[n].g_nap_CX_exc_L5 = g_nap_CX_exc_L5_array[n_ind]

CX_exc_L5.run_regularly('poiss = rand()')
CX_exc_L5.run_on_event('poisson', 't_last_poisson = t')
CX_exc_L5.run_regularly('poiss_2 = rand()')
CX_exc_L5.run_on_event('poisson_2', 't_last_poisson_2 = t')
# Neurongroup_list.append(CX_exc_L5)
# P_exc_L5 = PoissonInput(CX_exc_L5, 'V', 15, 1*Hz, weight = 1*mV)

spikes_exc_L5 = SpikeMonitor(CX_exc_L5)
# spikes_monitor_list.append(spikes_exc_L5)
voltage_monitor_exc_L5 = StateMonitor(CX_exc_L5, variables = 'V', record = True, dt = record_timestep_voltage*ms, dtype_to_record = np.float32)
# voltage_monitor_list.append(voltage_monitor_exc_L5)
current_monitor_exc_L5 = StateMonitor(CX_exc_L5, variables = currents_CX_exc + syn_current_list_L5, record = cells_to_record_current_indices_exc_CX, dt = record_timestep_current*ms, dtype_to_record = np.float32)
# current_monitor_list.append(current_monitor_exc_L5)
Isyn_monitor_exc_L5 = StateMonitor(CX_exc_L5, variables = 'Isyn', record = True, dt = record_timestep_Isyn*ms, dtype_to_record = np.float32)


# plost g_h strength spatially
g_h_array = CX_exc_L5.g_h_CX_exc_L5[:].reshape(2, grid_size, grid_size)
fig, ax = plt.subplots()
ax.imshow(np.sum(g_h_array, axis = 0), cmap = 'jet', vmin = 3, vmax =5)




#%% ----------------------------------------------------------- INH CELLS -----------------------------------------------
CX_inh_L5 = NeuronGroup(inh_cells_per_CX_layer, model = eqs_CX_inh_L5, method = 'rk4', threshold = 'V > thresh',
                reset = '''V = Ena
                            thresh = Ena
                            lastspike = t''',
                events = {'poisson' : 'poiss < poiss_rate_CX_inh_L5*dt * int(V < poisson_off_threshold)'})

CX_inh_L5.V = '-75*mV + rand()*5*mV'
CX_inh_L5.D = 0.01
CX_inh_L5.thresh = thresh_ss_CX_inh

CX_inh_L5.g_nal_CX_inh_L5 = g_nal_CX_inh_L5 + (np.random.randn(inh_cells_per_CX_layer) * g_nal_CX_inh_L5 * conductance_randomness)
CX_inh_L5.g_kl_CX_inh_L5 = g_kl_CX_inh_L5 + (np.random.randn(inh_cells_per_CX_layer) * g_kl_CX_inh_L5 * conductance_randomness)
CX_inh_L5.g_ks_CX_inh_L5 = g_ks_CX_inh_L5 + (np.random.randn(inh_cells_per_CX_layer) * g_ks_CX_inh_L5 * conductance_randomness)
CX_inh_L5.g_dk_CX_inh_L5 = g_dk_CX_inh_L5 + (np.random.randn(inh_cells_per_CX_layer) * g_dk_CX_inh_L5 * conductance_randomness)
CX_inh_L5.dk_time_constant_inh_L5 = dk_time_constant_inh_L5
CX_inh_L5.g_nap_CX_inh_L5 = g_nap_CX_inh_L5 + (np.random.randn(inh_cells_per_CX_layer) * g_nap_CX_inh_L5 * conductance_randomness)

CX_inh_L5.run_regularly('poiss = rand()')
CX_inh_L5.run_on_event('poisson', 't_last_poisson = t')
# Neurongroup_list.append(CX_inh_L5)
# P_inh_L5 = PoissonInput(CX_inh_L5, 'V', 2, 1*Hz, weight = 1*mV)

spikes_inh_L5 = SpikeMonitor(CX_inh_L5)
# spikes_monitor_list.append(spikes_inh_L5)
voltage_monitor_inh_L5 = StateMonitor(CX_inh_L5, variables = 'V', record = True, dt = record_timestep_voltage*ms, dtype_to_record = np.float32)
# voltage_monitor_list.append(voltage_monitor_inh_L5)
current_monitor_inh_L5 = StateMonitor(CX_inh_L5, variables = currents_CX_inh + syn_current_list_L5, record = cells_to_record_current_indices_inh_CX, dt = record_timestep_current*ms, dtype_to_record = np.float32)
# current_monitor_list.append(current_monitor_inh_L5)
Isyn_monitor_inh_L5 = StateMonitor(CX_inh_L5, variables = 'Isyn', record = True, dt = record_timestep_Isyn*ms, dtype_to_record = np.float32)




#%% ----------------------------------------------------------- SYNAPSES -----------------------------------------------

L5_L5_exc_exc_con_ampa = Synapses(CX_exc_L5, CX_exc_L5, model = eqs_syn_ampa_L5_L5_exc, on_pre = on_pre_eqs_ampa_L5_L5_exc, method = 'rk4')
L5_L5_exc_exc_con_ampa.connect(i = connection_indices_dict['L5_L5_exc_exc_con_ind'][0], j = connection_indices_dict['L5_L5_exc_exc_con_ind'][1])
L5_L5_exc_exc_con_ampa.delay = connection_distance_dict['L5_L5_exc_exc_con_ind']*delay_factor_L5_L5_exc + base_delay_L5_L5_exc
# L5_L5_exc_exc_con_ampa.delay = 1*ms + clip(0.25*randn(),0,1)*ms
L5_L5_exc_exc_con_ampa.g_max_ampa_L5_L5_exc_rand = g_max_ampa_L5_L5_exc + np.random.randn(connection_indices_dict['L5_L5_exc_exc_con_ind'].shape[1]) * g_max_ampa_L5_L5_exc * initial_synaptic_conductance_randomness
Synapse_list.append(L5_L5_exc_exc_con_ampa)

L5_L5_exc_exc_con_nmda = Synapses(CX_exc_L5, CX_exc_L5, model = eqs_syn_nmda_L5_L5_exc, on_pre = on_pre_eqs_nmda_L5_L5_exc, method = 'rk4')
L5_L5_exc_exc_con_nmda.connect(i = connection_indices_dict['L5_L5_exc_exc_con_ind'][0], j = connection_indices_dict['L5_L5_exc_exc_con_ind'][1])
L5_L5_exc_exc_con_nmda.delay = connection_distance_dict['L5_L5_exc_exc_con_ind']*delay_factor_L5_L5_exc + base_delay_L5_L5_exc
# L5_L5_exc_exc_con_nmda.delay = 1*ms + clip(0.25*randn(),0,1)*ms
L5_L5_exc_exc_con_nmda.g_max_nmda_L5_L5_exc_rand = g_max_nmda_L5_L5_exc + np.random.randn(connection_indices_dict['L5_L5_exc_exc_con_ind'].shape[1]) * g_max_nmda_L5_L5_exc * initial_synaptic_conductance_randomness
Synapse_list.append(L5_L5_exc_exc_con_nmda)

L5_L5_inh_exc_con_gaba_a = Synapses(CX_inh_L5, CX_exc_L5, model = eqs_syn_gaba_a_L5_L5_inh, on_pre = on_pre_eqs_gaba_a_L5_L5_inh, method = 'rk4')
L5_L5_inh_exc_con_gaba_a.connect(i = connection_indices_dict['L5_L5_inh_exc_con_ind'][0], j = connection_indices_dict['L5_L5_inh_exc_con_ind'][1])
L5_L5_inh_exc_con_gaba_a.delay = connection_distance_dict['L5_L5_inh_exc_con_ind']*delay_factor_L5_L5_inh + base_delay_L5_L5_inh
# L5_L5_inh_exc_con_gaba_a.delay = 0.75*ms + clip(0.1*randn(),0,1)*ms
L5_L5_inh_exc_con_gaba_a.g_max_gaba_a_L5_L5_inh_rand = g_max_gaba_a_L5_L5_inh + np.random.randn(connection_indices_dict['L5_L5_inh_exc_con_ind'].shape[1]) * g_max_gaba_a_L5_L5_inh * initial_synaptic_conductance_randomness
Synapse_list.append(L5_L5_inh_exc_con_gaba_a)

L5_L5_inh_inh_con_gaba_a = Synapses(CX_inh_L5, CX_inh_L5, model = eqs_syn_gaba_a_L5_L5_inh, on_pre = on_pre_eqs_gaba_a_L5_L5_inh, method = 'rk4')
L5_L5_inh_inh_con_gaba_a.connect(i = connection_indices_dict['L5_L5_inh_inh_con_ind'][0], j = connection_indices_dict['L5_L5_inh_inh_con_ind'][1])
L5_L5_inh_inh_con_gaba_a.delay = connection_distance_dict['L5_L5_inh_inh_con_ind']*delay_factor_L5_L5_inh + base_delay_L5_L5_inh
# L5_L5_inh_inh_con_gaba_a.delay = 0.75*ms + clip(0.1*randn(),0,1)*ms
L5_L5_inh_inh_con_gaba_a.g_max_gaba_a_L5_L5_inh_rand = g_max_gaba_a_L5_L5_inh + np.random.randn(connection_indices_dict['L5_L5_inh_inh_con_ind'].shape[1]) * g_max_gaba_a_L5_L5_inh * initial_synaptic_conductance_randomness
Synapse_list.append(L5_L5_inh_inh_con_gaba_a)

L5_L5_exc_inh_con_ampa = Synapses(CX_exc_L5, CX_inh_L5, model = eqs_syn_ampa_L5_L5_exc, on_pre = on_pre_eqs_ampa_L5_L5_exc, method = 'rk4')
L5_L5_exc_inh_con_ampa.connect(i = connection_indices_dict['L5_L5_exc_inh_con_ind'][0], j = connection_indices_dict['L5_L5_exc_inh_con_ind'][1])
L5_L5_exc_inh_con_ampa.delay = connection_distance_dict['L5_L5_exc_inh_con_ind']*delay_factor_L5_L5_exc + base_delay_L5_L5_exc
# L5_L5_exc_inh_con_ampa.delay = 1*ms + clip(0.25*randn(),0,1)*ms
L5_L5_exc_inh_con_ampa.g_max_ampa_L5_L5_exc_rand = g_max_ampa_L5_L5_exc + np.random.randn(connection_indices_dict['L5_L5_exc_inh_con_ind'].shape[1]) * g_max_ampa_L5_L5_exc * initial_synaptic_conductance_randomness
Synapse_list.append(L5_L5_exc_inh_con_ampa)

L5_L5_exc_inh_con_nmda = Synapses(CX_exc_L5, CX_inh_L5, model = eqs_syn_nmda_L5_L5_exc, on_pre = on_pre_eqs_nmda_L5_L5_exc, method = 'rk4')
L5_L5_exc_inh_con_nmda.connect(i = connection_indices_dict['L5_L5_exc_inh_con_ind'][0], j = connection_indices_dict['L5_L5_exc_inh_con_ind'][1])
L5_L5_exc_inh_con_nmda.delay = connection_distance_dict['L5_L5_exc_inh_con_ind']*delay_factor_L5_L5_exc + base_delay_L5_L5_exc
# L5_L5_exc_inh_con_nmda.delay = 1*ms + clip(0.25*randn(),0,1)*ms
L5_L5_exc_inh_con_nmda.g_max_nmda_L5_L5_exc_rand = g_max_nmda_L5_L5_exc + np.random.randn(connection_indices_dict['L5_L5_exc_inh_con_ind'].shape[1]) * g_max_nmda_L5_L5_exc * initial_synaptic_conductance_randomness
Synapse_list.append(L5_L5_exc_inh_con_nmda)

print('built layer 5')


#%% --------------------------------------------------- CREATE NETWORK -----------------------------------------------------------------------
    
network = Network(CX_exc_L5, CX_inh_L5,
                  Synapse_list, 
                  spikes_exc_L5, 
                  spikes_inh_L5,
                  voltage_monitor_exc_L5, 
                  voltage_monitor_inh_L5,
                  current_monitor_exc_L5,
                  current_monitor_inh_L5,
                  Isyn_monitor_exc_L5,
                  Isyn_monitor_inh_L5)

print('model built')

monitor_names = ['spikes_exc_L5', 
                'spikes_inh_L5',
                'voltage_monitor_exc_L5', 
                'voltage_monitor_inh_L5',
                'current_monitor_exc_L5', 
                'current_monitor_inh_L5',
                'Isyn_monitor_exc_L5',
                'Isyn_monitor_inh_L5',]
                


# @network_operation(dt=5*second)
# def change_parameter():
#     for n_ind, n in enumerate(list(IB_cells_indx)):
#         CX_exc_L5[n].g_h_CX_exc_L5 += 0.1
# parameter_monitor_1 = StateMonitor(CX_exc_L5, variables = 'g_h_CX_exc_L5', dt = 1*second, record = True)
# network.add(parameter_monitor_1)
# monitor_names.append('parameter_monitor_1')
# network.add(change_parameter)

# # --------------------------------------- decrease excitatory currents across the network
# L5_L5_exc_exc_con_ampa.run_regularly('step_increase += 0.01', dt = 10*second)
# L5_L5_exc_exc_con_ampa.step_increase = -0.01

# L5_L5_exc_exc_con_nmda.run_regularly('step_increase += 0.01', dt = 10*second)
# L5_L5_exc_exc_con_nmda.step_increase = -0.01

# L5_L5_exc_inh_con_ampa.run_regularly('step_increase += 0.01', dt = 10*second)
# L5_L5_exc_inh_con_ampa.step_increase = -0.01
# # parameter_monitor_1 = StateMonitor(L5_L5_exc_inh_con_ampa, variables = 'g_max_ampa_L5_L5_exc_rand', record = np.linspace(0,5000, 50, dtype = int), dt = 0.01*second)
# # network.add(parameter_monitor_1)
# # monitor_names.append('parameter_monitor_1')

# L5_L5_exc_inh_con_nmda.run_regularly('step_increase += 0.01', dt = 10*second)
# L5_L5_exc_inh_con_nmda.step_increase = -0.01
# # parameter_monitor_2 = StateMonitor(L5_L5_exc_inh_con_nmda, variables = 'g_max_nmda_L5_L5_exc_rand', record = np.linspace(0,5000, 50, dtype = int), dt = 0.01*second)
# # network.add(parameter_monitor_2)
# monitor_names.append('parameter_monitor_2')


parameter_monitor_1 = StateMonitor(L5_L5_exc_exc_con_ampa, variables = 'g_max_ampa_L5_L5_exc_rand', record = cells_to_record_current_indices_exc_CX, dt = 0.01*second)
network.add(parameter_monitor_1)
monitor_names.append('parameter_monitor_1')

parameter_monitor_2 = StateMonitor(L5_L5_exc_exc_con_nmda, variables = 'g_max_nmda_L5_L5_exc_rand', record = cells_to_record_current_indices_exc_CX, dt = 0.01*second)
network.add(parameter_monitor_2)
monitor_names.append('parameter_monitor_2')


# ---------------------------------------------------------------- SAVE params and connections indices
curr_time = datetime.today().strftime('%Y-%m-%d %H%M')
curr_results = os.path.join(results_dir, curr_time)
if ~os.path.isdir(curr_results):
    os.mkdir(curr_results)
os.chdir(curr_results)
np.savetxt('cells_to_record_current_indices_exc_CX.csv', cells_to_record_current_indices_exc_CX, delimiter = ',')
np.savetxt('cells_to_record_current_indices_inh_CX.csv', cells_to_record_current_indices_inh_CX, delimiter = ',')
with open('params.data', 'wb') as f:
    pickle.dump(p, f)
    with open("parameters.txt", 'w') as f: 
        for key, value in p.items(): 
            f.write(f'{key}:{value}\n')
    os.mkdir('connection_indices')
    os.chdir('connection_indices')
    for key, value in connection_indices_dict.items():
        savetxt(key, value, delimiter = ',')
print('saved parameters')


# g_h_CX_exc_L5_array -= 1.5
# #need to use for loop because subgroup creation can only use contiguous integers
# for n_ind, n in enumerate(list(IB_cells_indx)):
#     CX_exc_L5[n].g_h_CX_exc_L5 = g_h_CX_exc_L5_array[n_ind]

# CX_exc_L5.g_dk_CX_exc_L5 -= 1.5

# print('running model')
# network.run(15*second, report = 'text', profile = True)

# -------------------------------------------------- decrease excitatory currents at specific parts of the network
decrease_matrix_width = 5
decrease_matrix_step_size = int(grid_size/decrease_matrix_width)
decrease_matrix = -np.random.random([decrease_matrix_width,decrease_matrix_width])/4
os.chdir(curr_results)
np.savetxt('decrease_matrix.csv', decrease_matrix, delimiter = ',')
fig, ax = plt.subplots()
ax.imshow(decrease_matrix, vmin = -0.5, vmax = 0.5)

network.run(30*second, report = 'text', profile = True)

for run in range(1):

    # # find synaptic connection strengths within each area and decrease them accordingly
    exc_exc_ampa_step = copy.deepcopy(np.asarray(L5_L5_exc_exc_con_ampa.g_max_ampa_L5_L5_exc_rand))
    exc_exc_nmda_step = copy.deepcopy(np.asarray(L5_L5_exc_exc_con_nmda.g_max_nmda_L5_L5_exc_rand))
    exc_inh_ampa_step = copy.deepcopy(np.asarray(L5_L5_exc_inh_con_ampa.g_max_ampa_L5_L5_exc_rand))
    exc_inh_nmda_step = copy.deepcopy(np.asarray(L5_L5_exc_inh_con_nmda.g_max_nmda_L5_L5_exc_rand))
    
    for width_1 in range(decrease_matrix_width):
        for width_2 in range(decrease_matrix_width):
            decrease_factor = decrease_matrix[width_1, width_2]
            #cell indices within that subgrid
            cell_idx = grid_matrix[width_1*decrease_matrix_step_size:(width_1 + 1)*decrease_matrix_step_size, width_2*decrease_matrix_step_size:(width_2 + 1)*decrease_matrix_step_size].flatten()
            
            # exc_exc connections within that subgrid
            post_synaptic_cells = connection_indices_dict['L5_L5_exc_exc_con_ind'][1]
            connection_idx = np.sort(np.concatenate((np.where(np.in1d(post_synaptic_cells, cell_idx))[0], np.where(np.in1d(post_synaptic_cells, cell_idx+grid_size**2) == True)[0])))
            exc_exc_ampa_step[connection_idx] = exc_exc_ampa_step[connection_idx]*decrease_factor
            exc_exc_nmda_step[connection_idx] = exc_exc_nmda_step[connection_idx]*decrease_factor
    
            # exc_inh connections within that subgrid
            post_synaptic_cells = connection_indices_dict['L5_L5_exc_inh_con_ind'][1]
            connection_idx = (np.where(np.in1d(post_synaptic_cells, cell_idx))[0])
            exc_inh_ampa_step[connection_idx] = exc_inh_ampa_step[connection_idx]*decrease_factor
            exc_inh_nmda_step[connection_idx] = exc_inh_nmda_step[connection_idx]*decrease_factor
    
    L5_L5_exc_exc_con_ampa.step_increase += exc_exc_ampa_step
    L5_L5_exc_exc_con_nmda.step_increase += exc_exc_nmda_step
    L5_L5_exc_inh_con_ampa.step_increase += exc_inh_ampa_step
    L5_L5_exc_inh_con_nmda.step_increase += exc_inh_nmda_step
    
#     # g_h_CX_exc_L5_array += 1
#     # #need to use for loop because subgroup creation can only use contiguous integers
#     # for n_ind, n in enumerate(list(IB_cells_indx)):
#     #     CX_exc_L5[n].g_h_CX_exc_L5 = g_h_CX_exc_L5_array[n_ind]
        
#     CX_exc_L5.g_dk_CX_exc_L5 += 1.5
#     print('running model')

    network.run(30*second, report = 'text', profile = True)


# #plot new synaptic weights in each subgrid
# syn_avg_grid = np.zeros([decrease_matrix_width,decrease_matrix_width])
# for width_1 in range(decrease_matrix_width):
#     for width_2 in range(decrease_matrix_width):
#         cell_idx = grid_matrix[width_1*decrease_matrix_step_size:(width_1 + 1)*decrease_matrix_step_size, width_2*decrease_matrix_step_size:(width_2 + 1)*decrease_matrix_step_size].flatten()
        
#         post_synaptic_cells = connection_indices_dict['L5_L5_exc_exc_con_ind'][1]
#         connection_idx = np.sort(np.concatenate((np.where(np.in1d(post_synaptic_cells, cell_idx))[0], np.where(np.in1d(post_synaptic_cells, cell_idx+grid_size**2) == True)[0])))
#         syn_avg_grid[width_1, width_2] = np.mean(L5_L5_exc_exc_con_ampa.g_max_ampa_L5_L5_exc_rand[connection_idx])

# # CX_exc_L5.g_h_CX_exc_L5
# # CX_exc_L5.g_nap_CX_exc_L5 -= 0.5
# # CX_exc_L5.g_dk_CX_exc_L5 += 0.25
# for n_ind, n in enumerate(list(IB_cells_indx)):
#     # CX_exc_L5[n].g_h_CX_exc_L5 -= 0.2
#     print(CX_exc_L5[n].g_h_CX_exc_L5)

# L5_L5_exc_exc_con_ampa.step_increase = 0.03
# L5_L5_exc_exc_con_nmda.step_increase = 0.03
# L5_L5_exc_inh_con_ampa.step_increase = 0.03
# L5_L5_exc_inh_con_nmda.step_increase = 0.03

# network.run(10*second, report = 'text', profile = True)


#% ------------------------------------------------------ dumping on disk ---------------------------------------------------
os.chdir(curr_results)

#you can save the values as float 32 (maybe even 16) to save space. 
for monitor in monitor_names: 
    print(f'saving {monitor}')
    try:
        data = globals()[monitor].get_states()
        # # convert the voltage and current data into float16 (not time data) --> 4 decimal points
        # if 'spikes' not in monitor: # don't bother with reformatting spikes monitor as they take up very little space
        #     for key, values in data.items():
        #         print(key)
        #         if key == 't' or key == 'N':
        #             print('skipping t')
        #             continue
        #         elif key != 't' or key != 'N':
        #             data[key] = data[key].astype('float32')
        #dump on disk
        with open(f'{monitor}.data', 'wb') as f:
            pickle.dump(data, f)
        del data
    except KeyError:
        print(f'couldnt find {monitor}')
        continue

# network.store('after 5sec', filename = 'network_state_post_5sec')


#%% ------------------------------------------------------ dumping on disk and resetting the monitors ---------------------------------------------------
# #remove monitors from network and delete the monitors. When using network.remove() you can't use a string of the name, it will bug out and restart the kernel!
# for monitor in monitor_names:
#     try:
#         network.remove(globals()[monitor])
#         del globals()[monitor]
#     except NameError:
#         print(f'couldnt find {monitor}')
#         continue


# # redo the monitors and reintegrate them in the network
# for name in Neurongroup_names:
#     globals()[f'spikes_{name}'] = SpikeMonitor(globals()[name])
#     network.add(globals()[f'spikes_{name}'])
#     globals()[f'voltage_monitor_{name}'] = StateMonitor(globals()[name], variables = 'V', record = True)
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


#%% plot results
# make LFP
Isyn_total = np.sum(Isyn_monitor_exc_L5.Isyn, axis = 0)
LFP_total = -elephant.signal_processing.butter(neo.core.AnalogSignal(Isyn_total, units = 'V', sampling_rate = 1000/record_timestep_Isyn*pq.Hz), lowpass_frequency = 6, highpass_frequency = None).as_array()
fig, ax = plt.subplots()
ax.plot(np.linspace(0, Isyn_monitor_exc_L5.t[-1]/second, len(LFP_total)), LFP_total)

fig, ax = plt.subplots(figsize = (12,10))
for cell_ind, cell in enumerate([101,150,1300,901,902]):
    ax.plot(voltage_monitor_exc_L5.t/ms, voltage_monitor_exc_L5.V[cell,:].T + cell_ind*140*mV, color = 'k')
ax.set_xlim([5000,1000])
# plt.savefig('voltage traces examples.pdf', format = 'pdf', dpi = 1000)
# plt.savefig('voltage traces examples.jpg', format = 'jpg', dpi = 1000)

# grid_indices = np.arange(0, 28**2, 5, dtype = int)
# plot_grid_indices(grid_indices, 1, 28)

# plot(current_monitor_exc_L5.Isyn_ampa.T[:,2])
# plot(current_monitor_exc_L5.Igaba_a.T[:,2])


# diagonal_skip = 2
# diagonal_indces = np.array([i*(grid_size*diagonal_skip+diagonal_skip) for i in range(int(grid_size/diagonal_skip))])
# make_color_traces('voltage_monitor_exc_L5', diagonal_indces, x_offset = 0, y_offset = 0.1)

# fig, ax = plt.subplots()
# ax.plot(np.mean(parameter_monitor_1.g_max_ampa_L5_L5_exc_rand, axis = 0))
# ampa_conduct_matrix = np.zeros([decrease_matrix_width,decrease_matrix_width])
# for width_1 in range(decrease_matrix_width):
#     for width_2 in range(decrease_matrix_width):
#         cell_idx = grid_matrix[width_1*decrease_matrix_step_size:(width_1 + 1)*decrease_matrix_step_size, width_2*decrease_matrix_step_size:(width_2 + 1)*decrease_matrix_step_size].flatten()
#         ampa_conduct_matrix[width_1, width_2] = np.mean(parameter_monitor_1.g_max_ampa_L5_L5_exc_rand[cell_idx])

# plot_currents(current_monitor_exc_L5, 'exc_L5', neuron_to_plot = 2)
# # # # plot(voltage_monitor_T_exc.t/ms, voltage_monitor_exc_L4.V[3])
# # # plot(voltage_monitor_T_exc.t/ms, voltage_monitor_exc_L5.V[11])


stacked_traces = voltage_monitor_exc_L5.V/mV
stacked_traces = stacked_traces.reshape(exc_cells_per_CX_pnt,grid_size**2,-1)[1,:,:]
stacked_traces = stacked_traces[:,4000:9000]
fig, ax = plt.subplots(figsize = (10,2))
plot = ax.pcolormesh(stacked_traces, cmap = 'jet', vmin = -85, vmax = -50)
# ax.set_xlabel('time (s)', size = 16)
# ax.set_ylabel('neuron #', size = 16)
ax.set_xticks([])
ax.set_yticks([])
# fig.colorbar(plot, ax = ax, label = 'membrane voltage')
plt.tight_layout()
# plt.savefig('voltage exc traces L5 heatmap.pdf', dpi = 1000, format = 'pdf')
plt.savefig('voltage exc traces L5 heatmap HIGH.jpg', dpi = 1000, format = 'jpg')



fig, ax = plt.subplots(figsize = (15,3))
ax.plot(spikes_exc_L5.t/ms, spikes_exc_L5.i, '.k', markersize = 1)
ax.set_ylim([0, grid_size**2])
# ax.set_xlim([20000,30000])
# plt.savefig('spike raster.pdf', format = 'pdf', dpi = 1000)

# fig, ax = plt.subplots(figsize = (15,10))
# ax.plot(spikes_inh_L5.t/ms, spikes_inh_L5.i, '.k', markersize = 1)
# ax.set_ylim([0, grid_size**2])

# # # xlabel('Time (ms)')
# # # ylabel('Neuron index')

# # # # # plot(spikes_inh_L2.t/ms, spikes_inh_L2.i, '.k')
# # # # # xlabel('Time (ms)')
# # # # # ylabel('Neuron index')

# voltage_traces_heatmap(voltage_monitor_exc_L5, 'V', last_neuron_index = 1600)

# # # cycle_through_traces(voltage_monitor_exc_L2, 'V')

# # # plot(voltage_monitor_exc.t/ms, voltage_monitor_exc.V[1])

# from functions import *
# make_movie('voltage_monitor_exc_L4', fps = 120, exc_cells_per_CX_pnt = 2, grid_size = 30, vmax = -0)

# fig = plt.figure()
# for sweepNumber in data.sweepList:
#     data.setSweep(sweepNumber)
#     dataX = data.sweepX[:] + .025 * sweepNumber # x offset
#     dataY = data.sweepY[:] + 15 * sweepNumber # y offset
#     plt.plot(dataX, dataY, color=colors[sweepNumber], alpha=.5)
# plt.gca().axis('off')





# plotting animation heatmap of voltage traces together with sliding voltage trace, doesn't work very well but haven't tried saving it as movie yet.
# data_list = []
# for j in range(300):
#     data = voltage_monitor_exc_L2.V[:, j*166+100].reshape(exc_cells_per_CX_pnt, grid_size, grid_size)[1,:,:].astype('float32')
#     data_list.append(data) 
    
# # fig, axes = plt.subplots(2,1)
# def init_1():
#     sns.heatmap(data_list[0], cbar=False, cmap = 'RdYlBu_r', vmax = -0.03, vmin = -0.085, ax = axes[0])

# def animate_1(i):
#     # data = voltage_monitor_exc_L4.V[:, i*100].reshape(exc_cells_per_CX_pnt, grid_size, grid_size)[1,:,:]
#     data = data_list[i]
#     return sns.heatmap(data, cbar=False, cmap = 'RdYlBu_r', vmax = -0.03, vmin = -0.085, ax = axes[0])

# plt.rcParams['animation.ffmpeg_path'] = r'C:\FFmpeg\bin\ffmpeg.exe'
# FFwriter = animation.FFMpegWriter(fps = 60)
# anim_heatmap = animation.FuncAnimation(fig, animate_1, init_func=init_1, frames = 300, repeat = False)
# anim_heatmap.save('L2_1.mp4', writer=FFwriter)


# # fig = plt.figure()
# fig, axes = plt.subplots(2,1)
# axes[1].set_xlim([-2000, 2000])
# axes[1].set_ylim([-0.09, 0.05])
# # ax = plt.axes(xlim=(-2000, 2000), ylim=(-0.09, 0.05))
# line, = axes[1].plot([], [], lw=3)

# def init_2():
#     line.set_data([], [])
#     return line,

# def animate_2(i):
#     x = np.linspace(-2000, 2000, 4001)
#     y = voltage_monitor_exc_L2.V[100][i*166-2000 + 2000:i*166+2001 + 2000]
#     line.set_data(x, y)
#     return line,

# anim_line = animation.FuncAnimation(fig, animate_2, init_func=init_2, frames=100, repeat = False)
