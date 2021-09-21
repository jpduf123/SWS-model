# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 15:16:22 2021

@author: jpduf
"""
#%%
current_list = ['Inal', 'Ikl', 'Ih', 'It', 'Inap', 'Idk', 'Iks', 'Ikca', 'Isyn_ampa', 'Isyn_nmda', 'Isyn_gaba_a', 'Isyn_gaba_b']
int_current_list = ['Inal', 'Ikl', 'Ih', 'It', 'Inap', 'Idk', 'Iks', 'Ikca']
syn_current_list = ['Isyn_ampa', 'Isyn_nmda', 'Isyn_gaba_a', 'Isyn_gaba_b']

import numpy as np
from brian2 import *
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

class Struct:
    """
    just pass in keys and values to build the struct like Matlab
    """
    def __init__(self, **entries):
        self.__dict__.update(entries)
        


def make_connection_indices(source_cells_indx, target_cells_indx, p_max, radius, std_connections, grid_size, layers_for_progress_report = None, autapses = False):
    """
    function to give the connection matrix 2xN_con (first row = presyn indices, second row = postsyn indices) given the parameters in the paper
    I can't quite figure out exactly what the connectivity is in Tononi paper 
    i)strength = adjustment of synaptic conductance I guess??, ii) are the two exc neurons in the cortical columns of the same layer connected by definition? iii) how exactly are the topographic points calculated for the radius?. Bazhenov paper in contrast has quite simple connectivity with one-dimensional cortical layers and simple radii of connections
    I just say if it has a radius of 12 then it's a square with size 24x24 from that neuron, it might be wrong but doesn't specify in the paper
    
    input: 
        source_cells_per_point = how many cells per topographic point at the source? (e.g. 2 in cortex if doing excitatory connections)
        Layers_for_progress_report = include the input and output layers (e.g. l2-l4) if you want progress report
    """
    i = np.array([], dtype=int) # pre and post neuron numbers that are connected
    j = np.array([], dtype=int)
    
    # prob_scaling_factor = p_max/(1/(2*pi*std_connections**2)**0.5) #by how much to scale the gaussian curve given the pmax (which is the probability at the center of the gaussian curve) 
    
    neuron_pre = source_cells_indx #np.linspace(1,source_cell_number, source_cell_number, dtype = int).reshape(source_cells_per_point,30,30) #make an array map with neuron numbers. This makes n 30x30 arrays with neuron numbers 1-900 and 901-1800 etc.... Used for the topographic mapping 
    neuron_post = target_cells_indx #np.linspace(1,target_cell_number, target_cell_number, dtype = int).reshape(target_cells_per_point,30,30)
    
    for pre in np.nditer(neuron_pre):
        # for matrix and core in thalamus --> you need all 900 cells for the topographic indexing but I set those not to be done (i.e. matrix cells when doing core cells) as 0 in the main script
        if pre == 0:
            continue
        if layers_for_progress_report is not None:
            print('pre neuron ' + layers_for_progress_report + ' ' + str(pre) + '/' + str(neuron_pre.max())) # just progress report so I know it works
        indx_pre = np.where(neuron_pre == pre)# gives a tuple of arrays with each array containing the index in that dimension. First element is irrelevant for us as it is the index along the "depth" axis (i.e. how many neurons deep per topographic grid point)
        neurons_post_within_radius = neuron_post[:, np.clip((indx_pre[1][0]-(radius+1)),0,10000):np.clip((indx_pre[1][0]+(radius+1)),0,10000), np.clip((indx_pre[2][0]-(radius+1)),0,10000):np.clip((indx_pre[2][0]+(radius+1)),0,10000)] # which neurons fall within the connection radius (here a square of size 2xradius (radius+1 bc stop index not incl.)),I clip to 10000 arbitrarily
        
        ran = False
        for post in np.nditer(neurons_post_within_radius):
            # same again with matrix and core cells don't include those that are set to 0
            if post == 0:
                continue
            #autapses?
            if autapses is False and not ran:
                if pre == post:
                    ran = True
                    continue
            indx_post = np.where(neuron_post == post)
            distance = sqrt((indx_pre[1][0] - indx_post[1][0])**2 + (indx_pre[2][0] - indx_post[2][0])**2) #what is the distance between the pre and post neuron, use pythagorean theorem
            
            p = p_max*exp(-distance**2/(2*std_connections**2))
            # p = prob_scaling_factor*exp(-distance**2/(2*std_connections**2))/(2*pi*std_connections**2)**0.5 # probability of connection according to scaled gaussian
            if p > np.random.rand():
                i = np.hstack((i, pre))
                j = np.hstack((j, post))
                              
    return np.array([i, j])


def visualise_connectivity(S, line_plot = False, scatter_plot = True):
    '''
    to plot connectivity of the synapses. In the Tononi model connectivity is quite heavy so the line plot just gets completely saturated, therefore default False
    Parameters
    ----------
    S : Synapses to plot (one Synapse class instance).
    
    line_plot : TYPE, optional
        the line plot with i and j on opposite sides and lines connecting them. The default is False.
    scatter_plot : TYPE, optional
        classic scatter plot with i and j on x and y axis. The default is True.

    Returns
    -------
    plots showing connectivity

    '''
    Ns = len(S.source)
    Nt = len(S.target)
    figure(figsize=(10, 4))
    if line_plot and scatterplot:
        plt.subplot(121)
    if line_plot:
        plt.plot(zeros(Ns), arange(Ns), 'ok', ms=10)
        plt.plot(ones(Nt), arange(Nt), 'ok', ms=10)
        for i, j in zip(S.i, S.j):
            plt.plot([0, 1], [i, j], '-k')
        xticks([0, 1], ['Source', 'Target'])
        ylabel('Neuron index')
        xlim(-0.1, 1.1)
        ylim(-1, max(Ns, Nt))
    if line_plot and scatterplot:
        plt.subplot(122)
    if scatter_plot:
        plt.plot(S.i, S.j, 'ok')
        xlim(-1, Ns)
        ylim(-1, Nt)
        xlabel('Source neuron index')
        ylabel('Target neuron index')


def plot_equation(equation, variable, lower_bound = -100, upper_bound = 100):
    '''
    function to produce a plot of the voltage dependency curve of an equation, from -100 to +100mV.
    useful for example in plotting HH gating variables, NMDA voltage response curve etc...
    Equation has to be provided without units
    variable is the variable you want to plot that's in the equation
    '''
    a = []
    b = np.linspace(lower_bound, upper_bound, 10000)

    # var = variable #change the string of the variable to a variable name
    for x in b:
        vars()[variable] = x
        c = eval(equation)
        a.append(c)
    
    plt.plot(b,a)
    xlabel('variable')


def colored_voltage_traces(Monitor, voltage_variable, first_neuron_index = 0, last_neuron_index = None, min_colored = -0.08, max_colored = -0.05):
    '''
    to colormap the voltage trace of many neurons. 
    
    Parameters
    ----------
    Monitor : StateMonitor to plot
    first_neuron : first neuron to plot as index
    last_neuron : last neuron to plot
    min_colored : the voltage value that will have the min color (for U/D states -80)
    max_colored : the voltage value that will have the max color (for U/D states -50)

    Returns
    -------
    A colormap of the voltage traces.

    '''
    run = 0
    last_neuron = last_neuron_index if last_neuron_index is not None else Monitor.n_indices
    for neuron in np.arange(first_neuron_index, last_neuron):
        if run == 0:
            stacked_traces = getattr(Monitor, voltage_variable)[neuron]
        else:
            trace = getattr(Monitor, voltage_variable)[neuron]
            stacked_traces = np.vstack((stacked_traces,trace))
        run +=1
    fig, ax = plt.subplots()
    fig.set_size_inches(30, 10.5)
    plot = ax.pcolormesh(stacked_traces, cmap = 'RdYlBu_r', vmin = min_colored, vmax = max_colored)
    fig.colorbar(plot, ax = ax, label = 'membrane voltage')
    # figure out which x-axis ticks are the seconds
    xtick_pos, = np.where(np.mod(Monitor.t/second, 1) == 0) #where the remainder of the division of time vector and 1 is 0 (i.e. integer i.e. a whole second)
    ax.set_xticks(xtick_pos)
    xtick_labels = np.arange(int(len(Monitor.t)*(Monitor.clock.dt/second)))
    ax.set_xticklabels(xtick_labels, fontsize=15)
    xlabel('time in seconds', fontsize=15)
    ylabel('neuron number', fontsize=15)
 

def plot_currents(Monitor, cell_group_name, neuron_to_plot = 0, detailed_syn_only = False, all_currents_total = False, current_list = current_list, int_currents = True, syn_currents = True):
    '''
    Parameters
    ----------
    Monitor : StateMonitor with the currents
    cell_group_name = i.e. exc_L2, inh_L4, etc.. for the plot
    neuron_to_plot: index of neuron to plot in StateMonitor
    int_currents: plot intrinsic currents?
    syn_currents: plot synaptic currents?
    Returns
    -------
    plot with all the conductances in the cell. Click on legend to toggle currents on and off
    '''
    if int_currents == False:
        current_list = syn_current_list 
    elif syn_currents == False: 
        current_list = int_current_list    
    elif  detailed_syn_only == True:
        current_list = Monitor.record_variables.copy()
        del current_list[0:(Monitor.record_variables.index('Ipoiss') + 1)]
    elif all_currents_total == True:
        current_list = Monitor.record_variables
    else:
        current_list = current_list
    cell = f'{cell_group_name}' + ' ' + f'{neuron_to_plot}' #.split('=')[1]
    lines = []
    fig, ax = plt.subplots()
    for indx, current in enumerate(current_list):
        try:
            vars()['line' + str(indx)], = ax.plot(Monitor.t/ms, getattr(Monitor, current)[0], label = current) 
            lines.append(vars()['line' + str(indx)])
        except:
            print(f'no current {current} in this cell')
    xlabel('time ms')
    ylabel('current')
    ax.set_title(f'''{cell} 
                 click on legend line to toggle line on/off''')
    leg = ax.legend(fancybox=True, shadow=True)
    
    lined = {} #to map legend lines to originial lines
    for legline, origline in zip(leg.get_lines(), lines):
        legline.set_picker(True)  # Enable picking on the legend line.
        lined[legline] = origline
        
    def on_pick(event):
        # On the pick event, find the original line corresponding to the legend
        # proxy line, and toggle its visibility.
        legline = event.artist
        origline = lined[legline]
        visible = not origline.get_visible()
        origline.set_visible(visible)
        # Change the alpha on the line in the legend so we can see what lines
        # have been toggled.
        legline.set_alpha(1.0 if visible else 0.2)
        fig.canvas.draw()
    
    fig.canvas.mpl_connect('pick_event', on_pick) #pick_event is the event id for selecting an "artist" on figure


def calc_LFP(monitor):
    '''
    sum up the postsynaptic currents of all excitatory cells in a layer and inverse the signal to calculate the LFP, as in 2007 Tononi. 
    They have some constants in the equation but they are the same for every neuron so it just scales the LFP.
    In 2005 they simply average out the membrane potential to get an LFP-like signal but it's obviously not very interesting
    '''
    LFP = np.array([])
    for j in range(monitor.n_indices):
        Ij = monitor.Isyn[j]
        if j == 0:
            LFP = Ij
        else:
            LFP = np.add(Ij, LFP)
    # filter_cheby = cheby2()
    fig, ax = plt.subplots()
    ax.plot(monitor.t/ms, LFP)
    ax.invert_yaxis()
    
    
def cycle_through_traces(monitor, variable, n_rows = 4, n_cols = 4): 
    '''

    Parameters
    ----------
    monitor : State Monitor
        
    variable : Variable to plot
        
    n_rows : optional
               The default is 4.
    n_cols : optional
                The default is 4.

    Returns
    -------
    None.

    '''
    fig, ax = plt.subplots(n_rows, n_cols)
    for ind, ax1 in enumerate(ax.flatten()):
        ax1.plot(monitor.t/ms, monitor.V[ind])
        ax1.set_xlabel('time ms')
        ax1.set_title('Neuron ' + str(ind))
        
    class event_handling:
        ind = 0
        def next(self, event):
            self.ind += ax.size #increases by the number of cells in the plot
            for ind, ax1 in enumerate(ax.flatten()):
                ax1.clear()
                try:
                    ax1.plot(monitor.t/ms, monitor.V[ind + self.ind])
                    ax1.set_xlabel('time ms')
                    ax1.set_title('Neuron ' + str(ind + self.ind))
                except IndexError:
                    continue
            plt.draw()
    
        def prev(self, event):
            self.ind -= ax.size #increases by the number of cells in the plot
            for ind, ax1 in enumerate(ax.flatten()): 
                ax1.clear()
                ax1.plot(monitor.t/ms, monitor.V[ind + self.ind])
                ax1.set_xlabel('time ms')
                ax1.set_title('Neuron ' + str(ind + self.ind))
            plt.draw()
        
        def plot_enlarge(self, event):
              if event.inaxes in ax:
                 newfig, newax = plt.subplots()
                 plot_to_enlarge = fig.axes.index(event.inaxes)
                 newax.plot(ax.flatten()[plot_to_enlarge].get_lines()[0].get_xdata(), ax.flatten()[plot_to_enlarge].get_lines()[0].get_ydata())
                 newfig.canvas.manager.window.showMaximized()
                 
    callback = event_handling()
    axprev = plt.axes([0.7, 0.01, 0.075, 0.075])
    axnext = plt.axes([0.81, 0.01, 0.075, 0.075])
    bnext = Button(axnext, 'Next')
    bnext.on_clicked(callback.next)
    bprev = Button(axprev, 'Previous')
    bprev.on_clicked(callback.prev)
    axprev._button = bnext #create dummy reference (don't quite get why but I think it needs an explicit reference as an attribute of the plt.axes because the variable bnext is gone after function is called). Putting globals()['bnext'] = Button(axnext, 'Next') works too
    axnext._button = bprev
    fig.canvas.mpl_connect("button_press_event", callback.plot_enlarge)


def E_I_conductance(voltage_monitor, current_monitor, cell = 0, region = 'CX', E_ampa = 0, E_nmda = 0 , E_gaba_a_ctx = -70, E_gaba_a_T = -80, E_gaba_b = -90):
    v_trace = voltage_monitor.V[cell].copy()
    
    ampa = current_monitor.Isyn_ampa[cell].copy()
    nmda = current_monitor.Isyn_nmda[cell].copy()
    exc = np.add(ampa, nmda)
    gaba_a = current_monitor.Isyn_gaba_a[cell].copy()
    gaba_b = current_monitor.Isyn_gaba_b[cell].copy()
    inh = np.add(gaba_a, gaba_b)
    
    ampa_drive = v_trace - E_ampa
    nmda_drive = v_trace - E_nmda
    if region == 'CX':
        gaba_drive = v_trace - E_gaba_a_CX
    elif region == 'T':
        gaba_drive = v_trace - E_gaba_a_T
    
    #conductances:
    ampa_con = numpy.divide(ampa, ampa_drive)
    nmda_con = numpy.divide(nmda, nmda_drive)
    gaba_a_con = numpy.divide(gaba_a, gaba_drive)
    
    fig, ax = plt.subplots()
    ax.plot(voltage_monitor.t/ms, ampa_con)
    ax.plot(voltage_monitor.t/ms, nmda_con)
    ax.plot(voltage_monitor.t/ms, gaba_a_con)
    xlabel('time ms')
    ylabel('conductance')
    ax.set_title(f'''conductances of cell {cell}''')
    leg = ax.legend(fancybox=True, shadow=True)
    
    


    


    
    
    
    