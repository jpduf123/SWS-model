# -*- coding: utf-8 -*-
"""
Created on Thu May 13 15:33:35 2021

@author: Mann Lab
"""

# synaptic plasticity rule

from brian2 import *

# set_device('cpp_standalone')

N = 1000
taum = 10*ms
taupre = 20*ms
taupost = taupre
Ee = 0*mV
vt = -54*mV
vr = -60*mV
El = -74*mV
taue = 5*ms
F = 15*Hz
gmax = .01
dApre = .01
dApost = -dApre * taupre / taupost * 1.05
dApost *= gmax
dApre *= gmax

eqs_neurons = '''
dv/dt = (ge * (Ee-v) + El - v) / taum : volt
dge/dt = -ge / taue : 1
'''

input = PoissonGroup(N, rates=F)
neurons = NeuronGroup(1, eqs_neurons, threshold='v>vt', reset='v = vr',
                      method='euler')


# STDP rule here:
# w is the weight, which is added to the neuron every pre spike. Apre (positive) and Apost (negative) are the pre and postsynaptic traces
# At presynaptic spike, the dApre (0.1) is added to Apre. Apre then decays back to 0 over 20 ms. 
# At postsynaptic spike, dApost (something negative) is added to Apost. Apost decays back to 0 over time. 
# Apost gets added to the weight on presynaptic spike (--> makes the w smaller), and Apre on postsynaptic spike (making w bigger)
# so if w gets bigger or smaller is the difference in time between pre and post spike!

S = Synapses(input, neurons,
             '''w : 1
                dApre/dt = -Apre / taupre : 1 (event-driven)
                dApost/dt = -Apost / taupost : 1 (event-driven)''',
             on_pre='''ge += w
                    Apre += dApre
                    w = clip(w + Apost, 0, gmax)
                    t_pre = lastupdate''',
             on_post='''Apost += dApost
                     w = clip(w + Apre, 0, gmax)''',
             )

S.connect()
S.w = 'rand() * gmax'
mon_syn = StateMonitor(S, variables = ['w', 'Apost', 'Apre'], record=[0, 1])
mon_v = StateMonitor(neurons, 'v', record = True)
s_mon = SpikeMonitor(input)
S_cell_mon = SpikeMonitor(neurons)

run(20*second, report='text')

# plot(mon_v.t, mon_v.v[0])

subplot(211)
plot(s_mon.t, s_mon.i)
subplot(212)
plot(S_cell_mon.t, S_cell_mon.i, '.k')


# subplot(311)
# plot(S.w[0:2] / gmax, '.k')
# ylabel('Weight / gmax')
# xlabel('Synapse index')
# subplot(312)
# plot(mon.t/second, mon.w.T/gmax)
# xlabel('Time (s)')
# ylabel('Weight / gmax')
# subplot(313)
# hist(S.w / gmax, 20)
# xlabel('Weight / gmax')
# tight_layout()
# show()