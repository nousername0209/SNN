from brian2 import *
import imgutil
import numpy as np
import math

# Parameters
N = 4096 * 3
taum = 10 * ms
taue = 1 * ms
taupre = 20 * ms
taupost = taupre
Ee = 60 * mV
vt = 6 * mV
vr = 0 * mV
El = -7 * mV
gmax = 10
dApre = .01
dApost = -dApre * taupre / taupost * 1.05
dApost *= gmax
dApre *= gmax

# Neuron equations
eqs_neurons = '''
dv/dt = (ge * (Ee-v) + El - v + I*input) / taum : volt
dge/dt = -ge / taue : 1
I = (sin(2*pi*100*Hz*t+phi)*0.7 + 0.3) * volt : volt
input : 1
phi : 1
'''

image_array = imgutil.images_to_arrays('images')[0]

for j in range(3):
    # NeuronGroup
    neurons = NeuronGroup(N, eqs_neurons, threshold='v>vt', reset='v = vr',
                          method='euler')
    neurons.v = vr
    neurons.phi = np.array([(i // 4096) * math.pi / 3 for i in range(4096 * 3)])

    neurons.input = imgutil.encode(image_array[j * 3:j * 3 + 3]) / 6375  # Assign input

    # Synapses
    S = Synapses(neurons, neurons,
                 '''w : 1
                    dApre/dt = -Apre / taupre : 1 (event-driven)
                    dApost/dt = -Apost / taupost : 1 (event-driven)''',
                 on_pre='''ge += w/6000
                        Apre += dApre
                        w = clip(w + Apost, 0, gmax)''',
                 on_post='''Apost += dApost
                         w = clip(w + Apre, 0, gmax)''',
                 )
    S.connect()

    M = SpikeMonitor(neurons)
    for i in range(10):
        run(100 * ms, report='text')
        np.save(f'new_weights_012_{j + 1}_' + f'{i + 1}' + '.npy', S.w[:])

for j in range(3):
    # NeuronGroup
    neurons = NeuronGroup(N, eqs_neurons, threshold='v>vt', reset='v = vr',
                          method='euler')
    neurons.v = vr
    neurons.phi = np.array([(i // 4096) * math.pi / 3 for i in range(4096 * 3)])

    neurons.input = imgutil.encode([image_array[j], image_array[j + 3], image_array[j + 6]]) / 6375  # Assign input

    # Synapses
    S = Synapses(neurons, neurons,
                 '''w : 1
                    dApre/dt = -Apre / taupre : 1 (event-driven)
                    dApost/dt = -Apost / taupost : 1 (event-driven)''',
                 on_pre='''ge += w/6000
                        Apre += dApre
                        w = clip(w + Apost, 0, gmax)''',
                 on_post='''Apost += dApost
                         w = clip(w + Apre, 0, gmax)''',
                 )
    S.connect()

    M = SpikeMonitor(neurons)
    for i in range(10):
        run(100 * ms, report='text')
        np.save(f'new_weights_036_{j + 1}_' + f'{i + 1}' + '.npy', S.w[:])