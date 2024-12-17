from brian2 import *
import imgutil
import numpy as np
import pickle

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
I = (sin(2*pi*100*Hz*t)*0.7 + 0.3) * volt : volt
input : 1
'''

image_list = [0] * 10

for i in range(10):
    loaded_weights = np.load(f'new_weights_012_2_{i + 1}.npy')
    weight_rate = 0.000016 / (sum(loaded_weights[:]) / len(loaded_weights[:]))
    neurons = NeuronGroup(N, eqs_neurons, threshold='v>vt', reset='v = vr',
                          method='euler')
    neurons.v = vr
    neurons.input = imgutil.encode([imgutil.add_noise(imgutil.images_to_arrays('images')[0][3]),
                                    imgutil.draw_box(imgutil.images_to_arrays('images')[0][4]),
                                    imgutil.draw_box(imgutil.images_to_arrays('images')[0][5])]) / 6375

    S = Synapses(neurons, neurons,
                 '''w : 1''',
                 on_pre='''ge += w*weight_rate'''
                 )
    S.connect()
    print(weight_rate)
    S.w = loaded_weights
    M = SpikeMonitor(neurons)
    dec = []
    run(30 * ms, report='text')
    arr = np.array(
        (lambda x: [x[i] if i in x else 0 for i in range(4096 * 3)])(dict(zip(*np.unique(M.i, return_counts=True)))))
    image_list[i] = arr
    decoded = imgutil.decode(arr)
    imgutil.show_gray_images(decoded)

with open("new_image_345_345_noise.pickle", "wb") as f:
    pickle.dump(image_list, f)