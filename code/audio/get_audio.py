import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt    
from yin import yin                 

fs, hop, window = 48000, 480, 4800
shelf = np.zeros(hop, dtype=np.float32)
buffer = np.zeros(window, dtype=np.float32)
print(sd.query_devices())  # list all audio devices

def callback(data, frames, time, status):       # audio thread: just shelve the box
    shelf[:] = data[:, 0]

#Plotting
plt.ion()                                           
fig, ax = plt.subplots()                            
line, = ax.plot(buffer)                             
ax.set_ylim(-1.5, 1.5)                              

with sd.InputStream(device="Scarlett", samplerate=48000, channels=1, blocksize=480, callback=callback):   # "default" → "Scarlett"
    while True:
        buffer[:-hop] = buffer[hop:]          
        buffer[-hop:] = shelf    

        #Plotting              
        line.set_ydata(buffer)                      
        plt.pause(0.01)  
        f0 = yin(buffer, fs)
        print("no pitch" if f0 is None else f"{f0:.1f} Hz")                           