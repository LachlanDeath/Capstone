import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt    
from pitch_test_methods import pitch_diff, pitch_diff_mod, FRAME       

fs, hop, window = 48000, 480, FRAME
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

        f0 = pitch_diff(buffer, fs)
        f1 = pitch_diff_mod(buffer, fs) 
        if f0 is not None:
            print(f"{f0:.1f} Hz diff_mod: {f1:.1f} Hz")
                       