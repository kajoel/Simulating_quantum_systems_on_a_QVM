from core.data import load
import matplotlib.pyplot as plt

data, metadata = load()
p = plt.plot(data['workers'], data['time'], '.')
plt.legend(p, [metadata['machine']])
plt.xlabel('Workers')
plt.ylabel('Time/worker (s)')

