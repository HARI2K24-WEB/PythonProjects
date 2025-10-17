import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.fftpack import fft

# Parameters
RATE = 44100
CHUNK = 2048
CHANNELS = 1
# Desired maximum frequency to visualize (Hz). Will be capped to RATE/2 (Nyquist).
MAX_FREQ = 100000
# dB offset to apply to captured audio (negative to attenuate). -6 dB by default.
DB_OFFSET = -6.0
# Linear gain from dB offset
LINEAR_GAIN = 10 ** (DB_OFFSET / 20.0)

# State
buffer = np.zeros(CHUNK, dtype=np.float32)


def audio_callback(indata, frames, time, status):
    global buffer
    if status:
        print(status)
    audio = indata[:, 0]
    # shift buffer and append
    buffer = np.roll(buffer, -len(audio))
    buffer[-len(audio):] = audio


def start_stream():
    stream = sd.InputStream(callback=audio_callback, channels=CHANNELS, samplerate=RATE, blocksize=CHUNK//4)
    stream.start()
    return stream


def compute_spectrum(data):
    # Windowing
    window = np.hanning(len(data))
    data_win = data * window
    # FFT
    freqs = np.fft.rfftfreq(len(data_win), d=1.0/RATE)
    spec = np.abs(np.fft.rfft(data_win))
    return freqs, spec


def beat_energy(spec, freqs):
    # Bass: below 200 Hz
    bass_idx = freqs < 200
    bass = np.mean(spec[bass_idx]) if np.any(bass_idx) else 0
    mid_idx = (freqs >= 200) & (freqs < 2000)
    mid = np.mean(spec[mid_idx]) if np.any(mid_idx) else 0
    high_idx = freqs >= 2000
    high = np.mean(spec[high_idx]) if np.any(high_idx) else 0
    return bass, mid, high


def visualize():
    plt.style.use('dark_background')
    fig, (ax_wave, ax_spec) = plt.subplots(2, 1, figsize=(10, 6))
    x = np.linspace(-CHUNK/RATE, 0, CHUNK)
    line, = ax_wave.plot(x, buffer, lw=1)
    ax_wave.set_ylim(-1, 1)
    ax_wave.set_title('Waveform')

    freqs = np.fft.rfftfreq(CHUNK, d=1.0/RATE)
    # Cap the visualization frequency range to Nyquist (RATE/2)
    effective_max_freq = min(MAX_FREQ, RATE / 2)
    if MAX_FREQ > RATE / 2:
        print(f"WARNING: MAX_FREQ={MAX_FREQ} Hz exceeds Nyquist (RATE/2={RATE/2} Hz). Using {effective_max_freq} Hz instead.")
    # find index corresponding to effective_max_freq
    max_idx = np.searchsorted(freqs, effective_max_freq)
    if max_idx < 1:
        max_idx = 1
    bar_count = 60
    freq_bins = np.linspace(0, max_idx, bar_count+1, dtype=int)
    bars = ax_spec.bar(range(bar_count), np.zeros(bar_count), color='cyan')
    ax_spec.set_ylim(0, 100)
    ax_spec.set_xlabel(f'Frequency (0 - {effective_max_freq:.0f} Hz)')
    ax_spec.set_title('Spectrum (colorful bass/mid/high)')

    def update(frame):
        data = buffer.copy()
        # apply dB offset (gain)
        data = data * LINEAR_GAIN
        line.set_ydata(data)

        f, spec = compute_spectrum(data)
        bass, mid, high = beat_energy(spec, f)

        # Update bars
        magnitudes = []
        # only consider spectrum up to max_idx
        spec_cut = spec[:max_idx+1]
        for i in range(bar_count):
            lo = freq_bins[i]
            hi = freq_bins[i+1]
            # guard hi within range
            hi = min(hi, len(spec_cut))
            if hi <= lo:
                magnitudes.append(0.0)
            else:
                magnitudes.append(np.mean(spec_cut[lo:hi]))
        max_mag = max(magnitudes) if magnitudes else 1
        for rect, mag, i in zip(bars, magnitudes, range(bar_count)):
            h = mag / (max_mag + 1e-6) * 100
            rect.set_height(h)
            # color mapping: bass -> red, mid -> green, high -> blue
            frac = i / bar_count
            r = max(0, 1 - frac) * (bass / (max_mag + 1e-6))
            g = (1 - abs(frac - 0.5)*2) * (mid / (max_mag + 1e-6))
            b = max(0, frac) * (high / (max_mag + 1e-6))
            # normalize color
            maxc = max(r, g, b, 1e-6)
            r, g, b = r/maxc, g/maxc, b/maxc
            rect.set_color((r, g, b))

        # dynamic scaling
        ax_spec.set_ylim(0, max(100, max_mag*1.2))
        return [line] + list(bars)

    ani = FuncAnimation(fig, update, interval=30, blit=False)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    print('Starting audio stream...')
    stream = start_stream()
    try:
        visualize()
    except KeyboardInterrupt:
        pass
    finally:
        stream.stop()
        stream.close()
