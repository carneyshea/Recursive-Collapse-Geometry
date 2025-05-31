
import numpy as np
import librosa
from scipy.interpolate import interp1d

def load_and_preprocess_audio(audio_path, timesteps=100, scale=100):
    y, sr = librosa.load(audio_path, sr=None)
    y = y / np.max(np.abs(y))
    frame_length, hop_length = 2048, 512
    energy = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    interp_func = interp1d(np.linspace(0, 1, len(energy)), energy, kind='linear')
    resampled_energy = interp_func(np.linspace(0, 1, timesteps))
    return resampled_energy * scale, sr

def run_simulation(energy_series, grid_size=100, timesteps=100, delta_t=1.0,
                   alpha=0.6, beta=0.2, gamma=0.2, delta1=200, delta2=0.01, epsilon=3):
    Ψ = np.zeros((grid_size, grid_size, timesteps))
    ω = np.zeros((grid_size, grid_size, timesteps))
    flat_loop_mask = np.zeros((grid_size, grid_size, timesteps))

    for t in range(timesteps):
        x_rand = np.random.randint(epsilon, grid_size - epsilon)
        y_rand = np.random.randint(epsilon, grid_size - epsilon)
        ω[x_rand, y_rand, t] = energy_series[t]

    for t in range(1, timesteps - 1):
        for x in range(epsilon, grid_size - epsilon):
            for y in range(epsilon, grid_size - epsilon):
                local_energy = np.sum(ω[x-epsilon:x+epsilon+1, y-epsilon:y+epsilon+1, t] ** 2)
                Ψ_local = alpha * local_energy
                Ψ[x, y, t] = Ψ_local + beta * Ψ[x, y, t-1] + gamma * Ψ[x, y, t+1]

                dΨ_dt = abs(Ψ[x, y, t] - Ψ[x, y, t-1]) / delta_t
                if dΨ_dt < delta1 and local_energy > delta2:
                    flat_loop_mask[x, y, t] = 1

    return Ψ, flat_loop_mask
