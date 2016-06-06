import scipy.signal

def downsample(signal, current_samplerate, target_samplerate = 16000):
  if current_samplerate > target_samplerate:
    signal_duration = len(signal) / float(current_samplerate)
    num_samples_resampled = signal_duration * target_samplerate
    
    resampled_signal = scipy.signal.resample(signal, num_samples_resampled)
    return (resampled_signal, target_samplerate)
  
  elif current_samplerate < target_samplerate:
    raise ValueError("can't sample higher than the signal")
  
  else:
    return (signal, current_samplerate)