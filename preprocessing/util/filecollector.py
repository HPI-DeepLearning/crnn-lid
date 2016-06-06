import os

def collect(input_path, file_extension="wav"):
  
  collected = []
  for root, _, files in os.walk(input_path, followlinks=True):
    sound_files = filter(lambda sound_file: sound_file.endswith(file_extension), files)
    sound_files_absolute_paths = [os.path.join(root, sound_file) for sound_file in sound_files]
    collected.extend(sound_files_absolute_paths)
  return collected
