import os
import subprocess
import sys

def escape_characters(s):
  return (s
    .replace("$", "\\$")
    .replace("`", "\\`")
    )

def get_audio_length(f):
  command = "soxi -D \"%s\"" % escape_characters(f)
  return float(subprocess.check_output(command, shell=True))

if __name__ == '__main__':
  target_dir = sys.argv[1]
  files = os.listdir(target_dir)
  files = filter(lambda f: os.path.splitext(f)[1] == ".wav", files)
  total = sum([get_audio_length(os.path.join(target_dir, f)) for f in files])
  print total / 60. / 60.