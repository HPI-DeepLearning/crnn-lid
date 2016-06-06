import os
import subprocess
from audio_length import escape_characters
import argparse

filetypes_to_convert=[".mp3",".m4a", ".webm"]

def convert(filename):
  filename_extensionless, extension = os.path.splitext(filename)
  new_filename = "".join([filename_extensionless, ".wav"])
  if not os.path.exists(new_filename):
    command = "ffmpeg -i \"{}\" -ac 1 \"{}\"".format(escape_characters(filename), escape_characters(new_filename))
    subprocess.call(command, shell=True)


def walk_path(path):
  for root, dirs, files in os.walk(path):
    for sound_file in files:
      _, extension = os.path.splitext(sound_file)
      #print sound_file
      if extension in filetypes_to_convert:
        yield os.path.join(root, sound_file)
      else:
        continue


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--path', dest='path', help='Directory for the files to convert', required=True)
  args = parser.parse_args()

  for sound_file in walk_path(args.path):
    convert(sound_file)


