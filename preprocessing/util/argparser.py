import argparse
import os

def parse():
  parser = argparse.ArgumentParser()
  parser.add_argument('--inputPath', dest='input_path', default=os.getcwd(), help='Input Path to wav files', required=True)
  parser.add_argument('--outputPath', dest='output_path', default=os.path.join(os.getcwd(), "spectrograms"), required=True,
                      help='Output Path to wav files')

  args = parser.parse_args()
  _validate(args)
  return args

def _validate(args):
  # Input args validation
  if not os.path.exists(args.output_path):
      os.mkdir(args.output_path)

  if not os.path.isdir(args.output_path):
      sys.exit("Output path is not a directory.")

  if not os.path.isdir(args.input_path):
      sys.exit("Input path is not a directory.")
