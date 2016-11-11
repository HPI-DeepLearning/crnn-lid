# Convert all audio in the current (sub)folder to a sequence of WAV files
# Usage: ./segment_audio_sox.sh output_directory

# Clone directory structure
#find . -type d -exec mkdir -p -- $1/{} \;

find . -name "*.wav" -exec sox -V3 {} -c 1 -r 16000 $1/{}.wav silence 1 0.1 3% -1 3.0 3% trim 0 1 : newfile : restart \;

# -c 1 convert to mono
# trim 0 1 trim for 1 second beginning at 0 sec
# -w output as words (16bit)
# -r 16000 sampling rate
# silence 1 0.1 3% -1 3.0 3% # start at begining for min. 0.1 sec at 3% level, also trim in the file
# -V3 high verbosity

