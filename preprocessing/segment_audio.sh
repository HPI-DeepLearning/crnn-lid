# Convert all audio in the current (sub)folder to a sequence of WAV files
# Usage: ./segment_audio.sh output_directory

# Clone directory structure
find . -type d -exec mkdir -p -- $1/{} \;

# Do conversion
find . -iregex ".*\.\(mp3\|aac\|wav\)" -exec ffmpeg -i {} -map 0 -ar 16000 -f segment -segment_time 10 $1/{}_%03d.wav \;

# -map 0            : use first channel
# -ar 16000         : audio sample rate
# -f segment        : enable segmentation
# -segment_time 10  : segement length in seconds
# -af loudnorm=I=-16:TP=-1.5:LRA=11 : normalize volume using EBU R128 standard
