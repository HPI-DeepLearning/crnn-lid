#!/bin/bash

set -e

SCRIPT_ROOT="$(dirname $0)"
SOUND_ROOT="/home/pva1/DeepAudio/iLID-Data"

#declare -a SOURCES=(youtube dubsmash voxforge)
declare -a SOURCES=(youtube dubsmash voxforge)
declare -a LANGUAGES=(german english)

for SOURCE in "${SOURCES[@]}"
do
  for LANGUAGE in "${LANGUAGES[@]}"
  do
    SOUND_DIR=$SOUND_ROOT/$SOURCE/$LANGUAGE
    if [[ $(ls -d $SOUND_DIR/*/ ) ]]; then
      for SUBFOLDER in $(ls -d $SOUND_DIR/*/)
      do
        python2 $SCRIPT_ROOT/neg23/neg23.py $SUBFOLDER
      done
    else
        python2 $SCRIPT_ROOT/neg23/neg23.py $SOUND_DIR/
    fi    
  done
done