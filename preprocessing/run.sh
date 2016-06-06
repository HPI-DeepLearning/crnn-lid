#!/bin/bash
set -e

: "${SPARK_HOME:?Need to set SPARK_HOME non-empty}"

cd $(dirname $0)
python setup.py sdist --formats=zip

$SPARK_HOME/bin/spark-submit \
  --py-files dist/ilid-preprocessing-0.1.zip \
  --master 'local[4]' \
  sparkline.py $@
