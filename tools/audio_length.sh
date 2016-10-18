TmpFile=tmp.txt

find $1 -name "*.wav" -exec soxi -D {} \; > $TmpFile
awk '{ sum += $1 } END { print sum }' $TmpFile
rm $TmpFile