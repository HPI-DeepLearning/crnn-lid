
if [ $# -lt 1 ]; then
  echo "Usage: $0 <language>"
  exit 1
fi

source ./voxforge_download_urls
LANG=$1
eval VOXFORGE_DATA_URL=\$$LANG

if [ "${VOXFORGE_DATA_URL}x" = "x" ]; then
  echo "Can't find url for language $LANG"
  exit 1
fi

ZIPS=zipped_index_$LANG
TEMP_DIR=tmp

curl $VOXFORGE_DATA_URL | grep -o '<a .*href=.*>' | sed -e 's/<a /\n<a /g' | sed -e 's/<a .*href=['"'"'"]//' -e 's/["'"'"'].*$//' -e '/^$/ d' | grep tgz$ > $ZIPS

for ZIP in $(cat $ZIPS)
do
   URL=$VOXFORGE_DATA_URL/$ZIP
   wget --no-verbose --directory-prefix=$TEMP_DIR $URL
  ./extract_tgz.sh $TEMP_DIR/$ZIP $LANG
done

rm $ZIPS
