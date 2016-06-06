import os
import re
import shutil
import argparse

def clean(filename):
    withOutIllegalChars = re.sub("[^a-zA-Z0-9\.-_ ]", "", filename)
    return re.sub("[ ]{1,}", "_", withOutIllegalChars)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', dest='target', help='Directory for the filenames to be cleaned', required=True)
    args = parser.parse_args()

    os.chdir(args.target)

    for root, dirs, files in os.walk("."):
        for filename in files:
            new_filename = clean(filename)
            new_filepath = os.path.join(root, new_filename)
            old_filepath = os.path.join(root, filename)

            #print "%s -> %s" % (old_filepath, new_filepath)
            shutil.move(old_filepath, new_filepath)

