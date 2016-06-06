import os
import re

def truncate_extension(filename):
    return os.path.splitext(filename)[0]

def clean(filename):
    return re.sub("[^a-zA-Z0-9\.-_ ]", "", filename)