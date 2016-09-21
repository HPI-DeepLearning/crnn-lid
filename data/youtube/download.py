import subprocess
import os
import argparse
import glob
import string
import yaml
from collections import Counter

file_counter = Counter()

def read_yaml(file_name):
    with open(file_name, "r") as f:
        return yaml.load(f)


def clean_filename(filename):
    valid_chars = "-_%s%s" % (string.ascii_letters, string.digits)
    new_name = "".join(c for c in filename if c in valid_chars)
    new_name = new_name.replace(' ','_')
    return new_name


def download(language, source, source_name, source_type):

    output_path_raw = os.path.join(args.output_path, "raw", language, source_name)

    if os.path.exists(output_path_raw):
        print "skipping {0} because the target folder already exists".format(output_path_raw)
    else:
        print "Downloading {0} {1} to {2}".format(source_type, source_name, output_path_raw)
        command = """youtube-dl -i --max-downloads 1200 --extract-audio --audio-format wav {0} -o "{1}/%(title)s.%(ext)s" """.format(source, output_path_raw)
        subprocess.call(command, shell=True)


    # Use ffmpeg to convert and split WAV files into 10 second parts
    output_path_segmented = os.path.join(args.output_path, "segmented", language, source_name)
    if not os.path.exists(output_path_segmented):

        os.makedirs(output_path_segmented)
        files = glob.glob(os.path.join(output_path_raw, "*.wav"))

        for f in files:

            cleaned_filename = clean_filename(os.path.basename(f))
            output_filename = os.path.join(output_path_segmented, cleaned_filename[:-4] + "_%03d.wav")

            command = ["ffmpeg", "-y", "-i", f, "-map", "0", "-ac", "1", "-ar", "16000", "-f", "segment", "-segment_time", "10", output_filename]
            subprocess.call(command)

    file_counter[language] += len(glob.glob(os.path.join(output_path_segmented, "*.wav")))


def download_user(language, user):
    user_selector = "ytuser:%s" % user
    download(language, user_selector, user, "user")


def download_playlist(language, playlist_name, playlist_id):
    download(language, playlist_id, playlist_name, "playlist")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--output', dest='output_path', default=os.getcwd(), required=True)
    args = parser.parse_args()

    sources = read_yaml("sources.yml")
    for language, categories in sources.items():
        for user in categories["users"]:
            download_user(language, user)
        #for playlist_name, playlist_id in categories["playlists"].items():
        #  download_playlist(language, playlist_name, playlist_id)

    print file_counter
