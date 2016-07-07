import yaml
import subprocess
import os

def read_yaml(file_name):
  with open(file_name, "r") as f:
    return yaml.load(f)

def download(language, source, source_name, source_type):
  output_path = "{0}/{1}".format(language, source_name)
  if os.path.exists(output_path):
    print "skipping {0} {1} because the target folder already exists".format(source_type, source_name)
  else:
    print "downloading {0} {1}".format(source_type, source_name)
    command = """youtube-dl -i --max-downloads 500 --extract-audio --audio-format wav {0} -o "{1}/{2}/%(title)s.%(ext)s" """.format(source,language,source_name)
    subprocess.call(command, shell=True)

def download_user(language, user):
  user_selector = "ytuser:%s" % user
  download(language, user_selector, user, "user")

def download_playlist(language, playlist_name, playlist_id):
  download(language, playlist_id, playlist_name, "playlist")

if __name__ == '__main__':
  sources = read_yaml("sources.yml")
  for language, categories in sources.items():
    for user in categories["users"]:
      download_user(language, user)
    #for playlist_name, playlist_id in categories["playlists"].items():
    #  download_playlist(language, playlist_name, playlist_id)


