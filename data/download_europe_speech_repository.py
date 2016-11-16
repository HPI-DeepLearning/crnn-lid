import argparse
import subprocess
import os
from multiprocessing import Pool
from collections import defaultdict
from functools import partial

from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By


def get_urls_for_page(DRIVER, language_id, page):
    url = "https://webgate.ec.europa.eu/sr/search-speeches?language={0}&level=All&use=All&domain=All&type=All&combine=&combine_1=&video_reference=&page={1}".format(
        language_id, page)
    DRIVER.get(url)

    # Wait for site to load
    WebDriverWait(DRIVER, 5).until(EC.presence_of_element_located((By.CLASS_NAME, "title")))

    title_selector = "td.views-field.views-field-title a"
    title_elements = DRIVER.find_elements_by_css_selector(title_selector)
    hrefs = map(lambda el: el.get_attribute("href"), title_elements)

    return hrefs


def youtube_downloader(output_dir, url):
    command = """youtube-dl -i --max-downloads 1 --extract-audio --audio-format wav {} -o "{}/%(title)s.%(ext)s" """.format(
        url, output_dir)
    return subprocess.call(command, shell=True)


def download(urls, output_dir):
    pool = Pool(4)

    for language, url_list in urls.iteritems():
        dir = os.path.join(output_dir, language)
        pool.map(partial(youtube_downloader, dir), url_list)

    pool.close()
    pool.join()


if __name__ == '__main__':
    DRIVER = webdriver.Firefox()
    # DRIVER.set_window_size(1440, 900)

    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('--output_dir', '-o', metavar='Output directory', default=os.path.join(os.getcwd(), "eu-repo"),
                        type=str, help='Output directory.', dest="output_dir")
    args = PARSER.parse_args()

    LANGUAGE_ID_MAP = {
        # (language id, max page number)
        "english": (114, 37),
        "german": (120, 27),
        "french": (118, 30),
        "spanish": (112, 20)
    }

    urls = defaultdict(list)

    # for language, (language_id, max_page_number) in LANGUAGE_ID_MAP.items():
    #
    #     for i in range(0, max_page_number):
    #         urls[language] += get_urls_for_page(DRIVER, language_id, i)
    #
    # print urls

    DRIVER.close()

    import pickle

    urls = pickle.load(open("urls.pickle", "rb"))
    download(urls, args.output_dir)
