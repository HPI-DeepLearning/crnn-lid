import argparse
from yaml import load

from keras.models import load_model
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import data_loaders

def plot_with_labels(lowDWeights, labels, filename='tsne.png'):

    assert lowDWeights.shape[0] >= len(labels), "More labels than weights"
    plt.figure(figsize=(20, 20))  # in inches
    for i, label in enumerate(labels):
        x, y = lowDWeights[i,:]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')

    plt.savefig(filename)

def visualize_cluster(cli_args):

    config = load(open(cli_args.config, "rb"))

    # Load Data + Labels
    DataLoader = getattr(data_loaders, config["data_loader"])
    data_generator = DataLoader(config["validation_data_dir"], config)

    # Model Generation
    model = load_model(cli_args.model_dir)
    final_weights = model.output_layers[0].get_weights()[0] # input weights to final FC

    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=50)

    lowD_weights = tsne.fit_transform(final_weights)
    plot_with_labels(lowD_weights, config["label_names"])

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', dest='model_dir', required=True)
    parser.add_argument('--config', dest='config', required=True)
    cli_args = parser.parse_args()

    visualize_cluster(cli_args)
