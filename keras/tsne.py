import argparse
import os
import numpy as np
from yaml import load

import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt

from keras.models import Sequential, load_model
from sklearn.manifold import TSNE
from pandas import DataFrame
import data_loaders

def plot_with_labels(lowD_Weights, labels, label_names, filename):

    df = DataFrame({"x": lowD_Weights[:, 0], "y": lowD_Weights[:, 1], "label": labels})
    groups = df.groupby("label")

    # Plot
    fig, ax = plt.subplots()
    ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
    for label, group in groups:
        ax.plot(group.x, group.y, marker='o', linestyle='', ms=6, label=label_names[label])
    ax.legend(numpoints=1)

    # Hide Axis Labels and Ticks
    cur_axes = plt.gca()
    cur_axes.axes.get_xaxis().set_visible(False)
    cur_axes.axes.get_yaxis().set_visible(False)
    cur_axes.axes.get_xaxis().set_ticks([])
    cur_axes.axes.get_yaxis().set_ticks([])

    plt.show()

    fig = plt.gcf()
    fig.savefig(filename)

def visualize_cluster(cli_args):

    config = load(open(cli_args.config, "rb"))

    # Load Data + Labels
    DataLoader = getattr(data_loaders, config["data_loader"])
    data_generator = DataLoader(config["validation_data_dir"], config)

    # Model Generation
    ref_model = load_model(cli_args.model_file)

    # Create a new model and copy all but the last layer over
    model = Sequential()
    i = 0
    for layer in ref_model.layers[:-1]:
        try:
            model.add(layer)
            model.layers[i].set_weights(layer.get_weights())
        except ValueError:
            continue
        i += 1
    model.compile("adam", "categorical_crossentropy")

    probabilities = model.predict_generator(
        data_generator.get_data(should_shuffle=False, is_prediction=True),
        val_samples=data_generator.get_num_files(),
        nb_worker=1,  # parallelization messes up data order. careful!
        max_q_size=config["batch_size"],
        pickle_safe=True
    )
    print("Prob Shape ", probabilities.shape)

    limit = cli_args.limit


    # Save probs to file for analysis with external tools
    directory = os.path.dirname(cli_args.plot_name)
    np.savetxt(os.path.join(directory, "probabilities.tsv"), probabilities[:limit], delimiter='\t', newline='\n')
    file_names, labels = zip(*data_generator.images_label_pairs[:limit])
    df = DataFrame({"label": labels, "filename": file_names})
    df.to_csv(os.path.join(directory, "metadata.tsv"), "\t", "%.18e", header=True, index=False)

    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=cli_args.num_iter)
    lowD_weights = tsne.fit_transform(probabilities[:limit, :])

    plot_with_labels(lowD_weights, labels, config["label_names"], cli_args.plot_name)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', dest='model_file', required=True)
    parser.add_argument('--config', dest='config', required=True)
    parser.add_argument('--plot', dest='plot_name')
    parser.add_argument('--limit', dest='limit', default=2000, type=int)
    parser.add_argument('--iter', dest='num_iter', default=4000, type=int)
    cli_args = parser.parse_args()

    if cli_args.plot_name == None:
        cli_args.plot_name = os.path.join(os.path.dirname(cli_args.model_file), "tsne.pdf")

    visualize_cluster(cli_args)
