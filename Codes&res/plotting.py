import numpy as np
import matplotlib.pyplot as plt


def figsize(scale, nplots=1):
    fig_width_pt = 250.0  # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0 / 72.27  # Convert pt to inch
    #golden_mean = (np.sqrt(5.0) - 1.0) / 2.0  # Aesthetic ratio (you could change this)
    fig_width = fig_width_pt * inches_per_pt * scale  # width in inches
    # fig_height = nplots * fig_width * scale  # height in inches
    fig_size = [fig_width, fig_width]
    return fig_size


def newfig(width, nplots=1):
    fig = plt.figure(figsize=figsize(width, nplots))
    ax = fig.add_subplot(111)
    return fig, ax


def savefig(filename, crop=True):
    if crop == True:
        #        plt.savefig('{}.pgf'.format(filename), bbox_inches='tight', pad_inches=0)
        plt.savefig('{}.png'.format(filename), bbox_inches='tight')
    else:
        #        plt.savefig('{}.pgf'.format(filename))
        plt.savefig('{}.png'.format(filename))

