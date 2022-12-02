import numpy as np
import mne 
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.cm import ScalarMappable

def plot_tfr():
    pass

def plot_tfr_topo(tfr,
                  baseline,
                  vmin,
                  vmax,
                  colorbar=True,
                  figname=None,
                  figsize=(25, 15),
                  figsavepath=None):

    """
    Plot time-frequency representation as a topographical plot

    Parameters
    ----------
    tfr : AverageTFR object
        time-frequency representation derived from mne.time_frequency.tfr_morlet() function

    figsize : tuple
        figure size as (width, height)
        
    baseline : tuple
        determine baseline period as (start, stop)

    vmin : int
        minimum voltage value displayed in the figure

    vmax : int 
        maximum voltage value displayed in the figure

    colorbar : bool
        list of event ids of interest

    figname : string
        figure name in string

    figsavepath : string
        figure save path in string
 
    """

    example_epoch = tfr

    # topographical plot configuration
    layout = mne.channels.layout.find_layout(example_epoch.info)
    pos = layout.pos.copy()
    pos += np.array([0, 0, 1, 1])*0.02
    colorbar_pos = np.array([0.96, 0.25, 0.02, 0.5])

    fig = plt.figure(figsize=figsize)
    for ch, pos_ in enumerate(pos):
        ch_name = example_epoch.ch_names[ch]

        ax = fig.add_axes(pos_)

        # time-frequency plot for each channel
        tfr.plot(picks=[ch], 
                 baseline=baseline,
                 mode='zlogratio',
                 axes=ax,
                 vmax=vmax,
                 vmin=vmin,
                 show=False,
                 cmap='jet',
                 colorbar=False,
                 dB=False)

        # axes configuration    
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.axvline(0, linestyle='--', color='black')
        ax.title.set_text(ch_name)

        # set the plot properties for 'afz' channel only
        if ch in [0, 56]:
            ax.set_xlabel('Time(s)')
            ax.set_ylabel('Frequency(Hz)')
            ax.set_xticks([-4, -2, 0, 2, 4, 6])
            ax.set_xticklabels(['-4', '-2', '0', '2', '4', '6'])
            ax.set_yticks([5, 10, 20, 30, 40])
            ax.set_yticklabels(['5', '10', '20', '30', '40'])
        else:
            ax.tick_params(bottom=True, top=False, left=True, right=False,
                            labelbottom=False, labeltop=False, labelleft=False, labelright=False)

    # set colorbar for this figure
    if colorbar:
        ax = fig.add_axes(colorbar_pos)
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        fig.colorbar(ScalarMappable(norm=norm, cmap='jet'), cax=ax)
        ax.set_ylabel('')

    # set figure name
    if figname!=None:
        fig.suptitle(figname, fontsize=24)

    # save figure if needed
    if figsavepath!=None:
        fig.savefig(figsavepath)

    plt.close(fig)


def plot_topomap():
    pass

def plot_connectivity_matrix(conn,
        n_channels,
        figsize=(15, 15),
        figname=None,
        figsavepath=None):

    """
    Plot connectivity matrix

    Parameters
    ----------
    conn : array
        connectivity values derived from data_prep.epoch_to_conn

    n_channels : int
        number of channels
        
    figsize : tuple
        figure size as (width, height)

    figname : string
        figure name in string

    figsavepath : string
        figure save path in string
 
    """

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # make connectivity value into the symmetric matrix
    conn = conn + conn.T
    
    img = ax.imshow(conn, 
                extent=[0, n_channels, 0, n_channels], 
                cmap='jet', 
                vmin=0, 
                vmax=0.55)
    
    # center the ticks 
    ax.set_xticks(np.linspace(0, n_channels-1, n_channels)+0.5)
    ax.set_yticks(np.linspace(n_channels-1, 0, n_channels)+0.5)

    # create grid lines using minor ticks
    ax.set_xticks(np.linspace(n_channels-1, 0, n_channels), minor=True)
    ax.set_yticks(np.linspace(0, n_channels-1, n_channels), minor=True)
    ax.tick_params(which='minor', direction='in')
    ax.grid(visible=True, which='minor', color='k', linestyle='-', linewidth=1)

    # set tick parameters
    ax.xaxis.set_tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    ax.set_xticklabels(data.ch_names, rotation=90)
    ax.set_yticklabels(data.ch_names)

    # set each pixel to be squared
    ax.set_aspect("equal")

    # set figure name
    if figname!=None:
        ax.set_title(figname, fontsize=16, fontweight='bold')
    
    # customize the colorbar
    fig.colorbar(img, pad=0.02, shrink=0.85, aspect=30)

    # save figure
    if figsavepath!=None:
        fig.savefig(figsavepath)


def plot_topo_network():
    pass
    


