import matplotlib.pyplot as mpl
import numpy as np

def plot_timeseries(data, xlabel="Quarters", filename="", **kwargs):
    dims = (1,len(data))
    fig = mpl.figure(**kwargs)

    for i, (name, data) in enumerate(data.items()):
        mpl.subplot(*dims, i+1)
        mpl.plot(data)
        mpl.title(name)
        mpl.xlabel(xlabel)
        mpl.axhline(y=0, color="#808080", linestyle=":")
    
    mpl.tight_layout()
    if filename:
        mpl.savefig(filename, transparent=True)
    
    return fig

def plot_impulses(imp_data, labels, series, dims, xlabel="Quarters", T=None, filename="", **kwargs):
    fig = mpl.figure(**kwargs)
    for i, name in enumerate(series):
        mpl.subplot(*dims, i+1)
        
        for k, impulse_dict in imp_data.items():
            mpl.plot(impulse_dict.get(name)[:T], label=labels[k])
        
        mpl.title(name)
        mpl.xlabel(xlabel)
        mpl.axhline(y=0, color="#808080", linestyle=":")

        if i == 0:
            mpl.legend()

    mpl.tight_layout()
    if filename:
        mpl.savefig(filename, transparent=True)
    
    return fig

# TODO: streamline this as well...
def plot_decomp(Ds, data, shocks, series, xaxis, filename="", **kwargs):
    fig = mpl.figure(**kwargs)
    for i, o in enumerate(series):
        mpl.subplot(1,3,1+i)
        y_offset_pos, y_offset_neg = 0, 0

        for j, shock in enumerate(shocks):
            D = Ds[:, i, j]
            y_offset = (D > 0) * y_offset_pos + (D < 0) * y_offset_neg
            y_offset_pos_ = y_offset_pos + np.maximum(D,0)
            y_offset_neg_ = y_offset_neg - np.maximum(-D,0)
            mpl.fill_between(xaxis, y_offset_pos, y_offset_pos_, color=f"C{j}", label=shock)
            mpl.fill_between(xaxis, y_offset_neg, y_offset_neg_, color=f"C{j}")
            y_offset_pos = y_offset_pos_
            y_offset_neg = y_offset_neg_
        
        if data is not None:
            mpl.plot(xaxis, data[:, i], color="black")
        if i == 0:
            mpl.legend(framealpha=1)
        mpl.title(o)

    mpl.tight_layout()
    if filename:
        mpl.savefig(filename, transparent=True)

    return fig