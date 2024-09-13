import matplotlib.pyplot as mpl

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