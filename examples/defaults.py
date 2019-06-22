import matplotlib.pyplot as plt

colorcycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

# default tracker styles to be using in plotting
styles = {
    '1-D Gaussian': {
        #'label':r'1-D Gaussian ($\sigma=R$)',
        'color':'b',
        'linestyle':'--','linewidth':2,
        'marker':'o','markersize':4
    },
    '1-D Gaussian (Bastankhah)': {
        'color':'r',
        'linestyle':'--','linewidth':2,
        'marker':'o','markersize':4
    },
    '1-D Gaussian (ideal sigma)': {
        #'label':r'1-D Gaussian (x-mom deficit $\sigma$)',
        'color':'m',
        'linestyle':'--','linewidth':2,
        'marker':'o','markersize':4
    },
    '2-D Gaussian': {
        #'label':r'2-D Gaussian (opt $\sigma_x$,$\sigma_y$,$\theta$)',
        'color':'g',
        'linestyle':'--','linewidth':2,
        'marker':'o','markersize':4
    },
    
    'const area': {
        'color':colorcycle[0],
        'linestyle':'-','linewidth':2,
        'marker':'+','markersize':10
    },
    'const momentum deficit': {
        'color':colorcycle[1],
        'linestyle':'-','linewidth':2,
        'marker':'+','markersize':10
    },
    
    'min power': {
        'color':'y',
        'linestyle':'-','linewidth':2,
        'marker':'x','markersize':8
    },
}

