'''
func_plotting.py: Charting utilities for basic functions.

Author: David Abel
'''

# Python imports.
import math
import sys
import os
import matplotlib.pyplot as pyplot
import numpy

class PlotFunc(object):

    # Some nice markers and colors for plotting.
    _markers = ['o', 's', 'D', '^', '*', '+', 'p', 'x', 'v','|']
    _colors = [[240, 163, 255], [102, 120, 173], [113, 198, 113],\
                [197, 193, 170],[85, 85, 85], [198, 113, 113],\
                [142, 56, 142], [125, 158, 192],[184, 221, 255],\
                [153, 63, 0], [142, 142, 56], [56, 142, 142]]
    _colors = [[shade / 255.0 for shade in rgb] for rgb in _colors]

    def __init__(self, func, x_min=0, x_max=10, x_interval=1, param_dict={}, series_name="y", ci_func=None):
        '''
        Args:
            func (lambda : (x --> y))

        '''
        self.func = func
        self.x_min = x_min
        self.x_max = x_max
        self.x_interval = x_interval
        self.param_dict = param_dict
        self.series_name = series_name
        self.ci_func = ci_func

    def plot(self, series_id=0, log_scale=False):
        '''
        Args:
            series_id (int)
            log_scale (bool)
            use_legend (bool)

        Summary:
            Make a basic plot, passing in the xrange defined by self.x_min, self.x_max, self.x_interval
            to the self.func.
        '''
        x_range = numpy.arange(self.x_min, self.x_max, self.x_interval)
        if log_scale:
            y_range = [math.log(self.func(x)) for x in x_range] if self.param_dict is {} else [math.log(self.func(x, self.param_dict)) for x in x_range]
        else:
            y_range = [self.func(x) for x in x_range] if self.param_dict is {} else [self.func(x, self.param_dict) for x in x_range]
        
        # Confidence interval stuff.
        if self.ci_func is not None:
            conf_intervals = [self.ci_func(x) for x in x_range] if self.param_dict is {} else [self.ci_func(x, self.param_dict) for x in x_range]
            top = numpy.add(y_range, conf_intervals)
            bot = numpy.subtract(y_range, conf_intervals)
            pyplot.fill_between(x_range, top, bot, facecolor=PlotFunc._colors[series_id], edgecolor=PlotFunc._colors[series_id], alpha=0.25)

        pyplot.plot(x_range, y_range, marker=PlotFunc._markers[series_id], color=PlotFunc._colors[series_id], label=self.series_name)

def plot_funcs(funcs, file_name="func_plot", title="X versus Y", x_label="X", y_label="Y", log_scale=False, use_legend=True):
    '''
    Args:
        funcs (list of PlotFuncs)
        file_name (str)
        title (str)
        x_label (str)
        y_label (str)

    Summary:
        Plots a bunch of functions together on the same plot.
    '''
    for i, func in enumerate(funcs):
        func.plot(series_id=i, log_scale=log_scale)
    
    pyplot.rc('text', usetex=True)
    if use_legend:
        # Puts the legend into the best location in the plot and use a tight layout.
        pyplot.rcParams['legend.loc'] = 'best'
        pyplot.legend()
    pyplot.xlabel(x_label)
    pyplot.ylabel(y_label)
    title_suffix = " (Log Scale)" if log_scale else ""
    pyplot.title(title + title_suffix)
    pyplot.grid(True)
    pyplot.savefig("./" + file_name + ".pdf", format="pdf")
    os.system("open " + "./" + file_name + ".pdf")
    pyplot.cla()

# --- A Few Example Functions ---

def _geometric_func(x, param_dict):
    gamma = param_dict['gamma']
    return x /(1-gamma)

def _power_func(x, param_dict):
    power = param_dict['power']
    return x ** power

def main():
    power_two = PlotFunc(_power_func, param_dict={'power':3}, series_name='power of 3')
    power_three = PlotFunc(_power_func, param_dict={'power':2}, series_name='power of 2')

    plot_func([power_two, power_three])

if __name__ == "__main__":
    main()
