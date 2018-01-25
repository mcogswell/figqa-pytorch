import os.path as pth
import json
import numpy as np
import visdom

class VisdomVisualize():
    def __init__(self, server, port=8894, env_name='main',
                 config_file='.visdom_config.json'):
        '''
        Initialize a visdom server on server:port

        Override port and server using the local configuration from
        the json file at $config_file (containing a dict with optional
        keys 'server' and 'port').

        Credit: based on a visdom wrapper by Nirbhay Modhe
        '''
        print("Initializing visdom env [%s]"%env_name)
        if pth.exists(config_file):
            with open(config_file, 'r') as f:
                config = json.load(f)
                if 'server' in config:
                    server = config['server']
                if 'port' in config:
                    port = int(config['port'])
        self.viz = visdom.Visdom(
            port=port,
            env=env_name,
            server=server,
        )
        self.wins = {}

    @property
    def env(self):
        return self.viz.env

    @env.setter
    def env(self, env_name):
        self.viz.env = env_name

    def append_data(self, x, y, key, line_name, xlabel="Iterations",
                    ytype="linear"):
        '''
        Add or update a plot on the visdom server self.viz

        Plots and lines are created if they don't exist, otherwise
        they are updated.

        Arguments:
            x: Scalar -> X-coordinate on plot
            y: Scalar -> Y Value at x
            key: Name of plot/graph
            line_name: Name of line within plot/graph
            xlabel: Label for x-axis (default: # Iterations)
        '''
        if key in self.wins.keys():
            self.viz.updateTrace(
                X=np.array([x]),
                Y=np.array([y]),
                win=self.wins[key],
                name=line_name
            )
        else:
            self.wins[key] = self.viz.line(
                X=np.array([x]),
                Y=np.array([y]),
                opts=dict(
                    xlabel=xlabel,
                    ylabel=key,
                    ytype=ytype,
                    title=key,
                    marginleft=30,
                    marginright=30,
                    marginbottom=30,
                    margintop=30,
                    legend=[line_name]
                )
            )
