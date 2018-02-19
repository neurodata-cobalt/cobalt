from intern.remote.boss import BossRemote
from intern.resource.boss.resource import *
import numpy as np

class bossHandler:
    def __init__(self, collection_name):
        """
        Constructor
        """
        self.collection_name = collection_name
        try:
            self.rmt = BossRemote()
        except:
            print('Unexpected Error:', sys.exc_info()[0])
    
#     def get_collection_list(self):
#         return self.rmt.list_collections()

    def list_experiments(self):
        """
        List all the experiments available in current collection
        """
        exp_list = self.rmt.list_experiments(self.collection_name)
        return exp_list
    
    def select_experiment(self, experiment_name):
        """
        Select an experiment to be added to this handler
        """
        tmp = ExperimentResource(collection_name = self.collection_name, name = experiment_name)
        exp = self.rmt.get_project(tmp)
        self.experiment = exp
#         return exp
    
    def get_experiment(self):
        """
        Return the currently selected experiment for this handler
        """
        if hasattr(self,'experiment'):
            
            return self.experiment
        else:
            raise AttributeError('No experiment exists. First, select an experiment using select_experiment')
            
            
    def list_channels(self):
        """
        List all channel in currently selected experiment
        """
        return self.rmt.list_channels(self.collection_name, self.experiment.name)
        
    def select_channel(self, channel_name):
        
        """
        Select a channel to be added to this handler
        """
        self.channel = self.rmt.get_channel(chan_name= channel_name, coll_name=self.collection_name, exp_name=self.experiment.name)
    
    def get_coordinate_frame(self):
        """
        Get current experiment's coordinate frame
        """
        tmp = CoordinateFrameResource(name=self.experiment.coord_frame)
        coor = self.rmt.get_project(tmp)
        self.coordinate_frame = coor
        return coor
    
    def get_all(self):
        """
        Get a the entire channel image data at its native resolution
        """
        x_rng = [self.coordinate_frame.x_start , self.coordinate_frame.x_stop]
        y_rng = [self.coordinate_frame.y_start , self.coordinate_frame.y_stop]
        z_rng = [self.coordinate_frame.z_start , self.coordinate_frame.z_stop]
        
        return self.rmt.get_cutout(self.channel , 0 , x_rng , y_rng , z_rng)
        
    def get_cutout(self, x_range, y_range, z_range, resolution=0):
        """
        Return a cutout of the image data
        """
        return self.rmt.get_cutout(self.channel , resolution , x_range , y_range , z_range)
        
        
        
        
    
