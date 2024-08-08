
import os


class MyPath(object):
    @staticmethod
    def db_root_dir(database=''):
        db_names = {'msl', 'smap', 'smd', 'power', 'yahoo', 'kpi', 'swat', 'wadi'}
        assert(database in db_names)

        if database == 'msl' or database == 'smap':
            return 'datasets/MSL_SMAP'
        elif database == 'power':
            return 'datasets/Power'
        elif database == 'yahoo':
            return 'datasets/yahoo'
        elif database == 'smd':
            return 'datasets/SMD'
        elif database == 'swat':
            return 'datasets/SWAT'
        elif database == 'wadi':
            return 'datasets/WADI'
        elif database == 'kpi':
            return 'datasets/KPI'
        
        else:
            raise NotImplementedError
