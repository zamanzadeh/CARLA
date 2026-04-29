
import os


class MyPath(object):
    @staticmethod
    def db_root_dir(database=''):
        db_names = {'msl', 'smap', 'smd', 'power', 'yahoo', 'kpi', 'swat', 'wadi', 'gecco', 'swan', 'ucr'}
        assert(database in db_names)

        dataset_root = os.environ.get('CARLA_DATA_ROOT', r'd:\CARLA\datasets')


        if database == 'msl' or database == 'smap':
            return os.path.join(dataset_root, 'MSL_SMAP')
        elif database == 'ucr':
            return os.path.join(dataset_root, 'UCR')
        elif database == 'yahoo':
            return os.path.join(dataset_root, 'Yahoo')
        elif database == 'smd':
            return os.path.join(dataset_root, 'SMD')
        elif database == 'swat':
            return os.path.join(dataset_root, 'SWAT')
        elif database == 'wadi':
            return os.path.join(dataset_root, 'WADI')
        elif database == 'kpi':
            return os.path.join(dataset_root, 'KPI')
        elif database == 'swan':
            return os.path.join(dataset_root, 'Swan')
        elif database == 'gecco':
            return os.path.join(dataset_root, 'GECCO')

        else:
            raise NotImplementedError
