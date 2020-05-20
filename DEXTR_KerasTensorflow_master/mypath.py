import os


class Path(object):
    @staticmethod
    def models_dir():
        if os.path.isdir('DEXTR_KerasTensorflow_master'):
            return 'DEXTR_KerasTensorflow_master/models/'
        else:
            return 'dextr/DEXTR_KerasTensorflow_master/models/'
