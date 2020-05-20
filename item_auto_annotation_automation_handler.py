from Mask_RCNN_master.mrcnn.config import Config
import Mask_RCNN_master.mrcnn.model as modellib
from keras import backend as K
import tensorflow as tf
from PIL import Image
import numpy as np
import dtlpy as dl
import logging
import json
import cv2

logger = logging.getLogger('dataloop.maskrcnn')


class ItemAutoAnnotationHandler:
    def __init__(self, project_name, package_name):
        # download artifacts and weights
        project = dl.projects.get(project_name=project_name)
        artifact = project.artifacts.get(package_name=package_name, artifact_name='mask_rcnn_coco.h5')
        artifact.download(local_path='weights/maskrcnn')

        self.class_names = None
        self.model = None
        self.graph = None
        tf.keras.backend.clear_session()
        K.clear_session()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        self.graph = tf.get_default_graph()
        self.sess = tf.Session(config=config, graph=self.graph)

        K.set_session(self.sess)
        with self.sess.as_default():
            self.load_model(model_filepath='weights/maskrcnn/mask_rcnn_coco.h5',
                            config_filepath='weights/maskrcnn/configurations.json')

    def run(self, progress, item, config=None):
        """
        Write your main plugin function here

        :param progress:
        :param config:
        :param item:
        :return:
        """
        progress.logger.info('updating progress')
        progress.update(message='downloading image')
        progress.logger.info('downloading image')
        buffer_batch = item.download(save_locally=False)

        # check inputs
        if not isinstance(buffer_batch, list):
            buffer_batch = [buffer_batch]
        if config is None:
            config = dict()
        if 'annotation_type' not in config:
            config['annotation_type'] = 'segment'
        if 'confidence_th' not in config:
            config['confidence_th'] = 0.50
        if 'output_action' not in config:
            config['output_action'] = 'dict'
        logger.info('input config: %s' % config)

        #########################
        # load buffer to images #
        #########################
        img_batch = [np.asarray(Image.open(buf)) for buf in buffer_batch]

        # run single img for inference
        for img in img_batch:
            if len(img.shape) > 2:
                if img.shape[2] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            # Run detection
            progress.logger.info('updating progress')
            progress.update(message='running model')
            progress.logger.info('running model')
            with self.sess.as_default():
                results = self.model.detect([img], verbose=0)

            # Visualize results
            r = results[0]

            annotation_builder = item.annotations.builder()
            for i_det in range(len(r['class_ids'])):
                if r['scores'][i_det] < config['confidence_th']:
                    continue
                label = self.class_names[r['class_ids'][i_det]]
                ret, thresh = cv2.threshold(r['masks'][:, :, i_det].astype(np.uint8), 0.5, 255, 0)
                if config['annotation_type'] == 'binary':
                    annotation_definition = dl.Segmentation(geo=thresh,
                                                            label=label)
                elif config['annotation_type'] == 'segment':
                    annotation_definition = dl.Polygon.from_segmentation(mask=thresh,
                                                                         label=label)
                else:
                    continue

                annotation_builder.add(annotation_definition=annotation_definition)
            annotation_builder.upload()

        progress.logger.info('updating progress')
        progress.update(message='done')
        progress.logger.info('done')

    @staticmethod
    def load_model_configuration(configurations):
        # load default config
        default_config = Config()
        # set attributes from configuration yml
        for key, value in configurations['model_config'].items():
            v = value['value']
            if value['type'] == '<class \'tuple\'>':
                v = tuple(v)
            if value['type'] == '<class \'numpy.ndarray\'>':
                v = np.asarray(v)
            setattr(default_config, key, v)
        default_config.__init__()
        return default_config

    @staticmethod
    def dump_model_configuration(configurations, model_config):
        attributes = [i for i in dir(model_config) if not callable(i)]
        items_list = [a for a in attributes if a == a.upper()]
        if 'model_config' not in configurations:
            configurations['model_config'] = dict()
        for key in items_list:
            try:
                value = getattr(model_config, key)
                v_type = str(type(value))
                if isinstance(value, np.ndarray):
                    value = [float(v) if type(v) == np.float64 else int(v) for v in value]
                configurations['model_config'][key] = {'type': v_type,
                                                       'value': value}
            except Exception as err:
                logger.warning(err)
                logger.warning('unable to add key to configuration:', key)
        return configurations

    def load_model(self, model_filepath, config_filepath):
        with open(config_filepath, 'r') as f:
            configurations = json.load(f)

        config = self.load_model_configuration(configurations)
        config.display()

        # For inference!!! one image at a time
        config.IMAGES_PER_GPU = 1
        config.__init__()
        ##################

        # Create model object in inference mode.
        model = modellib.MaskRCNN(mode='inference', model_dir='weights/maskrcnn/logs', config=config)

        # Load weights trained on MS-COCO
        model.load_weights(model_filepath, by_name=True)

        self.class_names = configurations['labels']
        self.model = model
        # noinspection PyProtectedMember
        self.model.keras_model._make_predict_function()
