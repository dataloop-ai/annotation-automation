from DEXTR_KerasTensorflow_master.helpers import helpers as helpers
from DEXTR_KerasTensorflow_master.networks.dextr import DEXTR
from PIL import Image, ImageFile
from keras import backend as K
import tensorflow as tf
import numpy as np
import dtlpy as dl
import traceback
import logging
import time
import cv2

ImageFile.LOAD_TRUNCATED_IMAGES = True


class FourPointToSegmentationHandler:
    def __init__(self, project_name, package_name):
        # download weights
        project = dl.projects.get(project_name=project_name)
        artifact = project.artifacts.get(package_name=package_name, artifact_name='dextr_pascal-sbd.h5')
        artifact.download(local_path='weights/dextr')

        self.logger = None
        self.graph = None
        self.net = None
        self.pad = 50
        self.thresh = 0.8
        self.logger = logging.getLogger()
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

        # load model
        # Handle input and output args
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        self.graph = tf.get_default_graph()
        self.sess = tf.Session(config=config, graph=self.graph)
        K.set_session(self.sess)
        with self.sess.as_default():
            self.net = DEXTR(nb_classes=1,
                             resnet_layers=101,
                             input_shape=(512, 512),
                             weights_filepath='weights/dextr/dextr_pascal-sbd.h5',
                             num_input_channels=4,
                             classifier='psp',
                             sigmoid=True)

        # noinspection PyProtectedMember
        self.net.model._make_predict_function()

    # noinspection PyUnusedLocal
    def run(self, progress, item, annotations, config=None):
        progress.logger.info('updating progress')
        progress.update(message='downloading image')
        progress.logger.info('downloading image')
        buffer = item.download(save_locally=False)
        return self.execute(buffer=buffer,
                            progress=progress,
                            annotations=annotations,
                            config=config,
                            item=item)

    @staticmethod
    def check_input_config(config):
        config_default = {'input_type': 'id',
                          'annotation_type': 'binary',
                          'return_action': 'post',
                          'env': 'prod'}
        if config is None:
            return config_default

        if 'annotation_type' not in config:
            config['annotation_type'] = config_default['annotation_type']
        if 'return_action' not in config:
            config['return_action'] = config_default['return_action']
        if 'input_type' not in config:
            config['input_type'] = config_default['input_type']
        if 'env' not in config:
            config['env'] = config_default['env']
        return config

    def predict(self, image, points):
        """

        :param image:
        :param points:
        :return:
        """
        #  Crop image to the bounding box from the extreme points and resize
        bbox = helpers.get_bbox(image, points=points, pad=self.pad, zero_pad=True)
        crop_image = helpers.crop_from_bbox(image, bbox, zero_pad=True)
        resize_image = helpers.fixed_resize(crop_image, (512, 512)).astype(np.float32)

        #  Generate extreme point heat map normalized to image values
        extreme_points = points - [np.min(points[:, 0]), np.min(points[:, 1])] + [self.pad, self.pad]
        extreme_points = (512 * extreme_points * [1 / crop_image.shape[1], 1 / crop_image.shape[0]]).astype(np.int)
        extreme_heatmap = helpers.make_gt(resize_image, extreme_points, sigma=10)
        extreme_heatmap = helpers.cstm_normalize(extreme_heatmap, 255)

        #  Concatenate inputs and convert to tensor
        input_dextr = np.concatenate((resize_image, extreme_heatmap[:, :, np.newaxis]), axis=2)

        # Run a forward pass
        pred = self.net.model.predict(input_dextr[np.newaxis, ...])[0, :, :, 0]
        result = helpers.crop2fullmask(pred, bbox, im_size=image.shape[:2], zero_pad=True, relax=self.pad) > self.thresh

        return result

    def execute(self, buffer, progress, annotations, item, config=None):
        try:
            config = self.check_input_config(config)
            timing_dict = dict()
            tic = time.time()
            image = np.asarray(Image.open(buffer))
            timing_dict['loading_img'] = time.time() - tic
            if len(image.shape) == 2:  # not channels:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            else:
                pass

            progress.logger.info('updating progress')
            progress.update(message='running models')
            progress.logger.info('running models')
            #############
            # Run DEXTR #
            #############
            count = 1
            assert isinstance(item, dl.Item)
            annotation_builder = item.annotations.builder()
            for annotation in annotations:
                attributes = annotation['attributes']
                label = annotation['label']
                coordinates = annotation['coordinates']

                progress.logger.info('annotation {}/{}'.format(count, len(annotations)))

                pts = [[cor['x'], cor['y']] for cor in coordinates]
                points = np.round(pts).astype(np.int)
                tic = time.time()
                with self.sess.as_default():
                    mask = self.predict(image, points)
                timing_dict['model_operation_%d' % count] = time.time() - tic
                count += 1
                #####################
                # Create annotation #
                #####################
                if config['annotation_type'] == 'segment':
                    annotation_builder.add(annotation_definition=dl.Polygon.from_segmentation(
                        mask=mask,
                        label=label,
                        attributes=attributes))

                elif config['annotation_type'] == 'binary':
                    annotation_builder.add(annotation_definition=dl.Segmentation(
                        geo=mask,
                        label=label))
                else:
                    self.logger.exception('Unknown annotation type: %s' % config['annotation_type'])
                    raise ValueError
            annotation_builder.upload()
            progress.logger.info('updating progress')
            progress.update(message='done')
            progress.logger.info('done')

        except Exception as err:
            self.logger.exception('%s\n%s' % (err, traceback.format_exc()))
            raise
