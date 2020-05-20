from SiamMask_master.tools.test import generate_anchor, get_subwindow_tracking, siamese_track
from SiamMask_master.experiments.siammask_sharp.custom import Custom
from STEAL_master.models.casenet import casenet101 as CaseNet101
from SiamMask_master.utils.tracker_config import TrackerConfig
from SiamMask_master.utils.load_helper import load_pretrain
from SiamMask_master.utils.config_helper import load_config
from STEAL_master.utils.VisualizerBox import VisualizerBox
from STEAL_master.contours import ContourBox
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.utils.data
import numpy as np
import dtlpy as dl
import matplotlib
import logging
import torch
import tqdm
import time
import cv2
import os

matplotlib.use('TkAgg')
logger = logging.getLogger(__name__)


class BBoxToSegmentationHandler:

    def __init__(self, project_name, package_name):
        # init params
        self.show = False
        self.device = None
        self.cfg = None
        self.siammask = None
        self.n_classes = None
        self.net = None
        self.level_set_config_dict = None
        self.cbox = None

        project = dl.projects.get(project_name=project_name)

        artifact = project.artifacts.get(package_name=package_name, artifact_name='cityscapes_checkpoint.pt')
        artifact.download(local_path='weights/steal')

        artifact = project.artifacts.get(package_name=package_name, artifact_name='SiamMask_DAVIS.pth')
        artifact.download(local_path='weights/siam')

        # init models
        self.init_steal_model(weights_filepath='weights/steal/cityscapes_checkpoint.pt')
        self.init_siam_model(weights_filepath='weights/siam/SiamMask_DAVIS.pth',
                             config_filepath='weights/siam/config_davis.json')

    def init_steal_model(self, weights_filepath):
        # Initializing network and pretrained model.
        self.n_classes = 19

        net = CaseNet101(nclasses=self.n_classes)
        net = torch.nn.DataParallel(net.cuda())
        net.load_state_dict(torch.load(weights_filepath), strict=True)
        self.net = net

        # Initializing Contour Box
        self.level_set_config_dict = {
            'step_ckpts': [0, 30, 50, 120],
            'lambda_': 0.0,
            'alpha': 1,
            'smoothing': 1,
            'render_radius': -1,
            'is_gt_semantic': True,
            'method': 'MLS',
            'balloon': 0,
            'threshold': 0.99,
            'merge_weight': 0.5
        }

        self.cbox = ContourBox.LevelSetAlignment(n_workers=1,
                                                 fn_post_process_callback=None,
                                                 config=self.level_set_config_dict)

    def init_siam_model(self, weights_filepath, config_filepath):
        class TempLoader:
            resume = weights_filepath
            config = config_filepath
            cpu = False

        args = TempLoader()
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.backends.cudnn.benchmark = True

        # Setup Model
        self.cfg = load_config(args)

        self.siammask = Custom(anchors=self.cfg['anchors'])
        self.siammask = load_pretrain(self.siammask, args.resume)
        self.siammask.eval().to(self.device)

    # For fair comparison, this is inspired by the way CASENET inference procedure works
    @staticmethod
    def do_test(net, output_folder, image_list, n_classes=19, image_h=1024, image_w=2048, patch_h=512, patch_w=512,
                step_size_y=256, step_size_x=256, pad=16):
        num_cls = n_classes

        net.eval()
        if output_folder is not None:
            output_images_dir_iter = os.path.join(output_folder, 'val_images')
            if not os.path.isdir(output_images_dir_iter):
                os.makedirs(output_images_dir_iter)

        if ((2 * pad) % 8) != 0:
            raise ValueError('Pad number must be able to be divided by 8!')
        step_num_y = (image_h - patch_h + 0.0) / step_size_y
        step_num_x = (image_w - patch_w + 0.0) / step_size_x

        if round(step_num_y) != step_num_y:
            raise ValueError('Vertical sliding size can not be divided by step size!')

        if round(step_num_x) != step_num_x:
            raise ValueError('Horizontal sliding size can not be divided by step size!')

        step_num_y = int(step_num_y)
        step_num_x = int(step_num_x)
        mean_value = (104.008, 116.669, 122.675)  # BGR

        pred_set = []  # only used if output_folder is none.
        for idx_img in tqdm.tqdm(range(len(image_list))):
            in_ = image_list[idx_img].astype(np.float32)
            width, height, chn = in_.shape[1], in_.shape[0], in_.shape[2]
            im_array = cv2.copyMakeBorder(in_, pad, pad, pad, pad, cv2.BORDER_REFLECT)

            # Perform patch-by-patch testing
            score_pred = np.zeros((height, width, num_cls))
            mat_count = np.zeros((height, width, 1))
            for i in range(0, step_num_y + 1):
                offset_y = i * step_size_y
                for j in range(0, step_num_x + 1):
                    offset_x = j * step_size_x

                    # crop overlapped regions from the image
                    in_ = np.array(
                        im_array[offset_y:offset_y + patch_h + 2 * pad, offset_x:offset_x + patch_w + 2 * pad, :])
                    in_ -= np.array(mean_value)
                    in_ = in_.transpose((2, 0, 1))  # HxWx3 -> 3xHxW
                    # ---
                    in_ = torch.from_numpy(in_).cuda()
                    in_ = in_.unsqueeze(dim=0)  # 1x3xHxW

                    out_masks = net(in_)  #
                    prediction = torch.sigmoid(out_masks[0]).data.cpu().numpy()[0]
                    # add the prediction to score_pred and increase count by 1
                    score_pred[offset_y:offset_y + patch_h, offset_x:offset_x + patch_w, :] += \
                        np.transpose(prediction, (1, 2, 0))[pad:-pad, pad:-pad, :]
                    mat_count[offset_y:offset_y + patch_h, offset_x:offset_x + patch_w, 0] += 1.0

            score_pred = np.divide(score_pred, mat_count)

            if output_folder is None:
                pred_set.append(score_pred)
                continue

        return pred_set  # empty if output_folder exits

    def inference_initial_segment(self, im, x, y, w, h):
        target_pos = np.array([x + w / 2, y + h / 2])
        target_sz = np.array([w, h])
        state = dict()
        state['im_h'] = im.shape[0]
        state['im_w'] = im.shape[1]
        p = TrackerConfig()
        p.update(self.cfg['hp'], self.siammask.anchors)

        p.renew()

        p.scales = self.siammask.anchors['scales']
        p.ratios = self.siammask.anchors['ratios']
        p.anchor_num = self.siammask.anchor_num
        p.anchor = generate_anchor(self.siammask.anchors, p.score_size)
        avg_channels = np.mean(im, axis=(0, 1))

        wc_z = target_sz[0] + p.context_amount * sum(target_sz)
        hc_z = target_sz[1] + p.context_amount * sum(target_sz)
        s_z = round(np.sqrt(wc_z * hc_z))
        # initialize the exemplar
        z_crop = get_subwindow_tracking(im, target_pos, p.exemplar_size, s_z, avg_channels)

        z = Variable(z_crop.unsqueeze(0))
        self.siammask.template(z.to(self.device))

        if p.windowing == 'cosine':
            window = np.outer(np.hanning(p.score_size), np.hanning(p.score_size))
        elif p.windowing == 'uniform':
            window = np.ones((p.score_size, p.score_size))
        window = np.tile(window.flatten(), p.anchor_num)

        state['p'] = p
        state['net'] = self.siammask
        state['avg_chans'] = avg_channels
        state['window'] = window
        state['target_pos'] = target_pos
        state['target_sz'] = target_sz

        state = siamese_track(state, im, mask_enable=True, refine_enable=True, device=self.device)  # track
        return state

    def run_edge_fine_tune(self, rgb_image, segment_annotation):
        if self.show:
            plt.figure()
            plt.imshow(rgb_image)

        crop_top = segment_annotation.left - 32
        crop_bottom = segment_annotation.right + 32
        crop_left = segment_annotation.top - 32
        crop_right = segment_annotation.bottom + 32

        crop_top = np.maximum(crop_top, 0)
        crop_left = np.maximum(crop_left, 0)
        crop_bottom = np.minimum(crop_bottom, segment_annotation.geo.shape[0])
        crop_right = np.minimum(crop_right, segment_annotation.geo.shape[1])

        rgb_crop = rgb_image[crop_top: crop_bottom, crop_left:crop_right]
        mask_crop = segment_annotation.geo[crop_top: crop_bottom, crop_left:crop_right].astype(np.uint8)

        crop_h, crop_w = rgb_crop.shape[:2]

        if self.show:
            plt.figure()
            plt.imshow(rgb_crop)
            plt.figure()
            plt.imshow(mask_crop)

        rgb_crop = cv2.resize(rgb_crop,
                              tuple(np.multiply(np.round(np.divide(rgb_crop.shape[:2], 8)), 8).astype(int))[::-1])
        mask_crop = cv2.resize(mask_crop,
                               tuple(np.multiply(np.round(np.divide(mask_crop.shape[:2], 8)), 8).astype(int))[::-1])

        ###################
        # Inference and resizing.
        tic = time.time()
        pred = self.do_test(net=self.net,
                            output_folder=None,
                            image_list=[rgb_crop],
                            n_classes=self.n_classes,
                            image_h=rgb_crop.shape[0],
                            image_w=rgb_crop.shape[1],
                            patch_h=rgb_crop.shape[0],
                            patch_w=rgb_crop.shape[1],
                            pad=32)[0]
        print('------------------{}'.format(time.time() - tic))
        pred = cv2.resize(pred, (0, 0), fx=1, fy=1, interpolation=cv2.INTER_NEAREST).transpose((2, 0, 1))
        if self.show:
            plt.figure()
            plt.imshow(np.max(pred, axis=0))

        # Reading Coarse GT and Removing ignore classes.
        seg_coarse = [mask_crop]
        pred_ = np.max(pred, axis=0)[None, ...]

        # Alignment
        tic = time.time()
        output, _ = self.cbox({'seg': np.expand_dims(seg_coarse, 0), 'bdry': None},
                              np.expand_dims(pred_, 0))
        print('------------------{}'.format(time.time() - tic))

        # Visualization
        if self.show:
            vis_box = VisualizerBox(dataset_color='css4_fushia', plt_backend=None, fig_size=(15, 40))
            vis_box.set_output_folder(os.path.join('./output_dir', 'demo', 'vis'))

            plot_pairs = {'Coarse Label': seg_coarse, 'Semantic Edges': np.max(pred_, axis=0, keepdims=True)}

            for vis_step in range(len(self.level_set_config_dict['step_ckpts'])):
                masks_step = output[0, :, vis_step, :, :]
                vis_step = self.level_set_config_dict['step_ckpts'][vis_step]
                plot_pairs['(step_%.2i)' % vis_step] = masks_step

            vis_box.visualize(plot_pairs, background=rgb_image,
                              grid=True, merge_channels=True)

            plt.figure()
            plt.imshow(output[0, 0, -1, :, :])

        final = np.zeros(rgb_image.shape[:2])
        final[crop_top: crop_bottom, crop_left:crop_right] = cv2.resize(output[0, 0, -1, :, :],
                                                                        (crop_w, crop_h),
                                                                        interpolation=cv2.INTER_NEAREST)
        return final

    def run(self, item, annotations, config=None, progress=None):
        progress.logger.info('GPU available: {}'.format(torch.cuda.is_available()))
        if config is None:
            config = {}
        if 'return_type' not in config:
            config['return_type'] = 'segment'

        if config['return_type'] not in ['segment', 'binary']:
            raise ValueError('unknown return type: {}'.format(config['return_type']))

        tic_total = time.time()
        runtime_annotation = list()
        runtime_siam = list()
        runtime_steal = list()
        progress.logger.info('updating progress')
        progress.update(message='downloading item')
        progress.logger.info('downloading item')
        filepath = item.download(overwrite=True)
        try:
            im = cv2.imread(filepath)
            progress.logger.info('updating progress')
            progress.update(message='running model')
            progress.logger.info('running model')

            count = 1
            annotation_builder = item.annotations.builder()
            for annotation in annotations:
                coordinates = annotation['coordinates']
                label = annotation['label']
                attributes = annotation['attributes']

                count += 1
                tic_annotation = time.time()
                x = coordinates[0]['x']
                y = coordinates[0]['y']
                w = coordinates[1]['x'] - x
                h = coordinates[1]['y'] - y

                # initial segmentation
                tic_siam = time.time()
                state = self.inference_initial_segment(im=im, x=x, y=y, w=w, h=h)
                runtime_siam.append(time.time() - tic_siam)

                mask = state['mask'] > state['p'].seg_thr
                # coarse to fine
                tic_steal = time.time()
                final = 1 * mask
                runtime_steal.append(time.time() - tic_steal)

                ##########
                # Upload #
                ##########
                runtime_annotation.append(time.time() - tic_annotation)

                if config['return_type'] == 'binary':
                    annotation_builder.add(annotation_definition=dl.Segmentation(geo=final,
                                                                                 label=label,
                                                                                 attributes=attributes))

                elif config['return_type'] == 'segment':
                    annotation_builder.add(annotation_definition=dl.Polygon.from_segmentation(mask=final,
                                                                                              label=label,
                                                                                              attributes=attributes))
                else:
                    raise ValueError('Unknown return type: {}'.format(config['return_type']))

            annotation_builder.upload()

        finally:
            if os.path.isfile(filepath):
                os.remove(filepath)
        progress.logger.info('updating progress')
        progress.update(message='done')
        progress.logger.info('done')
        runtime_total = time.time() - tic_total
        progress.logger.info('Runtime:')
        progress.logger.info('Total: {:02.1f}s'.format(runtime_total))
        progress.logger.info('Mean annotations: {:02.1f}s'.format(np.mean(runtime_annotation)))
        progress.logger.info('Mean Siam: {:02.1f}s'.format(np.mean(runtime_siam)))
        progress.logger.info('Mean STEAL: {:02.1f}s'.format(np.mean(runtime_steal)))
        progress.logger.info('Num annotations: {}'.format(len(runtime_annotation)))
