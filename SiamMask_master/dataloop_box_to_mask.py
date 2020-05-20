import glob
import matplotlib.pyplot as plt
import time
import numpy as np
import cv2
import os
from SiamMask_master.utils.load_helper import load_pretrain
from SiamMask_master.tools.test import generate_anchor, get_subwindow_tracking, siamese_track
import torch
from torch.autograd import Variable
from SiamMask_master.utils.tracker_config import TrackerConfig
from SiamMask_master.utils.config_helper import load_config
from collections import namedtuple

from dtlpy import Annotation, Segmentation, Polygon, Segmentation


class Siam:
    def __init__(self):
        class temp:
            resume = r'E:\Shabtay\fonda_pytorch\SiamMask_master\models\SiamMask_DAVIS.pth'  # ,help='path to latest checkpoint (default: none)')
            config = r'E:\Shabtay\fonda_pytorch\SiamMask_master\experiments\siammask_sharp\config_davis.json'  # help='hyper-parameter of SiamMask in json format')
            # base_path=r'E:\Shabtay\fonda_pytorch\SiamMask_master\data\tennis' # help='datasets')
            base_path = r'C:\Users\Dataloop\.dataloop\datasets\5ce2b280fe3b45001c2c7a51\image\Arlozorov1'
            cpu = False  # help='cpu mode')

        args = temp()
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.backends.cudnn.benchmark = True

        # Setup Model
        self.cfg = load_config(args)
        from SiamMask_master.experiments.siammask_sharp.custom import Custom

        self.siammask = Custom(anchors=self.cfg['anchors'])
        if args.resume:
            assert os.path.isfile(args.resume), 'Please download {} first.'.format(args.resume)
            self.siammask = load_pretrain(self.siammask, args.resume)

        self.siammask.eval().to(self.device)

    def main(self, item, annotation):
        filepath = item.download(overwrite=True)
        im = cv2.imread(filepath)

        # Select ROI
        # cv2.namedWindow("SiamMask", cv2.WND_PROP_FULLSCREEN)
        # cv2.setWindowProperty("SiamMask", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        try:
            x = annotation.coordinates[0]['x']
            y = annotation.coordinates[0]['y']
            w = annotation.coordinates[1]['x'] - x
            h = annotation.coordinates[1]['y'] - y
        except:
            assert False

        tic = time.time()
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
        avg_chans = np.mean(im, axis=(0, 1))

        wc_z = target_sz[0] + p.context_amount * sum(target_sz)
        hc_z = target_sz[1] + p.context_amount * sum(target_sz)
        s_z = round(np.sqrt(wc_z * hc_z))
        # initialize the exemplar
        z_crop = get_subwindow_tracking(im, target_pos, p.exemplar_size, s_z, avg_chans)

        z = Variable(z_crop.unsqueeze(0))
        self.siammask.template(z.to(self.device))

        if p.windowing == 'cosine':
            window = np.outer(np.hanning(p.score_size), np.hanning(p.score_size))
        elif p.windowing == 'uniform':
            window = np.ones((p.score_size, p.score_size))
        window = np.tile(window.flatten(), p.anchor_num)

        state['p'] = p
        state['net'] = self.siammask
        state['avg_chans'] = avg_chans
        state['window'] = window
        state['target_pos'] = target_pos
        state['target_sz'] = target_sz

        state = siamese_track(state, im, mask_enable=True, refine_enable=True, device=self.device)  # track
        location = state['ploygon'].flatten()
        mask = state['mask'] > state['p'].seg_thr

        im[:, :, 2] = (mask > 0) * 255 + (mask == 0) * im[:, :, 2]
        cv2.polylines(im, [np.int0(location).reshape((-1, 1, 2))], True, (0, 255, 0), 3)
        # cv2.imshow('SiamMask', im)
        # key = cv2.waitKey(1)
        # if key > 0:
        #    break
        # plt.figure()
        # plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        # plt.figure()
        # plt.imshow(mask)
        builder = item.annotations.builder()
        builder.add(Segmentation(geo=mask,
                                 label=annotation.label))
        item.annotations.upload(builder)
        toc = time.time() - tic
        print('SiamMask Time: {:02.1f}s Speed:'.format(toc))


def test():
    # copy item
    import dtlpy as dlp

    # def c():
    #     dataset_from = dlp.datasets.get(dataset_id='5ce2b280fe3b45001c2c7a51')
    #     dataset_to = dlp.datasets.get(dataset_id='196d4fd9-49af-45a7-93f9-5ade105bcd21')
    #     item = dataset_from.items.get(item_id='5ce2c7696d144c001baf2a79')
    #     buffer = item.download()
    #     new_item = dataset_to.items.upload(filepath=buffer)
    #     new_item.annotations.upload(item.annotations.list().to_json())


    a = Siam()



    dataset = dlp.datasets.get(dataset_id='5d1df0db242027b83cd5ae92')
    item = dataset.items.get(item_id='5d1df2763370a305ce348d9e')

    #
    buffer = item.download(save_locally=False)

    annotations = item.annotations.list()
    for annotation in annotations:
        # print(annotation)
        if annotation.type == 'box':
            a.main(item=item, annotation=annotation)


# test()
