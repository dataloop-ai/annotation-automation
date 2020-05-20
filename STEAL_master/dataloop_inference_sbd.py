# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import torch
import numpy as np
import torch.utils.data
from STEAL_master.utils import dataloader
from STEAL_master.models.casenet import casenet101 as CaseNet101
import os
import cv2
import tqdm
import matplotlib.pyplot as plt

import argparse


def save_pred(im_info, predictions, num_cls, output_dir):
    org_height, org_width = im_info['orig_size']
    filename = os.path.basename(im_info['impath'][0])
    img_result_name = os.path.splitext(filename)[0] + '.png'
    im_list = list()
    for idx_cls in range(num_cls):
        score_pred = predictions.data.cpu().numpy()[0][idx_cls, 0:org_height, 0:org_width]
        im_list.append(score_pred)
        im = (score_pred * 255).astype(np.uint8)
        result_root = os.path.join(output_dir, 'class_' + str(idx_cls + 1))
        if not os.path.exists(result_root):
            os.makedirs(result_root)
        cv2.imwrite(os.path.join(result_root, img_result_name), im)
    return im_list


def do_test_sbd(net_, val_data_loader_, cuda, output_folder, n_classes):
    print('Running Inference....')
    net_.eval()
    output_images_dir = os.path.join(output_folder, 'val_images')

    if not os.path.isdir(output_images_dir):
        os.makedirs(output_images_dir)

    for i_batch, (im_info, input_img, input_gt) in tqdm.tqdm(enumerate(val_data_loader_), total=len(val_data_loader_)):
        if cuda:
            im = input_img.cuda(async=True)
        else:
            im = input_img

        out_masks = net_(im)

        prediction = torch.sigmoid(out_masks[0])
        save_pred(im_info, prediction, n_classes, output_images_dir)

    return 0


def main():
    # parser = argparse.ArgumentParser()
    #
    # parser.add_argument('--root_dir_val', type=str, default='./data/sbd/data_aug/')
    # parser.add_argument('--flist_val', type=str, default='./data/sbd/data_aug/val_list.txt')
    #
    # parser.add_argument('--ckpt',
    #                     type=str,
    #                     default='./checkpoints/sbd/model_checkpoint.pt')
    #
    # parser.add_argument('--output_folder', type=str,
    #                     default='./output/sbd/')
    # parser.add_argument('--dataset', type=str, default='sbd')
    #
    # parser.add_argument('--n_classes', type=int, default=20)
    #
    # args = parser.parse_args()

    class temp:
        img_filepath = r'E:\Shabtay\fonda_pytorch\STEAL_master\demo\images\augsburg_000000_001266_leftImg8bit.png'
        img_filepath = r'E:\Shabtay\fonda_pytorch\STEAL_master\demo\images\_Arlozorov1_0000000013.png'
        ckpt = r'E:\Shabtay\fonda_pytorch\STEAL_master\checkpoints\sbd\model_checkpoint.pt'
        dataset = 'sbd'
        n_classes = 20
        output_folder = r'E:\Shabtay\fonda_pytorch\STEAL_master\output'

    args = temp()
    print('****')
    print(args)
    print('****')

    output_folder = args.output_folder
    ckpt = args.ckpt

    # --
    n_classes = args.n_classes
    crop_size_val = 512

    def _read_image(input_image_path):
        crop_size = crop_size_val
        mean_value = (104.008, 116.669, 122.675)  # BGR
        original_im = cv2.imread(input_image_path).astype(np.float32)
        in_ = original_im
        width, height = in_.shape[1], in_.shape[0]
        if crop_size < width or crop_size < height:
            in_ = cv2.resize(in_, (0,0), fx=0.5, fy=0.5)
            print('Input image size must be smaller than crop size!')
        elif crop_size == width and crop_size == height:
            # ("WARNING *** skipping because of crop_size ")
            pass
        else:
            pad_x = crop_size - width
            pad_y = crop_size - height
            in_ = cv2.copyMakeBorder(in_, 0, pad_y, 0, pad_x, cv2.BORDER_CONSTANT, value=mean_value)
        in_ -= np.array(mean_value)
        in_ = in_.transpose((2, 0, 1))  # HxWx3 -> 3xHxW
        return in_, original_im

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    net = CaseNet101()
    net = torch.nn.DataParallel(net.cuda())

    print('loading ckpt :%s' % ckpt)
    net.load_state_dict(torch.load(ckpt), strict=True)

    print('Running Inference....')
    net.eval()
    output_images_dir = os.path.join(output_folder, 'val_images')

    if not os.path.isdir(output_images_dir):
        os.makedirs(output_images_dir)

    # val set
    im, orig = _read_image(args.img_filepath)
    im_info = {'impath': [args.img_filepath],
               'orig_size': im.shape[1:]}

    im = torch.from_numpy(im).cuda()
    im = im.unsqueeze(dim=0)  # 1x3xHxW

    out_masks = net(im)

    prediction = torch.sigmoid(out_masks[0])
    preds = save_pred(im_info, prediction, n_classes, output_images_dir)

    plt.figure()
    plt.imshow(np.max(preds, axis=0))
    plt.figure()
    plt.imshow(orig.astype(np.uint8))


if __name__ == '__main__':
    # main()
    pass