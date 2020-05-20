import torch
import numpy as np
import torch.utils.data
from STEAL_master.models.casenet import casenet101 as CaseNet101
from STEAL_master.coarse_to_fine.input_reader import InputReaderSemMatDemo
import cv2
from PIL import Image
from STEAL_master.contours import ContourBox
import STEAL_master.utils.vis_utils as vs_utils
import os
import tqdm

from STEAL_master.utils.VisualizerBox import VisualizerBox
import matplotlib.pyplot as plt


# For fair comparison, this is inspired by the way CASENET inference procedure works
def do_test(net, output_folder, image_list, n_classes=19, image_h=1024, image_w=2048, patch_h=512, patch_w=512,
            step_size_y=256, step_size_x=256, pad=16):
    num_cls = n_classes
    # image_h = 1024  # Need to pre-determine test image size
    # image_w = 2048  # Need to pre-determine test image size

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

        # if (height != image_h) or (width != image_w):
        #     raise ValueError('Input image size must be' + str(image_h) + 'x' + str(image_w) + '!')

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


def d_platform():
    import dtlpy as dlp
    item = dlp.datasets.get(dataset_id='196d4fd9-49af-45a7-93f9-5ade105bcd21').items.get(
        item_id='5d1367071720f11c5751c960')
    # item = dlp.datasets.get(dataset_id='5cb858b41e25dd0012d4c97b').items.get(item_id='5cb861b36896f900123d086b')
    annotations = item.annotations.list()
    mask = annotations.annotations[0].show(color=(1,), thickness=-1)
    return item.download(), mask


# Define input file path and coarse annotation file.
image_paths = [r'E:\Shabtay\fonda_pytorch\STEAL_master\demo\coarse_to_fine\augsburg_000000_001266_leftImg8bit.png']
# image_paths = [r"E:\Shabtay\fonda_pytorch\STEAL_master\demo\coarse_to_fine\_Arlozorov1_0000000013_.png"]

coarse_gt_paths = [r'E:\Shabtay\fonda_pytorch\STEAL_master\demo\coarse_to_fine\augsburg_000000_001266_leftImg8bit.mat']
classes_to_refine = [11, 12, 13, 14, 15, 16, 17, 18]

# Initializing network and pretrained model.
ckpt = r'E:\Shabtay\fonda_pytorch\STEAL_master\checkpoints\cityscapes\model_checkpoint.pt'
n_classes = 19
# ckpt = r'E:\Shabtay\fonda_pytorch\STEAL_master\checkpoints\sbd\model_checkpoint.pt'
# n_classes = 20


net = CaseNet101(nclasses=n_classes)
net = torch.nn.DataParallel(net.cuda())
net.load_state_dict(torch.load(ckpt), strict=True)

## Initializing Contour Box
level_set_config_dict = {
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

cbox = ContourBox.LevelSetAlignment(n_workers=1,
                                    fn_post_process_callback=None,
                                    config=level_set_config_dict)

###################

import dtlpy as dlp

dataset = dlp.datasets.get(dataset_id='5d1df0db242027b83cd5ae92')
item = dataset.items.get(item_id='5d1df2763370a305ce348d9e')

annotations = item.annotations.list()
for annotation in annotations:
    print(annotation.type)
    if annotation.type == 'binary':
        matched_annotation = annotation
        break

show = True
if False:
    rgb_image = Image.open(item.download())
    rgb_image = np.asarray(rgb_image)
    if show is True:
        plt.figure()
        plt.imshow(rgb_image)

    crop_top = matched_annotation.left - 32
    crop_bottom = matched_annotation.right + 32
    crop_left = matched_annotation.top - 32
    crop_right = matched_annotation.bottom + 32

    rgb_crop = rgb_image[crop_top: crop_bottom, crop_left:crop_right]
    mask_crop = matched_annotation.geo[crop_top: crop_bottom, crop_left:crop_right]
else:
    rgb_image = Image.open(r"E:\Shabtay\fonda_pytorch\ESPNet_master\data\augsburg_000000_001266_leftImg8bit.png")
    rgb_image = np.asarray(rgb_image)
    rgb_crop = rgb_image
    mask_crop = Image.open(r"E:\Shabtay\fonda_pytorch\ESPNet_master\results\augsburg_000000_001266_leftImg8bit.png")
    mask_crop = np.asarray(mask_crop)
    mask_crop = cv2.resize(mask_crop, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_NEAREST)

rgb_crop = cv2.resize(rgb_crop, tuple(np.multiply(np.round(np.divide(rgb_crop.shape[:2], 8)), 8).astype(int))[::-1])
mask_crop = cv2.resize(mask_crop, tuple(np.multiply(np.round(np.divide(mask_crop.shape[:2], 8)), 8).astype(int))[::-1])
if show is True:
    plt.figure()
    plt.imshow(rgb_crop)
    plt.figure()
    plt.imshow(mask_crop)
###################
# Inference and resizing.

if True:
    # pred = do_test(net, output_folder=None, image_list=[rgb_crop], n_classes=n_classes, patch_w=256, patch_h=256)[0]
    pred = do_test(net, output_folder=None, image_list=[rgb_crop], n_classes=n_classes,
                   image_h=rgb_crop.shape[0], image_w=rgb_crop.shape[1],
                   patch_h=rgb_crop.shape[0], patch_w=rgb_crop.shape[1],
                   pad=32)[0]
    pred = cv2.resize(pred, (0, 0), fx=1, fy=1, interpolation=cv2.INTER_NEAREST).transpose((2, 0, 1))

else:
    pred = cv2.imread(r"E:\Shabtay\fonda_tensorpack\HED\out-fused.png", 0) / 255.
    pred = pred[crop_top: crop_bottom, crop_left:crop_right]
    pred = cv2.resize(pred, tuple(np.multiply(np.round(np.divide(pred.shape[:2], 8)), 8).astype(int))[::-1])
    pred = pred[None, ...]

plt.figure()
plt.imshow(np.max(pred, axis=2))
# ---

# Reading Coarse GT and Removing ignore classes.
# seg_coarse = [cv2.resize(mask_crop, (2048, 1024), interpolation=cv2.INTER_NEAREST)]
seg_coarse = [mask_crop==i for i in range(30)]
pred_ = [np.max(pred, axis=2)for i in range(30)]

# Alignment
output, _ = cbox({'seg': np.expand_dims(seg_coarse, 0), 'bdry': None},
                 np.expand_dims(pred_, 0))

# Visualization
vis_box = VisualizerBox(dataset_color='css4_fushia', plt_backend=None, fig_size=(15, 40))
vis_box.set_output_folder(os.path.join('./output_dir', 'demo', 'vis'))

plot_pairs = {'Coarse Label': seg_coarse, 'Semantic Edges': np.max(pred_, axis=0, keepdims=True)}

for vis_step in range(len(level_set_config_dict['step_ckpts'])):
    masks_step = output[0, :, vis_step, :, :]
    vis_step = level_set_config_dict['step_ckpts'][vis_step]
    plot_pairs['(step_%.2i)' % vis_step] = masks_step

vis_box.visualize(plot_pairs, background=rgb_image,
                  grid=True, merge_channels=True)

plt.figure()
plt.imshow(output[0, 0, -1, :, :])

plt.figure()
plt.imshow(np.argmax(output[0, :, -1, :, :], axis=0).astype(np.uint8))
