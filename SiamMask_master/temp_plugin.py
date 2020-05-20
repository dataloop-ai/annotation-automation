# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
import glob
from SiamMask_master.tools.test import *
import time
import matplotlib.pyplot as plt


class temp:
    resume = r'E:\Shabtay\fonda_pytorch\SiamMask_master\models\SiamMask_DAVIS.pth'  # ,help='path to latest checkpoint (default: none)')
    config = r'E:\Shabtay\fonda_pytorch\SiamMask_master\experiments\siammask_sharp\config_davis.json'  # help='hyper-parameter of SiamMask in json format')
    # base_path=r'E:\Shabtay\fonda_pytorch\SiamMask_master\data\tennis' # help='datasets')
    # video_filepath = r"C:\Users\Dataloop\.dataloop\projects\Feb19_shelf_zed\datasets\try1\images\video\download.mp4"
    # video_filepath=r"C:\Users\Dataloop\.dataloop\projects\Eyezon_fixed\datasets\New Clips\clip2\ch34_25fps05.mp4"
    video_filepath = r"E:\Projects\Foresight\tracker\videoplayback.webm"
    cpu = False  # help='cpu mode')


args = temp()

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True

# Setup Model
cfg = load_config(args)
from SiamMask_master.experiments.siammask_sharp.custom import Custom

siammask = Custom(anchors=cfg['anchors'])
if args.resume:
    assert isfile(args.resume), 'Please download {} first.'.format(args.resume)
    siammask = load_pretrain(siammask, args.resume)

siammask.eval().to(device)

# Parse Image file
cap = cv2.VideoCapture(args.video_filepath)
assert cap.isOpened()
cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
ret, frame = cap.read()
# frame = cv2.resize(frame, (128, 128))

# Select ROI
cv2.namedWindow("SiamMask", cv2.WND_PROP_FULLSCREEN)
# cv2.setWindowProperty("SiamMask", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
try:
    init_rect = cv2.selectROI('SiamMask', frame, False, False)
    x, y, w, h = init_rect
except:
    exit()

toc = 0
count = 0
cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
show = True
mask_enable = False
debug = False
refine_enable = False

###########
# Init #

p = TrackerConfig()
p.update(cfg['hp'], siammask.anchors)
p.renew()

p.scales = siammask.anchors['scales']
p.ratios = siammask.anchors['ratios']
p.anchor_num = siammask.anchor_num
p.anchor = generate_anchor(siammask.anchors, p.score_size)

prev_frame = None
while cap.isOpened():
    ret, frame = cap.read()
    if prev_frame is None:
        prev_frame = frame
        ret, frame = cap.read()

    if not ret:
        break
    # frame = cv2.resize(frame, (128, 128))
    tic = time.time()

    ############
    # Template #
    ############
    im_h = frame.shape[0]
    im_w = frame.shape[1]
    target_pos = np.array([x + w / 2, y + h / 2])
    target_sz = np.array([w, h])
    avg_chans = np.mean(prev_frame, axis=(0, 1))

    wc_z = target_sz[0] + p.context_amount * sum(target_sz)
    hc_z = target_sz[1] + p.context_amount * sum(target_sz)
    s_z = round(np.sqrt(wc_z * hc_z))
    # initialize the exemplar
    z_crop = get_subwindow_tracking(prev_frame, target_pos, p.exemplar_size, s_z, avg_chans)

    z = Variable(z_crop.unsqueeze(0))
    # siammask.template(z.to(device))

    if p.windowing == 'cosine':
        window = np.outer(np.hanning(p.score_size), np.hanning(p.score_size))
    elif p.windowing == 'uniform':
        window = np.ones((p.score_size, p.score_size))
    window = np.tile(window.flatten(), p.anchor_num)

    #########
    # Match #
    #########

    wc_x = target_sz[1] + p.context_amount * sum(target_sz)
    hc_x = target_sz[0] + p.context_amount * sum(target_sz)
    s_x = np.sqrt(wc_x * hc_x)
    scale_x = p.exemplar_size / s_x
    d_search = (p.instance_size - p.exemplar_size) / 2
    pad = d_search / scale_x
    s_x = s_x + 2 * pad
    crop_box = [target_pos[0] - round(s_x) / 2, target_pos[1] - round(s_x) / 2, round(s_x), round(s_x)]

    if debug:
        im_debug = frame.copy()
        crop_box_int = np.int0(crop_box)
        cv2.rectangle(im_debug, (crop_box_int[0], crop_box_int[1]),
                      (crop_box_int[0] + crop_box_int[2], crop_box_int[1] + crop_box_int[3]), (255, 0, 0), 2)
        cv2.imshow('search area', im_debug)
        cv2.waitKey(0)

    # extract scaled crops for search region x at previous target position
    x_crop = Variable(get_subwindow_tracking(frame, target_pos, p.instance_size, round(s_x), avg_chans).unsqueeze(0))

    if mask_enable:
        score, delta, mask = siammask.track_mask(x_crop.to(device))
    else:
        search = siammask.features(x_crop.to(device))
        template = siammask.features(z.to(device))
        score, delta = siammask.rpn(template, search)

    delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1).data.cpu().numpy()
    score = F.softmax(score.permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0), dim=1).data[:, 1].cpu().numpy()

    delta[0, :] = delta[0, :] * p.anchor[:, 2] + p.anchor[:, 0]
    delta[1, :] = delta[1, :] * p.anchor[:, 3] + p.anchor[:, 1]
    delta[2, :] = np.exp(delta[2, :]) * p.anchor[:, 2]
    delta[3, :] = np.exp(delta[3, :]) * p.anchor[:, 3]


    def change(r):
        return np.maximum(r, 1. / r)


    def sz(w, h):
        pad = (w + h) * 0.5
        sz2 = (w + pad) * (h + pad)
        return np.sqrt(sz2)


    def sz_wh(wh):
        pad = (wh[0] + wh[1]) * 0.5
        sz2 = (wh[0] + pad) * (wh[1] + pad)
        return np.sqrt(sz2)


    # size penalty
    target_sz_in_crop = target_sz * scale_x
    s_c = change(sz(delta[2, :], delta[3, :]) / (sz_wh(target_sz_in_crop)))  # scale penalty
    r_c = change((target_sz_in_crop[0] / target_sz_in_crop[1]) / (delta[2, :] / delta[3, :]))  # ratio penalty

    penalty = np.exp(-(r_c * s_c - 1) * p.penalty_k)
    pscore = penalty * score

    # cos window (motion model)
    pscore = pscore * (1 - p.window_influence) + window * p.window_influence
    best_pscore_id = np.argmax(pscore)

    pred_in_crop = delta[:, best_pscore_id] / scale_x
    lr = penalty[best_pscore_id] * score[best_pscore_id] * p.lr  # lr for OTB

    res_x = pred_in_crop[0] + target_pos[0]
    res_y = pred_in_crop[1] + target_pos[1]

    res_w = target_sz[0] * (1 - lr) + pred_in_crop[2] * lr
    res_h = target_sz[1] * (1 - lr) + pred_in_crop[3] * lr

    target_pos = np.array([res_x, res_y])
    target_sz = np.array([res_w, res_h])

    # for Mask Branch
    if mask_enable:
        best_pscore_id_mask = np.unravel_index(best_pscore_id, (5, p.score_size, p.score_size))
        delta_x, delta_y = best_pscore_id_mask[2], best_pscore_id_mask[1]

        if refine_enable:
            mask = siammask.track_refine((delta_y, delta_x)).to(device).sigmoid().squeeze().view(
                p.out_size, p.out_size).cpu().data.numpy()
        else:
            mask = mask[0, :, delta_y, delta_x].sigmoid(). \
                squeeze().view(p.out_size, p.out_size).cpu().data.numpy()


        def crop_back(image, bbox, out_sz, padding=-1):
            a = (out_sz[0] - 1) / bbox[2]
            b = (out_sz[1] - 1) / bbox[3]
            c = -a * bbox[0]
            d = -b * bbox[1]
            mapping = np.array([[a, 0, c],
                                [0, b, d]]).astype(np.float)
            crop = cv2.warpAffine(image, mapping, (out_sz[0], out_sz[1]),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=padding)
            return crop


        s = crop_box[2] / p.instance_size
        sub_box = [crop_box[0] + (delta_x - p.base_size / 2) * p.total_stride * s,
                   crop_box[1] + (delta_y - p.base_size / 2) * p.total_stride * s,
                   s * p.exemplar_size, s * p.exemplar_size]
        s = p.out_size / sub_box[2]
        back_box = [-sub_box[0] * s, -sub_box[1] * s, im_w * s, im_h * s]
        mask_in_img = crop_back(mask, back_box, (im_w, im_h))

        target_mask = (mask_in_img > p.seg_thr).astype(np.uint8)
        _, contours, _ = cv2.findContours(target_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnt_area = [cv2.contourArea(cnt) for cnt in contours]
        if len(contours) != 0 and np.max(cnt_area) > 100:
            contour = contours[np.argmax(cnt_area)]  # use max area polygon
            polygon = contour.reshape(-1, 2)
            # pbox = cv2.boundingRect(polygon)  # Min Max Rectangle
            prbox = cv2.boxPoints(cv2.minAreaRect(polygon))  # Rotated Rectangle

            # box_in_img = pbox
            rbox_in_img = prbox
        else:  # empty mask
            location = cxy_wh_2_rect(target_pos, target_sz)
            rbox_in_img = np.array([[location[0], location[1]],
                                    [location[0] + location[2], location[1]],
                                    [location[0] + location[2], location[1] + location[3]],
                                    [location[0], location[1] + location[3]]])

    target_pos[0] = max(0, min(im_w, target_pos[0]))
    target_pos[1] = max(0, min(im_h, target_pos[1]))
    target_sz[0] = max(10, min(im_w, target_sz[0]))
    target_sz[1] = max(10, min(im_h, target_sz[1]))

    target_pos = target_pos
    target_sz = target_sz
    score = score[best_pscore_id]
    mask = mask_in_img if mask_enable else []
    ploygon = rbox_in_img if mask_enable else []
    ##########
    # To next
    left, top = target_pos - (target_sz / 2)
    right, bottom = target_pos + (target_sz / 2)
    x, y, w, h = left, top, right - left, bottom - top

    prev_frame = frame
    ###################
    #
    if show:
        if mask_enable:
            location = ploygon.flatten()
            mask = mask > p.seg_thr

            frame[:, :, 2] = (mask > 0) * 255 + (mask == 0) * frame[:, :, 2]
            cv2.polylines(frame, [np.int0(location).reshape((-1, 1, 2))], True, (0, 255, 0), 3)
        else:
            left, top = target_pos - (target_sz / 2)
            right, bottom = target_pos + (target_sz / 2)
            cv2.rectangle(frame, pt1=(int(left), int(top)), pt2=(int(right), int(bottom)), color=(0, 255, 0), thickness=3)
    cv2.putText(frame, text='FPS:{:.2f}'.format(1 / (time.time() - tic)), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=2, org=(50, 50), color=(0, 255, 0), thickness=3)
    cv2.imshow('SiamMask', frame)
    key = cv2.waitKey(1)
    if key > 0:
        break
    count += 1
    if count == 500:
        break

    toc += cv2.getTickCount() - tic
toc /= cv2.getTickFrequency()
fps = count / toc
print('SiamMask Time: {:02.1f}s Speed: {:3.1f}fps (with visulization!)'.format(toc, fps))
