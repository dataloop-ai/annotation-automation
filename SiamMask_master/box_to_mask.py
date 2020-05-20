# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
import glob
from SiamMask_master.tools.test import *
import matplotlib.pyplot as plt
import time


class temp:
    resume = r'E:\Shabtay\fonda_pytorch\SiamMask_master\models\SiamMask_DAVIS.pth'  # ,help='path to latest checkpoint (default: none)')
    config = r'E:\Shabtay\fonda_pytorch\SiamMask_master\experiments\siammask_sharp\config_davis.json'  # help='hyper-parameter of SiamMask in json format')
    # base_path=r'E:\Shabtay\fonda_pytorch\SiamMask_master\data\tennis' # help='datasets')
    base_path = r'C:\Users\Dataloop\.dataloop\datasets\5ce2b280fe3b45001c2c7a51\image\Arlozorov1'
    cpu = False  # help='cpu mode')


args = temp()


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


if __name__ == '__main__':
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
    img_files = sorted(glob.glob(join(args.base_path, '*.jp*')))
    ims = [cv2.imread(imf) for imf in img_files]

    # Select ROI
    # cv2.namedWindow("SiamMask", cv2.WND_PROP_FULLSCREEN)
    # cv2.setWindowProperty("SiamMask", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    try:
        init_rect = cv2.selectROI('SiamMask', ims[0], False, False)
        x, y, w, h = init_rect
    except:
        assert False

    tic = time.time()
    im = ims[0]
    target_pos = np.array([x + w / 2, y + h / 2])
    target_sz = np.array([w, h])
    state = dict()
    state['im_h'] = im.shape[0]
    state['im_w'] = im.shape[1]
    p = TrackerConfig()
    p.update(cfg['hp'], siammask.anchors)

    p.renew()

    p.scales = siammask.anchors['scales']
    p.ratios = siammask.anchors['ratios']
    p.anchor_num = siammask.anchor_num
    p.anchor = generate_anchor(siammask.anchors, p.score_size)
    avg_chans = np.mean(im, axis=(0, 1))

    wc_z = target_sz[0] + p.context_amount * sum(target_sz)
    hc_z = target_sz[1] + p.context_amount * sum(target_sz)
    s_z = round(np.sqrt(wc_z * hc_z))
    # initialize the exemplar
    z_crop = get_subwindow_tracking(im, target_pos, p.exemplar_size, s_z, avg_chans)

    z = Variable(z_crop.unsqueeze(0))
    siammask.template(z.to(device))

    if p.windowing == 'cosine':
        window = np.outer(np.hanning(p.score_size), np.hanning(p.score_size))
    elif p.windowing == 'uniform':
        window = np.ones((p.score_size, p.score_size))
    window = np.tile(window.flatten(), p.anchor_num)

    state['p'] = p
    state['net'] = siammask
    state['avg_chans'] = avg_chans
    state['window'] = window
    state['target_pos'] = target_pos
    state['target_sz'] = target_sz

    state = siamese_track(state, im, mask_enable=True, refine_enable=True, device=device)  # track
    location = state['ploygon'].flatten()
    mask = state['mask'] > state['p'].seg_thr

    im[:, :, 2] = (mask > 0) * 255 + (mask == 0) * im[:, :, 2]
    cv2.polylines(im, [np.int0(location).reshape((-1, 1, 2))], True, (0, 255, 0), 3)
    # cv2.imshow('SiamMask', im)
    # key = cv2.waitKey(1)
    # if key > 0:
    #    break
    plt.figure()
    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))

    toc = time.time() - tic
    print('SiamMask Time: {:02.1f}s Speed:'.format(toc))
