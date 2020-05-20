# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
import glob
from SiamMask_master.tools.test import *
import matplotlib.pyplot as plt

class temp:
    resume= r'E:\Shabtay\fonda_pytorch\SiamMask_master\models\SiamMask_DAVIS.pth' #,help='path to latest checkpoint (default: none)')
    config  = r'E:\Shabtay\fonda_pytorch\SiamMask_master\experiments\siammask_sharp\config_davis.json' # help='hyper-parameter of SiamMask in json format')
    #base_path=r'E:\Shabtay\fonda_pytorch\SiamMask_master\data\tennis' # help='datasets')
    base_path=r'C:\Users\Dataloop\.dataloop\datasets\5ce2b280fe3b45001c2c7a51\image\Arlozorov1'
    cpu = False# help='cpu mode')
args = temp()

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
    #cv2.namedWindow("SiamMask", cv2.WND_PROP_FULLSCREEN)
    # cv2.setWindowProperty("SiamMask", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    try:
        init_rect = cv2.selectROI('SiamMask', ims[0], False, False)
        x, y, w, h = init_rect
    except:
        exit()

    toc = 0
    for f, im in enumerate(ims):
        tic = cv2.getTickCount()
        if f == 0:  # init
            target_pos = np.array([x + w / 2, y + h / 2])
            target_sz = np.array([w, h])
            state = siamese_init(im, target_pos, target_sz, siammask, cfg['hp'], device=device)  # init tracker
        elif f > 0:  # tracking
            state = siamese_track(state, im, mask_enable=True, refine_enable=True, device=device)  # track
            location = state['ploygon'].flatten()
            mask = state['mask'] > state['p'].seg_thr

            im[:, :, 2] = (mask > 0) * 255 + (mask == 0) * im[:, :, 2]
            cv2.polylines(im, [np.int0(location).reshape((-1, 1, 2))], True, (0, 255, 0), 3)
            #cv2.imshow('SiamMask', im)
            #key = cv2.waitKey(1)
            #if key > 0:
            #    break
            plt.figure()
            plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))

        toc += cv2.getTickCount() - tic
    toc /= cv2.getTickFrequency()
    fps = f / toc
    print('SiamMask Time: {:02.1f}s Speed: {:3.1f}fps (with visulization!)'.format(toc, fps))
