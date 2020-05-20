# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
import glob
from SiamMask_master.tools.test import *
import matplotlib.pyplot as plt


class temp:
    resume = r'E:\Shabtay\fonda_pytorch\SiamMask_master\models\SiamMask_DAVIS.pth'  # ,help='path to latest checkpoint (default: none)')
    config = r'E:\Shabtay\fonda_pytorch\SiamMask_master\experiments\siammask_sharp\config_davis.json'  # help='hyper-parameter of SiamMask in json format')
    # base_path=r'E:\Shabtay\fonda_pytorch\SiamMask_master\data\tennis' # help='datasets')
    # video_filepath = r"C:\Users\Dataloop\.dataloop\projects\Feb19_shelf_zed\datasets\try1\images\video\download.mp4"
    # video_filepath=r"C:\Users\Dataloop\.dataloop\projects\Eyezon_fixed\datasets\New Clips\clip2\ch34_25fps05.mp4"
    # video_filepath = r"E:\Projects\Foresight\tracker\videoplayback.webm"
    video_filepath = r"E:\Projects\Foresight\tracker\VL_left.webm"
    cpu = False  # help='cpu mode')


args = temp()
enable_mask = True

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
    cap = cv2.VideoCapture(args.video_filepath)
    assert cap.isOpened()
    cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
    ret, frame = cap.read()
    # frame = cv2.resize(frame, (128, 128))

    # Select ROI
    cv2.namedWindow("SiamMask", cv2.WND_PROP_AUTOSIZE)
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
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # frame = cv2.resize(frame, (128, 128))
        tic = cv2.getTickCount()
        if count == 0:  # init
            first = False
            target_pos = np.array([x + w / 2, y + h / 2])
            target_sz = np.array([w, h])
            state = siamese_init(frame, target_pos, target_sz, siammask, cfg['hp'], device=device)  # init tracker
        else:  # tracking
            state = siamese_track(state, frame, mask_enable=enable_mask, refine_enable=True, device=device)  # track
            if show:
                if enable_mask:
                    location = state['ploygon'].flatten()
                    mask = state['mask'] > state['p'].seg_thr

                    frame[:, :, 2] = (mask > 0) * 255 + (mask == 0) * frame[:, :, 2]
                    cv2.polylines(frame, [np.int0(location).reshape((-1, 1, 2))], True, (0, 255, 0), 3)
                else:
                    left, top = state['target_pos'] - (state['target_sz'] / 2)
                    right, bottom= state['target_pos'] + (state['target_sz'] / 2)
                    cv2.rectangle(frame, pt1=(int(left), int(top)), pt2=(int(right), int(bottom)), color=(0, 255, 0), thickness=3)
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
