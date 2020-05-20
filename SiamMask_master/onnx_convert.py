from torch.autograd import Variable

import torch.onnx
import torchvision
import torch
# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
import glob
from SiamMask_master.tools.test import *
import matplotlib.pyplot as plt
import torch.nn as nn


class DataloopTracker(nn.Module):
    def __init__(self, siammask):
        super(DataloopTracker, self).__init__()
        self.siammask = siammask

    def forward(self, x, z):
        search = siammask.features(x.to(device))
        template = siammask.features(z.to(device))
        score, delta = siammask.rpn(template, search)
        return score, delta


class temp:
    resume = r'E:\Shabtay\fonda_pytorch\SiamMask_master\models\SiamMask_DAVIS.pth'  # ,help='path to latest checkpoint (default: none)')
    config = r'E:\Shabtay\fonda_pytorch\SiamMask_master\experiments\siammask_sharp\config_davis.json'  # help='hyper-parameter of SiamMask in json format')
    # base_path=r'E:\Shabtay\fonda_pytorch\SiamMask_master\data\tennis' # help='datasets')
    # video_filepath = r"C:\Users\Dataloop\.dataloop\projects\Feb19_shelf_zed\datasets\try1\images\video\download.mp4"
    # video_filepath=r"C:\Users\Dataloop\.dataloop\projects\Eyezon_fixed\datasets\New Clips\clip2\ch34_25fps05.mp4"
    video_filepath = r"E:\Projects\Foresight\tracker\videoplayback.webm"
    cpu = False  # help='cpu mode')


args = temp()
enable_mask = True

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

aa = DataloopTracker(siammask)
dummy_input_1 = torch.randn(1, 3, 256, 256, device='cuda')
dummy_input_2 = torch.randn(1, 3, 127, 127, device='cuda')
torch.onnx.export(aa, (dummy_input_1, dummy_input_2), r"e:\siam.onnx")

#################
import onnx

# Load the ONNX model
model = onnx.load(r"E:\Shabtay\js\onnxjs-demo-master\public\siam.onnx")

# Check that the IR is well formed
onnx.checker.check_model(model)

# Print a human readable representation of the graph
print(onnx.helper.printable_graph(model.graph))

###########################
import onnxruntime

ort_session = onnxruntime.InferenceSession(r"E:\siam.onnx")


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(dummy_input_1)}
ort_outs = ort_session.run(None, (ort_inputs, ort_inputs))

# compare ONNX Runtime and PyTorch results
np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

print("Exported model has been tested with ONNXRuntime, and the result looks good!")

########################
###
import torch
from pytorch2keras.converter import pytorch_to_keras

x = torch.randn(1, 3, 224, 224, requires_grad=False, device='cuda')
k_model = pytorch_to_keras(aa, x, [(3, None, None,)], verbose=True, names='short')
k_model.save(r'e:\keras.h5')
