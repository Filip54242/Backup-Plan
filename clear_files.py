import json
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch.backends.cudnn as cudnn
import torch.utils.data.distributed
import torchvision.transforms as transforms
from PIL import Image, ImageDraw

from core.loss import JointsMSELoss
from pose_hrnet import get_pose_net


def plot_result(input_image, output, radius=20):
    draw = ImageDraw.Draw(input_image)
    for heatmap in output:
        heatmap = cv2.resize(heatmap, dsize=(input_image.size), interpolation=cv2.INTER_CUBIC)
        x = int(max_index_matrix(heatmap, 0))
        y = int(max_index_matrix(heatmap, 1))
        draw.ellipse((y - radius, x - radius, y + radius, x + radius), fill=(255, 0, 0, 255))
    plt.imshow(input_image)


def normalize(list):
    return (list - min(list)) / (max(list) - min(list))


def max_index_matrix(matrix, coordinate):
    result = np.where(matrix == np.amax(matrix))
    return result[coordinate][0]


def create_coordinate_keypoint(model_output, coordinate):
    head_neck = [max_index_matrix(model_output[9], coordinate), max_index_matrix(model_output[8], coordinate)]
    right_hand = [max_index_matrix(model_output[12], coordinate), max_index_matrix(model_output[11],
                                                                                   coordinate), max_index_matrix(
        model_output[10], coordinate)]
    left_hand = [max_index_matrix(model_output[13], coordinate), max_index_matrix(model_output[14],
                                                                                  coordinate), max_index_matrix(
        model_output[15], coordinate)]
    torso = [max_index_matrix(model_output[7], coordinate), max_index_matrix(model_output[6], coordinate)]
    right_leg = [max_index_matrix(model_output[2], coordinate), max_index_matrix(model_output[1], coordinate),
                 max_index_matrix(
                     model_output[0], coordinate)]
    left_leg = [max_index_matrix(model_output[3], coordinate), max_index_matrix(model_output[4], coordinate),
                max_index_matrix(
                    model_output[5], coordinate)]
    body = head_neck + right_hand + left_hand + torso + right_leg + left_leg
    return body


def create_keypoints(model_output):
    return [list(normalize(create_coordinate_keypoint(model_output, 1))),
            list(normalize(create_coordinate_keypoint(model_output, 0)))]

def result_skeleton(input_image, output, radius=5):
    ids = {9: 0, 8: 1, 12: 2, 11: 3, 10: 4, 13: 5, 14: 6, 15: 7, 7: 8, 6: 9, 2: 10, 1: 11, 0: 12, 3: 13, 4: 14,
           5: 15}
    sticks = [[0, 1], [1, 8], [8, 2], [2, 3], [3, 4], [8, 5], [5, 6], [6, 7], [8, 9], [9, 10], [10, 11], [11, 12],
              [9, 13], [13, 14], [14, 15]]
    keypoint_locations = [None for index in range(len(output))]
    for index in range(len(output)):
        heatmap = cv2.resize(output[index], dsize=(input_image.size), interpolation=cv2.INTER_CUBIC)
        x = int(max_index_matrix(heatmap, 1))
        y = int(max_index_matrix(heatmap, 0))
        keypoint_locations[ids[index]] = (x, y)
    draw = ImageDraw.Draw(input_image)
    for item in sticks:
        draw.line((keypoint_locations[item[0]], keypoint_locations[item[1]]), fill=(255, 0, 0, 128), width=radius)
    return input_image

def get_all_files_from_path(path, extension):
    files = []
    for r, d, f in os.walk(path):
        for file in f:
            if extension in file:
                files.append(os.path.join(r, file))
    return files


cfg = {'OUTPUT_DIR': 'output', 'LOG_DIR': 'log', 'DATA_DIR': '', 'GPUS': (0), 'WORKERS': 24, 'PRINT_FREQ': 100,
       'AUTO_RESUME': True, 'PIN_MEMORY': True, 'RANK': 0,
       'CUDNN': {'BENCHMARK': True, 'DETERMINISTIC': False, 'ENABLED': True}, 'MODEL':
           {'NAME': 'pose_hrnet', 'INIT_WEIGHTS': True,
            'PRETRAINED': '/home/filip/Documents/image_to_2D/pose_hrnet_w32_256x256.pth',
            'NUM_JOINTS': 16, 'TAG_PER_JOINT': True, 'TARGET_TYPE': 'gaussian', 'IMAGE_SIZE': [256, 256],
            'HEATMAP_SIZE': [64, 64], 'SIGMA': 2, 'EXTRA': {'PRETRAINED_LAYERS': ['conv1', 'bn1', 'conv2', 'bn2',
                                                                                  'layer1', 'transition1',
                                                                                  'stage2', 'transition2',
                                                                                  'stage3', 'transition3',
                                                                                  'stage4'],
                                                            'FINAL_CONV_KERNEL': 1, 'STAGE2':
                                                                {'NUM_MODULES': 1, 'NUM_BRANCHES': 2, 'BLOCK': 'BASIC',
                                                                 'NUM_BLOCKS': [4, 4], 'NUM_CHANNELS': [32, 64],
                                                                 'FUSE_METHOD': 'SUM'}, 'STAGE3':
                                                                {'NUM_MODULES': 4, 'NUM_BRANCHES': 3, 'BLOCK': 'BASIC',
                                                                 'NUM_BLOCKS': [4, 4, 4],
                                                                 'NUM_CHANNELS': [32, 64, 128], 'FUSE_METHOD': 'SUM'},
                                                            'STAGE4':
                                                                {'NUM_MODULES': 3, 'NUM_BRANCHES': 4, 'BLOCK': 'BASIC',
                                                                 'NUM_BLOCKS': [4, 4, 4, 4],
                                                                 'NUM_CHANNELS': [32, 64, 128, 256],
                                                                 'FUSE_METHOD': 'SUM'}}},
       'LOSS': {'USE_OHKM': False, 'TOPK': 8, 'USE_TARGET_WEIGHT': True, 'USE_DIFFERENT_JOINTS_WEIGHT': False},
       'DATASET':
           {'ROOT': 'data/mpii/', 'DATASET': 'mpii', 'TRAIN_SET': 'train', 'TEST_SET': 'valid', 'DATA_FORMAT': 'jpg',
            'HYBRID_JOINTS_TYPE': '', 'SELECT_DATA': False, 'FLIP': True, 'SCALE_FACTOR': 0.25, 'ROT_FACTOR': 30,
            'PROB_HALF_BODY': -1.0, 'NUM_JOINTS_HALF_BODY': 8, 'COLOR_RGB': True}, 'TRAIN':
           {'LR_FACTOR': 0.1, 'LR_STEP': [170, 200], 'LR': 0.001, 'OPTIMIZER': 'adam', 'MOMENTUM': 0.9, 'WD': 0.0001,
            'NESTEROV': False, 'GAMMA1': 0.99, 'GAMMA2': 0.0, 'BEGIN_EPOCH': 0, 'END_EPOCH': 210, 'RESUME': False,
            'CHECKPOINT': '', 'BATCH_SIZE_PER_GPU': 32, 'SHUFFLE': True}, 'TEST':
           {'BATCH_SIZE_PER_GPU': 32, 'FLIP_TEST': True, 'POST_PROCESS': True, 'SHIFT_HEATMAP': True,
            'USE_GT_BBOX': False,
            'IMAGE_THRE': 0.1, 'NMS_THRE': 0.6, 'SOFT_NMS': False, 'OKS_THRE': 0.5, 'IN_VIS_THRE': 0.0,
            'COCO_BBOX_FILE': '', 'BBOX_THRE': 1.0,
            'MODEL_FILE': "/home/filip/Documents/image_to_2D/pose_hrnet_w32_256x256.pth"},
       'DEBUG':
           {'DEBUG': True, 'SAVE_BATCH_IMAGES_GT': True, 'SAVE_BATCH_IMAGES_PRED': True, 'SAVE_HEATMAPS_GT': True,
            'SAVE_HEATMAPS_PRED': True}}

# cudnn related setting
cudnn.benchmark = cfg['CUDNN']['BENCHMARK']
torch.backends.cudnn.deterministic = cfg['CUDNN']['DETERMINISTIC']
torch.backends.cudnn.enabled = cfg['CUDNN']['ENABLED']
model = get_pose_net(cfg, is_train=False).cuda()
model.load_state_dict(torch.load(cfg['TEST']['MODEL_FILE']), strict=False)

# define loss function (criterion) and optimizer
criterion = JointsMSELoss(use_target_weight=cfg['LOSS']['USE_TARGET_WEIGHT']).cuda()

test_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# Data loading code


files = get_all_files_from_path('/media/filip/HDD1/Images/', "jpg")
for file in files:
    print("File :", file)
    input = Image.open(file)
    first=input
    input = test_transforms(input)
    input = torch.unsqueeze(input, 0).cuda()
    with torch.no_grad():
        output = model(input).squeeze().detach().cpu().numpy()
    result = result_skeleton(first, output)
    cv2.imshow('Whirldata Window', np.asarray(result))
    k = cv2.waitKey(0)
    if k == ord('s'):
        cv2.destroyAllWindows()
    elif k==ord('d'):
        cv2.destroyAllWindows()
        os.remove(file)
