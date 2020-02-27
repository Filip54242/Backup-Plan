import cv2
import numpy as np
import torch.backends.cudnn as cudnn
import torch.utils.data.distributed
import torchvision.transforms as transforms
from PIL import Image, ImageDraw

from core.loss import JointsMSELoss
from pose_hrnet import get_pose_net


#video_or_camera="id00027.mp4"
video_or_camera=0
dataset = "MPII"

STICKS = {"MPII": [[0, 1], [1, 8], [8, 2], [2, 3], [3, 4], [8, 5], [5, 6], [6, 7], [8, 9], [9, 10], [10, 11], [11, 12],
                   [9, 13], [13, 14], [14, 15]],
          "COCO": [[0, 1], [1, 2], [0, 3], [3, 4], [0, 5], [0, 8], [8, 9], [9, 10], [5, 6], [6, 7], [5, 8], [8, 14],
                   [14, 11], [11, 5], [11, 12], [12, 13], [14, 15], [15, 16]]}
IDS = {"MPII": {9: 0, 8: 1, 12: 2, 11: 3, 10: 4, 13: 5, 14: 6, 15: 7, 7: 8, 6: 9, 2: 10, 1: 11, 0: 12, 3: 13, 4: 14,
                5: 15},
       "COCO": {0: 0, 1: 1, 3: 2, 2: 3, 4: 4, 6: 5, 8: 6, 10: 7, 5: 8, 7: 9, 9: 10, 12: 11, 14: 12, 16: 13, 11: 14,
                13: 15, 15: 16}}


CFGS = {
    "MPII": {'OUTPUT_DIR': 'output', 'LOG_DIR': 'log', 'DATA_DIR': '', 'GPUS': (0), 'WORKERS': 24, 'PRINT_FREQ': 100,
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
                                                                      {'NUM_MODULES': 1, 'NUM_BRANCHES': 2,
                                                                       'BLOCK': 'BASIC',
                                                                       'NUM_BLOCKS': [4, 4], 'NUM_CHANNELS': [32, 64],
                                                                       'FUSE_METHOD': 'SUM'}, 'STAGE3':
                                                                      {'NUM_MODULES': 4, 'NUM_BRANCHES': 3,
                                                                       'BLOCK': 'BASIC',
                                                                       'NUM_BLOCKS': [4, 4, 4],
                                                                       'NUM_CHANNELS': [32, 64, 128],
                                                                       'FUSE_METHOD': 'SUM'},
                                                                  'STAGE4':
                                                                      {'NUM_MODULES': 3, 'NUM_BRANCHES': 4,
                                                                       'BLOCK': 'BASIC',
                                                                       'NUM_BLOCKS': [4, 4, 4, 4],
                                                                       'NUM_CHANNELS': [32, 64, 128, 256],
                                                                       'FUSE_METHOD': 'SUM'}}},
             'LOSS': {'USE_OHKM': False, 'TOPK': 8, 'USE_TARGET_WEIGHT': True, 'USE_DIFFERENT_JOINTS_WEIGHT': False},
             'DATASET':
                 {'ROOT': 'data/mpii/', 'DATASET': 'mpii', 'TRAIN_SET': 'train', 'TEST_SET': 'valid',
                  'DATA_FORMAT': 'jpg',
                  'HYBRID_JOINTS_TYPE': '', 'SELECT_DATA': False, 'FLIP': True, 'SCALE_FACTOR': 0.25, 'ROT_FACTOR': 30,
                  'PROB_HALF_BODY': -1.0, 'NUM_JOINTS_HALF_BODY': 8, 'COLOR_RGB': True}, 'TRAIN':
                 {'LR_FACTOR': 0.1, 'LR_STEP': [170, 200], 'LR': 0.001, 'OPTIMIZER': 'adam', 'MOMENTUM': 0.9,
                  'WD': 0.0001,
                  'NESTEROV': False, 'GAMMA1': 0.99, 'GAMMA2': 0.0, 'BEGIN_EPOCH': 0, 'END_EPOCH': 210, 'RESUME': False,
                  'CHECKPOINT': '', 'BATCH_SIZE_PER_GPU': 32, 'SHUFFLE': True}, 'TEST':
                 {'BATCH_SIZE_PER_GPU': 32, 'FLIP_TEST': True, 'POST_PROCESS': True, 'SHIFT_HEATMAP': True,
                  'USE_GT_BBOX': False,
                  'IMAGE_THRE': 0.1, 'NMS_THRE': 0.6, 'SOFT_NMS': False, 'OKS_THRE': 0.5, 'IN_VIS_THRE': 0.0,
                  'COCO_BBOX_FILE': '', 'BBOX_THRE': 1.0,
                  'MODEL_FILE': "/home/filip/Documents/image_to_2D/pose_hrnet_w32_256x256.pth"},
             'DEBUG':
                 {'DEBUG': True, 'SAVE_BATCH_IMAGES_GT': True, 'SAVE_BATCH_IMAGES_PRED': True, 'SAVE_HEATMAPS_GT': True,
                  'SAVE_HEATMAPS_PRED': True}},
    "COCO": {'OUTPUT_DIR': 'output', 'LOG_DIR': 'log', 'DATA_DIR': '', 'GPUS': (0, 1, 2, 3), 'WORKERS': 24,
             'PRINT_FREQ': 100, 'AUTO_RESUME': True, 'PIN_MEMORY': True, 'RANK': 0,
             'CUDNN': {'BENCHMARK': True, 'DETERMINISTIC': False, 'ENABLED': True}, 'MODEL':
                 {'NAME': 'pose_hrnet', 'INIT_WEIGHTS': True,
                  'PRETRAINED': '/home/filip/Documents/image_to_2D/pose_hrnet_w48_384x288.pth',
                  'NUM_JOINTS': 17, 'TAG_PER_JOINT': True, 'TARGET_TYPE': 'gaussian', 'IMAGE_SIZE': [288, 384],
                  'HEATMAP_SIZE': [72, 96], 'SIGMA': 3, 'EXTRA': {'PRETRAINED_LAYERS': ['conv1', 'bn1', 'conv2',
                                                                                        'bn2', 'layer1',
                                                                                        'transition1', 'stage2',
                                                                                        'transition2', 'stage3',
                                                                                        'transition3', 'stage4'],
                                                                  'FINAL_CONV_KERNEL': 1, 'STAGE2':
                                                                      {'NUM_MODULES': 1, 'NUM_BRANCHES': 2,
                                                                       'BLOCK': 'BASIC', 'NUM_BLOCKS': [4, 4],
                                                                       'NUM_CHANNELS': [48, 96], 'FUSE_METHOD': 'SUM'},
                                                                  'STAGE3':
                                                                      {'NUM_MODULES': 4, 'NUM_BRANCHES': 3,
                                                                       'BLOCK': 'BASIC', 'NUM_BLOCKS': [4, 4, 4],
                                                                       'NUM_CHANNELS': [48, 96, 192],
                                                                       'FUSE_METHOD': 'SUM'}, 'STAGE4':
                                                                      {'NUM_MODULES': 3, 'NUM_BRANCHES': 4,
                                                                       'BLOCK': 'BASIC', 'NUM_BLOCKS': [4, 4, 4, 4],
                                                                       'NUM_CHANNELS': [48, 96, 192, 384],
                                                                       'FUSE_METHOD': 'SUM'}}}, 'LOSS':
                 {'USE_OHKM': False, 'TOPK': 8, 'USE_TARGET_WEIGHT': True, 'USE_DIFFERENT_JOINTS_WEIGHT': False},
             'DATASET':
                 {'ROOT': 'data/coco/', 'DATASET': 'coco', 'TRAIN_SET': 'train2017', 'TEST_SET': 'val2017',
                  'DATA_FORMAT': 'jpg', 'HYBRID_JOINTS_TYPE': '', 'SELECT_DATA': False, 'FLIP': True,
                  'SCALE_FACTOR': 0.35, 'ROT_FACTOR': 45, 'PROB_HALF_BODY': 0.3, 'NUM_JOINTS_HALF_BODY': 8,
                  'COLOR_RGB': True}, 'TRAIN':
                 {'LR_FACTOR': 0.1, 'LR_STEP': [170, 200], 'LR': 0.001, 'OPTIMIZER': 'adam', 'MOMENTUM': 0.9,
                  'WD': 0.0001,
                  'NESTEROV': False, 'GAMMA1': 0.99, 'GAMMA2': 0.0, 'BEGIN_EPOCH': 0, 'END_EPOCH': 210, 'RESUME': False,
                  'CHECKPOINT': '', 'BATCH_SIZE_PER_GPU': 24, 'SHUFFLE': True}, 'TEST':
                 {'BATCH_SIZE_PER_GPU': 24, 'FLIP_TEST': True, 'POST_PROCESS': True, 'SHIFT_HEATMAP': True,
                  'USE_GT_BBOX': False, 'IMAGE_THRE': 0.0, 'NMS_THRE': 1.0, 'SOFT_NMS': False, 'OKS_THRE': 0.9,
                  'IN_VIS_THRE': 0.2,
                  'COCO_BBOX_FILE': 'data/coco/person_detection_results/COCO_val2017_detections_AP_H_56_person.json',
                  'BBOX_THRE': 1.0,
                  'MODEL_FILE': '/home/filip/Documents/image_to_2D/pose_hrnet_w48_384x288.pth'},
             'DEBUG':
                 {'DEBUG': True, 'SAVE_BATCH_IMAGES_GT': True, 'SAVE_BATCH_IMAGES_PRED': True, 'SAVE_HEATMAPS_GT': True,
                  'SAVE_HEATMAPS_PRED': True}}
}


def output_valid(output, treshold=0.7):
    mean = 0
    for heatmap in output:
        mean += np.amax(heatmap)
    if mean / output.shape[0] > treshold:
        return True
    return False

sticks = STICKS[dataset]
ids = IDS[dataset]
cfg = CFGS[dataset]

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


def max_index_matrix(matrix, coordinate):
    result = np.where(matrix == np.amax(matrix))
    return result[coordinate][0]


def result(input_image, output, radius=5):
    keypoint_locations = [None for index in range(len(output))]
    for index in range(len(output)):
        heatmap = cv2.resize(output[index], dsize=(input_image. size), interpolation=cv2.INTER_CUBIC)
        x = int(max_index_matrix(heatmap, 1))
        y = int(max_index_matrix(heatmap, 0))
        keypoint_locations[ids[index]] = (x, y)
    draw = ImageDraw.Draw(input_image)
    for item in sticks:
        draw.line((keypoint_locations[item[0]], keypoint_locations[item[1]]), fill=(255, 0, 0, 128), width=radius)
    return input_image


capture = cv2.VideoCapture(video_or_camera)

while (True):

    ret, frame = capture.read()
    print(frame.shape)
    input = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    input_copy = input.copy()
    input = test_transforms(input)
    input = torch.unsqueeze(input, 0).cuda()
    with torch.no_grad():
        output = model(input).squeeze().detach().cpu().numpy()
    if output_valid(output):
        image=cv2.cvtColor(np.asarray(result(input_copy, output)), cv2.COLOR_RGB2BGR)
    else:
        image=cv2.cvtColor(np.asarray(input_copy), cv2.COLOR_RGB2BGR)
    cv2.imshow(dataset + '_Demo', image )

    if cv2.waitKey(1) == 27:
        break

capture.release()
cv2.destroyAllWindows()
