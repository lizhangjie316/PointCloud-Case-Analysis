import argparse
import os
from data_utils.ShapeNetDataLoader import PartNormalDataset
import torch
import datetime
import logging
from pathlib import Path
import sys
import importlib
import shutil
from tqdm import tqdm
import provider
import numpy as np
"""
训练所需设置参数：
--model pointnet2_part_seg_msg 
--normal 
--log_dir pointnet2_part_seg_msg
"""

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
# print(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))  # 项目的其实根目录

seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43], 'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}
seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}   {0:Airplane, 1:Airplane, ...49:Table}
for cat in seg_classes.keys():  # 将每个种类的 部件标签重置为方便理解的
    for label in seg_classes[cat]:
        seg_label_to_cat[label] = cat

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y

# --model pointnet2_part_seg_msg --normal --log_dir pointnet2_part_seg_msg
def parse_args():  # 参数解析
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='pointnet2_part_seg_msg', help='model name [default: pointnet2_part_seg_msg]')
    parser.add_argument('--batch_size', type=int, default=12, help='Batch Size during training [default: 16]')
    parser.add_argument('--epoch',  default=20, type=int, help='Epoch to run [default: 251]')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD [default: Adam]')
    parser.add_argument('--log_dir', type=str, default=None, help='Log path [default: None]')  # pointnet2_part_seg_msg
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay [default: 1e-4]')
    parser.add_argument('--npoint', type=int,  default=2048, help='Point Number [default: 2048]')
    parser.add_argument('--normal', action='store_true', default=False, help='Whether to use normal information [default: False]')
    parser.add_argument('--step_size', type=int,  default=20, help='Decay step for lr decay [default: every 20 epochs]')
    parser.add_argument('--lr_decay', type=float,  default=0.5, help='Decay rate for lr decay [default: 0.5]')

    return parser.parse_args()

def main(args):
    def log_string(str):
        logger.info(str)
        # print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu  # 0号GPU

    '''CREATE DIR'''  # 创建log存放的文件目录及文件夹，存储在log目录下
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath('part_seg')
    experiment_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)  # Namespace(batch_size=4, decay_rate=0.0001, epoch=251, gpu='0', learning_rate=0.001, log_dir='pointnet2_part_seg_msg', lr_decay=0.5, model='pointnet2_part_seg_msg', normal=True, npoint=2048, optimizer='Adam', step_size=20)

    root = 'data/shapenetcore_partanno_segmentation_benchmark_v0_normal/'

    # 开始处理数据集
    # 返回2048个点，并进行正则化   提前已经分配好了哪些是训练集，哪些作为测试集
    TRAIN_DATASET = PartNormalDataset(root = root, npoints=args.npoint, split='trainval', normal_channel=args.normal)

    # 按照batch_size进行组装数据
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size,shuffle=True, num_workers=4)
    # 测试数据同样处理
    TEST_DATASET = PartNormalDataset(root = root, npoints=args.npoint, split='test', normal_channel=args.normal)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size,shuffle=False, num_workers=4)

    log_string("The number of training data is: %d" % len(TRAIN_DATASET))  # 训练数据 13998
    log_string("The number of test data is: %d" %  len(TEST_DATASET))  # 测试数据 2874
    num_classes = 16
    num_part = 50
    '''MODEL LOADING'''
    MODEL = importlib.import_module(args.model)
    # 将模型和工具包都添加到log文件中
    shutil.copy('models/%s.py' % args.model, str(experiment_dir))
    shutil.copy('models/pointnet_util.py', str(experiment_dir))
    # 分类器，进行50分类，对2048个点都要进行分类
    classifier = MODEL.get_model(num_part, normal_channel=args.normal).cuda()
    criterion = MODEL.get_loss().cuda()  # 计算损失函数的方式


    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)

    try:  # 加载预训练的模型
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0
        classifier = classifier.apply(weights_init)

    if args.optimizer == 'Adam': #TODO 研究这些参数
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9)
    # 依据动量进行调整
    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size

    best_acc = 0
    global_epoch = 0
    best_class_avg_iou = 0
    best_inctance_avg_iou = 0

    # 开始进行迭代训练
    for epoch in range(start_epoch,args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        '''Adjust learning rate and BN momentum'''
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        log_string('Learning rate:%f' % lr)
        #TODO：？？？
        # param_groups 是一个list，里面每一个item都是字典；这项作用是给内部的lr项赋值为param上的lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        mean_correct = []
        # 0.1*（0.5^（epoch//20））  每20步，动量减小一次
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        print('BN momentum updated to: %f' % momentum)  # 0.100000
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x,momentum))

        '''learning one epoch'''
        for i, data in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
            points, label, target = data
            # print(points.shape) # (4,2048,6)
            # 数据增强：做一些微小扰动
            points = points.data.numpy()
            # print(points.shape)  # (4,2048,6)
            points[:,:, 0:3] = provider.random_scale_point_cloud(points[:,:, 0:3])
            points[:,:, 0:3] = provider.shift_point_cloud(points[:,:, 0:3])
            points = torch.Tensor(points)
            points, label, target = points.float().cuda(),label.long().cuda(), target.long().cuda()
            # print(points.shape)  # torch.Size([4, 2048, 6])
            # print(label.shape)  # torch.Size([4, 1])   每个样本的对应的种类标签
            # print(target.shape)  # torch.Size([4, 2048])   每个点的类别标签
            points = points.transpose(2, 1)
            # print(points.shape)  # torch.Size([4, 6, 2048])
            optimizer.zero_grad()
            classifier = classifier.train()
            seg_pred, trans_feat = classifier(points, to_categorical(label, num_classes))  # seg_pred  torch.Size([4, 2048, 50])   trans_feat：torch.Size([4, 1024, 1]) ？？？
            seg_pred = seg_pred.contiguous().view(-1, num_part) # torch.Size([8192, 50])
            target = target.view(-1, 1)[:, 0]  # 8192
            pred_choice = seg_pred.data.max(1)[1]  # 8192   预测的结果部件类别
            correct = pred_choice.eq(target.data).cpu().sum() # tensor(249)   即只有249个正确
            mean_correct.append(correct.item() / (args.batch_size * args.npoint))  # 平均正确率  0.0303955078125
            loss = criterion(seg_pred, target, trans_feat)
            loss.backward()
            optimizer.step()
        train_instance_acc = np.mean(mean_correct)  # 1个epoch 准确率  mean_correct的list中有 3500个值  13998个样本，一个batch处理4个,共需3500步 step
        log_string('Train accuracy is: %.5f' % train_instance_acc)  # 实例分割 准确率： 0.8502310616629464
        # 进行测试
        with torch.no_grad():  # 非训练过程，后续的tensor操作，不需要进行计算图的构建（计算过程的构建，以便梯度反向传播等操作），只用来进行测试
            test_metrics = {}
            total_correct = 0
            total_seen = 0
            total_seen_class = [0 for _ in range(num_part)]  # num_part个0组成的list
            total_correct_class = [0 for _ in range(num_part)]
            shape_ious = {cat: [] for cat in seg_classes.keys()}
            seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
            for cat in seg_classes.keys():
                for label in seg_classes[cat]:  # 每种的部件类别
                    seg_label_to_cat[label] = cat

            for batch_id, (points, label, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
                cur_batch_size, NUM_POINT, _ = points.size()  # torch.Size([4, 2048, 6])
                points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()
                points = points.transpose(2, 1)  # torch.Size([4, 6, 2048])
                classifier = classifier.eval()
                seg_pred, _ = classifier(points, to_categorical(label, num_classes)) # torch.Size([4, 2048, 50])
                cur_pred_val = seg_pred.cpu().data.numpy()  # (4, 2048, 50)
                cur_pred_val_logits = cur_pred_val
                cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)  # (4, 2048)
                target = target.cpu().data.numpy()  # 部件的类别一个batch中 所有点的类别  (4, 2048)
                for i in range(cur_batch_size):  # 对每个实例样本
                    cat = seg_label_to_cat[target[i, 0]]  # 获取每一个点其所对应的实例类别
                    logits = cur_pred_val_logits[i, :, :]    # (2048, 50)
                    cur_pred_val[i, :] = np.argmax(logits[:, seg_classes[cat]], 1) + seg_classes[cat][0]  # argmax 取出logits[:, seg_classes[cat]], 1)中元素最大值的索引，
                correct = np.sum(cur_pred_val == target)  # 7200
                total_correct += correct  # 当前正确点的总和
                total_seen += (cur_batch_size * NUM_POINT)  # 当前总的可见点，已经推理过的点
                # 每个部件进行统计
                for l in range(num_part):
                    total_seen_class[l] += np.sum(target == l)  # 每个部件类别总的 需要判断的点
                    total_correct_class[l] += (np.sum((cur_pred_val == l) & (target == l)))  # 每个部件类别正确的点

                for i in range(cur_batch_size):
                    segp = cur_pred_val[i, :]  # (4, 2048)
                    segl = target[i, :]
                    cat = seg_label_to_cat[segl[0]]  # 任意一个部件类别，即可确定一个实例类别
                    part_ious = [0.0 for _ in range(len(seg_classes[cat]))]  # 实例类别cat有多少个子类别，生成同尺寸的0.0的列表
                    for l in seg_classes[cat]:
                        if (np.sum(segl == l) == 0) and (np.sum(segp == l) == 0):  # part is not present, no prediction as well
                            part_ious[l - seg_classes[cat][0]] = 1.0
                        else:
                            part_ious[l - seg_classes[cat][0]] = np.sum((segl == l) & (segp == l)) / float(np.sum((segl == l) | (segp == l)))
                    shape_ious[cat].append(np.mean(part_ious))  # 每个样本的平均部件iou

            all_shape_ious = []
            for cat in shape_ious.keys():  # 计算所有shape的部件 实例iou
                for iou in shape_ious[cat]:
                    all_shape_ious.append(iou)
                shape_ious[cat] = np.mean(shape_ious[cat])  # 每个shape的平均实例iou
            mean_shape_ious = np.mean(list(shape_ious.values()))
            test_metrics['accuracy'] = total_correct / float(total_seen)
            test_metrics['class_avg_accuracy'] = np.mean(
                np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float))
            for cat in sorted(shape_ious.keys()):
                log_string('eval mIoU of %s %f' % (cat + ' ' * (14 - len(cat)), shape_ious[cat]))
            test_metrics['class_avg_iou'] = mean_shape_ious
            test_metrics['inctance_avg_iou'] = np.mean(all_shape_ious)


        log_string('Epoch %d test Accuracy: %f  Class avg mIOU: %f   Inctance avg mIOU: %f' % (
                 epoch+1, test_metrics['accuracy'],test_metrics['class_avg_iou'],test_metrics['inctance_avg_iou']))
        if (test_metrics['inctance_avg_iou'] >= best_inctance_avg_iou):
            logger.info('Save model...')
            savepath = str(checkpoints_dir) + '/best_model.pth'
            log_string('Saving at %s'% savepath)
            state = {
                'epoch': epoch,
                'train_acc': train_instance_acc,
                'test_acc': test_metrics['accuracy'],
                'class_avg_iou': test_metrics['class_avg_iou'],
                'inctance_avg_iou': test_metrics['inctance_avg_iou'],
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
            log_string('Saving model....')

        if test_metrics['accuracy'] > best_acc:
            best_acc = test_metrics['accuracy']
        if test_metrics['class_avg_iou'] > best_class_avg_iou:
            best_class_avg_iou = test_metrics['class_avg_iou']
        if test_metrics['inctance_avg_iou'] > best_inctance_avg_iou:
            best_inctance_avg_iou = test_metrics['inctance_avg_iou']
        log_string('Best accuracy is: %.5f'%best_acc)
        log_string('Best class avg mIOU is: %.5f'%best_class_avg_iou)
        log_string('Best inctance avg mIOU is: %.5f'%best_inctance_avg_iou)
        global_epoch+=1


# python train_partseg.py --model pointnet2_part_seg_msg --normal --log_dir pointnet2_part_seg_msg
if __name__ == '__main__':
    args = parse_args()
    main(args)



