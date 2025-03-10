import os
import time
import csv
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.optim
cudnn.benchmark = True

from models import ResNet
from metrics import AverageMeter, Result
from dataloaders.dense_to_sparse import UniformSampling, SimulatedStereo
import criteria
import utils
import dataloaders.transforms as transforms
import cv2
import math
from glob import glob
args = utils.parse_command()
print(args)

fieldnames = ['mse', 'rmse', 'absrel', 'lg10', 'mae',
                'delta1', 'delta2', 'delta3',
                'data_time', 'gpu_time']
best_result = Result()
best_result.set_to_worst()
to_tensor = transforms.ToTensor()
def create_data_loaders(args):
    # Data loading code
    print("=> creating data loaders ...")
    traindir = os.path.join('data', args.data, 'train')
    valdir = os.path.join('data', args.data, 'val')
    train_loader = None
    val_loader = None

    # sparsifier is a class for generating random sparse depth input from the ground truth
    sparsifier = None
    max_depth = args.max_depth if args.max_depth >= 0.0 else np.inf
    print("max depth "+ str(max_depth))
    if args.sparsifier == UniformSampling.name:
        sparsifier = UniformSampling(num_samples=args.num_samples, max_depth=max_depth)
    elif args.sparsifier == SimulatedStereo.name:
        sparsifier = SimulatedStereo(num_samples=args.num_samples, max_depth=max_depth)

    if args.data == 'nyudepthv2':
        from dataloaders.nyu_dataloader import NYUDataset
        if not args.evaluate:
            train_dataset = NYUDataset(traindir, type='train',
                modality=args.modality, sparsifier=sparsifier)
        print("===modality " + args.modality)
        val_dataset = NYUDataset(valdir, type='val',
            modality=args.modality, sparsifier=sparsifier)

    elif args.data == 'kitti':
        from dataloaders.kitti_dataloader import KITTIDataset
        if not args.evaluate:
            train_dataset = KITTIDataset(traindir, type='train',
                modality=args.modality, sparsifier=sparsifier)
        val_dataset = KITTIDataset(valdir, type='val',
            modality=args.modality, sparsifier=sparsifier)

    else:
        raise RuntimeError('Dataset not found.' +
                           'The dataset must be either of nyudepthv2 or kitti.')

    # set batch size to be 1 for validation
    val_loader = torch.utils.data.DataLoader(val_dataset,
        batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True)

    # put construction of train loader here, for those who are interested in testing only
    if not args.evaluate:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True, sampler=None,
            worker_init_fn=lambda work_id:np.random.seed(work_id))
            # worker_init_fn ensures different sampling patterns for each data loading thread

    print("=> data loaders created.")
    return train_loader, val_loader

def main():
    global args, best_result, output_directory, train_csv, test_csv

    # evaluation mode
    start_epoch = 0
    if args.inference:
        assert os.path.isfile(args.inference), \
        "=> no best model found at '{}'".format(args.inference)
        print("=> loading best model '{}'".format(args.inference))
        checkpoint = torch.load(args.inference, map_location=torch.device('cpu'))
        output_directory = os.path.dirname(args.inference)
        # args = checkpoint['args']
        model = checkpoint['model']
        print("=> loaded best model (epoch {})".format(checkpoint['epoch']))
        # inference(model,"256", "256_dense.png")
    #         parser.add_argument('--sparsepath', type=str, default='/home/menghe/Github/PEAC/sparse_point/0221/',
    #                     help='absolute path of sparse depth points')
    # parser.add_argument('--rgbpath', type=str, default="/home/menghe/Github/mediapipe/frames/0221/",
    #                     help='absolute path of frames')
        rgbpath = "/home/menghe/Github/mediapipe/frames/0221/"
        sparsepath = "/home/menghe/Github/PEAC/sparse_point/0221/"
        inference(model, args.rgbpath, args.sparsepath, torch.cuda.is_available())
        return

    elif args.evaluate:
        assert os.path.isfile(args.evaluate), \
        "=> no best model found at '{}'".format(args.evaluate)
        print("=> loading best model '{}'".format(args.evaluate))
        checkpoint = torch.load(args.evaluate)
        output_directory = os.path.dirname(args.evaluate)
        args = checkpoint['args']
        start_epoch = checkpoint['epoch'] + 1
        best_result = checkpoint['best_result']
        model = checkpoint['model']
        print("=> loaded best model (epoch {})".format(checkpoint['epoch']))
        _, val_loader = create_data_loaders(args)
        args.evaluate = True
        validate(val_loader, model, checkpoint['epoch'], write_to_file=False)
        return

    # optionally resume from a checkpoint
    elif args.resume:
        chkpt_path = args.resume
        assert os.path.isfile(chkpt_path), \
            "=> no checkpoint found at '{}'".format(chkpt_path)
        print("=> loading checkpoint '{}'".format(chkpt_path))
        checkpoint = torch.load(chkpt_path)
        args = checkpoint['args']
        start_epoch = checkpoint['epoch'] + 1
        best_result = checkpoint['best_result']
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']
        output_directory = os.path.dirname(os.path.abspath(chkpt_path))
        print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
        train_loader, val_loader = create_data_loaders(args)
        args.resume = True

    # create new model
    else:
        train_loader, val_loader = create_data_loaders(args)
        print("=> creating Model ({}-{}) ...".format(args.arch, args.decoder))
        in_channels = len(args.modality)
        if args.arch == 'resnet50':
            model = ResNet(layers=50, decoder=args.decoder, output_size=train_loader.dataset.output_size,
                in_channels=in_channels, pretrained=args.pretrained)
        elif args.arch == 'resnet18':
            model = ResNet(layers=18, decoder=args.decoder, output_size=train_loader.dataset.output_size,
                in_channels=in_channels, pretrained=args.pretrained)
        print("=> model created.")
        optimizer = torch.optim.SGD(model.parameters(), args.lr, \
            momentum=args.momentum, weight_decay=args.weight_decay)

        # model = torch.nn.DataParallel(model).cuda() # for multi-gpu training
        model = model.cuda()

    # define loss function (criterion) and optimizer
    if args.criterion == 'l2':
        criterion = criteria.MaskedMSELoss().cuda()
    elif args.criterion == 'l1':
        criterion = criteria.MaskedL1Loss().cuda()

    # create results folder, if not already exists
    output_directory = utils.get_output_directory(args)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    train_csv = os.path.join(output_directory, 'train.csv')
    test_csv = os.path.join(output_directory, 'test.csv')
    best_txt = os.path.join(output_directory, 'best.txt')

    # create new csv files with only header
    if not args.resume:
        with open(train_csv, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
        with open(test_csv, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

    for epoch in range(start_epoch, args.epochs):
        utils.adjust_learning_rate(optimizer, epoch, args.lr)
        train(train_loader, model, criterion, optimizer, epoch) # train for one epoch
        result, img_merge = validate(val_loader, model, epoch) # evaluate on validation set

        # remember best rmse and save checkpoint
        is_best = result.rmse < best_result.rmse
        if is_best:
            best_result = result
            with open(best_txt, 'w') as txtfile:
                txtfile.write("epoch={}\nmse={:.3f}\nrmse={:.3f}\nabsrel={:.3f}\nlg10={:.3f}\nmae={:.3f}\ndelta1={:.3f}\nt_gpu={:.4f}\n".
                    format(epoch, result.mse, result.rmse, result.absrel, result.lg10, result.mae, result.delta1, result.gpu_time))
            if img_merge is not None:
                img_filename = output_directory + '/results/comparison_best.png'
                utils.save_image(img_merge, img_filename)

        utils.save_checkpoint({
            'args': args,
            'epoch': epoch,
            'arch': args.arch,
            'model': model,
            'best_result': best_result,
            'optimizer' : optimizer,
        }, is_best, epoch, output_directory)


def train(train_loader, model, criterion, optimizer, epoch):
    average_meter = AverageMeter()
    model.train() # switch to train mode
    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        input, target = input.cuda(), target.cuda()
        torch.cuda.synchronize()
        data_time = time.time() - end

        # compute pred
        end = time.time()
        pred = model(input)
        loss = criterion(pred, target)
        optimizer.zero_grad()
        loss.backward() # compute gradient and do SGD step
        optimizer.step()
        torch.cuda.synchronize()
        gpu_time = time.time() - end

        # measure accuracy and record loss
        result = Result()
        result.evaluate(pred.data, target.data)
        average_meter.update(result, gpu_time, data_time, input.size(0))
        end = time.time()

        if (i + 1) % args.print_freq == 0:
            print('=> output: {}'.format(output_directory))
            print('Train Epoch: {0} [{1}/{2}]\t'
                  't_Data={data_time:.3f}({average.data_time:.3f}) '
                  't_GPU={gpu_time:.3f}({average.gpu_time:.3f})\n\t'
                  'RMSE={result.rmse:.2f}({average.rmse:.2f}) '
                  'MAE={result.mae:.2f}({average.mae:.2f}) '
                  'Delta1={result.delta1:.3f}({average.delta1:.3f}) '
                  'REL={result.absrel:.3f}({average.absrel:.3f}) '
                  'Lg10={result.lg10:.3f}({average.lg10:.3f}) '.format(
                  epoch, i+1, len(train_loader), data_time=data_time,
                  gpu_time=gpu_time, result=result, average=average_meter.average()))

    avg = average_meter.average()
    with open(train_csv, 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({'mse': avg.mse, 'rmse': avg.rmse, 'absrel': avg.absrel, 'lg10': avg.lg10,
            'mae': avg.mae, 'delta1': avg.delta1, 'delta2': avg.delta2, 'delta3': avg.delta3,
            'gpu_time': avg.gpu_time, 'data_time': avg.data_time})
def get_depth(sparse_file, out_size, b_filter = True, b_sample = True):
    try:
        sparse_file = open(sparse_file)                                                                                                      
    except IOError:
        print("sparse file not exist")
        return None

    lines = sparse_file.readlines()
    depth = np.zeros(out_size)
    cloud = np.zeros((480,640,3))

    if b_sample and len(lines)>300:
        cs = max(300, int(len(lines) * 0.7))
        lines = np.random.choice(lines, cs, replace=False)
    for line in lines: 
        nums = line.split(' ')
        vy, vx, vpx, vpy, vpz = float(nums[0]),float(nums[1]),float(nums[2]),float(nums[3]),float(nums[4]) 
        
        y = int (vy / 480* out_size[0])
        x = int(vx / 640 * out_size[1])
        depth[y][x] = math.sqrt(vpx**2 + vpy**2 + vpz**2)
        cloud[int (vy)][int(vx)] = [vpx, vpy, vpz]
    if b_filter:
        vs = np.unique(depth)
        vs.sort()
        print(vs)
        lf = vs[int(len(vs) * 0.1)]
        hf = vs[int(len(vs) * 0.9)]
        depth[np.where(np.logical_or((depth < lf), (depth > hf)))] = .0
        # if(len(lines) > 200):
    
    return depth, cloud

# def inference(model, filename, out_name):
def inference(model, rgb_path, sparse_path, b_gpu):
    # sparse_files = glob(sparse_path+"*.tiff")
    # print(sparse_files)
    sparse_files = glob(sparse_path + "*.txt")
    print(sparse_files)
    postfix = sparse_files[0].split('/')[-2]
    for sparse in sparse_files:
        #get frame
        # time_stamp = sparse.split('/')[-1].split('_')[0]
        time_stamp = sparse.split('/')[-1].split('.')[0]
        print(time_stamp)
        filename_rgb = rgb_path + time_stamp + ".png"
        print(filename_rgb)
        model.eval()
        rgb = cv2.imread(filename_rgb)
        if rgb is None:
            continue
        # rgb=cv2.flip( rgb, 1 )
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        # depth = cv2.imread(sparse, cv2.IMREAD_UNCHANGED)
        depth, cloud = get_depth(sparse, (228,304))
        if depth is None:
            continue
        np.save("res/"+postfix+"/"+time_stamp+".npz", cloud)
        
        #transform

        # transform = transforms.Compose([
        #     transforms.Resize(240.0 / 480),
        #     transforms.CenterCrop((228, 304)),
        # ])
        # rgb_np = transform(rgb)
        rgb_np = cv2.resize(rgb,(304,228))

        # cv2.imwrite("res/"+postfix+"/"+time_stamp+"_test.png", rgb_np)

        
        rgb_np = np.asfarray(rgb_np, dtype='float') / 255

        # depth_np = transform(depth)
        # depth_np = cv2.resize(depth,(304,228), interpolation=cv2.INTER_NEAREST)

        # depth_np = np.asfarray(depth_np, dtype='float')
        depth_np = depth
        print(rgb_np.shape)
        print(depth_np.shape)

        input_np = np.append(rgb_np, np.expand_dims(depth_np, axis=2), axis=2)
        print(input_np.shape)
        
        # input_np = np.transpose(input_np, (1, 2, 0))
        # print(input_np.shape)
        input_tensor = to_tensor(input_np)
        while input_tensor.dim() < 4:
            input_tensor = input_tensor.unsqueeze(0)
        if b_gpu:
            input_tensor = input_tensor.cuda()
        with torch.no_grad():
            print(input_tensor.size())
            pred = model(input_tensor)
        if b_gpu:    
            torch.cuda.synchronize()
        depth_pred_cpu = np.squeeze(pred.data.cpu().numpy())
        # print(depth_pred_cpu)
        # print("====")
        # print(np.max(depth_pred_cpu))

        res = depth_pred_cpu / np.max(depth_pred_cpu) * 255
        res = cv2.resize(res,(640,480))
        # res = res.astype(np.uint16)
        print(np.unique(res))

        cv2.imwrite("res/"+postfix+"/"+time_stamp+"_dense.png", res)

        img_merge = utils.merge_into_row(input_tensor[:,:3,:,:], input_tensor[:,3,:,:], pred)
        cv2.imwrite("res/"+postfix+"/"+time_stamp+"_comp.png", img_merge)




def validate(val_loader, model, epoch, write_to_file=True):
    average_meter = AverageMeter()
    model.eval() # switch to evaluate mode
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        input, target = input.cuda(), target.cuda()
        
        # trgb = input[:,:3,:,:]
        # tdepth = input[:,3:,:,:]
        # tdepth_cpu = np.squeeze(tdepth.data.cpu().numpy())
        # print(np.unique(tdepth_cpu))
        # print("====")
        # print(np.max(tdepth_cpu))

        torch.cuda.synchronize()
        data_time = time.time() - end

        # compute output
        end = time.time()
        with torch.no_grad():
            print(input.size())
            pred = model(input)
        torch.cuda.synchronize()
        gpu_time = time.time() - end

        # measure accuracy and record loss
        result = Result()
        result.evaluate(pred.data, target.data)
        average_meter.update(result, gpu_time, data_time, input.size(0))
        end = time.time()



        # save 8 images for visualization
        skip = 50
        if args.modality == 'd':
            img_merge = None
        else:
            if args.modality == 'rgb':
                rgb = input
            elif args.modality == 'rgbd':
                rgb = input[:,:3,:,:]
                depth = input[:,3:,:,:]

            if i == 0:
                if args.modality == 'rgbd':
                    img_merge = utils.merge_into_row_with_gt(rgb, depth, target, pred)
                    print("add row rgbd i=0 " + str(epoch))

                else:
                    img_merge = utils.merge_into_row(rgb, target, pred)
                    print("add row i=0" + str(epoch))

            elif (i < 8*skip) and (i % skip == 0):
                if args.modality == 'rgbd':
                    print("add row rgbd" + str(epoch))
                    row = utils.merge_into_row_with_gt(rgb, depth, target, pred)

                    filename = output_directory + '/results/pred_' + str(i) + '.png'
                    utils.save_image(row,filename)
                else:
                    row = utils.merge_into_row(rgb, target, pred)
                    print("add row " + str(epoch))
                img_merge = utils.add_row(img_merge, row)
            elif i == 8*skip:
                filename = output_directory + '/results/comparison_' + str(epoch) + '.png'
                utils.save_image(img_merge, filename)

        # if (i+1) % args.print_freq == 0:
        #     print('Test: [{0}/{1}]\t'
        #           't_GPU={gpu_time:.3f}({average.gpu_time:.3f})\n\t'
        #           'RMSE={result.rmse:.2f}({average.rmse:.2f}) '
        #           'MAE={result.mae:.2f}({average.mae:.2f}) '
        #           'Delta1={result.delta1:.3f}({average.delta1:.3f}) '
        #           'REL={result.absrel:.3f}({average.absrel:.3f}) '
        #           'Lg10={result.lg10:.3f}({average.lg10:.3f}) '.format(
        #            i+1, len(val_loader), gpu_time=gpu_time, result=result, average=average_meter.average()))

    avg = average_meter.average()

    print('\n*\n'
        'RMSE={average.rmse:.3f}\n'
        'MAE={average.mae:.3f}\n'
        'Delta1={average.delta1:.3f}\n'
        'REL={average.absrel:.3f}\n'
        'Lg10={average.lg10:.3f}\n'
        't_GPU={time:.3f}\n'.format(
        average=avg, time=avg.gpu_time))

    if write_to_file:
        with open(test_csv, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({'mse': avg.mse, 'rmse': avg.rmse, 'absrel': avg.absrel, 'lg10': avg.lg10,
                'mae': avg.mae, 'delta1': avg.delta1, 'delta2': avg.delta2, 'delta3': avg.delta3,
                'data_time': avg.data_time, 'gpu_time': avg.gpu_time})
    return avg, img_merge

if __name__ == '__main__':
    main()
