import torch
import configargparse
import random
import numpy as np
import os
import data_loader
import models
import utils
import sys
import itertools

def get_parser():
    """Get default arguments."""
    parser = configargparse.ArgumentParser(
        description="Physical-Informed Network config parser",
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
    )
    # general configuration
    parser.add_argument("--config", is_config_file=True, help="config file path")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=8)

    # network related
    parser.add_argument('--backbone', type=str, default='alexnet2d')
    parser.add_argument('--use_PIlayer', type=bool, default=True)
    parser.add_argument('--use_bottleneck', type=bool, default=False)

    # data loading related
    parser.add_argument('--data_dir', type=str, default=r'D:\DATABASE\ZXJ_GD\sample')
    parser.add_argument('--sim_data_dir', type=str, default=r'D:\DATABASE\ZXJ_GD\sample')
    parser.add_argument('--domain', type=str, default="10-B1-4_CH1-8")
    parser.add_argument('--num_class', type=int, default=6)
    parser.add_argument('--train_ratio', type=float, default=0.7)
    parser.add_argument('--graph', type=bool, default=False)
    parser.add_argument('--data_type', type=str, default="orig_sample")

    # training related
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_epoch', type=int, default=100)
    parser.add_argument('--max_iter', type=int, default=20000)
    parser.add_argument('--early_stop', type=int, default=0, help="Early stopping")
    parser.add_argument('--epoch_based_training', type=bool, default=False,
                        help="Epoch-based training / Iteration-based training")
    parser.add_argument("--n_iter_per_epoch", type=int, default=20, help="Used in Iteration-based training")

    # optimizer related
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)

    # learning rate scheduler related
    parser.add_argument('--lr_gamma', type=float, default=0.0003)
    parser.add_argument('--lr_decay', type=float, default=0.75)
    parser.add_argument('--lr_scheduler', type=bool, default=True)

    return parser


def set_random_seed(seed=0):
    # seed setting
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data(args):
    '''
    src_domain, tgt_domain data to load
    '''
    folder = os.path.join(args.data_dir, args.data_type, args.domain)
    train_loader = data_loader.load_data(
        folder, args.batch_size, train=True, ratio=args.train_ratio, graph=args.graph)
    test_loader = data_loader.load_data(
        folder, args.batch_size, train=False, ratio=args.train_ratio, graph=args.graph)
    return train_loader, test_loader


def get_model(args):
    # 人工造一些特征出来
    sim_feature = np.zeros([6, 4096])
    channel = list(np.arange(0, args.num_class))
    feature = [200, 400, 500, 950]
    feature_index = list(itertools.product(channel, feature))
    for i in feature_index:
        sim_feature[i] = 2
    # 这里需要一个仿真数据的加载
    model = models.PINet(num_class=args.num_class,
                         sim_feature=sim_feature,
                         base_net=args.backbone,
                         max_iter=args.max_iter,
                         use_bottleneck=args.use_bottleneck,
                         use_PIlayer=args.use_PIlayer).to(args.device)
    return model


def get_optimizer(model, args):
    initial_lr = args.lr if not args.lr_scheduler else 1.0
    params = model.get_parameters(initial_lr=initial_lr)
    # optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=False)
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    return optimizer


def get_scheduler(optimizer, args):
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (
        -args.lr_decay))
    return scheduler


def train(model, train_loader, test_loader, optimizer, lr_scheduler, args):
    print('train data name:', args.domain)
    n_batch = args.batch_size
    if n_batch == 0:
        n_batch = args.n_iter_per_epoch
    for repeat_n in range(0, 5):
        best_acc = 0
        stop = 0
        log = []
        for e in range(1, args.n_epoch + 1):
            model.train()
            train_loss_clf = utils.AverageMeter()
            iter_source = iter(train_loader)

            for f in range(args.n_iter_per_epoch):
                data_source, label_source, _ = next(iter_source)  # .next()
                data_source, label_source = data_source.to(
                    args.device), label_source.to(args.device)

                clf_loss, predict = model(data_source, label_source)

                optimizer.zero_grad()
                clf_loss.backward()
                optimizer.step()
                if lr_scheduler:
                    lr_scheduler.step()

                train_loss_clf.update(clf_loss.item())

            log.append(train_loss_clf.avg)

            info = 'Epoch: [{:2d}/{}], cls_loss: {:.4f}'.format(
                e, args.n_epoch, train_loss_clf.avg)
            # Test
            stop += 1
            test_acc, test_loss = test(model, test_loader, args)
            info += ', test_loss {:4f}, test_acc: {:.4f}'.format(test_loss, test_acc)
            np_log = np.array(log, dtype=float)
            np.savetxt('train_log.csv', np_log, delimiter=',', fmt='%.6f')
            if best_acc < test_acc:
                best_acc = test_acc
                stop = 0
            if best_acc == 100:
                print('Saving checkpoints', '/domain:', args.src_domain)
                ckpt = {'state_dict': model.state_dict()}
                train_ckpt_dir = r'D:\python_workfile\Physical_Informed\ckpt'\
                                 + '\\' + args.dataset_name + '\\'\
                                 + args.backbone + '\\' + args.data_type + '\\'
                if os.path.exists(train_ckpt_dir):
                    torch.save(ckpt, train_ckpt_dir + args.src_domain + '.tar')
                else:
                    os.makedirs(train_ckpt_dir)
                    torch.save(ckpt, train_ckpt_dir + args.src_domain + '.tar')
            if args.early_stop > 0 and stop >= args.early_stop:
                print(info)
                break
            print(info)
        print('Train result: {:.4f}'.format(best_acc), '\nRepeat num:', repeat_n)

    return best_acc


def test(model, test_loader, args):
    model.eval()
    iter_num = 10
    test_loss = utils.AverageMeter()
    correct = 0
    criterion = torch.nn.CrossEntropyLoss()
    len_test_dataset = iter_num * args.batch_size
    with torch.no_grad():
        i = 1
        for data in test_loader:
            data, target = data[0].to(args.device), data[1].to(args.device)
            s_output = model.predict(data)
            loss = criterion(s_output, target)
            test_loss.update(loss.item())
            pred = torch.max(s_output, 1)[1]
            correct += torch.sum(pred == target)
            i += 1
            if i > iter_num:
                break
    acc = 100. * correct / len_test_dataset

    return acc, test_loss.avg


if __name__ == '__main__':
    model_name = "PInet"
    dataset_name = "CWRU"
    # freq_sample, time-freq
    data_type = "freq_sample"
    backbone = "alexnet"

    config_dir = r'D:\python_workfile\Physical_Informed\YAML'
    data_dir = r'D:\DATABASE\ZXJ_GD\sample'
    sim_data_dir = r''

    domain_name = "100-B1-4_CH1-8"

    sys.argv[1] = '--config'
    sys.argv[2] = config_dir + '\\' + model_name+'.yaml'
    info_list = []

    parser = get_parser()
    args = parser.parse_args()
    setattr(args, "data_dir", data_dir)
    setattr(args, "dataset_name", dataset_name)
    setattr(args, "data_type", data_type)
    setattr(args, "backbone", backbone)
    setattr(args, "graph", False)
    setattr(args, "sim_data_dir", sim_data_dir)
    setattr(args, "device", torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    print(args)
    set_random_seed(args.seed)

    train_loader, test_loader = load_data(args)

    model = get_model(args)
    optimizer = get_optimizer(model, args)

    if args.lr_scheduler:
        scheduler = get_scheduler(optimizer, args)
    else:
        scheduler = None

    best_acc = train(model, train_loader, test_loader, optimizer, scheduler, args)

