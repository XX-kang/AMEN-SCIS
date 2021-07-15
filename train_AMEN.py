# coding:utf-8
import os
import time
import torch
import random
import numpy as np
import argparse

import gen_attention
from torch.utils import data
from os.path import join as pjoin
from torch import optim
from torch.nn import functional as F
from torch.backends import cudnn
import cv2
import math
from ptclassifaction.metrics import averageMeter, runningScore
from ptclassifaction.utils import make_dir, get_logger,generate_yaml_doc_ruamel,append_yaml_doc_ruamel
from ptclassifaction.loader import get_loader
from ptclassifaction.models import get_model
import save_result
os.environ["CUDA_VISIBLE_DEVICES"] = '6'
devices_ids = 0

def get_arguments(data_path,data_name,channels,classes,model_arch):
    parser = argparse.ArgumentParser(description="Pytorch Classification Master")
    #parser.add_argument("--config-name", type=str, default='BK_CAM',help="")
    parser.add_argument("--model-arch", type=str, default=model_arch, help="")
    parser.add_argument("--data-path", type=str, default=data_path, help="")
    parser.add_argument("--data-name", type=str, default=data_name, help="")
    parser.add_argument("--dataset", type=str, default='imagefolder',help="")
    #parser.add_argument("--cuda_devices", type=str, default='3', help="")
    parser.add_argument("--train-split", type=str, default='train', help="")
    parser.add_argument("--test-split", type=str, default='test', help="")
    parser.add_argument("--n-classes", type=int, default=classes, help="")
    parser.add_argument("--img-rows", type=int, default=256, help="")
    parser.add_argument("--img-cols", type=int, default=256, help="")
    parser.add_argument("--input-channels", type=int, default=channels, help="")
    # parser.add_argument("--fold-series", type=str, default='1', help="")
    parser.add_argument("--seed", type=int, default=1334, help="")
    parser.add_argument("--train-iters", type=int, default=100, help="")
    parser.add_argument("--batch-size", type=int, default=32, help="")
    parser.add_argument("--val-interval", type=int, default=10, help="")
    parser.add_argument("--n-workers", type=int, default=16, help="")
    parser.add_argument("--print-interval", type=int, default=1, help="")
    parser.add_argument("--optimizer-name", type=str, default='sdg', help="")
    parser.add_argument("--lr", type=float, default=0.0001, help="")
    parser.add_argument("--weight-decay", type=float, default=0.08, help="")
    parser.add_argument("--momentum", type=float, default=0.99, help="")
    parser.add_argument("--loss-name", type=str, default='cross_entropy', help="")
    parser.add_argument("--pkl-path", type=str, default='./pkls', help="")
    parser.add_argument("--resume", type=str, default='', help="")

    return parser.parse_args()


def net(args, flag, trained_model, alpha, output_dir,sample_weight,train_num,test_num):

    # Setup seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    cudnn.benchmark = False  # if benchmark=True, deterministic will be False
    cudnn.deterministic = True

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(0)
    # Setup Dataloader
    train_loader = get_loader(args.dataset)(
        args.data_path,
        is_transform=True,
        split=args.train_split,
        img_size=(args.img_rows, args.img_cols),
        n_classes=args.n_classes,
        sample_weight = sample_weight
    )

    test_loader = get_loader(args.dataset)(
        args.data_path,
        is_transform=True,
        split=args.test_split,
        img_size=(args.img_rows, args.img_cols),
        n_classes=args.n_classes,
        sample_weight=sample_weight
    )

    trainloader = data.DataLoader(
        train_loader,
        batch_size=args.batch_size,
        num_workers=args.n_workers,
        shuffle=True,
    )

    testloader = data.DataLoader(
        test_loader,
        batch_size=args.batch_size,
        num_workers=0
    )

    attentiontrainloader = data.DataLoader(
        train_loader,
        batch_size=1,
        num_workers=0
    )

    attentiontestloader = data.DataLoader(
        test_loader,
        batch_size=1,
        num_workers=0
    )
    # Setup Metrics
    running_metrics_val = runningScore(args.n_classes)

    # Setup Model
    if flag == 0:
        #print("no!")
        model = get_model(args.model_arch, pretrained=True, num_classes=args.n_classes,
                          input_channels=args.input_channels).to(device)
    else:
        #print("yes!")
        model = torch.load("best_model.pth")
    optimizer = optim.SGD(model.parameters(), lr=0.001)


    model = model.to(device)
    loss_fn = torch.nn.CrossEntropyLoss()

    val_loss_meter = averageMeter()
    time_meter = averageMeter()

    best_acc = -100
    epoch_iter = 0
    running_loss = 0.0
    '''train'''
    error_rate = 0
    Z = 0
    classifier_weight = 0
    Pre_train = np.zeros((train_num,1))
    Label_train = np.zeros((train_num,1))
    for epoch in range(epoch_iter, args.train_iters):
        # train_inter
        pre_train = []
        label_train = []
        error_rate = 0
        for (train_images, train_cla_labels, _,sample_weight_bs) in trainloader:
            start_ts = time.time()
            model.train()
            train_images = train_images.to(device)
            train_cla_labels = train_cla_labels.to(device)
            optimizer = optim.SGD(model.parameters(), lr=0.01 * 1.0 / (1.0 + 0.1 * epoch))
            #optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9,weight_decay=0.08)
            #optimizer = optim.Adam(model.parameters(),lr= 0.001, betas=(0.9,0.99),eps= 1e-08,weight_decay=0)
            optimizer.zero_grad()
            outputs = model(train_images)
            _, predict = torch.max(outputs.data, 1)
            for i in range(predict.shape[0]):
                pre_train.append(predict[i].data.item())
                label_train.append(train_cla_labels[i].item())
            outputs_weight = train_num * outputs * sample_weight_bs.to(device)
            loss = loss_fn(outputs_weight, train_cla_labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            time_meter.update(time.time() - start_ts)
        print("epoch",epoch)

        # print_interval
        if (epoch + 1) % args.print_interval == 0:
            fmt_str = "Iter [{:d}/{:d}]  Loss: {:.4f}  Time/Image: {:.4f}".format(
                epoch + 1,
                args.train_iters,
                running_loss / args.print_interval,
                time_meter.avg / args.batch_size,
            )
            running_loss = 0.0
            #print(fmt_str)
            time_meter.reset()

        if epoch+1 == args.train_iters:
            Pre_train = np.array(pre_train)[:]
            Label_train = np.array(label_train)[:]
            for i in range(Pre_train.shape[0]):
                error_rate = error_rate + sample_weight[i] * bool(Pre_train[i] ^ Label_train[i])
            #print("error_rate", error_rate)
            classifier_weight = 0.5 * math.log((1 - error_rate) / error_rate)
            #print("classifier_weight", classifier_weight)
            for i in range(train_num):
                Z = Z + sample_weight[i] * math.exp(-classifier_weight * Pre_train[i] * Label_train[i])
            for i in range(train_num):
                sample_weight[i] = (sample_weight[i] / Z) * math.exp(-classifier_weight * Pre_train[i] * Label_train[i])
        if (epoch + 1) % args.val_interval == 0 or epoch + 1 == args.train_iters:
            '''test'''
            Pre_test = np.zeros((test_num, 1))
            Label_test = np.zeros((test_num, 1))
            model.eval()
            Weight_output = torch.Tensor(args.batch_size, 1)
            with torch.no_grad():
                pre_test = []
                label_test = []
                flag_local = 0
                for (val_images, val_labels, _, _) in testloader:
                    val_images = val_images.to(device)
                    val_labels = val_labels.to(device)
                    outputs = model(val_images)
                    weight_outputs = classifier_weight * outputs
                    if flag_local == 0:
                        Weight_output = weight_outputs.cuda().data.cpu().numpy()
                        flag_local = 1
                    else:
                        Weight_output = np.concatenate((Weight_output, weight_outputs.cuda().data.cpu().numpy()), 0)
                    val_loss = loss_fn(input=outputs, target=val_labels)
                    _, predict = torch.max(outputs.data, 1)
                    for i in range(predict.shape[0]):
                        pre_test.append(predict[i].data.item())
                        label_test.append(val_labels[i].data.item())
                    running_metrics_val.update(predict.cuda().data.cpu().numpy(),
                                               val_labels.cuda().data.cpu().numpy())
                    val_loss_meter.update(val_loss.item())
                Pre_test = np.array(pre_test)[:]
                Label_test = np.array(label_test)[:]
            score = running_metrics_val.get_scores()
            print("score", score)
            if score["Overall Acc: \t"] > best_acc:
                Score = score
                best_acc = score["Overall Acc: \t"]
                torch.save(model, "best_model.pth")
            val_loss_meter.reset()
            running_metrics_val.reset()

    '''Attention'''
    attention_map(args, attentiontrainloader, device, model, alpha, output_dir)
    attention_map(args, attentiontestloader, device, model, alpha, output_dir)
    return model, error_rate, classifier_weight, sample_weight, Label_train, Pre_train, Label_test, Pre_test,Weight_output,Score


def attention_map(args,loader,device,model,alpha,output_dir):
    with torch.no_grad():
        for (test_images, test_labels, test_images_name,_) in loader:
            test_images = test_images.to(device)
            features_blo = []
            def hook_feature(module, input, outputs):
                features_blo.append(outputs)

            #print(args.model_arch)
            if args.model_arch == 'densenet121' or args.model_arch == 'vgg16_cam' or args.model_arch == 'vgg19_cam':
                model._modules.get('features').register_forward_hook(hook_feature)
            if args.model_arch == 'googlenet':
                model._modules.get('inception5b').register_forward_hook(hook_feature)
            if args.model_arch == 'resnet50' or args.model_arch == 'resnet101':
                model._modules.get('layer4').register_forward_hook(hook_feature)
            logit = model(test_images)
            # get the softmax weight
            params = list(model.parameters())
            weight_fc1 = params[-2].data.squeeze()
            #weight_fc1 = params[-4].data.squeeze()
            h_x = F.softmax(logit, dim=1).data.squeeze()
            probs, class_idx = h_x.sort(0, True)
            class_idx = class_idx.cuda().data.cpu().numpy()
            feature_conv = features_blo[0]
            bz, nc, h, w = feature_conv.shape
            output_cam = []
            for idx in class_idx:
                cam = torch.matmul(weight_fc1[idx], feature_conv.reshape((nc, h * w)))
                cam = cam.reshape(1, 1, h, w)
                cam = cam - torch.min(cam)  # Normalization
                cam = cam / torch.max(cam) * 255
                cam = np.squeeze(np.uint8(cam.cuda().data.cpu().numpy()))
                output_cam.append(cam)
            heatmap = cv2.applyColorMap(cv2.resize(output_cam[0], (args.img_rows, args.img_cols)), cv2.COLORMAP_JET)
            img = np.squeeze(np.uint8(test_images.cuda().data.cpu().numpy() * 255))
            img = img.swapaxes(0, 1).swapaxes(1, 2)
            result = heatmap * alpha + img
            img_result_name = pjoin(output_dir,''.join(test_images_name)+'.png')
            #print(img_result_name)
            path = img_result_name.rsplit("/", 1)[0]
            #print(path)
            if not os.path.exists(path):
                os.makedirs(path)
            cv2.imwrite(img_result_name, result)

def test_Adaboost(train_num,test_num,modelarch1,sample_weight,model):
    # train_num = 1979(100X)
    # test_num = 272
    # train_num = 1772(200X)
    # test_num = 241
    # train_num = 1899(40X)
    # test_num = 252
    n_class = 2
    '''branch1'''
    datapath = "data/BCU_2160/"
    output_dir1 = None
    args = get_arguments(datapath, "BCU", 1, n_class, modelarch1)
    # init sample_weight
    model, _, _, sample_weight, _, _, Label_test, _, Weight_output, score = net(args,
                                                                                    alpha=alpha1,
                                                                                    flag=0,
                                                                                    trained_model=model,
                                                                                    output_dir=output_dir1,
                                                                                    sample_weight=sample_weight,
                                                                                    train_num=train_num,
                                                                                    test_num=test_num)
    save_result.generate_excel("test_Pooling.xls", score, "b1_" + modelarch1 + "_" + str(alpha1))
    return sample_weight, model


def run(modelarch1,modelarch2,modelarch3,alpha1,alpha2,alpha3):
    # train_num = 1979
    # test_num = 272
    #train_num = 1732(400X)
    #test_num = 225
    #train_num = 1772(200X)
    #test_num = 241
    # train_num = 1899
    # test_num = 252
    # n_class = 8
    train_num = 2160
    test_num = 30
    n_class = 2
    '''branch1'''
    datapath = "data/BCU_2160_3/"
    output_dir1 = "output/BCU_2160_3/branch1_" + modelarch1 + '_' + str(alpha1) + '/'
    args = get_arguments(datapath, "BCU_2160_3", 3, n_class, modelarch1)
    # init sample_weight
    sample_weight0 = np.ones((train_num, 1)) / train_num
    model1, _, _, sample_weight1, _, _, Label_test, _, Weight_output1, score1 = net(args,
                                                                       alpha=alpha1,
                                                                       flag=0,
                                                                       trained_model=None,
                                                                       output_dir=output_dir1,
                                                                       sample_weight=sample_weight0,
                                                                       train_num=train_num,
                                                                       test_num=test_num)
    save_result.generate_excel("test_Pooling.xls",score1, "b1_" + modelarch1 + "_" + str(alpha1))

    '''branch2'''
    # datapath2 = "data/BreakHis/400X/"
    output_dir2 = "output/BCU_2160_3/branch2_" + modelarch2 + '_' + str(alpha2) + '/'
    args = get_arguments(output_dir1, "BCU_2160_3", 3, n_class, modelarch2)
    model2, _, _, sample_weight2, _, _, _, _, Weight_output2, score2 = net(args,
                                                              alpha=alpha2,
                                                              flag=1,
                                                              trained_model=model1,
                                                              output_dir=output_dir2,
                                                              sample_weight=sample_weight1,
                                                              train_num=train_num,
                                                              test_num=test_num)
    save_result.generate_excel("test_Pooling.xls",score2, "b2_" + modelarch2 + "_" + str(alpha2))

    '''branch3'''
    # datapath3 = "data/BreakHis/400X/"
    output_dir3 = "output/BCU_2160_3/branch3_" + modelarch3 + '_' + str(alpha3) + '/'
    args = get_arguments(output_dir2, "BCU_2160_3", 3, n_class, modelarch3)
    model3, _, _, _, _, _, _, _, Weight_output3, score3 = net(args,
                                                              alpha=alpha3,
                                                              flag=1,
                                                              trained_model=model2,
                                                              output_dir=output_dir3,
                                                              sample_weight=sample_weight2,
                                                              train_num=train_num,
                                                              test_num=test_num)
    save_result.generate_excel("test_Pooling.xls",score3, "b3_" + modelarch3 + "_" + str(alpha3))

    '''Adaboost'''
    Output = Weight_output1 + Weight_output2 + Weight_output3
    _, class_idx = torch.sort(torch.from_numpy(Output), 1, True)
    # Setup Metrics
    running_metrics_val = runningScore(n_class)
    running_metrics_val.update(Label_test,
                               class_idx[:, 0].numpy())
    score = running_metrics_val.get_scores()
    save_result.generate_excel("test_Pooling.xls",score, "boosting")
    print("score", score)

if __name__ == "__main__":
    #0.01,, 0.001, 0.0005, 0.0001, 0.00005, 0.00001
    alpha1 = [0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001]
    alpha2 = [0.01,0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001]
    #'resnet101','googlenet','densenet121','resnet50',, "vgg19_cam"
    model_arch = "densenet121"

    """main"""
    # for m in model_arch:
    #     for i in alpha1:
    #         for j in alpha2:
    #             print("model:",m,"alpha1:",i,"alpha2:",j)
    #             run(m,m,m,i,j,j)

    """test_Adaboost"""
    # train_num = 1732(400X)
    # test_num = 225
    # train_num = 2160
    # test_num = 30
    # sample_weight = np.ones((train_num, 1)) / train_num
    # print("1")
    # sample_weight1, model1 = test_Adaboost(train_num, test_num, "densenet121", sample_weight, None)
    # print("2")
    # sample_weight2, model2 = test_Adaboost(train_num, test_num, "densenet121", sample_weight1, model1)
    # print("3")
    # sample_weight3, model3 = test_Adaboost(train_num, test_num, "densenet121", sample_weight2, model2)
    # print("4")
    # sample_weight1, model1 = test_Adaboost(train_num, test_num, "googlenet", sample_weight, None)
    # print("5")
    # sample_weight2, model2 = test_Adaboost(train_num, test_num, "googlenet", sample_weight1, model1)
    # print("6")
    # sample_weight3, model3 = test_Adaboost(train_num, test_num, "googlenet", sample_weight2, model2)
    """test_pooling"""
    run("densenet121", "densenet121", "densenet121", 0.001, 0.001, 0.001)