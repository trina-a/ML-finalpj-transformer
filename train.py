import ml_collections
import argparse
from modeling import VisionTransformer
from load_cifar100 import get_loader
import torch
import os
import numpy as np
import json

def get_config():
    '''
    配置transformer的模型的参数
    '''
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size':2}) # 原论文为16
    config.hidden_size =504 #原论文为768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim =2016 # 原论文为3072
    config.transformer.num_heads = 12
    config.transformer.num_layers = 12
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None
    return config


def save_model(args, model,epoch_index):
    '''
    保存每个epoch训练的模型
    '''
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(args.output_dir, "epoch%s_checkpoint.bin" % epoch_index)
    torch.save(model_to_save.state_dict(), model_checkpoint)



#实例化模型
def getVisionTransformers_model(args):
    config=get_config()#获取模型的配置文件
    num_classes = 10 if args.dataset == "cifar10" else 100
    model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=num_classes)
    model.to(args.device)
    return args,model


#用测试集评估模型的训练好坏
def eval(args,model,test_loader):
    eval_loss=0.0
    total_acc=0.0
    model.eval()
    loss_function = torch.nn.CrossEntropyLoss()
    for i,batch in enumerate(test_loader):
        batch = tuple(t.to(args.device) for t in batch)
        x, y = batch
        with torch.no_grad():
            logits,_= model(x)#model返回的是（bs,num_classes）和weight
            batch_loss=loss_function(logits,y)
            #记录误差
            eval_loss+=batch_loss.item()
            #记录准确率
            _,preds= logits.max(1)
            num_correct=(preds==y).sum().item()
            total_acc+=num_correct

    loss=eval_loss/len(test_loader)
    acc=total_acc/(len(test_loader)*args.eval_batch_size)
    return loss,acc





def train(args,model,method):
    print("load dataset.........................")
    #加载数据
    train_loader, test_loader = get_loader(args)
    # Prepare optimizer and scheduler
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.learning_rate,
                                momentum=0.9,
                                weight_decay=args.weight_decay)

    print("training.........................")
    #设置测试损失list,和测试acc 列表
    val_loss_list=[]
    val_acc_list=[]
    #设置训练损失list,和训练acc列表
    train_loss_list=[]
    train_acc_list = []
    for i in range(args.total_epoch):
        model.train()
        train_loss=0
        for step, batch in enumerate(train_loader):
            batch = tuple(t.to(args.device) for t in batch)
            x, y = batch
            loss = model(x, y,method)
            train_loss +=loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # 每一个iteration记录一次loss和accuracy
            train_loss_list.append(loss.item())
            with torch.no_grad():
                logits,_=model(x)
                _,preds=logits.max(1)
                mini_batch_acc=np.array((preds==y), dtype="float64").mean()
                train_acc_list.append(mini_batch_acc)

            # 每100个iteration输出一次
            if step % 100 == 99:
                print('epoch: %d  minibatch: %d ====== mean loss:%.3f  train_accuracy:%.4f' % (
                    i + 1, i + 1, train_loss / 100, mini_batch_acc))
                train_loss=0

        # # 每个epoch保存一次模型参数
        # save_model(args, model,i)
        # 每训练一个epoch,用当前训练的模型对验证集进行测试
        eval_loss, eval_acc = eval(args, model, test_loader)
        #将每一个测试集验证的结果加入列表
        val_loss_list.append(eval_loss)
        val_acc_list.append(eval_acc)
        print("val Epoch:{},eval_loss:{},eval_acc:{}".format(i, eval_loss, eval_acc))

    # 保存loss,acc和模型
    filename1 = args.output_dir+method + '_train_loss_list.json'
    with open(filename1, 'w') as file_obj:
        json.dump(train_loss_list, file_obj)
    filename2 = args.output_dir+method + '_train_acc_list.json'
    with open(filename2, 'w') as file_obj:
        json.dump(train_acc_list, file_obj)
    filename3 = args.output_dir+method +'_val_loss_list.json'
    with open(filename3, 'w') as file_obj:
        json.dump(val_loss_list, file_obj)
    filename4 = args.output_dir+method +'_val_acc_list.json'
    with open(filename4, 'w') as file_obj:
        json.dump(val_acc_list, file_obj)
    torch.save(model, args.output_dir+method + '.pth')  # 保存模型


def main(method):
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar100",
                        help="Which downstream task.")
    parser.add_argument("--output_dir", default="./output", type=str,
                        help="The output directory where checkpoints will be written.")
    parser.add_argument("--img_size", default=32, type=int,help="Resolution size")
    parser.add_argument("--train_batch_size", default=64, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate", default=3e-2, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--total_epoch", default=50, type=int,
                        help="Total number of training epochs to perform.")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    args,modle=getVisionTransformers_model(args)
    train(args,modle,method=method)

if __name__ == "__main__":
    main('baseline')
    main('mixup')
    main('cutmix')
    main('cutout')