import os
import torch
import torch.nn as nn
import json
import numpy as np
import math
import copy
# from dataset import TrainOrchidDataset,ValOrchidDataset
# from config_eval import get_args
from torchvision import transforms
import pandas as pd
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import os
from apex import amp
import torch.distributed as dist


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt

def get_args():
    parser = argparse.ArgumentParser("")
    # New Added! 05/11
    parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
                                                 "ViT-L_32", "ViT-H_14"],
                        default="ViT-B_16",
                        help="Which variant to use.")

    # checkpoint_L = ["./records/"+name for name in os.listdir("./records") if os.path.isdir("./records/"+name)]
    # checkpoint = max(checkpoint_L, key=os.path.getctime) 
    parser.add_argument("--pretrained_path", default=f"Best/sample_smooth0.07_acc0.938356_checkpoint.bin", type=str)
    parser.add_argument("--data_root", default="./dataset", type=str) 
    parser.add_argument("--data_size", default=448, type=int)
    # parser.add_argument("--pretrained_dir", type=str, default="pretrained_models/ViT-B_16.npz",
    #                     help="Where to search for pretrained ViT models.")
    # parser.add_argument("--img_size", default=448, type=int)
    # parser.add_argument("--num_rows", default=0, type=int)
    # parser.add_argument("--num_cols", default=0, type=int)
    # parser.add_argument("--sub_data_size", default=32, type=int)

    parser.add_argument("--model_name", default="TransFG", type=str, 
        choices=["efficientnet-b7", 'resnet-50', 'vit-b16', 'swin-vit-p4w12', 'TransFG'])
    # parser.add_argument("--optimizer_name", default="sgd", type=str, 
    #     choices=["sgd", 'adamw'])
    
    # parser.add_argument("--use_fpn", default=True, type=bool)
    # parser.add_argument("--use_ori", default=False, type=bool)
    # parser.add_argument("--use_gcn", default=True, type=bool)
    # parser.add_argument("--use_layers", 
    #     default=[True, True, True, True], type=list)
    # parser.add_argument("--use_selections", 
    #     default=[True, True, True, True], type=list)
    # # [2048, 512, 128, 32] for CUB200-2011
    # # [256, 128, 64, 32] for NABirds
    # parser.add_argument("--num_selects",
    #     default=[2048, 512, 128, 32], type=list)
    # parser.add_argument("--global_feature_dim", default=1536, type=int)
    
    # loader
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--batch_size", default=8, type=int)
    
    # about model building
    parser.add_argument("--num_classes", default=219, type=int)
    parser.add_argument("--test_global_top_confs", default=[1,3,5], type=list)
    parser.add_argument("--tta",default=False, type=bool)
    parser.add_argument("--tv",help = 'test with voting after csv is done',default=False, action='store_true')
    args = parser.parse_args()

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return args

class TestOrchidDataset(Dataset):
    def __init__(self, root_dir='./dataset/', csv_file='dataset/val_label.csv', transform=None, is_test=False):
        self.x = []
        self.y = []
        self.filenames = []
        self.transform = transform
        sample_submission = pd.read_csv(csv_file)
        for idx, filename in enumerate(sample_submission['filename']):
            # print("filename: ", filename, "Label: ",sample_submission['category'][idx])
            filepath = os.path.join(root_dir, filename)
            label = 0 if is_test else sample_submission['category'][idx]    # If no ground truth, label = 0
            self.filenames.append(filename)
            self.x.append(filepath)
            self.y.append(label)
        self.num_classes = max(self.y)+1
        
    def __len__(self):
        return len(self.x)

    # TODO: NOTICE! return filenames now.
    # Return: [filenames, PIL imgs, labels] (1 batch)
    def __getitem__(self, index):
        image = Image.open(self.x[index]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return self.filenames[index], image, self.y[index]
        # return image, self.y[index]

def get_test_loader():
    test_transform=transforms.Compose([
                                transforms.Resize((600, 600), Image.BILINEAR),
                                transforms.CenterCrop((448, 448)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    testset = TestOrchidDataset(root_dir='./dataset/', csv_file='dataset/val_label.csv', transform=test_transform)
    test_sampler = SequentialSampler(testset)
    test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=8,
                             num_workers=4,
                             pin_memory=True) if testset is not None else None
    return test_loader

def set_environment(args):
    args = get_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # test_set = ValOrchidDataset(args.data_root,val_transforms) # data_transform
    # test_loader = torch.utils.data.DataLoader(test_set, num_workers=1, shuffle=False, batch_size=args.batch_size)
    # args, model = setup(args)
    # print("test samples: {}, test batchs: {}".format(len(test_set), len(test_loader)))

    test_loader = get_test_loader()
    
    if args.model_name == "TransFG":
        from models.modeling import VisionTransformer, CONFIGS
        config = CONFIGS[args.model_type]
        print(config)
        # config.split = args.split
        config.split = 'overlap'
        # config.slide_step = args.slide_step
        model = VisionTransformer(config, args.data_size, zero_head=True, num_classes=args.num_classes, smoothing_value=0)
        # model.load_from(np.load(args.pretrained_dir)) # No need pretrained model now
        checkpoint = torch.load(args.pretrained_path)
        model.load_state_dict(checkpoint['model'])
        model.to(args.device)

        # TODO: 看這再衝三小? No need?
        # if args.fp16:
        #     model, optimizer = amp.initialize(models=model,
        #                                   optimizers=optimizer,
        #                                   opt_level=args.fp16_opt_level)
        # amp._amp_state.loss_scalers[0]._loss_scale = 2**20

    return test_loader, model

def valid(args, model, writer, test_loader, global_step):
    # Validation!
    eval_losses = AverageMeter()

    print("***** Running Validation *****")
    print("  Num steps = ",  len(test_loader))
    # print("  Batch size = %d", args.eval_batch_size)

    model.eval()
    all_preds = np.array([])
    all_filenames = []
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True)
                        #   disable=args.local_rank not in [-1, 0])
    # epoch_iterator = tqdm(total=len(test_loader), ascii=True)
    loss_fct = torch.nn.CrossEntropyLoss()
    for step, batch in enumerate(epoch_iterator):
        # batch = tuple(t.to(args.device) for t in batch) # To device
        filenames, x, y = batch
        x = x.to(args.device)
        y = y.to(args.device)
        with torch.no_grad():
            logits = model(x)

            eval_loss = loss_fct(logits, y)
            eval_loss = eval_loss.mean()
            eval_losses.update(eval_loss.item())

            preds = torch.argmax(logits, dim=-1)    # Can do ensemble here!
        print(f'filenames: {filenames}, preds: {preds}, len(filenames): {len(filenames)}')
        all_preds = np.append(all_preds, preds.detach().cpu().numpy())
        for f in filenames:
            all_filenames.append(f)


        # if len(all_preds) == 0:
        #     all_preds.append(preds.detach().cpu().numpy())
        #     all_label.append(y.detach().cpu().numpy())
        # else:
        #     all_preds[0] = np.append(
        #         all_preds[0], preds.detach().cpu().numpy(), axis=0
        #     )
        #     all_label[0] = np.append(
        #         all_label[0], y.detach().cpu().numpy(), axis=0
        #     )
        # epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)

    # all_preds, all_label = all_preds[0], all_label[0]
    # accuracy = simple_accuracy(all_preds, all_label)
    # accuracy = torch.tensor(accuracy).to(args.device)
    # dist.barrier()
    # val_accuracy = reduce_mean(accuracy, nprocs=1)  # TODO: WTF?
    # val_accuracy = val_accuracy.detach().cpu().numpy()

    # print("\n")
    # print("Validation Results")
    # print("Global Steps: %d" % global_step)
    # print("Valid Loss: %2.5f" % eval_losses.avg)
    # print("Valid Accuracy: %2.5f" % val_accuracy)
        
    # return val_accuracy
    return all_filenames, all_preds.tolist()

def write_csv(all_filenames, all_preds):
    data_dict = {"filename": pd.Series(all_filenames),
                 "category": pd.Series(all_preds)}
    df = pd.DataFrame(data_dict)
    df.to_csv(("./Final_submission.csv"), index=False)


def test(args, model, test_loader):
    with torch.no_grad():
        all_filenames, all_preds = valid(args, model, 0, test_loader, global_step=0)
        write_csv(all_filenames, all_preds)


# TODO: 街上彥伯的Inference 方式
if __name__ == "__main__":
    args = get_args()
    if args.tv:
        answer = []
        correct = 0
        df = pd.read_csv('./Final_submission.csv')
        num = len(df['filename'])
        for idx in range(num):
            ans = df.iloc[idx][2:].mode()[0] # start from voter_1
            answer.append(ans)
        '''     # Get Accuracy in following lines
            # ans1 = df['voter_1'][idx]
            # if ans != ans1:
            #     print(idx+2)
            if ans == df['category'][idx]:
                correct+=1
        print(correct/num)
        ''' 
        output_file = "./Vote_Final_submission.csv"
        df = pd.read_csv('dataset/val_label.csv')
        df['category'] = answer
        df.to_csv(output_file, index=False)
        print("finished!!")       
    else:    
        test_loader, model = set_environment(args)
        test(args, model, test_loader)
        

"""
def evaluate(model, device, test_loader):
    correct=0
    TP=0
    TN=0
    FP=0
    FN=0
    with torch.set_grad_enabled(False):
        model.eval()
        for idx,(data,label) in enumerate(test_loader):
            data = data.to(device,dtype=torch.float)
            label = label.to(device,dtype=torch.long)
            predict = model(data)
            pred = torch.max(predict,1).indices
            #correct += pred.eq(label).cpu().sum().item()
            for j in range(data.size()[0]):
                #print ("{} pred label: {} ,true label:{}" .format(len(pred),pred[j],int(label[j])))
                if (int (pred[j]) == int (label[j])):
                    correct +=1
                if (int (pred[j]) == 1 and int (label[j]) ==  1):
                    TP += 1
                if (int (pred[j]) == 0 and int (label[j]) ==  0):
                    TN += 1
                if (int (pred[j]) == 1 and int (label[j]) ==  0):
                    FP += 1
                if (int (pred[j]) == 0 and int (label[j]) ==  1):
                    FN += 1
        print ("TP : " , TP)
        print ("TN : " , TN)
        print ("FP : " , FP)
        print ("FN : " , FN)

        # Plot confusion_matrix (True \ Pred)
        confusion_matrix = [[TP, FN], 
                            [FP, TN]]
        df_cm = pd.DataFrame(confusion_matrix, index=["NORMAL", "PNEUMONIA"],
                  columns=["NORMAL", "PNEUMONIA"])      
        sn.heatmap(df_cm, annot=True)
        plt.xlabel("Prediction")
        plt.ylabel("Ground Truth")
        plt.title('Confusion Matrix')            
        plt.savefig('test_confusion_matrix.png')


        print ("num_correct :",correct ," / " , len(test_loader.dataset))
        Recall = TP/(TP+FN)
        print ("Recall : " ,  Recall )

        Precision = TP/(TP+FP)
        print ("Preecision : " ,  Precision )

        F1_score = 2 * Precision * Recall / (Precision + Recall)
        print ("F1 - score : " , F1_score)

        correct = (correct/len(test_loader.dataset))*100.
        print ("Accuracy : " , correct ,"%")

    return correct , Recall , Precision , F1_score

if __name__=="__main__":
    IMAGE_SIZE=(128,128) # (256, 256) 
    batch_size=128
    learning_rate = 0.001 # 0.01
    epochs=30
    num_classes=2

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print (device)

    best_model_wts = "./resnext50_img128_adam0.001_ep24_acc94.39_BEST.pt"
    # train_path='archive/chest_xray/train'
    test_path='archive/chest_xray/test'
    val_path='archive/chest_xray/val'

    model.to(device)

    accuracy, Recall , Precision , F1_score = evaluate(model, device, test_loader)
    

    if args.tv:
        answer = []
        correct = 0
        df = pd.read_csv('./Final_submission.csv')
        num = len(df['filename'])
        for idx in range(num):
            ans = df.iloc[idx][2:].mode()[0] # start from voter_1
            answer.append(ans)

        output_file = "./Vote_Final_submission.csv"
        df = pd.read_csv('dataset/val_label.csv')
        df['category'] = answer
        df.to_csv(output_file, index=False)
        print("finished!!")       
    else:    
        test_loader, model = set_environment(args)
        test(args, model, test_loader)

"""
