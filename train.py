from datasets import make_dataset
from losses import make_loss
from models import make_model
from processor import train_model
from solvers import make_solver
import torch
import pandas as pd
from utils.remv_row import remv_row
from utils.logger import setup_logger
import argparse


parser = argparse.ArgumentParser(description='argparse tutorial')
parser.add_argument('--path_train', default = '/vinbrain/samtb/Liver_Tumor_Cls/dataset/csv/updated_csv/version2/liver_tumor_ver2_merge_train.csv')
parser.add_argument('--path_val', default = '/vinbrain/samtb/Liver_Tumor_Cls/dataset/csv/updated_csv/version2/liver_tumor_ver2_merge_test.csv')
parser.add_argument('--fn_fig', default = '/vinbrain/samtb/Liver_Tumor_Cls/source2Dmodel/experiments/log/updated_data/intersect_ver1_ven_bigtumor_128channels_den_pretrain_2fc_nz_clean_code.png')
parser.add_argument('--fn_log', default='/vinbrain/samtb/Liver_Tumor_Cls/source2Dmodel/experiments/log/updated_data/intersect_ver1_ven_bigtumor_128channels_den_pretrain_2fc_nz_clean_code.txt')
parser.add_argument('--fn_model', default='/vinbrain/samtb/Liver_Tumor_Cls/source2Dmodel/experiments/models/updated_data/intersect_ver1_ven_bigtumor_128channels_den_pretrain_2fc_nz_clean_code')
parser.add_argument('--input_size', default=224)
parser.add_argument('--n_slices', default=128)
parser.add_argument('--batch_size', default=16)
parser.add_argument('--num_workers', default=2)
parser.add_argument('--model_type', default='classifier')
parser.add_argument('--in_channels', default=128)
parser.add_argument('--device', default= torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
parser.add_argument('--loss_type', default='crossentropy')
parser.add_argument('--optimizer_type', default='SGD')
parser.add_argument('--lr_scheduler_type', default='onecyclelr')
parser.add_argument('--n_epochs', default=80)
parser.add_argument('--lr', default=1e-3)
parser.add_argument('--use_amp', default=True)
parser.add_argument('--milestones', default=(30, 50))
parser.add_argument('--is_train', default=True)
    
config = parser.parse_args()


logger = setup_logger("classification", config.fn_model, if_train=config.is_train)
logger.info("Saving model in the path :{}".format(config.fn_model))
logger.info(config)
logger.info("Running with config:\n{}".format(config))


traindf = pd.read_csv(config.path_train, dtype = {"date": str, "date_reverse": str})
valdf = pd.read_csv(config.path_val, dtype = {"date": str, "date_reverse": str})
exp_case = ["N19-0383646" , "N19-0401629",  "B07-0002524", "N18-0244874"]
traindf = remv_row(traindf,exp_case)
valdf = remv_row(valdf,exp_case)
traindf = traindf[traindf["regist_available"]==True]
valdf = valdf[valdf["regist_available"]==True]

train_dl = make_dataset(traindf, input_shape = (config.input_size,config.input_size), n_slices = config.n_slices, is_train=True)
val_dl = make_dataset(valdf, input_shape = (config.input_size,config.input_size), n_slices = config.n_slices, is_train = False)

train_dataloader = torch.utils.data.DataLoader(train_dl, shuffle = True, batch_size = config.batch_size, num_workers = config.num_workers, drop_last=False)
val_dataloader = torch.utils.data.DataLoader(val_dl, shuffle = False, batch_size = config.batch_size, num_workers = config.num_workers, drop_last=False)

loss_func = make_loss(config)
model = make_model(config).to(config.device, dtype = torch.double)
optimizer, lr_scheduler = make_solver(config, model, train_dataloader)

model.train()
model_conv = train_model(model, loss_func, optimizer, lr_scheduler, train_dataloader, val_dataloader, config)