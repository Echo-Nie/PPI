import csv
import nni
import time
import json
import math
import copy
import argparse
import warnings
import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler  # 添加混合精度训练所需的导入

from utils import *
from models import *
from dataloader import *

warnings.filterwarnings("ignore", category=Warning)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 添加GPU内存监测函数
def get_gpu_memory_usage():
    """返回当前GPU内存使用情况（MB）"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0

def train(model, ppi_g, prot_embed, ppi_list, labels, index, batch_size, optimizer, loss_fn, epoch, scaler=None):  # 添加scaler参数

    f1_sum = 0.0
    loss_sum = 0.0

    batch_num = math.ceil(len(index) / batch_size)
    random.shuffle(index)

    model.train()

    for batch in range(batch_num):
        if batch == batch_num - 1:
            train_idx = index[batch * batch_size:]
        else:
            train_idx = index[batch * batch_size : (batch+1) * batch_size]

        # 使用混合精度训练
        with autocast(enabled=scaler is not None):  # 只在scaler不为None时启用
            output = model(ppi_g, prot_embed, ppi_list, train_idx)
            loss = loss_fn(output, labels[train_idx])

        if scaler is not None:
            # 使用scaler进行反向传播和优化
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # 原始的反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_sum += loss.item()
        # 修改这里，确保输出转换为float32类型
        output_float = output.detach().cpu().float()  # 显式转换为float32
        f1_score = evaluat_metrics(output_float, labels[train_idx].detach().cpu())
        f1_sum += f1_score

        # print("Epoch: {}, Batch: {}/{} | Train Loss: {:.5f}, F1-score: {:.5f}".format(epoch, batch+1, batch_num, loss.item(), f1_score))

    return loss_sum / batch_num, f1_sum / batch_num


def evaluator(model, ppi_g, prot_embed, ppi_list, labels, index, batch_size, mode='metric'):

    eval_output_list = []
    eval_labels_list = []

    batch_num = math.ceil(len(index) / batch_size)

    model.eval()

    with torch.no_grad():
        for batch in range(batch_num):
            if batch == batch_num - 1:
                eval_idx = index[batch * batch_size:]
            else:
                eval_idx = index[batch * batch_size : (batch+1) * batch_size]

            output = model(ppi_g, prot_embed, ppi_list, eval_idx)
            # 确保输出转换为float32
            eval_output_list.append(output.detach().cpu().float())  # 显式转换为float32
            eval_labels_list.append(labels[eval_idx].detach().cpu())
            
        f1_score = evaluat_metrics(torch.cat(eval_output_list, dim=0), torch.cat(eval_labels_list, dim=0))

    if mode == 'metric':
        return f1_score
    elif mode == 'output':
        return torch.cat(eval_output_list, dim=0), torch.cat(eval_labels_list, dim=0)
    


def pretrain_vae():

    if args.pre_train is None:
        protein_data, ppi_g, ppi_list, labels, ppi_split_dict = load_data(param['dataset'], param['split_mode'], param['seed'])
    else:
        protein_data = load_pretrain_data(args.pre_train)

    output_dir = "../results/{}/{}/VAE/".format(param['dataset'], timestamp)
    check_writable(output_dir, overwrite=False)
    log_file = open(os.path.join(output_dir, "train_log.txt"), 'a+')
    with open(os.path.join(output_dir, "config.json"), 'a+') as tf:
        json.dump(param, tf, indent=2)

    vae_dataloader = DataLoader(protein_data, batch_size=512, shuffle=True, collate_fn=collate)
    vae_model = CodeBook(param, DataLoader(protein_data, batch_size=512, shuffle=False, collate_fn=collate)).to(device)
    vae_optimizer = torch.optim.Adam(vae_model.parameters(), lr=float(param['learning_rate']), weight_decay=float(param['weight_decay']))
    
    # 添加混合精度训练的scaler
    scaler = GradScaler() if torch.cuda.is_available() else None
    
    for epoch in range(1, param["pre_epoch"] + 1):
        for iter_num, batch_graph in enumerate(vae_dataloader):

            batch_graph.to(device)

            # 使用混合精度训练
            with autocast(enabled=scaler is not None):
                z, e, e_q_loss, recon_loss, mask_loss = vae_model(batch_graph)
                loss_vae = e_q_loss + recon_loss + mask_loss * param['mask_loss']

            if scaler is not None:
                vae_optimizer.zero_grad()
                scaler.scale(loss_vae).backward()
                scaler.step(vae_optimizer)
                scaler.update()
            else:
                vae_optimizer.zero_grad()
                loss_vae.backward()
                vae_optimizer.step()

            if (epoch - 1) % param['log_num'] == 0 and iter_num == 0:
                print("\033[0;30;43m Pre-training VQ-VAE | Epoch: {}, Batch: {} | Train Loss: {:.5f} | {:.5f} {:.5f} {:.5f}\033[0m".format(epoch, iter_num, loss_vae.item(), e_q_loss.item(), recon_loss.item(), mask_loss.item()))
                log_file.write("Pre-training VQ-VAE | Epoch: {}, Batch: {} | Train Loss: {:.5f} | {:.5f} {:.5f} {:.5f}\n".format(epoch, iter_num, loss_vae.item(), e_q_loss.item(), recon_loss.item(), mask_loss.item()))
                log_file.flush()

    torch.save(vae_model.state_dict(), os.path.join(output_dir, f'vae_model.ckpt'))

    del vae_model
    torch.cuda.empty_cache()


def main():

    protein_data, ppi_g, ppi_list, labels, ppi_split_dict = load_data(param['dataset'], param['split_mode'], param['seed'])
    
    vae_model = CodeBook(param, DataLoader(protein_data, batch_size=512, shuffle=False, collate_fn=collate)).to(device)
    if args.ckpt_path is None:
        vae_model.load_state_dict(torch.load(os.path.join("../results/{}/{}/VAE/".format(param['dataset'], timestamp), f'vae_model.ckpt')))
    else:
        vae_model.load_state_dict(torch.load(args.ckpt_path))
    prot_embed = vae_model.Protein_Encoder.forward(vae_model.vq_layer).to(device)

    del vae_model
    torch.cuda.empty_cache()

    output_dir = "../results/{}/{}/SEES_{}/".format(param['dataset'], timestamp, param['seed'])
    check_writable(output_dir, overwrite=False)
    log_file = open(os.path.join(output_dir, "train_log.txt"), 'a+')

    model = GIN(param).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(param['learning_rate']), weight_decay=float(param['weight_decay']))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    loss_fn = nn.BCEWithLogitsLoss().to(device)
    
    # 添加混合精度训练的scaler
    scaler = GradScaler() if torch.cuda.is_available() else None
    
    # 如果启用了混合精度训练，打印提示信息
    if scaler is not None:
        print("\033[0;30;42m 已启用混合精度训练(AMP) \033[0m")
        log_file.write("已启用混合精度训练(AMP)\n")
        log_file.flush()
    
    # 记录初始内存使用
    initial_memory = get_gpu_memory_usage()
    print(f"\033[0;30;42m 初始GPU内存使用: {initial_memory:.2f} MB \033[0m")
    log_file.write(f"初始GPU内存使用: {initial_memory:.2f} MB\n")
    log_file.flush()

    es = 0
    val_best = 0
    test_val = 0
    test_best = 0
    best_epoch = 0
    
    # 添加时间记录变量
    start_time = time.time()
    last_time_check = start_time

    for epoch in range(1, param["max_epoch"] + 1):
        
        # 传入scaler参数
        train_loss, train_f1_score = train(model, ppi_g, prot_embed, ppi_list, labels, ppi_split_dict['train_index'], param['batch_size'], optimizer, loss_fn, epoch, scaler)
        
        scheduler.step(train_loss)

        if (epoch - 1) % param['log_num'] == 0:

            val_f1_score = evaluator(model, ppi_g, prot_embed, ppi_list, labels, ppi_split_dict['val_index'], param['batch_size'])
            test_f1_score = evaluator(model, ppi_g, prot_embed, ppi_list, labels, ppi_split_dict['test_index'], param['batch_size'])

            if test_f1_score > test_best:
                test_best = test_f1_score

            if val_f1_score >= val_best:
                val_best = val_f1_score
                test_val = test_f1_score
                state = copy.deepcopy(model.state_dict())
                es = 0
                best_epoch = epoch
            else:
                es += 1

            print("\033[0;30;46m Epoch: {}, Train Loss: {:.5f} | Train: {:.4f}, Val: {:.4f}, Test: {:.4f} | Val Best: {:.4f}, Test Val: {:.4f}, Test Best: {:.4f} | Best Epoch: {}\033[0m".format(
                    epoch, train_loss, train_f1_score, val_f1_score, test_f1_score, val_best, test_val, test_best, best_epoch))
            log_file.write(" Epoch: {}, Train Loss: {:.5f} | Train: {:.4f}, Val: {:.4f}, Test: {:.4f} | Val Best: {:.4f}, Test Val: {:.4f}, Test Best: {:.4f} | Best Epoch: {}\n".format(
                    epoch, train_loss, train_f1_score, val_f1_score, test_f1_score, val_best, test_val, test_best, best_epoch))
            log_file.flush()
            
            # 每10个epoch打印一次时间统计
            if epoch % 10 == 0:
                current_time = time.time()
                total_elapsed = current_time - start_time
                last_10_elapsed = current_time - last_time_check
                
                # 添加内存使用统计
                current_memory = get_gpu_memory_usage()
                memory_diff = current_memory - initial_memory
                
                print("\033[0;30;43m 时间统计 | 总训练时间: {:.2f}分钟 | 最近10个epoch: {:.2f}分钟 | 平均每个epoch: {:.2f}秒 \033[0m".format(
                    total_elapsed/60, last_10_elapsed/60, last_10_elapsed/10))
                print("\033[0;30;43m 内存统计 | 当前GPU内存: {:.2f} MB | 与初始相比: {:.2f} MB \033[0m".format(
                    current_memory, memory_diff))
                
                log_file.write("时间统计 | 总训练时间: {:.2f}分钟 | 最近10个epoch: {:.2f}分钟 | 平均每个epoch: {:.2f}秒\n".format(
                    total_elapsed/60, last_10_elapsed/60, last_10_elapsed/10))
                log_file.write("内存统计 | 当前GPU内存: {:.2f} MB | 与初始相比: {:.2f} MB\n".format(
                    current_memory, memory_diff))
                log_file.flush()
                
                # 更新上次检查时间
                last_time_check = current_time

            if es == 500:
                print("Early stopping!")
                break

    # 打印总训练时间
    total_time = time.time() - start_time
    print("\033[0;30;43m 训练完成 | 总训练时间: {:.2f}分钟 ({:.2f}小时) | 总epoch数: {} \033[0m".format(
        total_time/60, total_time/3600, epoch))
    log_file.write("训练完成 | 总训练时间: {:.2f}分钟 ({:.2f}小时) | 总epoch数: {}\n".format(
        total_time/60, total_time/3600, epoch))
    log_file.flush()

    torch.save(state, os.path.join(output_dir, "model_state.pth"))
    log_file.close()

    model.load_state_dict(state)
    eval_output, eval_labels = evaluator(model, ppi_g, prot_embed, ppi_list, labels, ppi_split_dict['test_index'], param['batch_size'], 'output')

    np.save(os.path.join(output_dir, "eval_output.npy"), eval_output.detach().cpu().numpy())
    np.save(os.path.join(output_dir, "eval_labels.npy"), eval_labels.detach().cpu().numpy())

    # jsobj = json.dumps(ppi_split_dict)
    # with open(os.path.join(output_dir, "ppi_split_dict.json"), 'w') as f:
    #     f.write(jsobj)
    #     f.close()

    return test_f1_score, test_val, test_best, best_epoch



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch DGL implementation")
    parser.add_argument("--dataset", type=str, default="SHS27k")
    parser.add_argument("--split_mode", type=str, default="random")
    parser.add_argument("--input_dim", type=int, default=7)
    parser.add_argument("--output_dim", type=int, default=7)
    parser.add_argument("--ppi_hidden_dim", type=int, default=512)
    parser.add_argument("--prot_hidden_dim", type=int, default=128)
    parser.add_argument("--ppi_num_layers", type=int, default=2)
    parser.add_argument("--prot_num_layers", type=int, default=4)
    
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--max_epoch", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--dropout_ratio", type=float, default=0.0)
    
    parser.add_argument("--pre_epoch", type=int, default=50)
    parser.add_argument("--commitment_cost", type=float, default=0.25)
    parser.add_argument("--num_embeddings", type=int, default=512)
    parser.add_argument("--mask_ratio", type=float, default=0.15)
    parser.add_argument("--sce_scale", type=float, default=1.5)
    parser.add_argument("--mask_loss", type=float, default=1.0)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_num", type=int, default=1)
    parser.add_argument("--data_mode", type=int, default=0)
    parser.add_argument("--data_split_mode", type=int, default=0)
    parser.add_argument("--pre_train", type=str, default=None)
    parser.add_argument("--ckpt_path", type=str, default=None)

    args = parser.parse_args()
    param = args.__dict__
    param.update(nni.get_next_parameter())
    timestamp = time.strftime("%Y-%m-%d %H-%M-%S") + f"-%3d" % ((time.time() - int(time.time())) * 1000)

    if os.path.exists("../configs/param_configs.json"):
        param = json.loads(open("../configs/param_configs.json", 'r').read())[param['dataset']][param['split_mode']]

    if param['data_mode'] == 0:
        param['dataset'] = 'SHS27k'
    elif param['data_mode'] == 1:
        param['dataset'] = 'SHS148k'
    elif param['data_mode'] == 2:
        param['dataset'] = 'STRING'

    if param['data_split_mode'] == 0:
        param['split_mode'] = 'random'
    elif param['data_split_mode'] == 1:
        param['split_mode'] = 'bfs'
    elif param['data_split_mode'] == 2:
        param['split_mode'] = 'dfs'

    set_seed(param['seed'])
    if args.ckpt_path is None:
        pretrain_vae()
    test_acc, test_val, test_best, best_epoch = main()
    nni.report_final_result(test_val)

    outFile = open('../PerformMetrics_Metrics.csv','a+', newline='')
    writer = csv.writer(outFile, dialect='excel')
    results = [timestamp]
    for v, k in param.items():
        results.append(k)
    
    results.append(str(test_acc))
    results.append(str(test_val))
    results.append(str(test_best))
    writer.writerow(results)