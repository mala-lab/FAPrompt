import AnomalyCLIP_lib
import torch
import argparse
import torch.nn.functional as F
from FAPrompt import AnomalyCLIP_PromptLearner
from loss import FocalLoss, BinaryDiceLoss, BinaryFocalLoss
from utils import normalize
from dataset import Dataset
from logger import get_logger
from tqdm import tqdm
import numpy as np
import torch.nn as nn
from collections import OrderedDict
import os
import random
from utils import get_transform
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train(args):
    logger = get_logger(args.save_path)

    preprocess, target_transform = get_transform(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    AnomalyCLIP_parameters = {"Prompt_length": args.n_ctx, "learnabel_text_embedding_depth": args.depth, "learnabel_text_embedding_length": args.t_n_ctx}

    model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device=device, design_details = AnomalyCLIP_parameters)
    model.eval()

    train_data = Dataset(root=args.train_data_path, transform=preprocess, target_transform=target_transform, dataset_name = args.dataset)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    prompt_learner = AnomalyCLIP_PromptLearner(model.to("cpu"), AnomalyCLIP_parameters)
    prompt_learner.to(device)

    count = count_parameters(prompt_learner)
    print(count)

    model.to(device)
    model.visual.DAPM_replace(DPAM_layer = 20)
    optimizer = torch.optim.Adam(list(prompt_learner.parameters()), lr=args.learning_rate, betas=(0.5, 0.999))

    # losses
    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()
    loss_fun = BinaryFocalLoss()
    loss_mse = torch.nn.MSELoss()

    model.eval()
    prompt_learner.train()
    for epoch in tqdm(range(args.epoch)):
        model.eval()
        prompt_learner.train()
        loss_list = []
        image_loss_list = []
        image_loss_list2 = []
        loss_ab_list = []

        for items in tqdm(train_dataloader):


            image = items['img'].to(device)
            label =  items['anomaly']
            class_name = items['cls_name']

            gt = items['img_mask'].squeeze().to(device)
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0

            with torch.no_grad():
                image_features, patch_features = model.encode_image(image, args.features_list, DPAM_layer = 20)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            prompts_pos, prompts_neg, tokenized_prompt_pos, tokenized_prompt_neg, compound_prompts_text, _ = prompt_learner.forward()
            text_features_pos = model.encode_text_learn(prompts_pos, tokenized_prompt_pos, compound_prompts_text).float()
            text_features_neg = model.encode_text_learn(prompts_neg, tokenized_prompt_neg, compound_prompts_text).float()

            loss_ab = 0
            for k in range(len(text_features_neg) - 1):
                for m in range(k + 1, len(text_features_neg)):
                    loss_ab = loss_ab + abs(torch.dot(text_features_neg[k], text_features_neg[m]) / (
                                torch.norm(text_features_neg[k]) * torch.norm(text_features_neg[m])))

            loss_ab_list.append(loss_ab.item())

            text_features_neg = torch.mean(text_features_neg, dim=0, keepdim=True)
            text_features = torch.cat([text_features_pos, text_features_neg])
            text_features = torch.stack(torch.chunk(text_features, dim=0, chunks=2), dim=1)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            patch_features = patch_features / patch_features.norm(dim=-1, keepdim=True)  # [1370, 8, 768]
            similarity, _ = AnomalyCLIP_lib.compute_similarity_ori(patch_features, text_features[0])
            similarity_map1 = AnomalyCLIP_lib.get_similarity_map(similarity[1:, :], args.image_size)
            map_max_score1 = similarity[1:, :, 1].max(dim=0).values

            pk_value, pk_indx = torch.topk(similarity[1:, :, 1], k=10, dim = 0, largest=True, sorted=True)
            pk_indx = pk_indx.permute(1, 0)
            selected_tokens = []
            for i in range(len(pk_indx)):
                selected_tokens2 = []
                for k in range(len(pk_indx[0])):
                    tmp_token = patch_features[1:, :, :].permute(1, 0, 2)[i, k, :]
                    selected_tokens2.append(tmp_token)
                selected_tokens2 = torch.stack(selected_tokens2)
                selected_tokens.append(selected_tokens2)
            selected_tokens = torch.stack(selected_tokens)

            text_features_list = []
            loss_bias = 0.0
            for i in range(len(selected_tokens)):
                prompts_pos, prompts_neg, tokenized_prompt_pos, tokenized_prompt_neg, compound_prompts_text, bias = prompt_learner.forward(selected_tokens = selected_tokens[i])
                text_features_pos = model.encode_text_learn(prompts_pos, tokenized_prompt_pos, compound_prompts_text).float()
                text_features_neg = model.encode_text_learn(prompts_neg, tokenized_prompt_neg, compound_prompts_text).float()
                text_features_neg = torch.mean(text_features_neg, dim=0, keepdim=True)

                text_features = torch.cat([text_features_pos, text_features_neg])
                text_features_list.append(text_features)

                bias_gt0 = torch.zeros(768).to(device)
                if label[i] == 0:
                    loss_tmp = loss_mse(bias.squeeze(0), bias_gt0)
                    loss_bias = loss_bias + loss_tmp

            text_features = torch.stack(text_features_list)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            similarity, _ = AnomalyCLIP_lib.compute_similarity(patch_features, text_features)
            similarity_map2 = AnomalyCLIP_lib.get_similarity_map(similarity[1:, :], args.image_size)
            map_max_score2 = similarity[1:, :, 1].max(dim=0).values

            text_probs = image_features.unsqueeze(1) @ text_features.permute(0, 2, 1)
            text_probs = text_probs[:, 0, ...] / 0.07
            image_loss = F.cross_entropy(text_probs.squeeze(), label.long().to(device))
            image_loss_list.append(image_loss.item())

            tmp = (text_probs).softmax(-1)
            tmp = tmp[:, 1]

            map_max_score = (2 * map_max_score1 + map_max_score2)/3
            score2 = 0.5 * (tmp + map_max_score)
            image_loss2 = loss_fun(score2, label.float().to(device))
            image_loss_list2.append(image_loss2.item())

            loss = 0
            loss += loss_focal(similarity_map1, gt)
            loss += loss_dice(similarity_map1[:, 1, :, :], gt)
            loss += loss_dice(similarity_map1[:, 0, :, :], 1-gt)

            loss += 0.5 * loss_focal(similarity_map2, gt)
            loss += 0.5 * loss_dice(similarity_map2[:, 1, :, :], gt)
            loss += 0.5 * loss_dice(similarity_map2[:, 0, :, :], 1 - gt)

            optimizer.zero_grad()
            (loss + image_loss2 + 0.1*loss_ab + loss_bias).backward()
            optimizer.step()
            loss_list.append(loss.item())


        # logs
        if (epoch + 1) % args.print_freq == 0:
            logger.info('epoch [{}/{}], loss:{:.4f}, image_loss:{:.4f}, image_loss2:{:.4f}'.format(epoch + 1, args.epoch, np.mean(loss_list), np.mean(image_loss_list), np.mean(image_loss_list2)))

        # save model
        if (epoch + 1) % args.save_freq == 0:
            ckp_path = os.path.join(args.save_path, 'epoch_' + str(epoch + 1) + '.pth')
            torch.save({"prompt_learner": prompt_learner.state_dict()}, ckp_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("AnomalyCLIP", add_help=True)
    parser.add_argument("--train_data_path", type=str, default="./data/visa", help="train dataset path")
    parser.add_argument("--save_path", type=str, default='./checkpoint', help='path to save results')


    parser.add_argument("--dataset", type=str, default='mvtec', help="train dataset name")

    parser.add_argument("--depth", type=int, default=9, help="image size")
    parser.add_argument("--n_ctx", type=int, default=12, help="zero shot")
    parser.add_argument("--t_n_ctx", type=int, default=4, help="zero shot")
    parser.add_argument("--feature_map_layer", type=int, nargs="+", default=[0, 1, 2, 3], help="zero shot")
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="features used")

    parser.add_argument("--epoch", type=int, default=15, help="epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--image_size", type=int, default=518, help="image size")
    parser.add_argument("--print_freq", type=int, default=1, help="print frequency")
    parser.add_argument("--save_freq", type=int, default=1, help="save frequency")
    parser.add_argument("--seed", type=int, default=111, help="random seed")
    args = parser.parse_args()
    setup_seed(args.seed)
    train(args)
