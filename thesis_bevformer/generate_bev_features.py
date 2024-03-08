import torch
from torch.utils.tensorboard import SummaryWriter
from nuscenes.nuscenes import NuScenes

from datetime import datetime

from thesis_bevformer.utils.utils_blip import BertLMHeadModel, BertConfig, init_tokenizer
from thesis_bevformer.utils.utils_bevformer import build_bevformer, build_data_loader
from thesis_bevformer.BEVCapGen import BEVCapGen


def main():
    torch.cuda.empty_cache()

    # NuScenes for loading the scene captions
    nuscenes_trainval = None
    # nuscenes_trainval = NuScenes(version='v1.0-trainval', dataroot='./data/nuscenes', verbose=True)

    # BEVFormer config, model and data laoders
    bevformer_cfg = "./projects/configs/bevformer/bevformer_tiny.py"
    bevformer_ckpt = "./ckpts/bevformer_tiny_epoch_24.pth"
    bevformer = build_bevformer(bevformer_cfg, bevformer_ckpt)
    train_loader = build_data_loader(bevformer_cfg, mode="train")
    # val_loader = build_data_loader(bevformer_cfg, mode="val")

    # BLIP configs, model and tokenizer
    med_config_path = "./thesis_bevformer/configs/med_config.json"
    med_config = BertConfig.from_json_file(med_config_path)
    blip_lm_head = BertLMHeadModel(config=med_config)
    blip_tokenizer = init_tokenizer()

    # Initializing BEVCapGen
    bcg = BEVCapGen(
        bev_encoder=bevformer,
        text_decoder=blip_lm_head,
        tokenizer=blip_tokenizer,
        bev_feature_size=256,
        bev_area=50*50,
        hidden_size=med_config.hidden_size,
        encoder_width=med_config.encoder_width,
        device="cuda"
        )
    
    for i, data in enumerate(train_loader):
        print(f"\r Generating BEV Feature: {i+1}/{len(train_loader)}", end="")
            
        sample_idx = data["img_metas"][0].data[0][0]["sample_idx"]
        bev = bcg.get_bev_embeds(data)

        torch.save(bev, f"./thesis_bevformer/data/bev_features/tiny/train/{sample_idx}.pt")
    

if __name__ == "__main__":
    main()