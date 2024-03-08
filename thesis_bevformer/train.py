import torch
from torch.utils.tensorboard import SummaryWriter
from nuscenes.nuscenes import NuScenes

from datetime import datetime

from thesis_bevformer.utils.utils_blip import BertLMHeadModel, BertConfig, init_tokenizer
from thesis_bevformer.utils.utils_bevformer import build_bevformer, build_data_loader
from thesis_bevformer.BEVCapGen import BEVCapGen


def evaluate(model, nusc, dataloader, ep, writer, log_file, log_every=1000):
    
    model.eval()
    with torch.no_grad():
        total_val_loss = 0.0
        
        for i, data in enumerate(dataloader):
            scene_token = data["img_metas"][0].data[0][0]["scene_token"]
            caption = nusc.get("scene", scene_token)["description"]

            result = model(data, caption)
            val_loss = result["loss"]
            total_val_loss += val_loss.item()

            if i % log_every == 0:
                generate(model, data, caption, val_loss, ep, i, log_file)
        
        writer.add_scalar("Loss/Val", total_val_loss / len(dataloader), ep)
    model.train()


def generate(model, data, caption, loss, ep, step, log_file):
    
    output = model.generate(data)
    print(f"Epoch: {ep}, Step: {step}", file=log_file, flush=True)
    print(f"GT:\t{caption}", file=log_file, flush=True)
    print(f"Out:\t{output}", file=log_file, flush=True)
    print(f"Loss:\t{loss.item()}", file=log_file, flush=True)

def train_step(model, nusc, data, optimizer, ep, step, g_step, writer, log_file, log_every=5000):
    
    # scene_token = data["img_metas"][0].data[0][0]["scene_token"]
    scene_tokens = [img_metas["scene_token"] for img_metas in data["img_metas"][0].data[0]]
    # captions = [nusc.get("scene", scene_token)["description"] for scene_token in scene_tokens]
    captions = ["Test caption", "A red car is passing by", "Just to test"]

    result = model(data, captions)
    loss = result["loss"]

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    writer.add_scalar("Loss/Train", loss.item(), g_step)

    if step % log_every == 0:
        generate(model, data, caption, loss, ep, step, log_file)



def train(model, nusc, dataloader_train, dataloader_val, optimizer, writer, log_file, max_epoch):
    
    global_step = 0
    
    for epoch in range(max_epoch):
        for i, data in enumerate(dataloader_train):
            train_step(model, nusc, data, optimizer, epoch, i, global_step, writer, log_file)
            global_step += 1
        
        evaluate(model, nusc, dataloader_val, epoch, writer, log_file)

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
    val_loader = build_data_loader(bevformer_cfg, mode="val")

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

    bcg.load_blip_decoder_ckpt("./ckpts/model_base_capfilt_large.pth")

    adam_w = torch.optim.AdamW(params=bcg.parameters(), lr=1e-4, weight_decay=0.05)
    
    run_name = "experiment_1"
    todays_date = datetime.now().strftime("%d-%m")
    sum_writer = SummaryWriter(log_dir=f"runs/{todays_date}_{run_name}")

    with open(f"./thesis_bevformer/logs/log_{todays_date}_{run_name}.txt", "w") as log:
        train(model=bcg,
            nusc=nuscenes_trainval,
            dataloader_train=train_loader,
            dataloader_val=val_loader,
            optimizer=adam_w,
            writer=sum_writer,
            log_file=log,
            max_epoch=5)

if __name__ == "__main__":
    main()