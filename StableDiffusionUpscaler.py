import io
import cv2
import torch
import numpy as np
import safetensors.torch
from PIL import Image
from omegaconf import OmegaConf
from einops import repeat, rearrange
from pytorch_lightning import seed_everything
from imwatermark import WatermarkEncoder
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler
from ldm.models.diffusion.ddpm import LatentUpscaleDiffusion, LatentUpscaleFinetuneDiffusion
from ldm.util import exists, instantiate_from_config

torch.set_grad_enabled(False)

def put_watermark(img, wm_encoder = None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img

def load_model_from_config(config, ckpt, verbose = False):
    print(f"Загрузка модели из {ckpt}")
    if ckpt[ckpt.rfind('.'):] == ".safetensors":
        pl_sd = safetensors.torch.load_file(ckpt, device = "cpu")
    else:
        pl_sd = torch.load(ckpt, map_location = "cpu")
    if "global_step" in pl_sd:
        print(f"Глобальный шаг: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"] if "state_dict" in pl_sd else pl_sd
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict = False)
    if len(m) > 0 and verbose:
        print("Пропущенные параметры:\n", m)
    if len(u) > 0 and verbose:
        print("Некорректные параматры:")
        print(u)
    model.cuda()
    model.eval()
    return model

def load_img(binary_data, max_dim):
    image = Image.open(io.BytesIO(binary_data)).convert("RGB")
    orig_w, orig_h = image.size
    print(f"Загружено входное изображение размера ({orig_w}, {orig_h})")
    cur_dim = orig_w * orig_h
    if cur_dim > max_dim:
        k = cur_dim / max_dim
        sk = float(k ** (0.5))
        w, h = int(orig_w / sk), int(orig_h / sk)
    else:
        w, h = orig_w, orig_h
    w, h = map(lambda x: x - x % 64, (w, h))  # изменение размера в целое число, кратное 64-м
    if w == 0 and orig_w != 0:
        w = 64
    if h == 0 and orig_h != 0:
        h = 64
    if (w, h) != (orig_w, orig_h):
        image = image.resize((w, h), resample = Image.LANCZOS)
        print(f"Размер изображения изменён на ({w}, {h} (w, h))")
    else:
        print(f"Размер исходного изображения не был изменён")
    return image

def Stable_diffusion_upscaler(binary_data, prompt, opt):
    image = load_img(binary_data, opt["max_dim"])
    w, h = image.size
    checkpoint_path = "weights\\"
    config_path = "configs\\"
    config_list = ["x4-upscaling.yaml"]
    image = load_img(binary_data, opt["max_dim"])
    config = OmegaConf.load(config_path + config_list[0])
    model = load_model_from_config(config, checkpoint_path + opt["ckpt"], opt["verbose"])
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    if opt["sampler"] == "plms":
        sampler = PLMSSampler(model, device = device)
    elif opt["sampler"] == "dpm":
        sampler = DPMSolverSampler(model, device = device)
    else:
        sampler = DDIMSampler(model, device = device)
    noise_level = None
    if isinstance(sampler.model, LatentUpscaleDiffusion):
        noise_level = opt["noise"]
    noise_level = torch.Tensor(opt["num_samples"] * [noise_level]).to(sampler.model.device).long()
    sampler.make_schedule(opt["steps"], ddim_discretize = opt["discretize"], ddim_eta = opt["eta"], verbose = opt["verbose"])
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = sampler.model
    seed_everything(opt["seed"])
    prng = np.random.RandomState(opt["seed"])
    start_code = prng.randn(opt["num_samples"], model.channels, h, w)
    start_code = torch.from_numpy(start_code).to(device = device, dtype = torch.float32)
    with torch.no_grad(), torch.autocast("cuda"):
        image = np.array(image.convert("RGB"))
        image = torch.from_numpy(image).to(dtype = torch.float32) / 127.5 - 1.0
        batch = {
            "lr": rearrange(image, "h w c -> 1 c h w"),
            "txt": opt["num_samples"] * [prompt],
        }
        batch["lr"] = repeat(batch["lr"].to(device = device), "1 ... -> n ...", n = opt["num_samples"])
        c = model.cond_stage_model.encode(batch["txt"])
        c_cat = list()
        if isinstance(model, LatentUpscaleFinetuneDiffusion):
            for ck in model.concat_keys:
                cc = batch[ck]
                if exists(model.reshuffle_patch_size):
                    assert isinstance(model.reshuffle_patch_size, int)
                    cc = rearrange(cc, "b c (p1 h) (p2 w) -> b (p1 p2 c) h w", p1 = model.reshuffle_patch_size, p2 = model.reshuffle_patch_size)
                c_cat.append(cc)
            c_cat = torch.cat(c_cat, dim = 1)
            # Условное
            cond = {"c_concat": [c_cat], "c_crossattn": [c]}
            # Безусловное условие
            uc_cross = model.get_unconditional_conditioning(opt["num_samples"], "")
            uc_full = {"c_concat": [c_cat], "c_crossattn": [uc_cross]}
        elif isinstance(model, LatentUpscaleDiffusion):
            x_low = batch[model.low_scale_key]
            x_low = x_low.to(memory_format = torch.contiguous_format).float()
            x_augment, noise_level = model.low_scale_model(x_low, noise_level)
            cond = {
                "c_concat": [x_augment],
                "c_crossattn": [c], 
                "c_adm": noise_level
            }
            # Безусловное условие
            uc_cross = model.get_unconditional_conditioning(opt["num_samples"], "")
            uc_full = {
                "c_concat": [x_augment],
                "c_crossattn": [uc_cross],
                "c_adm": noise_level
            }
        else:
            raise NotImplementedError()
        shape = [model.channels, h, w]
        samples, _ = sampler.sample(opt["steps"], opt["num_samples"], shape, cond, verbose = opt["verbose"], discretize = opt["discretize"], eta = opt["eta"], unconditional_guidance_scale = opt["scale"], unconditional_conditioning = uc_full, x_T = start_code)
    with torch.no_grad():
        x_samples_ddim = model.decode_first_stage(samples)
    result = torch.clamp((x_samples_ddim + 1.0) / 2.0, min = 0.0, max = 1.0)
    result = result.cpu().numpy().transpose(0, 2, 3, 1) * 255
    b_data_list = []
    if opt["add_watermark"]:
        wm = "SDV2"
        wm_encoder = WatermarkEncoder()
        wm_encoder.set_watermark("bytes", wm.encode("utf-8"))
        for img in result:
            image = put_watermark(Image.fromarray(img.astype(np.uint8)), wm_encoder)
            buf = io.BytesIO()
            image.save(buf, format = "PNG")
            b_data = buf.getvalue()
            image.close
            b_data_list.append(b_data)
    else:
        for img in result:
            image = Image.fromarray(img.astype(np.uint8))
            buf = io.BytesIO()
            image.save(buf, format = "PNG")
            b_data = buf.getvalue()
            image.close
            b_data_list.append(b_data)
    torch.cuda.empty_cache()
    return b_data_list

if __name__ == '__main__':
    params = {
        "steps": 75,                            #Шаги, от 2 до 200
        "seed": 42,                             #От 0 до 1000000
        "noise": 20,                            #Удаление шума (от 0 до 350)  
        "sampler": "ddim",                      #Выбор обработчика (доступны "ddim", "dpm", "plms"). У plms крайне зашумлённые результаты генерации
        "outscale": 4,                          #Во сколько раз увеличивать изображение (сейчас не работает)
        "num_samples": 2,                       #Сколько изображений возвращать (от 1 до 4)
        "eta": 0.0,                             #значения от 0.0 до 1.0, η = 0.0 соответствует детерминированной выборке
        "discretize": "uniform",                #Дискретизатор обработчика (доступны "uniform" и "quad"), только при eta > 0.0
        "scale": 10.0,                          #От 0.1 до 30.0
        "ckpt": "x4-upscaler-ema.safetensors",  #Выбор весов модели ("x4-upscaler-ema.safetensors")   
        "add_watermark": False,                 #Добавлять невидимую вотермарку
        "verbose": False,                       #Системный параметр
        "max_dim": pow(1024, 2)                 #Я не могу генерировать на своей видюхе картинки больше 1024 на 1024
    }

    with open("img.png", "rb") as f:
        init_img_binary_data = f.read()
    prompt = "Digital hight resolution photo"
    binary_data_list = Stable_diffusion_upscaler(init_img_binary_data, prompt, params)
    i = 0
    for binary_data in binary_data_list:
        Image.open(io.BytesIO(binary_data)).save("big" + str(i) + ".png")
        i += 1