# Usage: python uvr.py --model_name onnx_dereverb_By_FoxJoy --inp_path ./music/voice.wav --save_root_vocal ./vocal --save_root_inst ./inst --agg 10 --format0 wav
# Usage: python uvr.py --model_name HP2_all_vocals --inp_path ./music/voice.wav --save_root_vocal ./vocal --save_root_inst ./inst --agg 10 --format0 wav
from infer_uvr5 import _audio_pre_, _audio_pre_new
import ffmpeg
import os,traceback
import torch
from multiprocessing import cpu_count
from i18n import I18nAuto

class Config:
    def __init__(self,device,is_half):
        self.device = device
        self.is_half = is_half
        self.n_cpu = 0
        self.gpu_name = None
        self.gpu_mem = None
        self.x_pad, self.x_query, self.x_center, self.x_max = self.device_config()

    def device_config(self) -> tuple:
        if torch.cuda.is_available():
            i_device = int(self.device.split(":")[-1])
            self.gpu_name = torch.cuda.get_device_name(i_device)
            if (
                ("16" in self.gpu_name and "V100" not in self.gpu_name.upper())
                or "P40" in self.gpu_name.upper()
                or "1060" in self.gpu_name
                or "1070" in self.gpu_name
                or "1080" in self.gpu_name
            ):
                print("16系/10系显卡和P40强制单精度")
                self.is_half = False
                for config_file in ["32k.json", "40k.json", "48k.json"]:
                    with open(f"configs/{config_file}", "r") as f:
                        strr = f.read().replace("true", "false")
                    with open(f"configs/{config_file}", "w") as f:
                        f.write(strr)
                with open("trainset_preprocess_pipeline_print.py", "r") as f:
                    strr = f.read().replace("3.7", "3.0")
                with open("trainset_preprocess_pipeline_print.py", "w") as f:
                    f.write(strr)
            else:
                self.gpu_name = None
            self.gpu_mem = int(
                torch.cuda.get_device_properties(i_device).total_memory
                / 1024
                / 1024
                / 1024
                + 0.4
            )
            if self.gpu_mem <= 4:
                with open("trainset_preprocess_pipeline_print.py", "r") as f:
                    strr = f.read().replace("3.7", "3.0")
                with open("trainset_preprocess_pipeline_print.py", "w") as f:
                    f.write(strr)
        elif torch.backends.mps.is_available():
            print("没有发现支持的N卡, 使用MPS进行推理")
            self.device = "mps"
        else:
            print("没有发现支持的N卡, 使用CPU进行推理")
            self.device = "cpu"
            self.is_half = True

        if self.n_cpu == 0:
            self.n_cpu = cpu_count()

        if self.is_half:
            # 6G显存配置
            x_pad = 3
            x_query = 10
            x_center = 60
            x_max = 65
        else:
            # 5G显存配置
            x_pad = 1
            x_query = 6
            x_center = 38
            x_max = 41

        if self.gpu_mem != None and self.gpu_mem <= 4:
            x_pad = 1
            x_query = 5
            x_center = 30
            x_max = 32

        return x_pad, x_query, x_center, x_max
    
i18n = I18nAuto()

torch.manual_seed(114514)

dim_c = 4
ngpu = torch.cuda.device_count()
gpu_infos = []
mem = []
if_gpu_ok = False

if torch.cuda.is_available() or ngpu != 0:
    for i in range(ngpu):
        gpu_name = torch.cuda.get_device_name(i)
        if any(
            value in gpu_name.upper()
            for value in [
                "10",
                "16",
                "20",
                "30",
                "40",
                "A2",
                "A3",
                "A4",
                "P4",
                "A50",
                "500",
                "A60",
                "70",
                "80",
                "90",
                "M4",
                "T4",
                "TITAN",
            ]
        ):
            # A10#A100#V100#A40#P40#M40#K80#A4500
            if_gpu_ok = True  # 至少有一张能用的N卡
            gpu_infos.append("%s\t%s" % (i, gpu_name))
            mem.append(
                int(
                    torch.cuda.get_device_properties(i).total_memory
                    / 1024
                    / 1024
                    / 1024
                    + 0.4
                )
            )

if if_gpu_ok and len(gpu_infos) > 0:
    gpu_info = "\n".join(gpu_infos)
    default_batch_size = min(mem) // 2
else:
    gpu_info = i18n("很遗憾您这没有能用的显卡来支持您训练")
    default_batch_size = 1
gpus = "-".join([i[0] for i in gpu_infos])
    
device = 'cuda:0'
is_half = True

config=Config(device,is_half)
weight_uvr5_root = "uvr5_weights"
now_dir = os.getcwd()
tmp = os.path.join(now_dir, "TEMP")


def uvr(model_name, inp_path, save_root_vocal, paths, save_root_ins, agg, format0):    
    infos = []
    try:
        save_root_vocal = (
            save_root_vocal.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        )
        save_root_ins = (
            save_root_ins.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        )
        if model_name == "onnx_dereverb_By_FoxJoy":
            from MDXNet import MDXNetDereverb
            pre_fun = MDXNetDereverb(15)
        else:
            func = _audio_pre_ if "DeEcho" not in model_name else _audio_pre_new
            pre_fun = func(
                agg=int(agg),
                model_path=os.path.join(weight_uvr5_root, model_name + ".pth"),
                device=config.device,
                is_half=config.is_half,
            )

        need_reformat = 1
        done = 0
        try:
            info = ffmpeg.probe(inp_path, cmd="ffprobe")
            if (
                info["streams"][0]["channels"] == 2
                and info["streams"][0]["sample_rate"] == "44100"
            ):
                need_reformat = 0
                pre_fun._path_audio_(
                    inp_path, save_root_ins, save_root_vocal, format0
                )
                done = 1
        except:
            need_reformat = 1
            traceback.print_exc()
        if need_reformat == 1:
            tmp_path = "%s/%s.reformatted.wav" % (tmp, os.path.basename(inp_path))
            os.system(
                "ffmpeg -i %s -vn -acodec pcm_s16le -ac 2 -ar 44100 %s -y"
                % (inp_path, tmp_path)
            )
            inp_path = tmp_path
        try:
            if done == 0:
                pre_fun._path_audio_(
                    inp_path, save_root_ins, save_root_vocal, format0
                )
            infos.append("%s->Success" % (os.path.basename(inp_path)))
            yield "\n".join(infos)
        except:
            infos.append(
                "%s->%s" % (os.path.basename(inp_path), traceback.format_exc())
            )
            yield "\n".join(infos)
    except:
        infos.append(traceback.format_exc())
        yield "\n".join(infos)
    finally:
        try:
            if model_name == "onnx_dereverb_By_FoxJoy":
                del pre_fun.pred.model
                del pre_fun.pred.model_
            else:
                del pre_fun.model
                del pre_fun
        except:
            traceback.print_exc()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    yield "\n".join(infos)

import argparse
'''
python uvr.py --model_name onnx_dereverb_By_FoxJoy --inp_path ./music/voice.wav --save_root_vocal ./vocal --save_root_inst ./inst --agg 10 --format0 wav
'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str , help='model name', required=True)
    parser.add_argument('--inp_path', type=str , help='input path', required=True)
    parser.add_argument('--save_root_vocal', type=str , help='save root vocal', required=True)
    parser.add_argument('--save_root_inst', type=str , help='save root inst', required=True)
    parser.add_argument('--agg', type=int , help='agg', required=True)
    parser.add_argument('--format0', type=str , help='format0', required=True)
    args = parser.parse_args()
    
    # uvr
    result = uvr(model_name=args.model_name, inp_path=args.inp_path, save_root_vocal=args.save_root_vocal, paths=None, save_root_ins=args.save_root_inst, agg=args.agg, format0=args.format0)
        
    for output in result:
        print(output)

