from typing import Union, Tuple, List
from dynamic_network_architectures.building_blocks.helper import get_matching_batchnorm
from torch import nn
import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.models import clip_model_3d
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.models import unet_decoder

import os

from huggingface_hub import hf_hub_download


def download_file(
    repo_id: str,
    filename: str,
    local_dir: str,
):
    os.makedirs(local_dir, exist_ok=True)
    local_file_path = hf_hub_download(
        repo_id=repo_id, filename=filename, local_dir=local_dir
    )
    print(f"{filename} downloaded and saved to {local_file_path}")
    return local_file_path

class nnUNetTrainerMerlin(nnUNetTrainer):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True, # Ignored, kept for compatibility
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        
        print(self.configuration_manager)
        
    
    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        model_config = {
        "architecture": "i3_resnet_clinical_longformer",
        "text_encoder": "clinical_longformer",
        "use_ehr": True
        }
        model = clip_model_3d.Clip3D(model_config)

        # Load in Merlin weights
        file_path = download_file(
            repo_id="stanfordmimi/Merlin",
            filename="i3_resnet_clinical_longformer_best_clip_04-02-2024_23-21-36_epoch_99.pt",
            local_dir=os.path.join(os.path.dirname(__file__), "models"),
        )
        checkpoint = torch.load(file_path)
        model_state_dict = model.state_dict()
        filtered_checkpoint = {k: v for k, v in checkpoint.items() if k in model_state_dict and model_state_dict[k].size() == v.size()}
        missing, unexpected = model.load_state_dict(filtered_checkpoint, strict=False)
        print("Missing keys: ", missing)
        print("Unexpected keys: ", unexpected)

        model = model.encode_image
        decoder = unet_decoder.UNetDecoder(num_classes=num_output_channels, deep_supervision=enable_deep_supervision)
        model = torch.nn.Sequential(model, decoder)
        
        for name, param in model.named_parameters():
            print(name)

        return model
        

    def set_deep_supervision_enabled(self, enabled: bool):
        """
        This function is specific for the default architecture in nnU-Net. If you change the architecture, there are
        chances you need to change this as well!
        """
        pass
