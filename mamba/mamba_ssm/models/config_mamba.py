from dataclasses import dataclass, field
import copy
from typing import Any, Dict, List, Optional, Tuple, Union

from transformers import PretrainedConfig

@dataclass
class MambaConfig:

    d_model: int = 2560
    n_layer: int = 64
    vocab_size: int = 50277
    ssm_cfg: dict = field(default_factory=dict)
    rms_norm: bool = True
    residual_in_fp32: bool = True
    fused_add_norm: bool = True
    pad_vocab_size_multiple: int = 8
    tie_embeddings: bool = True
    model_type:str = "mamba"

    def to_dict(self):
        """将配置对象的属性转换为字典"""
        return {attr: getattr(self, attr) for attr in self.__dict__ if not attr.startswith("_")}
    
    # def to_dict(self) -> Dict[str, Any]:
    #     """
    #     Serializes this instance to a Python dictionary.

    #     Returns:
    #         `Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance.
    #     """
    #     output = copy.deepcopy(self.__dict__)
    #     if hasattr(self.__class__, "model_type"):
    #         output["model_type"] = self.__class__.model_type
    #     if "_auto_class" in output:
    #         del output["_auto_class"]
    #     if "_commit_hash" in output:
    #         del output["_commit_hash"]
    #     if "_attn_implementation_internal" in output:
    #         del output["_attn_implementation_internal"]



    #     for key, value in output.items():
    #         # Deal with nested configs like CLIP
    #         if isinstance(value, PretrainedConfig):
    #             value = value.to_dict()

    #         output[key] = value

    #     if hasattr(self, "quantization_config"):
    #         output["quantization_config"] = (
    #             self.quantization_config.to_dict()
    #             if not isinstance(self.quantization_config, dict)
    #             else self.quantization_config
    #         )

    #         # pop the `_pre_quantization_dtype` as torch.dtypes are not serializable.
    #         _ = output.pop("_pre_quantization_dtype", None)

    #     self.dict_torch_dtype_to_str(output)

    #     return output
    
    def dict_torch_dtype_to_str(self, d: Dict[str, Any]) -> None:
        """
        Checks whether the passed dictionary and its nested dicts have a *torch_dtype* key and if it's not None,
        converts torch.dtype to a string of just the type. For example, `torch.float32` get converted into *"float32"*
        string, which can then be stored in the json format.
        """
        if d.get("torch_dtype", None) is not None and not isinstance(d["torch_dtype"], str):
            d["torch_dtype"] = str(d["torch_dtype"]).split(".")[1]
        for value in d.values():
            if isinstance(value, dict):
                self.dict_torch_dtype_to_str(value)