import copy

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, AutoConfig
from safetensors.torch import load_file, save_file
from huggingface_hub import snapshot_download
from nltk.tokenize import wordpunct_tokenize
import torchvision
import math
from captor.models.i3res import I3ResNet
import json 
import os

CONFIG_NAME = "config.json"
WEIGHTS_NAME = "model.safetensors"
TOKENIZER_DIR = "tokenizer"

class ImageEncoder(nn.Module):
    def __init__(self, path_weights_resnet152, ImageEmbedding: bool = False, nb_channels = 1, hook=False, use_checkpoints=True, return_skips=False):
        super().__init__()
        self.ImageEmbedding = ImageEmbedding
        resnet = torchvision.models.resnet152(weights=None)
        if path_weights_resnet152 is not None:  # training-from-scratch only
            resnet.load_state_dict(torch.load(path_weights_resnet152, map_location="cpu"))

        self.i3_resnet = I3ResNet(
            copy.deepcopy(resnet), class_nb=1692, conv_class=True, return_skips=return_skips, ImageEmbedding=self.ImageEmbedding, nb_channels=nb_channels, hook=hook, use_checkpoints=use_checkpoints
        )

    def forward(self, image):
        if self.ImageEmbedding:
            contrastive_features = self.i3_resnet(image)
            return contrastive_features
        else:
            contrastive_features, ehr_features = self.i3_resnet(image)
            return contrastive_features, ehr_features


class TextEncoder(nn.Module):
    def __init__(self, path_clinical_long_former=None, text_config=None):
        super().__init__()
        self.tokenizer = None
        if path_clinical_long_former is not None:      # training: pretrained init
            self.tokenizer = AutoTokenizer.from_pretrained(path_clinical_long_former)
            self.text_encoder = AutoModel.from_pretrained(path_clinical_long_former)
        elif text_config is not None:                  # inference: random init, weights loaded later
            cfg = dict(text_config)
            model_type = cfg.pop("model_type")
            self.text_encoder = AutoModel.from_config(AutoConfig.for_model(model_type, **cfg))
        else:
            raise ValueError("Provide either text_config or path_clinical_long_former")
        self.text_encoder.gradient_checkpointing_enable()
        self.linear_layer = nn.Linear(768, 512)

    def forward(self, text_labels):
        text_labels = [sanitize_report(text) for text in text_labels]
        inputs = self.tokenizer(
            text_labels,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        )
        inputs = {k: v.to(self.text_encoder.device) for k, v in inputs.items()}
        text_embeddings = self.text_encoder(**inputs).last_hidden_state[:, 0, :]
        text_embeddings = self.linear_layer(text_embeddings)
        return text_embeddings


class MerlinArchitecture(nn.Module):
    def __init__(self, init_logit_scale: float = math.log(1/0.07), ImageEmbedding: bool = False, nb_channels = 1, hook=False, use_checkpoints=True, return_skips=False, path_weights_resnet152=None, path_clinical_long_former=None, train_logit_scale=False, text_config=None): 
        super().__init__()
        self.ImageEmbedding = ImageEmbedding
        self.encode_image = ImageEncoder(ImageEmbedding=self.ImageEmbedding, nb_channels=nb_channels, hook=hook, use_checkpoints=use_checkpoints, return_skips=return_skips, path_weights_resnet152=path_weights_resnet152)
        self.encode_text = TextEncoder(path_clinical_long_former=path_clinical_long_former, text_config=text_config)
        self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)
        self.hook = hook
        self.return_skips = return_skips
        self.use_checkpoints= use_checkpoints
        if train_logit_scale==True:
            self.logit_scale_max = math.log(100)
        
        # else:
        #     # Valeur fixe (buffer non entraînable, suit device avec .to())
        #     self.register_buffer("logit_scale", torch.ones([]) * init_logit_scale)
        self.config = dict(
            init_logit_scale=init_logit_scale, ImageEmbedding=ImageEmbedding, nb_channels=nb_channels,
            hook=hook, use_checkpoints=use_checkpoints, return_skips=return_skips,
            train_logit_scale=train_logit_scale,
            text_config=json.loads(self.encode_text.text_encoder.config.to_json_string()),
        )

    def forward(self, image, text=None):
        if self.ImageEmbedding and text is None:
            image_features = self.encode_image(image)
            return image_features
        elif self.ImageEmbedding and text is not None:
            raise ValueError("Text input not required for image embedding")
        elif text is None:
            raise ValueError("Text input required for Image and Text embedding")


        image_features, ehr_features = self.encode_image(image)
        text_features = self.encode_text(text)

        if len(image_features.shape) == 1:
            image_features = image_features.unsqueeze(0)
        if len(text_features.shape) == 1:
            text_features = text_features.unsqueeze(0)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return (
            image_features,
            ehr_features,
            text_features,
        )
    
    
    @torch.no_grad()
    def get_logit_scale(self, clamp: bool = True, exp: bool = True):
        """
        Récupère la température (logit_scale), avec options de clamp et exp.
        Utilisé en éval, mais peut être appelé en train aussi.
        """
        ls = self.logit_scale
        if clamp:
            # clamp dans le graphe : si ls est Parameter, renvoie un Tensor clampé ;
            # si ls est buffer, idem.
            ls = torch.clamp(ls, max=self.logit_scale_max)
        if exp:
            ls = ls.exp()
        return ls
    
    def save_pretrained(self, save_directory: str, save_tokenizer: bool = True):
        os.makedirs(save_directory, exist_ok=True)
        with open(os.path.join(save_directory, CONFIG_NAME), "w") as f:
            json.dump(self.config, f, indent=2)
        state = {k: v.detach().cpu().contiguous() for k, v in self.state_dict().items()}
        save_file(state, os.path.join(save_directory, WEIGHTS_NAME), metadata={"format": "pt"})
        if save_tokenizer and self.encode_text.tokenizer is not None:
            self.encode_text.tokenizer.save_pretrained(os.path.join(save_directory, TOKENIZER_DIR))

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *, subfolder: str = None,
                        revision=None, cache_dir=None, token=None, local_files_only=False,
                        device="cpu", **config_overrides):
        root = str(pretrained_model_name_or_path)
        if not os.path.isdir(root):
            # download ONLY what this checkpoint needs
            patterns = [f"{TOKENIZER_DIR}/*"]
            if subfolder:
                patterns += [f"{subfolder}/{CONFIG_NAME}", f"{subfolder}/{WEIGHTS_NAME}"]
            else:
                patterns += [CONFIG_NAME, WEIGHTS_NAME]
            root = snapshot_download(repo_id=root, revision=revision, cache_dir=cache_dir,
                                    token=token, local_files_only=local_files_only,
                                    allow_patterns=patterns)

        model_dir = os.path.join(root, subfolder) if subfolder else root
        # tokenizer: checkpoint-specific if present, else the shared one at repo root
        tok_dir = os.path.join(model_dir, TOKENIZER_DIR)
        if not os.path.isdir(tok_dir):
            tok_dir = os.path.join(root, TOKENIZER_DIR)

        with open(os.path.join(model_dir, CONFIG_NAME)) as f:
            config = json.load(f)
        config.update(config_overrides)

        model = cls(**config)
        model.load_state_dict(load_file(os.path.join(model_dir, WEIGHTS_NAME)), strict=True)
        model.encode_text.tokenizer = AutoTokenizer.from_pretrained(tok_dir)
        return model.to(device).eval()




def sanitize_report(report):
    report = report.lower()
    return " ".join(wordpunct_tokenize(report))
