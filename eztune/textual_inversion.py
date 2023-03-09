import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.extend(str(Path(__file__).parent / x) for x in ["../", "../../../"])
import pytorch_lightning as pl
import torch
from pytorch_lightning.cli import LightningCLI
from typing import List, Optional
import os
import wandb
import dadaptation


class DataModule(pl.LightningDataModule):
    def __init__(self, *, data_dir: str, placeholder_token: str, templates: List[str]):
        super().__init__()
        self.data_dir = data_dir
        self.placeholder_token = placeholder_token
        self.templates = templates

    def train_dataloader(self):
        import modules
        from modules import shared
        import tempfile, os

        with tempfile.TemporaryDirectory() as d:
            template_file = os.path.join(d, "tmp.txt")
            with open(template_file, "w") as f:
                for line in self.templates:
                    print(line.strip(), file=f)

            ds = modules.textual_inversion.dataset.PersonalizedBase(
                data_root=self.data_dir,
                width=-1,
                height=-1,
                repeats=1,
                placeholder_token=self.placeholder_token,
                model=shared.sd_model,
                cond_model=shared.sd_model.cond_stage_model,
                device=shared.devices.device,
                template_file=template_file,
                batch_size=1,
                gradient_step=1,
                shuffle_tags=False,
                tag_drop_out=0,
                latent_sampling_method="once",
                varsize=True,
            )
        dl = modules.textual_inversion.dataset.PersonalizedDataLoader(
            ds,
            latent_sampling_method="once",
            batch_size=1,
            pin_memory=False,
        )
        return dl


class HiddenModule:
    def __init__(self, module):
        self.module = module


class DeltaLinear(torch.nn.Module):
    def __init__(self, orig: torch.nn.Linear) -> None:
        super().__init__()
        if orig.bias is not None:
            self.bias = torch.nn.Parameter(torch.zeros_like(orig.bias, dtype=float))
        self.weight = torch.nn.Parameter(torch.zeros_like(orig.weight, dtype=float))
        self.orig = HiddenModule(orig)

    def forward(self, x):
        import torch.nn.functional as F

        orig = self.orig.module
        weight = orig.weight + self.weight.to(orig.weight)
        bias = None if orig.bias is None else orig.bias + self.bias.to(orig.bias)
        return F.linear(x, weight, bias)


def create_embed(name, init_text, num_vectors_per_token):
    from modules import shared

    cond_model = shared.sd_model.cond_stage_model
    embedded = cond_model.encode_embedding_init_text(
        init_text or "*", num_vectors_per_token
    )
    vec = torch.zeros(
        (num_vectors_per_token, embedded.shape[1]), device=shared.devices.device
    )

    # Only copy if we provided an init_text, otherwise keep vectors as zeros
    if init_text:
        for i in range(num_vectors_per_token):
            vec[i] = embedded[i * int(embedded.shape[0]) // num_vectors_per_token]
    import modules.textual_inversion.textual_inversion

    return modules.textual_inversion.textual_inversion.Embedding(vec, name)


def decompose(A, top_sum=0.5):
    """Low rank approximation of a 2D tensor, keeping only
    largest singular values that sums to topsum * (original sum)"""
    U, S, Vh = torch.linalg.svd(A.float(), full_matrices=False)
    r = max(
        1, sum(torch.cumsum(S, 0) < sum(S) * top_sum * 1.0001)
    )  # 1.0001 because floats
    return U[:, :r] @ torch.diag(S)[:r, :r], Vh[:r]


class StableDiffusionTuner(pl.LightningModule):
    def __init__(
        self,
        *,
        embedding_list: List[str],
        embedding_length: int,
        embedding_init: str,
        embedding_lr: float = 5e-3,
        model_lr: float = 1e-5,
        model_unfrozen_regex: str = r"2.to_[kv]",
        fixed_image_log_params: Optional[dict],
        optimizer_params: Optional[dict] = None,
        optimizer_params_embeds: Optional[dict] = None,
        optimizer_params_weights: Optional[dict] = None,
        log_random_image: bool = True,
        log_image_every_nsteps: int = 0,
        export_every_nsteps: int = 0,
    ):
        super().__init__()
        from modules import sd_hijack

        self.embeds = torch.nn.ParameterDict()
        we = sd_hijack.model_hijack.embedding_db.word_embeddings
        for e in embedding_list:
            v = we[e].vec = torch.nn.Parameter(we[e].vec)
            self.embeds[e] = v
        import re
        from modules import shared

        self.model_weights = torch.nn.ModuleDict()
        for k, v in shared.sd_model.named_modules():
            if model_unfrozen_regex and re.search(model_unfrozen_regex, k):
                module_path, _dot, kp = k.rpartition(".")
                assert type(v) == torch.nn.Linear
                parent = shared.sd_model.get_submodule(module_path)
                self.model_weights[k.replace(".", ">")] = DeltaLinear(v)
                setattr(parent, kp, self.model_weights[k.replace(".", ">")])

        self.log_random_image = log_random_image
        self.fixed_image_log_params = fixed_image_log_params or {}
        self.embedding_lr = embedding_lr
        self.model_lr = model_lr
        self.log_image_every_nsteps = log_image_every_nsteps
        self.export_every_nsteps = export_every_nsteps
        self.optimizer_params = optimizer_params
        self.optimizer_params_embeds = optimizer_params_embeds or {}
        self.optimizer_params_weights = optimizer_params_weights or {}
        wandb.init(project="tuning")

    def _gen_image(self, **kwargs):
        from modules import processing, shared

        print("prompt:", kwargs["prompt"])
        p = processing.StableDiffusionProcessingTxt2Img(
            sd_model=shared.sd_model,
            do_not_save_grid=True,
            do_not_save_samples=True,
            do_not_reload_embeddings=True,
            width=512,
            height=512,
            steps=20,
            sampler_name="DPM++ 2M",
            **kwargs,
        )
        with torch.random.fork_rng():
            processed = processing.process_images(p)
        image = processed.images[0]
        p.close()
        shared.total_tqdm.clear()
        return image

    def _log_image(self, prompt: str):
        from torch.utils.tensorboard.writer import SummaryWriter

        tensorboard: SummaryWriter = self.logger.experiment
        from torchvision.transforms.functional import to_tensor
        ims = []
        if self.log_random_image:
            rand_image = self._gen_image(prompt=prompt)
            ims.append(rand_image)
            tensorboard.add_image(
                "imgs/random", to_tensor(rand_image), global_step=self.global_step
            )
        if self.fixed_image_log_params:
            fixed_image = self._gen_image(**self.fixed_image_log_params)
            ims.append(fixed_image)
            tensorboard.add_image(
                "imgs/fixed", to_tensor(fixed_image), global_step=self.global_step
            )
        return ims

    def export(
        self, path: str, weight_dtype: torch.dtype = torch.float16, top_sum: float = 1.0
    ):
        from safetensors.torch import save_file
        import json

        tensors = {}
        embeddings = []
        weights = {}
        for k, v in self.embeds.items():
            tensors[f"embeddings/{k}"] = v
            embeddings.append(k)
        for module_path, delta_module in self.model_weights.items():
            for k, v in delta_module.named_parameters():
                param_path = f"{module_path.replace('>', '.')}.{k}"
                if top_sum == 1:  # no factorization
                    tensors[f"weights/{param_path}"] = v.to(weight_dtype)
                    weights[param_path] = "delta"
                else:
                    (
                        tensors[f"weights/{param_path}.US"],
                        tensors[f"weights/{param_path}.Vh"],
                    ) = map(
                        lambda a: a.to(weight_dtype).contiguous(), decompose(v, top_sum)
                    )
                    weights[param_path] = "delta_factors"
        metadata = dict(version="0.1.0", embeddings=embeddings, weights=weights)
        save_file(tensors, path, {"tuner": json.dumps(metadata)})

    def training_step(self, batch, batch_idx):
        from modules import shared

        with shared.devices.autocast():
            x = batch.latent_sample.to(shared.devices.device, non_blocking=False)
            c = shared.sd_model.cond_stage_model(batch.cond_text)
            loss = shared.sd_model(x, c)[0]
        logdict = {}
        self.log("loss/train", loss.item())
        logdict['loss'] = loss.item()
        if (
            self.log_image_every_nsteps
            and (self.global_step + 1) % self.log_image_every_nsteps == 0
        ):
            ims = self._log_image(batch.cond_text[0])
            if ims:
                logims = [wandb.Image(im, caption=f"{i+1}") for i, im in enumerate(ims)]
                logdict['val img'] = logims
        wandb.log(logdict)

        if (
            self.export_every_nsteps
            and (self.global_step + 1) % self.export_every_nsteps == 0
        ):
            path = os.path.join(
                self._trainer.log_dir, f"{self.global_step}.tuner.safetensors"
            )
            print(path)
            self.export(path, top_sum=0.5)

        return loss

    def configure_optimizers(self):
        param_groups = []
        if self.embeds:
            param_groups.append(
                dict(params=list(self.embeds.values()), **self.optimizer_params_embeds)
            )
        if self.model_weights:
            param_groups.append(
                dict(
                    params=list(self.model_weights.parameters()),
                    **self.optimizer_params_weights
                )
            )

        return torch.optim.AdamW(param_groups, **(self.optimizer_params or {}))


class EZCLI(LightningCLI):
    def before_instantiate_classes(self):
        from modules import sd_hijack, shared

        embed_cfg = self.config.fit.model
        el = embed_cfg.embedding_list
        for embed in el:
            embed = create_embed(
                embed, embed_cfg.embedding_init, embed_cfg.embedding_length
            )
            sd_hijack.model_hijack.embedding_db.register_embedding(
                embed, shared.sd_model
            )
        print(f"{len(el)} text embedding(s) registered: {', '.join(el)}")


def cli_main():
    import sys

    argv = [*sys.argv]
    sys.argv = ["webui.py", "--disable-console-progressbars"]
    torch_load = torch.load  # somehow webui breaks checkpoint loading
    import webui

    webui.initialize()
    torch.load = torch_load
    sys.argv = argv
    EZCLI(StableDiffusionTuner, DataModule, trainer_defaults={"log_every_n_steps": 1})


if __name__ == "__main__":
    cli_main()
