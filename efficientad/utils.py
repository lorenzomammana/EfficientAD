import torch
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
from typing import Optional


@torch.no_grad()
def map_normalization(
    validation_loader: DataLoader,
    teacher: nn.Module,
    student: nn.Module,
    autoencoder: nn.Module,
    teacher_mean: torch.Tensor,
    teacher_std: torch.Tensor,
    out_channels: int,
    device: torch.device,
    desc: str = "Map normalization",
):
    maps_st = []
    maps_ae = []
    # ignore augmented ae image
    for image, _ in tqdm(validation_loader, desc=desc):
        _, map_st, map_ae = predict(
            image=image.to(device),
            teacher=teacher,
            student=student,
            autoencoder=autoencoder,
            teacher_mean=teacher_mean,
            teacher_std=teacher_std,
            out_channels=out_channels,
        )
        maps_st.append(map_st)
        maps_ae.append(map_ae)

    maps_st = torch.cat(maps_st)
    maps_ae = torch.cat(maps_ae)
    q_st_start = torch.quantile(maps_st, q=0.9)
    q_st_end = torch.quantile(maps_st, q=0.995)
    q_ae_start = torch.quantile(maps_ae, q=0.9)
    q_ae_end = torch.quantile(maps_ae, q=0.995)
    return q_st_start, q_st_end, q_ae_start, q_ae_end


@torch.no_grad()
def predict(
    image: torch.Tensor,
    teacher: nn.Module,
    student: nn.Module,
    autoencoder: nn.Module,
    teacher_mean: torch.Tensor,
    teacher_std: torch.Tensor,
    out_channels: int,
    q_st_start: Optional[torch.Tensor] = None,
    q_st_end: Optional[torch.Tensor] = None,
    q_ae_start: Optional[torch.Tensor] = None,
    q_ae_end: Optional[torch.Tensor] = None,
):
    teacher_output = teacher(image)
    teacher_output = (teacher_output - teacher_mean) / teacher_std
    student_output = student(image)
    autoencoder_output = autoencoder(image)
    map_st = torch.mean((teacher_output - student_output[:, :out_channels]) ** 2, dim=1, keepdim=True)
    map_ae = torch.mean((autoencoder_output - student_output[:, out_channels:]) ** 2, dim=1, keepdim=True)
    if q_st_start is not None:
        map_st = 0.1 * (map_st - q_st_start) / (q_st_end - q_st_start)
    if q_ae_start is not None:
        map_ae = 0.1 * (map_ae - q_ae_start) / (q_ae_end - q_ae_start)
    map_combined = 0.5 * map_st + 0.5 * map_ae
    return map_combined, map_st, map_ae


@torch.no_grad()
def teacher_normalization(teacher: nn.Module, train_loader: DataLoader, device: torch.device):
    mean_outputs = []
    teacher_outputs = []
    for (train_image, _), _ in tqdm(train_loader, desc="Computing mean of features"):
        teacher_output = teacher(train_image.to(device))
        mean_output = torch.mean(teacher_output, dim=[0, 2, 3])
        mean_outputs.append(mean_output)
        teacher_outputs.append(teacher_output)
    channel_mean = torch.mean(torch.stack(mean_outputs), dim=0)
    channel_mean = channel_mean[None, :, None, None]

    mean_distances = []
    for teacher_output in tqdm(teacher_outputs, desc="Computing std of features"):
        distance = (teacher_output - channel_mean) ** 2
        mean_distance = torch.mean(distance, dim=[0, 2, 3])
        mean_distances.append(mean_distance)

    channel_var = torch.mean(torch.stack(mean_distances), dim=0)
    channel_var = channel_var[None, :, None, None]
    channel_std = torch.sqrt(channel_var)

    return channel_mean, channel_std
