from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MovingAverage(nn.Module):
    def __init__(self, kernel_size: int, stride: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        pad = (self.kernel_size - 1) // 2
        front = x[:, 0:1, :].repeat(1, pad, 1)
        end = x[:, -1:, :].repeat(1, pad, 1)
        x_pad = torch.cat([front, x, end], dim=1)

        if mask is None:
            return self.avg(x_pad.permute(0, 2, 1)).permute(0, 2, 1)

        mask = mask.to(dtype=x.dtype)
        front_m = mask[:, 0:1, :].repeat(1, pad, 1)
        end_m = mask[:, -1:, :].repeat(1, pad, 1)
        mask_pad = torch.cat([front_m, mask, end_m], dim=1)
        valid = 1.0 - mask_pad

        b, t, n = x_pad.shape
        x_reshape = x_pad.permute(0, 2, 1).reshape(b * n, 1, t)
        v_reshape = valid.permute(0, 2, 1).reshape(b * n, 1, t)
        weight = torch.ones(1, 1, self.kernel_size, device=x.device, dtype=x.dtype)
        num = F.conv1d(x_reshape * v_reshape, weight, stride=1)
        den = F.conv1d(v_reshape, weight, stride=1)
        mean = num / (den + 1e-6)
        return mean.reshape(b, n, -1).permute(0, 2, 1)


class SeriesDecomposition(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.moving_avg = MovingAverage(kernel_size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        trend = self.moving_avg(x, mask=mask)
        seasonal = x - trend
        return seasonal, trend


class FeedForward(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(hidden_dim, 4 * hidden_dim, kernel_size=(1, 1), bias=True),
            nn.GELU(),
            nn.Dropout(p=0.15),
            nn.Conv2d(4 * hidden_dim, hidden_dim, kernel_size=(1, 1), bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x) + x


class CityPrompt(nn.Module):
    def __init__(self, input_dim: int, prompt_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, prompt_dim),
        )

    def forward(self, city_desc: torch.Tensor) -> torch.Tensor:
        return self.net(city_desc)


class BaseBranch(nn.Module):
    def __init__(
        self,
        seq_len: int,
        horizon: int,
        num_layers: int,
        model_dim: int,
        prompt_dim: int,
        tod_size: int,
        kernel_size: int,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.horizon = horizon
        self.embed_dim = model_dim
        self.prompt_dim = prompt_dim
        self.hidden_dim = self.embed_dim + 2 * self.prompt_dim
        self.time_of_day_size = tod_size
        self.day_of_week_size = 7

        self.time_in_day_emb = nn.Parameter(torch.empty(self.time_of_day_size, self.prompt_dim))
        self.day_in_week_emb = nn.Parameter(torch.empty(self.day_of_week_size, self.prompt_dim))
        nn.init.xavier_uniform_(self.time_in_day_emb)
        nn.init.xavier_uniform_(self.day_in_week_emb)

        self.decomposition = SeriesDecomposition(kernel_size)
        self.time_series_seasonal = nn.Conv1d(seq_len, self.embed_dim, kernel_size=1, bias=True)
        self.time_series_trend = nn.Conv1d(seq_len, self.embed_dim, kernel_size=1, bias=True)
        self.encoder = nn.Sequential(*[FeedForward(self.hidden_dim) for _ in range(num_layers)])
        self.prediction_head = nn.Conv2d(self.hidden_dim, horizon, kernel_size=(1, 1), bias=True)
        self.reconstruction_head = nn.Conv2d(self.hidden_dim, seq_len, kernel_size=(1, 1), bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1))

    @staticmethod
    def _safe_time_index(data: torch.Tensor, size: int) -> torch.Tensor:
        data = torch.clamp(data, 0.0, 1.0 - 1e-6)
        idx = (data * size).long()
        return torch.clamp(idx, 0, size - 1)

    def forward(
        self,
        history_data: torch.Tensor,
        return_reconstruction: bool = False,
        mae_mask: torch.Tensor | None = None,
    ):
        tod_idx = self._safe_time_index(history_data[:, -1, :, 1], self.time_of_day_size)
        dow_idx = self._safe_time_index(history_data[:, -1, :, 2], self.day_of_week_size)
        tod_emb = self.time_in_day_emb[tod_idx].transpose(1, 2).unsqueeze(-1)
        dow_emb = self.day_in_week_emb[dow_idx].transpose(1, 2).unsqueeze(-1)

        speed = history_data[..., 0]
        seasonal, trend = self.decomposition(speed, mask=mae_mask if return_reconstruction else None)
        series_emb = (self.time_series_seasonal(seasonal) + self.time_series_trend(trend)).unsqueeze(-1)

        hidden = torch.cat([series_emb, tod_emb, dow_emb], dim=1)
        h = hidden.transpose(1, -1)
        hidden = self.encoder(hidden)
        z = hidden.transpose(1, -1)
        prediction = self.prediction_head(hidden)

        if return_reconstruction:
            reconstruction = self.reconstruction_head(hidden).squeeze(-1)
            return h, z, prediction, reconstruction
        return h, z, prediction

    def residual_projection(self, hidden_tokens: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(hidden_tokens.transpose(1, -1))
        return hidden.transpose(1, -1)


class TAPR(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_prototypes: int,
        num_heads: int,
        prompt_dim: int = 0,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if feature_dim % num_heads != 0:
            raise ValueError("feature_dim must be divisible by num_heads.")
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.num_prototypes = num_prototypes
        self.head_dim = feature_dim // num_heads
        self.prototypes = nn.Parameter(torch.randn(num_heads, num_prototypes, self.head_dim))
        self.prompt_to_bias = nn.Linear(prompt_dim, num_prototypes) if prompt_dim > 0 else None
        self.value = nn.Conv2d(feature_dim, feature_dim, kernel_size=(1, 1))
        self.ffn = nn.Sequential(
            nn.Conv2d(2 * feature_dim, 8 * feature_dim, kernel_size=(1, 1)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(8 * feature_dim, feature_dim, kernel_size=(1, 1)),
        )
        self.norm = nn.BatchNorm2d(feature_dim)

    def forward(
        self,
        tokens: torch.Tensor,
        city_prompt: torch.Tensor | None = None,
        return_assignment: bool = False,
    ):
        input_tokens = tokens.permute(0, 3, 1, 2)
        b, f, t, n = input_tokens.shape

        q = self.value(input_tokens)
        q = torch.stack(torch.split(q, self.head_dim, dim=1), dim=1)
        logits = torch.einsum("hkd,bhdtn->bhktn", self.prototypes, q).transpose(-2, -3)
        logits = logits / (self.head_dim ** 0.5)

        if city_prompt is not None and self.prompt_to_bias is not None:
            bias = self.prompt_to_bias(city_prompt).view(b, 1, 1, self.num_prototypes, 1)
            logits = logits + bias

        prototype_to_node = torch.softmax(logits, dim=-1)
        node_to_prototype = torch.softmax(logits, dim=-2)

        v = torch.stack(torch.split(input_tokens, self.head_dim, dim=1), dim=1)
        v = torch.einsum("bhftn,bhtkn->bhftk", v, prototype_to_node)
        v = torch.einsum("bhftk,bhtkn->bhftn", v, node_to_prototype)
        v = v.transpose(0, 1).reshape(b, f, t, n)

        output = torch.cat([input_tokens - v, v], dim=1)
        output = self.ffn(output)
        output = self.norm(output + input_tokens)
        output = output.permute(0, 2, 3, 1)

        if return_assignment:
            return output, prototype_to_node
        return output


class TAPSTAR(nn.Module):
    def __init__(
        self,
        seq_len: int,
        horizon: int,
        tod_size: int,
        num_layers: int = 3,
        model_dim: int = 64,
        prompt_dim: int = 32,
        kernel_size: int = 3,
        num_prototypes: int = 8,
        num_heads: int = 8,
        use_city_prompt: bool = True,
        city_desc_dim: int = 16,
        city_prompt_hidden: int = 64,
        decoder_hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.base = BaseBranch(seq_len, horizon, num_layers, model_dim, prompt_dim, tod_size, kernel_size)
        self.use_city_prompt = use_city_prompt
        self.city_prompt = CityPrompt(city_desc_dim, prompt_dim, city_prompt_hidden) if use_city_prompt else None
        self.tapr = TAPR(self.base.hidden_dim, num_prototypes, num_heads, prompt_dim if use_city_prompt else 0)
        self.decoder = nn.Sequential(
            nn.Linear(self.base.hidden_dim, decoder_hidden_dim),
            nn.GELU(),
            nn.Linear(decoder_hidden_dim, horizon),
        )

    def encode_prompt(self, city_desc: torch.Tensor | None) -> torch.Tensor | None:
        if self.city_prompt is None or city_desc is None:
            return None
        return self.city_prompt(city_desc)

    def forward(self, history_data: torch.Tensor, city_desc: torch.Tensor | None = None, return_parts: bool = False):
        city_prompt = self.encode_prompt(city_desc)
        h, z, base_prediction = self.base(history_data)
        tapr_tokens = self.tapr(z, city_prompt=city_prompt)
        residual_tokens = self.base.residual_projection(h - tapr_tokens)
        residual_prediction = self.decoder(residual_tokens).transpose(1, -1)
        if return_parts:
            return base_prediction, residual_prediction
        return base_prediction + residual_prediction


class TAPRPretrainer(nn.Module):
    def __init__(
        self,
        seq_len: int,
        tod_size: int,
        num_layers: int = 3,
        model_dim: int = 64,
        prompt_dim: int = 32,
        kernel_size: int = 3,
        num_prototypes: int = 8,
        num_heads: int = 8,
        use_city_prompt: bool = True,
        city_desc_dim: int = 16,
        city_prompt_hidden: int = 64,
    ) -> None:
        super().__init__()
        self.base = BaseBranch(seq_len, seq_len, num_layers, model_dim, prompt_dim, tod_size, kernel_size)
        self.use_city_prompt = use_city_prompt
        self.city_prompt = CityPrompt(city_desc_dim, prompt_dim, city_prompt_hidden) if use_city_prompt else None
        self.tapr = TAPR(self.base.hidden_dim, num_prototypes, num_heads, prompt_dim if use_city_prompt else 0)
        self.residual_head = nn.Conv2d(self.base.hidden_dim, seq_len, kernel_size=(1, 1), bias=True)

    def encode_prompt(self, city_desc: torch.Tensor | None) -> torch.Tensor | None:
        if self.city_prompt is None or city_desc is None:
            return None
        return self.city_prompt(city_desc)

    def get_city_prompt(self, city_desc: torch.Tensor | None) -> torch.Tensor | None:
        return self.encode_prompt(city_desc)

    def forward(
        self,
        history_data: torch.Tensor,
        mae_mask: torch.Tensor | None = None,
        city_desc: torch.Tensor | None = None,
        return_assignment: bool = False,
    ):
        city_prompt = self.encode_prompt(city_desc)
        _, z, _, coarse_reconstruction = self.base(
            history_data,
            return_reconstruction=True,
            mae_mask=mae_mask,
        )
        if return_assignment:
            tapr_tokens, assignment = self.tapr(z, city_prompt=city_prompt, return_assignment=True)
        else:
            tapr_tokens = self.tapr(z, city_prompt=city_prompt)
            assignment = None
        delta_hat = self.residual_head(tapr_tokens.transpose(1, -1)).squeeze(-1)
        speed_hat = coarse_reconstruction + delta_hat
        if return_assignment:
            return coarse_reconstruction, delta_hat, speed_hat, z, assignment
        return coarse_reconstruction, delta_hat, speed_hat
