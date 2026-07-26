from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ResidualParameterization:
    """Map latent residual decisions onto the full control-rate horizon."""

    rollout_horizon: int
    decision_horizon: int
    mode: str
    basis: torch.Tensor
    interior_pseudoinverse: torch.Tensor | None

    @classmethod
    def create(
        cls,
        *,
        rollout_horizon: int,
        decision_horizon: int,
        mode: str,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> "ResidualParameterization":
        if rollout_horizon <= 0:
            raise ValueError("rollout_horizon must be positive")
        if mode not in {"full", "linear_control_points"}:
            raise ValueError("residual parameterization must be 'full' or 'linear_control_points'")
        if mode == "full":
            if decision_horizon != rollout_horizon:
                raise ValueError("full residual parameterization requires decision_horizon == rollout_horizon")
            basis = torch.eye(rollout_horizon, device=device, dtype=dtype)
            return cls(rollout_horizon, decision_horizon, mode, basis, None)
        if not 2 <= decision_horizon <= rollout_horizon:
            raise ValueError("linear control points require 2 <= decision_horizon <= rollout_horizon")
        identity = torch.eye(decision_horizon, device=device, dtype=dtype).unsqueeze(0)
        basis = F.interpolate(
            identity,
            size=rollout_horizon,
            mode="linear",
            align_corners=True,
        ).squeeze(0).transpose(0, 1).contiguous()
        interior = basis[:, 1:-1]
        interior_pseudoinverse = (
            torch.linalg.pinv(interior)
            if decision_horizon > 2
            else torch.empty((0, rollout_horizon), device=device, dtype=dtype)
        )
        return cls(
            rollout_horizon,
            decision_horizon,
            mode,
            basis,
            interior_pseudoinverse,
        )

    def _basis_like(self, reference: torch.Tensor) -> torch.Tensor:
        return self.basis.to(device=reference.device, dtype=reference.dtype)

    def expand(self, control_points: torch.Tensor) -> torch.Tensor:
        squeeze = control_points.ndim == 2
        points = control_points.unsqueeze(0) if squeeze else control_points
        if points.ndim != 3 or points.shape[1] != self.decision_horizon:
            raise ValueError(
                "control_points must have shape [batch, decision_horizon, joints] "
                f"or [decision_horizon, joints], got {tuple(control_points.shape)}"
            )
        expanded = torch.einsum("hk,bkj->bhj", self._basis_like(points), points)
        return expanded[0] if squeeze else expanded

    def compress(self, sequence: torch.Tensor) -> torch.Tensor:
        """Least-squares fit with exact first and last control points."""
        squeeze = sequence.ndim == 2
        values = sequence.unsqueeze(0) if squeeze else sequence
        if values.ndim != 3 or values.shape[1] != self.rollout_horizon:
            raise ValueError(
                "sequence must have shape [batch, rollout_horizon, joints] "
                f"or [rollout_horizon, joints], got {tuple(sequence.shape)}"
            )
        if self.mode == "full":
            result = values.clone()
        else:
            basis = self._basis_like(values)
            first = values[:, 0]
            last = values[:, -1]
            if self.decision_horizon == 2:
                result = torch.stack([first, last], dim=1)
            else:
                target = (
                    values
                    - basis[:, 0].view(1, -1, 1) * first.unsqueeze(1)
                    - basis[:, -1].view(1, -1, 1) * last.unsqueeze(1)
                )
                pinv = self.interior_pseudoinverse
                assert pinv is not None
                interior = torch.einsum(
                    "kh,bhj->bkj",
                    pinv.to(device=values.device, dtype=values.dtype),
                    target,
                )
                result = torch.cat([first.unsqueeze(1), interior, last.unsqueeze(1)], dim=1)
        result = torch.clamp(result, -1.0, 1.0)
        return result[0] if squeeze else result

    def shift(self, control_points: torch.Tensor, shift_steps: int) -> torch.Tensor:
        if shift_steps <= 0:
            return control_points.clone()
        full = self.expand(control_points)
        squeeze = full.ndim == 2
        values = full.unsqueeze(0) if squeeze else full
        shift = min(int(shift_steps), self.rollout_horizon)
        if shift >= self.rollout_horizon:
            shifted = values[:, -1:].expand(-1, self.rollout_horizon, -1)
        else:
            tail = values[:, -1:].expand(-1, shift, -1)
            shifted = torch.cat([values[:, shift:], tail], dim=1)
        compressed = self.compress(shifted)
        return compressed[0] if squeeze and compressed.ndim == 3 else compressed

    def reconstruction_error(self, control_points: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        reconstructed = self.expand(control_points)
        numerator = torch.linalg.vector_norm(reconstructed - target, dim=(-2, -1))
        denominator = torch.linalg.vector_norm(target, dim=(-2, -1)).clamp_min(1e-8)
        return numerator / denominator
