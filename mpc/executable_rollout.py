"""Fixed-shape executable-command rollouts for the CUDA planner.

The command projector and learned dynamics are causally interleaved: the
measured position predicted at step ``t`` is an input to the projector at
step ``t+1``.  The time loop therefore cannot be vectorised away, but it can
be kept entirely on the GPU.  On CUDA, fixed-shape calls are captured in a
CUDA Graph so Python and per-step kernel-launch overhead are paid only during
warm-up.  The eager path remains the numerical reference and the CPU
fallback.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import warnings

import torch

from neural_dynamics.rollout import rollout_dynamics_step
from robot_runtime.executable_command import (
    ExecutableCommandSpec,
    step_executable_command_torch,
)


@dataclass(frozen=True)
class ExecutableRolloutOutput:
    """Device-resident output of one projected learned-dynamics rollout."""

    q_ref_sequences: torch.Tensor
    expected_raw_sequences: torch.Tensor
    pred_states: torch.Tensor
    final_history: torch.Tensor
    final_q_ref: torch.Tensor
    final_velocity: torch.Tensor
    fallback_mask: torch.Tensor


def _forward_rollout(
    model: torch.nn.Module,
    normalizer: Any,
    model_type: str,
    initial_history: torch.Tensor,
    requested_q_ref: torch.Tensor,
    previous_q_ref: torch.Tensor,
    previous_velocity: torch.Tensor,
    fallback_q_ref: torch.Tensor,
    expected_raw: torch.Tensor,
    expected_raw_mask: torch.Tensor,
    fail_closed: bool,
    state_dim: int,
    target_mode: str,
    control_dt: float,
    spec: ExecutableCommandSpec,
    vectors: dict[str, torch.Tensor],
    exact: bool,
) -> ExecutableRolloutOutput:
    """Run the interleaved projector/GRU loop without host interaction."""
    batch_size, horizon, action_dim = requested_q_ref.shape
    history = initial_history.clone()
    predicted_state = history[:, -1, :state_dim]
    previous_q = previous_q_ref.clone()
    previous_v = previous_velocity.clone()
    q_ref_sequences = torch.empty(
        (batch_size, horizon, action_dim),
        device=requested_q_ref.device,
        dtype=requested_q_ref.dtype,
    )
    raw_sequences = torch.empty(
        (batch_size, horizon, action_dim),
        device=requested_q_ref.device,
        dtype=torch.int64,
    )
    pred_states = torch.empty(
        (batch_size, horizon + 1, state_dim),
        device=requested_q_ref.device,
        dtype=predicted_state.dtype,
    )
    pred_states[:, 0] = predicted_state
    fallback_mask = torch.zeros(
        (batch_size, horizon), device=requested_q_ref.device, dtype=torch.bool
    )
    for step_index in range(horizon):
        requested_i = requested_q_ref[:, step_index]
        _, raw_i, transmitted_i, velocity_i = step_executable_command_torch(
            requested_i,
            predicted_state[:, : action_dim],
            previous_q,
            previous_v,
            spec,
            exact=exact,
            vectors=vectors,
        )
        if fail_closed:
            # Scheduled packets carry the raw count that was scored by the
            # planner.  Re-project the current Direct nominal in parallel and
            # select it on-device when the scheduled packet is stale.  The
            # branch is tensorized so it never synchronizes the worker thread.
            _, fallback_raw_i, fallback_transmitted_i, fallback_velocity_i = step_executable_command_torch(
                fallback_q_ref[:, step_index],
                predicted_state[:, : action_dim],
                previous_q,
                previous_v,
                spec,
                exact=exact,
                vectors=vectors,
            )
            scheduled_i = expected_raw_mask[:, step_index]
            mismatch_i = scheduled_i & torch.any(raw_i != expected_raw[:, step_index], dim=-1)
            transmitted_i = torch.where(mismatch_i[:, None], fallback_transmitted_i, transmitted_i)
            raw_i = torch.where(mismatch_i[:, None], fallback_raw_i, raw_i)
            velocity_i = torch.where(mismatch_i[:, None], fallback_velocity_i, velocity_i)
            fallback_mask[:, step_index] = mismatch_i
        predicted_state, history = rollout_dynamics_step(
            model,
            normalizer,
            model_type,
            history,
            predicted_state,
            transmitted_i,
            state_dim,
            target_mode,
            control_dt,
        )
        q_ref_sequences[:, step_index] = transmitted_i
        raw_sequences[:, step_index] = raw_i
        pred_states[:, step_index + 1] = predicted_state
        previous_q, previous_v = transmitted_i, velocity_i
    return ExecutableRolloutOutput(
        q_ref_sequences=q_ref_sequences,
        expected_raw_sequences=raw_sequences,
        pred_states=pred_states,
        final_history=history,
        final_q_ref=previous_q,
        final_velocity=previous_v,
        fallback_mask=fallback_mask,
    )


class _CudaGraphRunner:
    """One static-shape CUDA Graph and its reusable input/output buffers."""

    def __init__(
        self,
        owner: "ExecutableRolloutEngine",
        batch_size: int,
        horizon: int,
        exact: bool,
        initial_history: torch.Tensor,
        requested_q_ref: torch.Tensor,
        previous_q_ref: torch.Tensor,
        previous_velocity: torch.Tensor,
        fallback_q_ref: torch.Tensor,
        expected_raw: torch.Tensor,
        expected_raw_mask: torch.Tensor,
        fail_closed: bool,
    ) -> None:
        self.owner = owner
        self.batch_size = int(batch_size)
        self.horizon = int(horizon)
        self.exact = bool(exact)
        self.fail_closed = bool(fail_closed)
        self.device = requested_q_ref.device
        self.static_initial_history = torch.zeros_like(initial_history.contiguous())
        self.static_requested_q_ref = torch.zeros_like(requested_q_ref.contiguous())
        self.static_previous_q_ref = torch.zeros_like(previous_q_ref.contiguous())
        self.static_previous_velocity = torch.zeros_like(previous_velocity.contiguous())
        self.static_fallback_q_ref = torch.zeros_like(requested_q_ref.contiguous())
        self.static_expected_raw = torch.zeros_like(
            requested_q_ref, dtype=torch.int64, device=requested_q_ref.device
        )
        self.static_expected_raw_mask = torch.zeros(
            (self.batch_size, self.horizon), dtype=torch.bool, device=requested_q_ref.device
        )
        self._capture_stream = torch.cuda.Stream(device=self.device)
        self._graph = torch.cuda.CUDAGraph()
        self._outputs: ExecutableRolloutOutput | None = None
        self._capture()

    def _run_static(self) -> ExecutableRolloutOutput:
        return _forward_rollout(
            self.owner.model,
            self.owner.normalizer,
            self.owner.model_type,
            self.static_initial_history,
            self.static_requested_q_ref,
            self.static_previous_q_ref,
            self.static_previous_velocity,
            self.static_fallback_q_ref,
            self.static_expected_raw,
            self.static_expected_raw_mask,
            self.fail_closed,
            self.owner.state_dim,
            self.owner.target_mode,
            self.owner.control_dt,
            self.owner.spec,
            self.owner._vectors(self.device, self.exact),
            self.exact,
        )

    def _capture(self) -> None:
        # cuDNN RNN kernels require a few eager warm-ups before capture.  All
        # allocations made here belong to the graph's private memory pool.
        with torch.inference_mode(), torch.cuda.stream(self._capture_stream):
            for _ in range(3):
                self._outputs = self._run_static()
        self._capture_stream.synchronize()
        with torch.inference_mode(), torch.cuda.graph(self._graph, stream=self._capture_stream):
            self._outputs = self._run_static()
        self._capture_stream.synchronize()
        if self._outputs is None:
            raise RuntimeError("CUDA Graph capture returned no rollout output")

    def replay(
        self,
        initial_history: torch.Tensor,
        requested_q_ref: torch.Tensor,
        previous_q_ref: torch.Tensor,
        previous_velocity: torch.Tensor,
        fallback_q_ref: torch.Tensor,
        expected_raw: torch.Tensor,
        expected_raw_mask: torch.Tensor,
    ) -> ExecutableRolloutOutput:
        if tuple(initial_history.shape) != tuple(self.static_initial_history.shape):
            raise ValueError("CUDA Graph initial_history shape changed")
        if tuple(requested_q_ref.shape) != tuple(self.static_requested_q_ref.shape):
            raise ValueError("CUDA Graph requested_q_ref shape changed")
        self.static_initial_history.copy_(initial_history)
        self.static_requested_q_ref.copy_(requested_q_ref)
        self.static_previous_q_ref.copy_(previous_q_ref)
        self.static_previous_velocity.copy_(previous_velocity)
        self.static_fallback_q_ref.copy_(fallback_q_ref)
        self.static_expected_raw.copy_(expected_raw)
        self.static_expected_raw_mask.copy_(expected_raw_mask)
        with torch.inference_mode():
            self._graph.replay()
        assert self._outputs is not None
        return self._outputs


class ExecutableRolloutEngine:
    """GPU-resident executable rollout with eager/CUDA-Graph backends.

    ``backend='auto'`` uses CUDA Graphs on CUDA and eager execution elsewhere.
    ``backend='cuda_graph'`` is strict and raises if capture is unavailable;
    this is useful for production latency runs.  The graph cache is keyed by
    ``(batch, horizon, exact)`` so candidate scoring and exact final-pool
    validation never trigger shape-specialisation churn.
    """

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        normalizer: Any,
        model_type: str,
        state_dim: int,
        target_mode: str,
        control_dt: float,
        spec: ExecutableCommandSpec,
        backend: str = "auto",
    ) -> None:
        if backend not in {"auto", "cuda_graph", "eager"}:
            raise ValueError("backend must be 'auto', 'cuda_graph', or 'eager'")
        self.model = model
        self.normalizer = normalizer
        self.model_type = model_type
        self.state_dim = int(state_dim)
        self.target_mode = target_mode
        self.control_dt = float(control_dt)
        self.spec = spec
        self.backend = backend
        self._graph_runners: dict[tuple[int, int, bool, bool], _CudaGraphRunner] = {}
        self._vectors_cache: dict[tuple[str, torch.dtype], dict[str, torch.Tensor]] = {}
        self.capture_failures = 0
        # ``StandardNormalizer`` stores plain tensors rather than registered
        # module buffers.  Materialise them on the model device once so CUDA
        # Graph capture never records a CPU->GPU ``.to()`` copy per GRU step.
        model_device = next(self.model.parameters()).device
        for name in ("state_mean", "state_std", "action_mean", "action_std", "delta_mean", "delta_std"):
            value = getattr(self.normalizer, name, None)
            if isinstance(value, torch.Tensor):
                setattr(self.normalizer, name, value.to(device=model_device, dtype=torch.float32))

    @property
    def use_cuda_graph(self) -> bool:
        return self.backend != "eager" and next(self.model.parameters()).is_cuda

    def _vectors(self, device: torch.device, exact: bool) -> dict[str, torch.Tensor]:
        dtype = torch.float64 if exact else torch.float32
        key = (str(device), dtype)
        vectors = self._vectors_cache.get(key)
        if vectors is None:
            vectors = self.spec.torch_vectors(device=device, dtype=dtype)
            self._vectors_cache[key] = vectors
        return vectors

    def _eager(
        self,
        initial_history: torch.Tensor,
        requested_q_ref: torch.Tensor,
        previous_q_ref: torch.Tensor,
        previous_velocity: torch.Tensor,
        fallback_q_ref: torch.Tensor,
        expected_raw: torch.Tensor,
        expected_raw_mask: torch.Tensor,
        fail_closed: bool,
        exact: bool,
    ) -> ExecutableRolloutOutput:
        with torch.inference_mode():
            return _forward_rollout(
                self.model,
                self.normalizer,
                self.model_type,
                initial_history,
                requested_q_ref,
                previous_q_ref,
                previous_velocity,
                fallback_q_ref,
                expected_raw,
                expected_raw_mask,
                fail_closed,
                self.state_dim,
                self.target_mode,
                self.control_dt,
                self.spec,
                self._vectors(requested_q_ref.device, exact),
                exact,
            )

    def run(
        self,
        *,
        initial_history: torch.Tensor,
        requested_q_ref: torch.Tensor,
        previous_q_ref: torch.Tensor,
        previous_velocity: torch.Tensor,
        fallback_q_ref: torch.Tensor | None = None,
        expected_raw: torch.Tensor | None = None,
        expected_raw_mask: torch.Tensor | None = None,
        fail_closed: bool = False,
        exact: bool = False,
    ) -> ExecutableRolloutOutput:
        if requested_q_ref.ndim != 3:
            raise ValueError("requested_q_ref must have shape [batch, horizon, joints]")
        if initial_history.ndim != 3:
            raise ValueError("initial_history must have shape [batch, history, token_dim]")
        batch_size, horizon, action_dim = requested_q_ref.shape
        if action_dim != self.spec.n_joints:
            raise ValueError("requested_q_ref joint dimension does not match command spec")
        if initial_history.shape[0] != batch_size:
            raise ValueError("history and command batch sizes must match")
        if previous_q_ref.shape != (batch_size, action_dim) or previous_velocity.shape != previous_q_ref.shape:
            raise ValueError("previous command state has an invalid shape")
        initial_history = initial_history.contiguous()
        requested_q_ref = requested_q_ref.contiguous()
        previous_q_ref = previous_q_ref.contiguous()
        previous_velocity = previous_velocity.contiguous()
        if fallback_q_ref is None:
            fallback_q_ref = requested_q_ref
        if expected_raw is None:
            expected_raw = torch.zeros(
                (batch_size, horizon, action_dim), dtype=torch.int64, device=requested_q_ref.device
            )
        if expected_raw_mask is None:
            expected_raw_mask = torch.zeros(
                (batch_size, horizon), dtype=torch.bool, device=requested_q_ref.device
            )
        fallback_q_ref = fallback_q_ref.contiguous()
        expected_raw = expected_raw.contiguous()
        expected_raw_mask = expected_raw_mask.contiguous()
        if fallback_q_ref.shape != requested_q_ref.shape:
            raise ValueError("fallback_q_ref must match requested_q_ref shape")
        if expected_raw.shape != (batch_size, horizon, action_dim):
            raise ValueError("expected_raw must have shape [batch, horizon, joints]")
        if expected_raw_mask.shape != (batch_size, horizon):
            raise ValueError("expected_raw_mask must have shape [batch, horizon]")
        if not self.use_cuda_graph:
            if self.backend == "cuda_graph" and requested_q_ref.device.type != "cuda":
                raise RuntimeError("cuda_graph backend requires CUDA tensors")
            return self._eager(
                initial_history, requested_q_ref, previous_q_ref, previous_velocity,
                fallback_q_ref, expected_raw, expected_raw_mask, fail_closed, exact,
            )
        key = (int(batch_size), int(horizon), bool(exact), bool(fail_closed))
        runner = self._graph_runners.get(key)
        if runner is None:
            try:
                runner = _CudaGraphRunner(
                    self,
                    batch_size,
                    horizon,
                    exact,
                    initial_history,
                    requested_q_ref,
                    previous_q_ref,
                    previous_velocity,
                    fallback_q_ref,
                    expected_raw,
                    expected_raw_mask,
                    fail_closed,
                )
                self._graph_runners[key] = runner
            except Exception as exc:
                self.capture_failures += 1
                if self.backend == "cuda_graph":
                    raise
                warnings.warn(
                    f"executable rollout CUDA Graph capture failed; falling back to eager: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return self._eager(
                    initial_history, requested_q_ref, previous_q_ref, previous_velocity,
                    fallback_q_ref, expected_raw, expected_raw_mask, fail_closed, exact,
                )
        return runner.replay(
            initial_history, requested_q_ref, previous_q_ref, previous_velocity,
            fallback_q_ref, expected_raw, expected_raw_mask,
        )

    def warmup(self, shapes: list[tuple[int, int, bool]], template: torch.Tensor) -> None:
        """Capture known production shapes before the worker announces ready."""
        if not self.use_cuda_graph:
            return
        batch_history = template.unsqueeze(0) if template.ndim == 2 else template
        if batch_history.ndim != 3:
            raise ValueError("warmup template must have shape [history, token] or [batch, history, token]")
        for batch_size, horizon, exact in shapes:
            history = batch_history[:1].expand(batch_size, -1, -1).clone()
            requested = torch.zeros(
                (batch_size, horizon, self.spec.n_joints), device=template.device, dtype=template.dtype
            )
            previous = torch.zeros((batch_size, self.spec.n_joints), device=template.device, dtype=template.dtype)
            velocity = torch.zeros_like(previous)
            self.run(
                initial_history=history,
                requested_q_ref=requested,
                previous_q_ref=previous,
                previous_velocity=velocity,
                fallback_q_ref=requested,
                expected_raw=torch.zeros(
                    (batch_size, horizon, self.spec.n_joints), dtype=torch.int64, device=template.device
                ),
                expected_raw_mask=torch.zeros(
                    (batch_size, horizon), dtype=torch.bool, device=template.device
                ),
                fail_closed=False,
                exact=exact,
            )
