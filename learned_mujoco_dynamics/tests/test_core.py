import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import mujoco

from learned_dynamics.dataset import DynamicsDataset, RolloutDynamicsDataset, load_npz_dataset, split_dataset
from learned_dynamics.dataset_merge import merge_npz_datasets
from learned_dynamics.models import GRUDynamics, MLPDynamics, TransformerDynamics
from learned_dynamics.mujoco_env import MuJoCoArmEnv
from learned_dynamics.normalization import StandardNormalizer
from learned_dynamics.parallel_collector import sample_smooth_action, save_dataset, validate_append_dataset
from learned_dynamics.paths import DEFAULT_MODEL_XML, resolve_project_path
from learned_dynamics.train_utils import load_checkpoint, require_resume_checkpoint, save_checkpoint
from learned_dynamics2.mujoco_env import MuJoCoArmEnv as MuJoCoArmEnvV2
from learned_dynamics2.dataset import (
    DynamicsDataset as DynamicsDatasetV2,
    RolloutDynamicsDataset as RolloutDynamicsDatasetV2,
    split_dataset as split_dataset_v2,
)


ROOT = Path(__file__).resolve().parents[1]
EVAL_DYNAMICS_SPEC = importlib.util.spec_from_file_location("eval_dynamics", ROOT / "scripts" / "eval_dynamics.py")
if EVAL_DYNAMICS_SPEC is None or EVAL_DYNAMICS_SPEC.loader is None:
    raise RuntimeError("Could not load scripts/eval_dynamics.py")
EVAL_DYNAMICS = importlib.util.module_from_spec(EVAL_DYNAMICS_SPEC)
EVAL_DYNAMICS_SPEC.loader.exec_module(EVAL_DYNAMICS)
resolve_history_len = EVAL_DYNAMICS.resolve_history_len
parse_horizons = EVAL_DYNAMICS.parse_horizons
state_labels = EVAL_DYNAMICS.state_labels
per_dimension_rmse = EVAL_DYNAMICS.per_dimension_rmse
build_sequence_history = EVAL_DYNAMICS.build_sequence_history
predict_open_loop_segment = EVAL_DYNAMICS.predict_open_loop_segment
summarize_prediction = EVAL_DYNAMICS.summarize_prediction

EVAL_DYNAMICS2_SPEC = importlib.util.spec_from_file_location("eval_dynamics2", ROOT / "scripts2" / "eval_dynamics.py")
if EVAL_DYNAMICS2_SPEC is None or EVAL_DYNAMICS2_SPEC.loader is None:
    raise RuntimeError("Could not load scripts2/eval_dynamics.py")
EVAL_DYNAMICS2 = importlib.util.module_from_spec(EVAL_DYNAMICS2_SPEC)
EVAL_DYNAMICS2_SPEC.loader.exec_module(EVAL_DYNAMICS2)

DIAG_ROLLOUT_SPIKES_SPEC = importlib.util.spec_from_file_location(
    "diagnose_rollout_spikes", ROOT / "scripts" / "diagnose_rollout_spikes.py"
)
if DIAG_ROLLOUT_SPIKES_SPEC is None or DIAG_ROLLOUT_SPIKES_SPEC.loader is None:
    raise RuntimeError("Could not load scripts/diagnose_rollout_spikes.py")
DIAG_ROLLOUT_SPIKES = importlib.util.module_from_spec(DIAG_ROLLOUT_SPIKES_SPEC)
DIAG_ROLLOUT_SPIKES_SPEC.loader.exec_module(DIAG_ROLLOUT_SPIKES)

DIAG_DYNAMICS_SPEC = importlib.util.spec_from_file_location(
    "diagnose_dynamics_data", ROOT / "scripts" / "diagnose_dynamics_data.py"
)
if DIAG_DYNAMICS_SPEC is None or DIAG_DYNAMICS_SPEC.loader is None:
    raise RuntimeError("Could not load scripts/diagnose_dynamics_data.py")
DIAG_DYNAMICS = importlib.util.module_from_spec(DIAG_DYNAMICS_SPEC)
DIAG_DYNAMICS_SPEC.loader.exec_module(DIAG_DYNAMICS)

ROLLOUT_VISUALIZE_SPEC = importlib.util.spec_from_file_location(
    "rollout_visualize", ROOT / "scripts" / "rollout_visualize.py"
)
if ROLLOUT_VISUALIZE_SPEC is None or ROLLOUT_VISUALIZE_SPEC.loader is None:
    raise RuntimeError("Could not load scripts/rollout_visualize.py")
ROLLOUT_VISUALIZE = importlib.util.module_from_spec(ROLLOUT_VISUALIZE_SPEC)
ROLLOUT_VISUALIZE_SPEC.loader.exec_module(ROLLOUT_VISUALIZE)

TUNE_POSITION_KP_SPEC = importlib.util.spec_from_file_location("tune_position_kp", ROOT / "tests" / "tune_position_kp.py")
if TUNE_POSITION_KP_SPEC is None or TUNE_POSITION_KP_SPEC.loader is None:
    raise RuntimeError("Could not load tests/tune_position_kp.py")
TUNE_POSITION_KP = importlib.util.module_from_spec(TUNE_POSITION_KP_SPEC)
TUNE_POSITION_KP_SPEC.loader.exec_module(TUNE_POSITION_KP)

TRAIN_DYNAMICS_SPEC = importlib.util.spec_from_file_location("train_dynamics", ROOT / "scripts" / "train_dynamics.py")
if TRAIN_DYNAMICS_SPEC is None or TRAIN_DYNAMICS_SPEC.loader is None:
    raise RuntimeError("Could not load scripts/train_dynamics.py")
TRAIN_DYNAMICS = importlib.util.module_from_spec(TRAIN_DYNAMICS_SPEC)
TRAIN_DYNAMICS_SPEC.loader.exec_module(TRAIN_DYNAMICS)
weighted_delta_loss = TRAIN_DYNAMICS.weighted_delta_loss
rollout_state_loss = TRAIN_DYNAMICS.rollout_state_loss
format_delta_rmse = TRAIN_DYNAMICS.format_delta_rmse
parse_extra_weights = TRAIN_DYNAMICS.parse_extra_weights
reconstruct_next_state = TRAIN_DYNAMICS.reconstruct_next_state

TRAIN_DYNAMICS2_SPEC = importlib.util.spec_from_file_location("train_dynamics2", ROOT / "scripts2" / "train_dynamics.py")
if TRAIN_DYNAMICS2_SPEC is None or TRAIN_DYNAMICS2_SPEC.loader is None:
    raise RuntimeError("Could not load scripts2/train_dynamics.py")
TRAIN_DYNAMICS2 = importlib.util.module_from_spec(TRAIN_DYNAMICS2_SPEC)
TRAIN_DYNAMICS2_SPEC.loader.exec_module(TRAIN_DYNAMICS2)

CONVERT_DAE_SPEC = importlib.util.spec_from_file_location(
    "convert_collada_to_stl", ROOT / "scripts" / "convert_collada_to_stl.py"
)
if CONVERT_DAE_SPEC is None or CONVERT_DAE_SPEC.loader is None:
    raise RuntimeError("Could not load scripts/convert_collada_to_stl.py")
CONVERT_DAE = importlib.util.module_from_spec(CONVERT_DAE_SPEC)
CONVERT_DAE_SPEC.loader.exec_module(CONVERT_DAE)
extract_collada_triangles = CONVERT_DAE.extract_collada_triangles


class CoreBehaviorTests(unittest.TestCase):
    def test_default_model_path_is_project_relative(self) -> None:
        project_root = Path("/tmp/example_project")

        resolved = resolve_project_path(DEFAULT_MODEL_XML, project_root)

        self.assertEqual(resolved, project_root / "ABB_IRB2400.xml")

    def test_env_reports_missing_xml_path(self) -> None:
        missing_path = Path(tempfile.gettempdir()) / "does_not_exist_robot.xml"

        with self.assertRaisesRegex(FileNotFoundError, "MuJoCo XML file does not exist"):
            MuJoCoArmEnv(model_xml=str(missing_path), n_joints=6)

    def test_irb2400_xml_uses_explicit_reasonable_body_inertials(self) -> None:
        env = MuJoCoArmEnv(model_xml=str(ROOT / "ABB_IRB2400.xml"), n_joints=6)
        try:
            model = env.model
            names = {
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, idx): idx
                for idx in range(model.nbody)
            }
            total_mass = float(np.sum(model.body_mass))

            self.assertGreater(total_mass, 360.0)
            self.assertLess(total_mass, 400.0)
            self.assertGreater(float(model.body_mass[names["link_5"]]), 8.0)
            self.assertGreater(float(model.body_mass[names["link_6"]]), 5.0)
            self.assertGreater(float(np.min(model.body_inertia[names["link_5"]])), 0.05)
            self.assertGreater(float(np.min(model.body_inertia[names["link_6"]])), 0.01)
        finally:
            env.close()

    def test_irb2400_xml_uses_visual_mesh_assets(self) -> None:
        xml = (ROOT / "ABB_IRB2400.xml").read_text(encoding="utf-8")

        self.assertIn('file="link_5_visual.stl"', xml)
        self.assertIn('file="link_6_visual.stl"', xml)
        self.assertTrue((ROOT / "abb_irb2400_assets" / "link_5_visual.stl").exists())
        self.assertTrue((ROOT / "abb_irb2400_assets" / "link_6_visual.stl").exists())

    def test_irb2400_xml_uses_position_actuators_matching_joint_ranges(self) -> None:
        xml = (ROOT / "ABB_IRB2400.xml").read_text(encoding="utf-8")
        joint_names, joint_ranges = TUNE_POSITION_KP.joint_specs_from_xml(xml)
        model = mujoco.MjModel.from_xml_path(str(ROOT / "ABB_IRB2400.xml"))

        self.assertEqual(xml.count("<position "), 6)
        self.assertNotIn("<velocity", xml)
        self.assertEqual(model.nu, 6)
        for actuator_id, (joint_name, joint_range) in enumerate(zip(joint_names, joint_ranges)):
            actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
            self.assertEqual(actuator_name, f"{joint_name}_position")
            self.assertTrue(np.allclose(model.actuator_ctrlrange[actuator_id], np.asarray(joint_range)))

    def test_default_envs_apply_gravity_compensation_for_position_control(self) -> None:
        for env_class in (MuJoCoArmEnv, MuJoCoArmEnvV2):
            env = env_class(model_xml=str(ROOT / "ABB_IRB2400.xml"), n_joints=6, frame_skip=1)
            try:
                self.assertEqual(env.control_mode, "position")
                self.assertTrue(env.gravity_compensation)

                mujoco.mj_resetData(env.model, env.data)
                env.data.qpos[:6] = np.asarray([0.0, 0.45, -0.2, 0.1, -0.15, 0.2])
                env.data.qvel[:6] = 0.0
                mujoco.mj_forward(env.model, env.data)
                q_ref = np.asarray(env.data.qpos[:6], dtype=np.float64).copy()
                expected = env._gravity_compensation_force()

                env.step(q_ref)

                self.assertTrue(np.allclose(env.data.qfrc_applied[:6], expected, rtol=1e-6, atol=1e-6))
                self.assertEqual(expected[5], 0.0)
                self.assertEqual(env.data.qfrc_applied[5], 0.0)
            finally:
                env.close()

    def test_envs_reject_velocity_control_mode(self) -> None:
        for env_class in (MuJoCoArmEnv, MuJoCoArmEnvV2):
            with self.assertRaisesRegex(ValueError, "position"):
                env_class(model_xml=str(ROOT / "ABB_IRB2400.xml"), n_joints=6, control_mode="velocity")

    def test_envs_report_bottom_controller_torque_components(self) -> None:
        for env_class in (MuJoCoArmEnv, MuJoCoArmEnvV2):
            env = env_class(model_xml=str(ROOT / "ABB_IRB2400.xml"), n_joints=6, frame_skip=1)
            try:
                env.reset_random()
                q_ref = np.asarray(env.data.qpos[:6], dtype=np.float64) + np.asarray(
                    [0.05, -0.04, 0.03, -0.02, 0.01, -0.005],
                    dtype=np.float64,
                )

                components = env.compute_torque_components(q_ref)

                self.assertEqual(set(components), {"actuator_tau", "gravity_tau", "total_tau"})
                for value in components.values():
                    self.assertEqual(value.shape, (6,))
                    self.assertEqual(value.dtype, np.dtype("float64"))
                self.assertTrue(
                    np.allclose(components["total_tau"], components["actuator_tau"] + components["gravity_tau"])
                )
                self.assertEqual(components["gravity_tau"][5], 0.0)
                self.assertEqual(components["total_tau"][5], components["actuator_tau"][5])
            finally:
                env.close()

    def test_collada_converter_applies_node_matrix_to_polylist_vertices(self) -> None:
        dae = """<?xml version="1.0"?>
<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema">
  <library_geometries>
    <geometry id="g"><mesh>
      <source id="g-positions">
        <float_array id="g-positions-array" count="9">1 2 3 2 2 3 1 3 3</float_array>
        <technique_common><accessor source="#g-positions-array" count="3" stride="3"/></technique_common>
      </source>
      <vertices id="g-vertices"><input semantic="POSITION" source="#g-positions"/></vertices>
      <polylist count="1">
        <input semantic="VERTEX" source="#g-vertices" offset="0"/>
        <vcount>3</vcount>
        <p>0 1 2</p>
      </polylist>
    </mesh></geometry>
  </library_geometries>
  <library_visual_scenes>
    <visual_scene id="Scene">
      <node><matrix>1 0 0 -1 0 1 0 -2 0 0 1 -3 0 0 0 1</matrix><instance_geometry url="#g"/></node>
    </visual_scene>
  </library_visual_scenes>
  <scene><instance_visual_scene url="#Scene"/></scene>
</COLLADA>
"""
        with tempfile.NamedTemporaryFile("w", suffix=".dae") as file:
            file.write(dae)
            file.flush()
            triangles = extract_collada_triangles(Path(file.name))

        self.assertEqual(len(triangles), 1)
        self.assertTrue(np.allclose(triangles[0], np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)))

    def test_mlp_dataset_returns_single_step_delta_target(self) -> None:
        states = np.array([[0.0, 1.0], [1.0, 3.0]], dtype=np.float32)
        actions = np.array([[0.5], [0.25]], dtype=np.float32)
        next_states = np.array([[1.0, 2.0], [2.0, 4.0]], dtype=np.float32)

        dataset = DynamicsDataset(states, actions, next_states, model_type="mlp")
        sample_input, sample_target = dataset[0]

        self.assertEqual(tuple(sample_input.shape), (3,))
        self.assertTrue(torch.allclose(sample_input, torch.tensor([0.0, 1.0, 0.5])))
        self.assertTrue(torch.allclose(sample_target, torch.tensor([1.0, 1.0])))

    def test_mlp_dataset_can_return_delta_dq_target(self) -> None:
        states = np.array([[1.0, 2.0, 0.5, -0.5]], dtype=np.float32)
        actions = np.array([[0.25, -0.25]], dtype=np.float32)
        next_states = np.array([[1.1, 1.9, 0.75, -0.25]], dtype=np.float32)

        dataset = DynamicsDataset(states, actions, next_states, model_type="mlp", target_mode="delta_dq")
        sample_input, sample_target = dataset[0]

        self.assertEqual(tuple(sample_input.shape), (6,))
        self.assertEqual(tuple(sample_target.shape), (2,))
        self.assertTrue(torch.allclose(sample_target, torch.tensor([0.25, 0.25])))

    def test_sequence_dataset_uses_history_window_and_delta_target(self) -> None:
        states = np.arange(12, dtype=np.float32).reshape(6, 2)
        actions = np.arange(6, dtype=np.float32).reshape(6, 1)
        next_states = states + 1.0

        dataset = DynamicsDataset(states, actions, next_states, model_type="gru", history_len=3)
        sample_input, sample_target = dataset[0]

        self.assertEqual(len(dataset), 4)
        self.assertEqual(tuple(sample_input.shape), (3, 3))
        self.assertTrue(torch.allclose(sample_target, torch.ones(2)))

    def test_sequence_dataset_with_episode_ids_does_not_cross_episode_boundary(self) -> None:
        states = np.arange(16, dtype=np.float32).reshape(8, 2)
        actions = np.arange(8, dtype=np.float32).reshape(8, 1)
        next_states = states + 1.0
        episode_ids = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)

        dataset = DynamicsDataset(
            states,
            actions,
            next_states,
            model_type="transformer",
            history_len=3,
            episode_ids=episode_ids,
        )
        first_input, _ = dataset[0]
        second_input, _ = dataset[1]
        third_input, _ = dataset[2]
        fourth_input, _ = dataset[3]

        self.assertIsInstance(dataset.sequence_indices, np.ndarray)
        self.assertEqual(len(dataset), 4)
        self.assertTrue(torch.allclose(first_input[:, 0], torch.tensor([0.0, 2.0, 4.0])))
        self.assertTrue(torch.allclose(second_input[:, 0], torch.tensor([2.0, 4.0, 6.0])))
        self.assertTrue(torch.allclose(third_input[:, 0], torch.tensor([8.0, 10.0, 12.0])))
        self.assertTrue(torch.allclose(fourth_input[:, 0], torch.tensor([10.0, 12.0, 14.0])))

    def test_sequence_dataset_builds_vectorized_indices_for_many_episode_windows(self) -> None:
        episode_len = 20
        num_episodes = 1000
        samples = episode_len * num_episodes
        states = np.arange(samples * 2, dtype=np.float32).reshape(samples, 2)
        actions = np.arange(samples, dtype=np.float32).reshape(samples, 1)
        next_states = states + 1.0
        episode_ids = np.repeat(np.arange(num_episodes, dtype=np.int64), episode_len)

        dataset = DynamicsDataset(
            states,
            actions,
            next_states,
            model_type="transformer",
            history_len=16,
            episode_ids=episode_ids,
        )

        self.assertIsInstance(dataset.sequence_indices, np.ndarray)
        self.assertEqual(dataset.sequence_indices.dtype, np.int64)
        self.assertEqual(len(dataset), num_episodes * (episode_len - 16 + 1))

    def test_rollout_dataset_stops_before_episode_boundary(self) -> None:
        episode_len = 1000
        history_len = 16
        rollout_steps = 100
        states = np.arange(episode_len * 2, dtype=np.float32).reshape(episode_len, 2)
        actions = np.arange(episode_len, dtype=np.float32).reshape(episode_len, 1)
        next_states = states + 1.0
        episode_ids = np.zeros(episode_len, dtype=np.int64)

        dataset = RolloutDynamicsDataset(
            states,
            actions,
            next_states,
            model_type="transformer",
            history_len=history_len,
            episode_ids=episode_ids,
            target_mode="delta_state",
            rollout_steps=rollout_steps,
        )

        self.assertEqual(len(dataset), 886)
        self.assertEqual(int(dataset.sequence_indices[-1]), 885)
        x, y, rollout_actions, rollout_next_states = dataset[-1]
        self.assertTrue(torch.allclose(x[-1, :2], torch.as_tensor(states[900])))
        self.assertTrue(torch.allclose(y, torch.ones(2)))
        self.assertEqual(tuple(rollout_actions.shape), (rollout_steps, 1))
        self.assertTrue(torch.allclose(rollout_actions[-1], torch.as_tensor(actions[999])))
        self.assertTrue(torch.allclose(rollout_next_states[-1], torch.as_tensor(next_states[999])))

    def test_rollout_dataset_rejects_episodes_shorter_than_window(self) -> None:
        states = np.zeros((10, 2), dtype=np.float32)
        actions = np.zeros((10, 1), dtype=np.float32)
        next_states = states.copy()
        episode_ids = np.zeros(10, dtype=np.int64)

        with self.assertRaisesRegex(ValueError, "No valid rollout windows"):
            RolloutDynamicsDataset(
                states,
                actions,
                next_states,
                model_type="transformer",
                history_len=8,
                episode_ids=episode_ids,
                rollout_steps=5,
            )

    def test_load_npz_dataset_reads_optional_episode_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sequence_data.npz"
            states = np.arange(12, dtype=np.float32).reshape(6, 2)
            actions = np.arange(6, dtype=np.float32).reshape(6, 1)
            next_states = states + 1.0
            episode_ids = np.repeat(np.arange(2, dtype=np.int64), 3)
            np.savez(path, states=states, actions=actions, next_states=next_states, episode_ids=episode_ids)

            dataset = load_npz_dataset(path, model_type="transformer", history_len=2)

        self.assertIsNotNone(dataset.episode_ids)
        self.assertEqual(len(dataset), 4)

    def test_split_dataset_keeps_episode_ids_disjoint_when_available(self) -> None:
        states = np.arange(60, dtype=np.float32).reshape(30, 2)
        actions = np.arange(30, dtype=np.float32).reshape(30, 1)
        next_states = states + 1.0
        episode_ids = np.repeat(np.arange(10, dtype=np.int64), 3)
        dataset = DynamicsDataset(
            states,
            actions,
            next_states,
            model_type="transformer",
            history_len=2,
            episode_ids=episode_ids,
        )

        train_set, val_set = split_dataset(dataset, val_fraction=0.2, seed=0)
        train_episodes = {int(dataset.episode_ids[dataset.sequence_indices[idx]]) for idx in train_set.indices}
        val_episodes = {int(dataset.episode_ids[dataset.sequence_indices[idx]]) for idx in val_set.indices}

        self.assertTrue(train_episodes)
        self.assertTrue(val_episodes)
        self.assertTrue(train_episodes.isdisjoint(val_episodes))

    def test_split_dataset_can_stride_training_windows_but_keep_full_validation(self) -> None:
        states = np.arange(96, dtype=np.float32).reshape(48, 2)
        actions = np.arange(48, dtype=np.float32).reshape(48, 1)
        next_states = states + 1.0
        episode_ids = np.repeat(np.arange(4, dtype=np.int64), 12)
        dataset = DynamicsDataset(
            states,
            actions,
            next_states,
            model_type="transformer",
            history_len=4,
            episode_ids=episode_ids,
        )

        train_set, val_set = split_dataset(
            dataset,
            val_fraction=0.25,
            seed=0,
            train_sample_stride=3,
            val_sample_stride=1,
        )
        train_starts = dataset.sequence_indices[train_set.indices]
        val_starts = dataset.sequence_indices[val_set.indices]

        self.assertTrue(np.all(train_starts % 3 == 0))
        self.assertEqual(len(train_set), 9)
        self.assertEqual(len(val_set), 9)
        self.assertEqual(np.diff(val_starts).tolist(), [1] * (len(val_starts) - 1))

    def test_split_dataset_can_show_episode_progress(self) -> None:
        states = np.arange(96, dtype=np.float32).reshape(48, 2)
        actions = np.arange(48, dtype=np.float32).reshape(48, 1)
        next_states = states + 1.0
        episode_ids = np.repeat(np.arange(4, dtype=np.int64), 12)
        dataset = DynamicsDataset(
            states,
            actions,
            next_states,
            model_type="transformer",
            history_len=4,
            episode_ids=episode_ids,
        )

        with patch("learned_dynamics.dataset.tqdm", side_effect=lambda iterable, **_: iterable) as progress:
            split_dataset(
                dataset,
                val_fraction=0.25,
                seed=0,
                train_sample_stride=3,
                val_sample_stride=1,
                show_progress=True,
            )

        self.assertTrue(progress.called)
        self.assertEqual(progress.call_args.kwargs["desc"], "split episodes")

    def test_v2_split_dataset_can_stride_training_windows_with_direct_loss_steps(self) -> None:
        states = np.arange(120, dtype=np.float32).reshape(60, 2)
        actions = np.arange(60, dtype=np.float32).reshape(60, 1)
        next_states = states + 1.0
        episode_ids = np.repeat(np.arange(5, dtype=np.int64), 12)
        dataset = DynamicsDatasetV2(
            states,
            actions,
            next_states,
            model_type="transformer",
            history_len=4,
            episode_ids=episode_ids,
            direct_loss_steps=2,
        )

        train_set, val_set = split_dataset_v2(
            dataset,
            val_fraction=0.2,
            seed=0,
            train_sample_stride=3,
            val_sample_stride=1,
        )
        train_starts = dataset.sequence_indices[train_set.indices]
        val_starts = dataset.sequence_indices[val_set.indices]

        self.assertTrue(np.all(train_starts % 3 == 0))
        self.assertEqual(len(train_set), 12)
        self.assertEqual(len(val_set), 8)
        self.assertEqual(np.diff(val_starts).tolist(), [1] * (len(val_starts) - 1))

    def test_v2_split_dataset_can_show_episode_progress(self) -> None:
        states = np.arange(120, dtype=np.float32).reshape(60, 2)
        actions = np.arange(60, dtype=np.float32).reshape(60, 1)
        next_states = states + 1.0
        episode_ids = np.repeat(np.arange(5, dtype=np.int64), 12)
        dataset = DynamicsDatasetV2(
            states,
            actions,
            next_states,
            model_type="transformer",
            history_len=4,
            episode_ids=episode_ids,
            direct_loss_steps=2,
        )

        with patch("learned_dynamics2.dataset.tqdm", side_effect=lambda iterable, **_: iterable) as progress:
            split_dataset_v2(
                dataset,
                val_fraction=0.2,
                seed=0,
                train_sample_stride=3,
                val_sample_stride=1,
                show_progress=True,
            )

        self.assertTrue(progress.called)
        self.assertEqual(progress.call_args.kwargs["desc"], "split episodes")

    def test_rollout_dataset_stride_keeps_rollout_steps_contiguous(self) -> None:
        states = np.arange(320, dtype=np.float32).reshape(160, 2)
        actions = np.arange(160, dtype=np.float32).reshape(160, 1)
        next_states = states + 1.0
        episode_ids = np.repeat(np.arange(4, dtype=np.int64), 40)
        dataset = RolloutDynamicsDataset(
            states,
            actions,
            next_states,
            model_type="transformer",
            history_len=4,
            episode_ids=episode_ids,
            rollout_steps=5,
        )

        train_set, _ = split_dataset(
            dataset,
            val_fraction=0.25,
            seed=0,
            train_sample_stride=4,
            val_sample_stride=1,
        )
        sample_index = int(train_set.indices[1])
        start = int(dataset.sequence_indices[sample_index])
        current_index = start + dataset.history_len - 1
        x, _, rollout_actions, rollout_next_states = dataset[sample_index]

        self.assertEqual(start % 4, 0)
        self.assertTrue(torch.allclose(x[-1, :2], torch.as_tensor(states[current_index])))
        self.assertTrue(torch.allclose(rollout_actions[:, 0], torch.as_tensor(actions[current_index : current_index + 5, 0])))
        self.assertTrue(
            torch.allclose(rollout_next_states[:, 0], torch.as_tensor(next_states[current_index : current_index + 5, 0]))
        )

    def test_v2_rollout_dataset_stride_keeps_rollout_steps_contiguous(self) -> None:
        states = np.arange(320, dtype=np.float32).reshape(160, 2)
        actions = np.arange(160, dtype=np.float32).reshape(160, 1)
        next_states = states + 1.0
        episode_ids = np.repeat(np.arange(4, dtype=np.int64), 40)
        dataset = RolloutDynamicsDatasetV2(
            states,
            actions,
            next_states,
            model_type="transformer",
            history_len=4,
            episode_ids=episode_ids,
            rollout_steps=5,
            direct_loss_steps=2,
        )

        train_set, _ = split_dataset_v2(
            dataset,
            val_fraction=0.25,
            seed=0,
            train_sample_stride=4,
            val_sample_stride=1,
        )
        sample_index = int(train_set.indices[1])
        start = int(dataset.sequence_indices[sample_index])
        current_index = start + dataset.history_len - 1
        x, y, rollout_actions, rollout_next_states = dataset[sample_index]

        self.assertEqual(start % 4, 0)
        self.assertEqual(tuple(y.shape), (2, 2))
        self.assertTrue(torch.allclose(x[-1, :2], torch.as_tensor(states[current_index])))
        self.assertTrue(torch.allclose(rollout_actions[:, 0], torch.as_tensor(actions[current_index : current_index + 5, 0])))
        self.assertTrue(
            torch.allclose(rollout_next_states[:, 0], torch.as_tensor(next_states[current_index : current_index + 5, 0]))
        )

    def test_normalizer_round_trips_tensors(self) -> None:
        states = torch.tensor([[0.0, 2.0], [2.0, 4.0]])
        actions = torch.tensor([[1.0], [3.0]])
        deltas = torch.tensor([[0.5, -0.5], [1.5, -1.5]])

        normalizer = StandardNormalizer()
        normalizer.fit(states, actions, deltas)

        restored = normalizer.denormalize_delta(normalizer.normalize_delta(deltas))
        self.assertTrue(torch.allclose(restored, deltas, atol=1e-6))

    def test_weighted_delta_loss_applies_q_and_dq_weights(self) -> None:
        pred = torch.tensor([[1.0, 3.0, 6.0, 10.0]], dtype=torch.float32)
        target = torch.tensor([[0.0, 1.0, 3.0, 6.0]], dtype=torch.float32)

        loss = weighted_delta_loss(
            pred,
            target,
            n_joints=2,
            q_weight=2.0,
            dq_weight=0.5,
            q_extra_weights=None,
            dq_extra_weights=None,
        )

        q_loss = (1.0**2 + 2.0**2) * 2.0
        dq_loss = (3.0**2 + 4.0**2) * 0.5
        self.assertAlmostEqual(float(loss), q_loss + dq_loss)

    def test_weighted_delta_loss_applies_per_dimension_dq_weights(self) -> None:
        pred = torch.tensor([[1.0, 3.0, 6.0, 10.0]], dtype=torch.float32)
        target = torch.tensor([[0.0, 1.0, 3.0, 6.0]], dtype=torch.float32)

        loss = weighted_delta_loss(
            pred,
            target,
            n_joints=2,
            q_weight=1.0,
            dq_weight=1.0,
            q_extra_weights=None,
            dq_extra_weights=torch.tensor([2.0, 5.0], dtype=torch.float32),
        )

        q_loss = 1.0**2 + 2.0**2
        dq_loss = 3.0**2 * 2.0 + 4.0**2 * 5.0
        self.assertAlmostEqual(float(loss), q_loss + dq_loss)

    def test_rollout_state_loss_is_zero_for_perfect_predictions(self) -> None:
        normalizer = StandardNormalizer()
        states = torch.tensor([[0.0, 0.0], [1.0, 2.0]], dtype=torch.float32)
        actions = torch.zeros((2, 1), dtype=torch.float32)
        deltas = torch.zeros((2, 2), dtype=torch.float32)
        normalizer.fit(states, actions, deltas)
        pred = torch.tensor([[[1.0, 2.0], [2.0, 4.0]]], dtype=torch.float32)
        truth = pred.clone()

        loss = rollout_state_loss(pred, truth, normalizer, discount=0.9)

        self.assertAlmostEqual(float(loss), 0.0)

    def test_rollout_state_loss_discounts_later_steps(self) -> None:
        normalizer = StandardNormalizer()
        states = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
        actions = torch.zeros((2, 1), dtype=torch.float32)
        deltas = torch.zeros((2, 2), dtype=torch.float32)
        normalizer.fit(states, actions, deltas)
        pred = torch.tensor([[[1.5, 1.0], [1.0, 2.0]]], dtype=torch.float32)
        truth = torch.tensor([[[1.0, 1.0], [1.0, 1.0]]], dtype=torch.float32)

        loss = rollout_state_loss(pred, truth, normalizer, discount=0.5)

        self.assertAlmostEqual(float(loss), 3.0)

    def test_parse_extra_weights_accepts_none_or_one_value_per_joint(self) -> None:
        self.assertIsNone(parse_extra_weights(None, n_joints=3, name="dq_extra_weights"))
        parsed = parse_extra_weights("1, 2.5,3", n_joints=3, name="dq_extra_weights")

        self.assertIsNotNone(parsed)
        self.assertTrue(torch.allclose(parsed, torch.tensor([1.0, 2.5, 3.0])))

        with self.assertRaisesRegex(ValueError, "Expected 3"):
            parse_extra_weights("1,2", n_joints=3, name="dq_extra_weights")

    def test_format_delta_rmse_reports_q_and_dq_labels(self) -> None:
        rmse = torch.tensor([0.1, 0.2, 1.0, 2.0], dtype=torch.float32)

        formatted = format_delta_rmse("val", rmse, n_joints=2)

        self.assertIn("val_q0_rmse=0.100000", formatted)
        self.assertIn("val_q1_rmse=0.200000", formatted)
        self.assertIn("val_dq0_rmse=1.000000", formatted)
        self.assertIn("val_dq1_rmse=2.000000", formatted)

    def test_models_return_delta_with_state_dimension(self) -> None:
        state_dim = 12
        action_dim = 6
        batch_size = 4
        history_len = 5

        mlp = MLPDynamics(state_dim=state_dim, action_dim=action_dim)
        gru = GRUDynamics(state_dim=state_dim, action_dim=action_dim)
        transformer = TransformerDynamics(state_dim=state_dim, action_dim=action_dim, max_history_len=history_len)

        self.assertEqual(tuple(mlp(torch.zeros(batch_size, state_dim + action_dim)).shape), (batch_size, state_dim))
        seq = torch.zeros(batch_size, history_len, state_dim + action_dim)
        self.assertEqual(tuple(gru(seq).shape), (batch_size, state_dim))
        self.assertEqual(tuple(transformer(seq).shape), (batch_size, state_dim))

    def test_models_can_return_delta_dq_output_dimension(self) -> None:
        state_dim = 12
        action_dim = 6
        output_dim = 6
        batch_size = 4
        history_len = 5

        mlp = MLPDynamics(state_dim=state_dim, action_dim=action_dim, output_dim=output_dim)
        gru = GRUDynamics(state_dim=state_dim, action_dim=action_dim, output_dim=output_dim)
        transformer = TransformerDynamics(
            state_dim=state_dim,
            action_dim=action_dim,
            output_dim=output_dim,
            max_history_len=history_len,
        )

        self.assertEqual(tuple(mlp(torch.zeros(batch_size, state_dim + action_dim)).shape), (batch_size, output_dim))
        seq = torch.zeros(batch_size, history_len, state_dim + action_dim)
        self.assertEqual(tuple(gru(seq).shape), (batch_size, output_dim))
        self.assertEqual(tuple(transformer(seq).shape), (batch_size, output_dim))

    def test_reconstruct_next_state_integrates_delta_dq_semi_implicitly(self) -> None:
        state = torch.tensor([[1.0, 2.0, 0.5, -0.5]])
        delta_dq = torch.tensor([[0.25, 0.75]])

        next_state = reconstruct_next_state(state, delta_dq, target_mode="delta_dq", control_dt=0.1, n_joints=2)

        self.assertTrue(torch.allclose(next_state, torch.tensor([[1.075, 2.025, 0.75, 0.25]])))

    def test_sample_smooth_action_clips_to_per_joint_range(self) -> None:
        rng = np.random.default_rng(0)
        previous = np.array([0.0, 0.0], dtype=np.float32)
        low = np.array([-0.2, -2.0], dtype=np.float32)
        high = np.array([0.2, 2.0], dtype=np.float32)

        action = sample_smooth_action(rng, previous, action_std=100.0, n_joints=2, action_low=low, action_high=high)

        self.assertLessEqual(action[0], 0.2)
        self.assertGreaterEqual(action[0], -0.2)
        self.assertLessEqual(action[1], 2.0)
        self.assertGreaterEqual(action[1], -2.0)

    def test_rollout_visualization_samples_position_targets_with_env_action_range(self) -> None:
        rng = np.random.default_rng(0)
        previous = np.array([0.0, 0.0], dtype=np.float32)
        action_low = np.array([-0.2, -2.0], dtype=np.float32)
        action_high = np.array([0.2, 2.0], dtype=np.float32)

        action = ROLLOUT_VISUALIZE.sample_visualization_action(
            rng,
            previous,
            action_std=100.0,
            n_joints=2,
            action_low=action_low,
            action_high=action_high,
        )

        self.assertLessEqual(action[0], 0.2)
        self.assertGreaterEqual(action[0], -0.2)
        self.assertLessEqual(action[1], 2.0)
        self.assertGreaterEqual(action[1], -2.0)

    def test_diagnose_qacc_substep_records_apply_gravity_compensation(self) -> None:
        env = MuJoCoArmEnv(model_xml=str(ROOT / "ABB_IRB2400.xml"), n_joints=6, frame_skip=2, seed=0)
        try:
            env.reset_random()
            action = np.asarray(env.data.qpos[:6], dtype=np.float64).copy()

            record = DIAG_DYNAMICS.step_with_alignment_records(env, action, 6)

            self.assertEqual(record.control_delta.shape, (6,))
            self.assertEqual(record.summed_substep_delta.shape, (6,))
            self.assertEqual(record.post_step_qacc_delta.shape, (6,))
            self.assertEqual(env.data.qfrc_applied[5], 0.0)
            self.assertTrue(np.any(np.abs(env.data.qfrc_applied[:5]) > 1e-9))
        finally:
            env.close()

    def test_train_scripts_default_to_delta_dq_targets(self) -> None:
        argv = ["train_dynamics.py", "--data_path", "dummy.npz"]
        with patch.object(sys, "argv", argv):
            args = TRAIN_DYNAMICS.parse_args()
        with patch.object(sys, "argv", argv):
            args_v2 = TRAIN_DYNAMICS2.parse_args()

        self.assertEqual(args.target_mode, "delta_dq")
        self.assertEqual(args_v2.target_mode, "delta_dq")

    def test_default_config_uses_delta_dq_target_mode(self) -> None:
        config_text = (ROOT / "configs" / "default.yaml").read_text(encoding="utf-8")

        self.assertIn("target_mode: delta_dq", config_text)
        self.assertNotIn("target_mode: delta_state", config_text)

    def test_checkpoint_round_trips_full_training_state(self) -> None:
        model = MLPDynamics(state_dim=2, action_dim=1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        config = {"model_type": "mlp", "state_dim": 2, "action_dim": 1, "history_len": 1}
        metadata = {"epoch": 3, "best_val": 0.25, "checkpoint_type": "latest"}

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.pt"
            save_checkpoint(path, model, config, optimizer=optimizer, scaler=scaler, metadata=metadata)

            checkpoint = load_checkpoint(path)

        self.assertIn("optimizer_state_dict", checkpoint)
        self.assertIn("scaler_state_dict", checkpoint)
        self.assertEqual(checkpoint["config"], config)
        self.assertEqual(checkpoint["metadata"], metadata)

    def test_old_checkpoint_format_still_loads(self) -> None:
        model = MLPDynamics(state_dim=2, action_dim=1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "old.pt"
            torch.save({"model_state_dict": model.state_dict(), "config": {"model_type": "mlp"}}, path)

            checkpoint = load_checkpoint(path)

        self.assertIn("model_state_dict", checkpoint)
        self.assertNotIn("optimizer_state_dict", checkpoint)

    def test_resume_requires_full_training_state(self) -> None:
        model = MLPDynamics(state_dim=2, action_dim=1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "old.pt"
            torch.save({"model_state_dict": model.state_dict(), "config": {"model_type": "mlp"}}, path)
            checkpoint = load_checkpoint(path)

        with self.assertRaisesRegex(ValueError, "not a full resume checkpoint"):
            require_resume_checkpoint(checkpoint)

    def test_train_script_resumes_latest_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_path = tmp_path / "tiny_data.npz"
            save_dir = tmp_path / "checkpoints"
            self._write_tiny_dataset(data_path)

            self._run_train(data_path, save_dir, "--epochs", "1")
            run_dir = self._single_run_dir(save_dir)
            latest = run_dir / "latest_model.pt"
            self.assertTrue(latest.exists())

            self._run_train(data_path, save_dir, "--epochs", "2", "--resume_checkpoint", str(latest))
            checkpoint = load_checkpoint(latest)

        self.assertEqual(checkpoint["metadata"]["epoch"], 2)
        self.assertEqual(checkpoint["metadata"]["checkpoint_type"], "latest")

    def test_train_script_initializes_from_checkpoint_in_new_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_path = tmp_path / "tiny_data.npz"
            save_dir = tmp_path / "checkpoints"
            init_save_dir = tmp_path / "init_checkpoints"
            self._write_tiny_dataset(data_path)

            self._run_train(data_path, save_dir, "--epochs", "1")
            source = self._single_run_dir(save_dir) / "best_model.pt"
            self.assertTrue(source.exists())

            self._run_train(
                data_path,
                init_save_dir,
                "--epochs",
                "1",
                "--init_from_checkpoint",
                str(source),
            )
            run_dir = self._single_run_dir(init_save_dir)
            checkpoint = load_checkpoint(run_dir / "latest_model.pt")

        self.assertEqual(checkpoint["metadata"]["epoch"], 1)
        self.assertEqual(checkpoint["metadata"]["checkpoint_type"], "latest")
        self.assertEqual(checkpoint["metadata"]["init_from_checkpoint"], str(source))

    def test_train_script_with_rollout_loss_saves_rollout_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_path = tmp_path / "tiny_data.npz"
            save_dir = tmp_path / "checkpoints"
            self._write_tiny_dataset(data_path, include_episode_ids=True)

            self._run_train(
                data_path,
                save_dir,
                "--epochs",
                "1",
                "--rollout_loss_steps",
                "3",
                "--rollout_loss_weight",
                "0.1",
            )
            run_dir = self._single_run_dir(save_dir)
            checkpoint = load_checkpoint(run_dir / "latest_model.pt")

            self.assertTrue((run_dir / "best_rollout_model.pt").exists())
            self.assertEqual(checkpoint["config"]["rollout_loss_steps"], 3)
            self.assertEqual(checkpoint["config"]["rollout_loss_weight"], 0.1)

    def test_train_script_saves_sample_stride_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_path = tmp_path / "tiny_data.npz"
            save_dir = tmp_path / "checkpoints"
            self._write_tiny_dataset(data_path, include_episode_ids=True)

            self._run_train(
                data_path,
                save_dir,
                "--epochs",
                "1",
                "--train_sample_stride",
                "3",
                "--val_sample_stride",
                "1",
            )
            run_dir = self._single_run_dir(save_dir)
            checkpoint = load_checkpoint(run_dir / "latest_model.pt")

        self.assertEqual(checkpoint["config"]["train_sample_stride"], 3)
        self.assertEqual(checkpoint["config"]["val_sample_stride"], 1)

    def test_train_script_v2_saves_sample_stride_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_path = tmp_path / "tiny_data.npz"
            save_dir = tmp_path / "checkpoints"
            self._write_tiny_dataset(data_path, include_episode_ids=True)

            self._run_train_v2(
                data_path,
                save_dir,
                "--epochs",
                "1",
                "--train_sample_stride",
                "3",
                "--val_sample_stride",
                "1",
            )
            run_dir = self._single_run_dir(save_dir)
            checkpoint = load_checkpoint(run_dir / "latest_model.pt")

        self.assertEqual(checkpoint["config"]["train_sample_stride"], 3)
        self.assertEqual(checkpoint["config"]["val_sample_stride"], 1)

    def test_merge_npz_datasets_concatenates_required_arrays(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            first = tmp_path / "first.npz"
            second = tmp_path / "second.npz"
            output = tmp_path / "merged.npz"
            np.savez(
                first,
                states=np.ones((2, 4), dtype=np.float32),
                actions=np.ones((2, 2), dtype=np.float32),
                next_states=np.full((2, 4), 2.0, dtype=np.float32),
            )
            np.savez(
                second,
                states=np.full((3, 4), 3.0, dtype=np.float32),
                actions=np.full((3, 2), 4.0, dtype=np.float32),
                next_states=np.full((3, 4), 5.0, dtype=np.float32),
            )

            shapes = merge_npz_datasets([first, second], output)
            merged = np.load(output)

        self.assertEqual(shapes["states"], (5, 4))
        self.assertEqual(shapes["actions"], (5, 2))
        self.assertEqual(shapes["next_states"], (5, 4))
        self.assertTrue(np.allclose(merged["states"][:2], 1.0))
        self.assertTrue(np.allclose(merged["states"][2:], 3.0))

    def test_save_dataset_appends_to_existing_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.npz"
            save_dataset(
                path,
                np.ones((2, 4), dtype=np.float32),
                np.ones((2, 2), dtype=np.float32),
                np.full((2, 4), 2.0, dtype=np.float32),
            )

            save_dataset(
                path,
                np.full((3, 4), 3.0, dtype=np.float32),
                np.full((3, 2), 4.0, dtype=np.float32),
                np.full((3, 4), 5.0, dtype=np.float32),
                append=True,
            )
            merged = np.load(path)

        self.assertEqual(merged["states"].shape, (5, 4))
        self.assertEqual(merged["actions"].shape, (5, 2))
        self.assertEqual(merged["next_states"].shape, (5, 4))
        self.assertTrue(np.allclose(merged["states"][:2], 1.0))
        self.assertTrue(np.allclose(merged["states"][2:], 3.0))

    def test_save_dataset_offsets_episode_ids_when_appending(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sequence_data.npz"
            save_dataset(
                path,
                np.ones((4, 4), dtype=np.float32),
                np.ones((4, 2), dtype=np.float32),
                np.full((4, 4), 2.0, dtype=np.float32),
                episode_ids=np.array([0, 0, 1, 1], dtype=np.int64),
            )

            save_dataset(
                path,
                np.full((2, 4), 3.0, dtype=np.float32),
                np.full((2, 2), 4.0, dtype=np.float32),
                np.full((2, 4), 5.0, dtype=np.float32),
                episode_ids=np.array([0, 0], dtype=np.int64),
                append=True,
            )
            merged = np.load(path)

        self.assertEqual(merged["episode_ids"].tolist(), [0, 0, 1, 1, 2, 2])

    def test_validate_append_dataset_rejects_incomplete_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "broken.npz"
            np.savez(path, states=np.ones((2, 4), dtype=np.float32))

            with self.assertRaisesRegex(KeyError, "missing arrays"):
                validate_append_dataset(path)

    def test_resolve_history_len_prefers_checkpoint_config_for_sequence_models(self) -> None:
        self.assertEqual(resolve_history_len("transformer", requested_history_len=1, config={"history_len": 16}), 16)
        self.assertEqual(resolve_history_len("gru", requested_history_len=8, config={"history_len": 16}), 8)
        self.assertEqual(resolve_history_len("mlp", requested_history_len=16, config={"history_len": 16}), 1)

    def test_eval_parse_args_exposes_diagnostic_options_with_compatible_defaults(self) -> None:
        args = EVAL_DYNAMICS.parse_args(
            [
                "--checkpoint",
                "model.pt",
                "--normalizer",
                "normalizer.pt",
                "--model_type",
                "transformer",
            ]
        )

        self.assertEqual(args.action_std, 0.3)
        self.assertEqual(args.warmup_steps, 0)
        self.assertEqual(args.horizons, "1,5,10,20,50,200")
        self.assertFalse(args.teacher_forcing)

    def test_parse_horizons_rejects_non_positive_values(self) -> None:
        self.assertEqual(parse_horizons("1, 5,10"), [1, 5, 10])

        with self.assertRaisesRegex(ValueError, "positive"):
            parse_horizons("1,0,5")

    def test_state_labels_and_per_dimension_rmse_report_q_and_dq(self) -> None:
        truth = np.array([[1.0, 2.0, 10.0, 20.0], [3.0, 4.0, 30.0, 40.0]], dtype=np.float32)
        pred = np.array([[0.0, 4.0, 7.0, 24.0], [5.0, 0.0, 36.0, 32.0]], dtype=np.float32)

        labels = state_labels(2)
        rmse = per_dimension_rmse(truth, pred)

        self.assertEqual(labels, ["q0", "q1", "dq0", "dq1"])
        self.assertTrue(np.allclose(rmse, np.array([np.sqrt(2.5), np.sqrt(10.0), np.sqrt(22.5), np.sqrt(40.0)])))

    def test_summarize_prediction_reports_aggregate_q_and_dq_rmse(self) -> None:
        truth = np.array([[1.0, 2.0, 10.0, 20.0], [3.0, 4.0, 30.0, 40.0]], dtype=np.float32)
        pred = np.array([[0.0, 4.0, 7.0, 24.0], [5.0, 0.0, 36.0, 32.0]], dtype=np.float32)

        summary = summarize_prediction(truth, pred, state_labels(2))

        self.assertAlmostEqual(summary["q_rmse"], float(np.sqrt((1.0 + 4.0 + 4.0 + 16.0) / 4.0)))
        self.assertAlmostEqual(summary["dq_rmse"], float(np.sqrt((9.0 + 16.0 + 36.0 + 64.0) / 4.0)))

    def test_build_sequence_history_uses_warmup_window_without_padding_when_available(self) -> None:
        entries = [np.array([idx, idx + 10], dtype=np.float32) for idx in range(5)]

        history = build_sequence_history(entries, current_index=4, history_len=3)

        self.assertEqual(len(history), 3)
        self.assertTrue(np.allclose(np.stack(history), np.array([[2, 12], [3, 13], [4, 14]], dtype=np.float32)))

    def test_build_sequence_history_repeats_first_entry_when_history_is_short(self) -> None:
        entries = [np.array([1.0, 2.0], dtype=np.float32), np.array([3.0, 4.0], dtype=np.float32)]

        history = build_sequence_history(entries, current_index=1, history_len=4)

        self.assertEqual(len(history), 4)
        self.assertTrue(
            np.allclose(
                np.stack(history),
                np.array([[1, 2], [1, 2], [1, 2], [3, 4]], dtype=np.float32),
            )
        )

    def test_open_loop_segment_can_record_next_states_for_horizon_metrics(self) -> None:
        class ConstantDeltaModel(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.tensor([[1.0, 2.0]], dtype=torch.float32, device=x.device)

        class IdentityNormalizer:
            def normalize_single_input(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
                return torch.cat([states, actions], dim=-1)

            def denormalize_delta(self, deltas: torch.Tensor) -> torch.Tensor:
                return deltas

        true_states = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
        actions = np.array([[0.5], [0.25]], dtype=np.float32)
        model = ConstantDeltaModel()
        normalizer = IdentityNormalizer()

        initial_records = predict_open_loop_segment(
            model,
            normalizer,
            "mlp",
            true_states,
            actions,
            start_index=0,
            rollout_len=1,
            history_len=1,
            state_dim=2,
            device=torch.device("cpu"),
        )
        next_records = predict_open_loop_segment(
            model,
            normalizer,
            "mlp",
            true_states,
            actions,
            start_index=0,
            rollout_len=1,
            history_len=1,
            state_dim=2,
            device=torch.device("cpu"),
            record_next_states=True,
        )

        self.assertTrue(np.allclose(initial_records, np.array([[10.0, 20.0]], dtype=np.float32)))
        self.assertTrue(np.allclose(next_records, np.array([[11.0, 22.0]], dtype=np.float32)))

    def test_plot_rollout_can_save_teacher_forcing_figures_separately(self) -> None:
        truth = np.array([[1.0, 10.0], [2.0, 20.0]], dtype=np.float32)
        pred = np.array([[1.5, 10.5], [2.5, 20.5]], dtype=np.float32)

        with tempfile.TemporaryDirectory() as tmp:
            save_dir = Path(tmp)

            EVAL_DYNAMICS.plot_rollout(truth, pred, n_joints=1, rollout_idx=7, save_dir=save_dir, prefix="teacher_forcing")

            self.assertTrue((save_dir / "teacher_forcing_007_q.png").exists())
            self.assertTrue((save_dir / "teacher_forcing_007_dq.png").exists())
            self.assertTrue((save_dir / "teacher_forcing_007_error.png").exists())

    def test_eval_scripts_plot_bottom_controller_torque_components(self) -> None:
        torque_components = {
            "total_tau": np.array([[1.0, 2.0], [1.5, 2.5]], dtype=np.float64),
            "actuator_tau": np.array([[0.25, 0.5], [0.75, 1.0]], dtype=np.float64),
            "gravity_tau": np.array([[0.75, 1.5], [0.75, 1.5]], dtype=np.float64),
        }

        for module, prefix in ((EVAL_DYNAMICS, "rollout"), (EVAL_DYNAMICS2, "rollout_v2")):
            with tempfile.TemporaryDirectory() as tmp:
                save_dir = Path(tmp)

                module.plot_torque_components(torque_components, n_joints=2, rollout_idx=7, save_dir=save_dir, prefix=prefix)

                self.assertTrue((save_dir / f"{prefix}_007_torque.png").exists())

    def test_diagnostic_joint_limit_summary_reports_near_limit_and_violation_rates(self) -> None:
        states = np.array(
            [
                [0.0, 1.00, 0.0, 0.0],
                [0.1, 1.86, 0.2, 0.0],
                [0.2, 1.95, 0.4, 0.0],
                [0.3, -1.76, 0.6, 0.0],
            ],
            dtype=np.float32,
        )
        joint_ranges = np.array([[-3.0, 3.0], [-1.7453, 1.9199]], dtype=np.float32)

        summary = DIAG_ROLLOUT_SPIKES.summarize_joint_limits(
            states,
            joint_ranges,
            n_joints=2,
            limit_margin=0.05,
            near_limit_overrides={1: 1.85},
        )

        self.assertAlmostEqual(summary["q1_near_upper_rate"], 0.5)
        self.assertAlmostEqual(summary["q1_violation_rate"], 0.5)
        self.assertLess(summary["q1_min_limit_distance"], 0.0)
        self.assertAlmostEqual(summary["q0_near_upper_rate"], 0.0)

    def test_diagnostic_dataset_summary_handles_bad_npz_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bad_path = Path(tmp) / "bad.npz"
            bad_path.write_text("not a npz", encoding="utf-8")
            joint_ranges = np.array([[-1.0, 1.0]], dtype=np.float32)

            summary = DIAG_ROLLOUT_SPIKES.summarize_dataset_file(
                bad_path,
                n_joints=1,
                joint_ranges=joint_ranges,
                limit_margin=0.05,
                near_limit_overrides={},
                max_dataset_samples=None,
            )

        self.assertEqual(summary["status"], "bad_npz")
        self.assertEqual(summary["path"], str(bad_path))

    def test_diagnostic_spike_summary_identifies_dominant_error_and_causes(self) -> None:
        truth = np.array(
            [
                [0.0, 1.00, 0.0, 0.0],
                [0.0, 1.91, 0.1, 5.0],
                [0.0, 1.92, 0.1, 4.0],
            ],
            dtype=np.float32,
        )
        pred = np.array(
            [
                [0.0, 1.00, 0.0, 0.0],
                [0.0, 1.80, 0.1, 2.0],
                [0.0, 1.81, 0.1, 3.0],
            ],
            dtype=np.float32,
        )
        actions = np.array([[0.0], [0.2], [0.21]], dtype=np.float32)
        teacher_pred = truth.copy()
        teacher_pred[1, 3] = 4.9
        teacher_pred[2, 3] = 3.0
        joint_ranges = np.array([[-3.0, 3.0], [-1.7453, 1.9199]], dtype=np.float32)
        jacobian = {"sigma_min": np.array([0.1, 0.005, 0.2]), "condition": np.array([10.0, 250.0, 8.0])}

        summary = DIAG_ROLLOUT_SPIKES.summarize_rollout_spike(
            rollout_idx=7,
            truth=truth,
            pred=pred,
            actions=actions,
            joint_ranges=joint_ranges,
            labels=["q0", "q1", "dq0", "dq1"],
            jacobian=jacobian,
            teacher_pred=teacher_pred,
            limit_margin=0.05,
            action_jump_percentile=95.0,
            high_speed_percentile=90.0,
            singularity_cond_threshold=200.0,
            singularity_sigma_threshold=0.01,
        )

        self.assertEqual(summary["rollout"], 7)
        self.assertEqual(summary["peak_step"], 1)
        self.assertEqual(summary["dominant_error_0"], "dq1")
        self.assertTrue(summary["near_joint_limit"])
        self.assertTrue(summary["high_speed"])
        self.assertTrue(summary["possible_singularity"])
        self.assertFalse(summary["teacher_forcing_bad"])

    def test_kp_tuning_parses_human_joint_selection(self) -> None:
        self.assertEqual(TUNE_POSITION_KP.parse_joint_selection("all", n_joints=6), [0, 1, 2, 3, 4, 5])
        self.assertEqual(TUNE_POSITION_KP.parse_joint_selection("2,4", n_joints=6), [1, 3])

        with self.assertRaisesRegex(ValueError, "between 1 and 6"):
            TUNE_POSITION_KP.parse_joint_selection("0", n_joints=6)

    def test_kp_tuning_builds_position_actuators_with_unit_damping_ratio(self) -> None:
        source = """
<mujoco model="tiny">
  <worldbody>
    <body name="b">
      <joint name="joint_1" type="hinge" range="-1 1"/>
      <joint name="joint_2" type="hinge" range="-2 2"/>
    </body>
  </worldbody>
  <actuator>
    <velocity name="joint_1_velocity" joint="joint_1" kv="40" ctrllimited="true" ctrlrange="-1 1"/>
    <velocity name="joint_2_velocity" joint="joint_2" kv="40" ctrllimited="true" ctrlrange="-1 1"/>
  </actuator>
</mujoco>
"""
        xml = TUNE_POSITION_KP.build_position_actuator_xml(
            source,
            joint_names=["joint_1", "joint_2"],
            joint_ranges=[(-1.0, 1.0), (-2.0, 2.0)],
            kps=[800.0, 1200.0],
        )

        self.assertIn('<position name="joint_1_position" joint="joint_1" kp="800', xml)
        self.assertIn('dampratio="1"', xml)
        self.assertIn('ctrlrange="-2 2"', xml)
        self.assertNotIn("<velocity", xml)

    def test_kp_tuning_records_and_plots_torque_components(self) -> None:
        response = TUNE_POSITION_KP.run_response(
            ROOT / "ABB_IRB2400.xml",
            joint_idx=1,
            kp=1200.0,
            base_kp=1200.0,
            duration=0.01,
            frame_skip=1,
            step_fraction=0.02,
            gravity_compensation=True,
        )

        for key in ("total_tau", "actuator_tau", "gravity_tau"):
            self.assertIn(key, response)
            self.assertEqual(response[key].shape, response["q"].shape)
        self.assertTrue(np.allclose(response["total_tau"], response["actuator_tau"] + response["gravity_tau"]))
        self.assertTrue(np.all(response["gravity_tau"][:, 5] == 0.0))
        self.assertTrue(np.all(response["applied"][:, 5] == 0.0))

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            TUNE_POSITION_KP.plot_joint_torque_sweep(
                output_dir,
                joint_idx=1,
                responses=[response],
                metrics=[{"kp": 1200.0}],
            )

            self.assertTrue((output_dir / "joint_02_kp_sweep_torque_gravity_comp.png").exists())

    @staticmethod
    def _write_tiny_dataset(path: Path, include_episode_ids: bool = False) -> None:
        states = np.arange(80, dtype=np.float32).reshape(40, 2) / 100.0
        actions = np.linspace(-0.5, 0.5, 40, dtype=np.float32).reshape(40, 1)
        next_states = states + np.concatenate([actions, -actions], axis=1) * 0.01
        if include_episode_ids:
            np.savez(
                path,
                states=states,
                actions=actions,
                next_states=next_states,
                episode_ids=np.zeros(40, dtype=np.int64),
            )
        else:
            np.savez(path, states=states, actions=actions, next_states=next_states)

    @staticmethod
    def _run_train(data_path: Path, save_dir: Path, *extra_args: str) -> None:
        command = [
            sys.executable,
            str(ROOT / "scripts" / "train_dynamics.py"),
            "--data_path",
            str(data_path),
            "--model_type",
            "mlp",
            "--batch_size",
            "8",
            "--save_dir",
            str(save_dir),
            *extra_args,
        ]
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
        if result.returncode != 0:
            raise AssertionError(f"train command failed\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}")

    @staticmethod
    def _run_train_v2(data_path: Path, save_dir: Path, *extra_args: str) -> None:
        command = [
            sys.executable,
            str(ROOT / "scripts2" / "train_dynamics.py"),
            "--data_path",
            str(data_path),
            "--model_type",
            "mlp",
            "--batch_size",
            "8",
            "--save_dir",
            str(save_dir),
            *extra_args,
        ]
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
        if result.returncode != 0:
            raise AssertionError(f"train v2 command failed\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}")

    @staticmethod
    def _single_run_dir(save_dir: Path) -> Path:
        run_dirs = [path for path in save_dir.iterdir() if path.is_dir()]
        if len(run_dirs) != 1:
            raise AssertionError(f"expected one run dir in {save_dir}, got {run_dirs}")
        return run_dirs[0]


if __name__ == "__main__":
    unittest.main()
