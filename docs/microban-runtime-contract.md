# Microban native runtime compatibility contract

This contract fixes the KHR keyboard-policy launch and input architecture to
the Rhoban/microban implementation. KHR-specific model differences do not
authorize a different viewer or input transport.

## Pinned reference

- Repository: `https://github.com/Rhoban/microban`
- Commit: `d0a128d180ecfce69eaaf31b295a0fce31280bfd`
- Launch command: `make sim`
- Expanded command:
  `PYTHONPATH=src uv run --group sim src/sim/sim_main.py --hz 50`

## Required path

| Stage | Microban source | KHRBan counterpart | Required invariant |
|---|---|---|---|
| Entry | `src/sim/sim_main.py` | `khrban.keyboard_policy` | One local simulator process owns input, policy, physics, and viewer. |
| Input state | `MuJoCoInputSource` | `KeyboardVelocityState` | `V`, arrows, `X`, `R`, and `Q` keep Microban meanings. |
| Controller/viewer | `MuJoCoController` | `MicrobanKeyboardViewer` | Native MuJoCo passive viewer only. |
| Window creation | `mujoco.viewer.launch_passive` | MjLab `NativeMujocoViewer`, which calls the same API | No browser or streamed replacement. |
| Key transport | viewer `key_callback` | viewer `_safe_key_callback` | GLFW events enter through the MuJoCo window, not the terminal or OS-wide hooks. |

## KHR-only differences allowed

- KHR URDF/MJCF geometry and inertial properties.
- KRS-2552 actuator parameters.
- KHR joint names, directions, home pose, and frozen joints.
- KHR policy observation/action dimensions and checkpoint loading.
- KHR-specific physical safety ranges and model-size-dependent thresholds.

## Forbidden substitutions

- Viser or another browser-based policy viewer.
- A web keyboard bridge or browser hotkeys.
- Video streaming of a hidden/headless simulator.
- Terminal-focused keyboard polling.
- A viewer backend selected only to hide a Windows/WSL display failure.

Windows display failures are final-environment failures. They must be handled
at the Windows/WSL graphics layer while preserving the native Microban path.
