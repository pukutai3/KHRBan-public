# KHR / microban model dimension and reward audit

## Decision

The current KHR standing task must keep its KHR-specific reward and termination
configuration. Its evaluation terms are based on orientation, joint angles,
joint angular velocities, actions, and actuator torque, so no link-length
threshold needs to be changed.

The Microban velocity-task configuration must **not** be copied unchanged into
a future KHR walking task. In particular, its foot-distance and root-height
thresholds encode Microban geometry and frame conventions.

## Sources and measurement convention

- KHRBan model commit: `b9473ca7121152fe483d3de999b0295d3a50e7dc`
- Local read-only microban commit: `d0a128d180ecfce69eaaf31b295a0fce31280bfd`
- Microban training reference (`Rhoban/mjlab_microban`) main commit checked:
  `d594a6088bb7b6600fe8098321169031fbca680c`

Distances below are Euclidean distances between MuJoCo joint-axis anchors in
the servo mounting-zero configuration. Bilateral widths are measured between
the corresponding left and right axes. The direct KHR shoulder-roll-to-elbow
distance spans its additional shoulder-yaw joint, which Microban does not have.

## Dimension comparison

| Measurement | KHR | microban | KHR difference |
|---|---:|---:|---:|
| Shoulder-pitch axis spacing | 97.800 mm | 101.000 mm | -3.2% |
| Head axis to shoulder midpoint | 11.750 mm | 45.046 mm | -73.9% |
| Shoulder midpoint to hip midpoint | 87.509 mm | 81.000 mm | +8.0% |
| Hip-yaw axis spacing | 43.400 mm | 72.000 mm | -39.7% |
| Shoulder pitch to shoulder roll | 29.548 mm | 24.300 mm | +21.6% |
| Shoulder roll to elbow, direct | 87.883 mm | 55.432 mm | +58.5% |
| Hip yaw to hip roll | 28.504 mm | 29.291 mm | -2.7% |
| Hip roll to hip pitch | 50.799 mm | 41.140 mm | +23.5% |
| Hip pitch to knee (link A) | 65.000 mm | 62.201 mm | +4.5% |
| Knee to ankle pitch (link B) | 65.002 mm | 61.734 mm | +5.3% |
| Ankle pitch to ankle roll | 52.291 mm | 41.140 mm | +27.1% |
| Pitch-chain A + B | 130.002 mm | 123.935 mm | +4.9% |
| Model mass | 1.458 kg | 0.819 kg | +78.1% |

At each model's configured home pose, the measured horizontal foot-reference
spacing is 42.533 mm for the KHR foot-body origins and 93.075 mm for the
Microban foot sites. These references are not identical definitions, so the
values are suitable for rejecting a direct threshold copy, not for final KHR
foot-collision clearance certification.

## Current KHR standing evaluation

| Term | Dimension dependency | Decision |
|---|---|---|
| `alive` | None | Keep |
| `upright` | Gravity direction / orientation | Keep |
| `posture` | Joint-angle error; averaged over joints | Keep |
| `joint_velocity` | Joint angular velocity | Keep |
| `action_rate` | Policy action change | Keep |
| `torque` | Dynamics and actuator capability, not link-length threshold | Keep with the KRS-2552 BAM model |
| `fell_over` | 45-degree body-orientation limit | Keep |
| `leg_flexion_ratio` | Normalized angle from link A and B vectors | Keep; vector lengths cancel |

The KHR model has 22 actuators while Microban has 19. MjLab's posture reward
uses a mean over joints, while the velocity, action-rate, and torque penalties
use sums. This changes aggregate scaling with the DOF count, but the present
KHR weights are an independent standing baseline and were not copied from the
Microban walking task.

## Required KHR changes for a future velocity task

The current repository does not yet contain a KHR velocity task, so these
values must be applied when that task is added rather than inserted as unused
configuration now.

| Microban setting | Why it cannot be copied | KHR handling |
|---|---|---|
| Root-height fall limit `0.10 m` | Microban's root is `0.168 m` high at home; KHR's exported root frame is `0.010 m` high | Continue orientation termination, or evaluate a named torso/COM frame and calibrate its KHR threshold |
| Minimum foot distance `0.08 m` | Microban home foot sites are 93.075 mm apart; KHR foot-body origins are 42.533 mm apart | Add explicit KHR foot sites/collision geoms first, then derive the limit from their clearance |
| Foot clearance/swing target `0.020 m` | Absolute length | Initial geometric scaling gives about `0.021 m` from the +4.9% KHR pitch-chain length; validate in simulation |
| Foot scan radius `0.040 m` | Absolute length and foot geometry | Initial geometric scaling gives about `0.042 m`; validate against the KHR foot collision mesh |
| Angular momentum penalty | Depends on mass distribution and size | Retune for KHR; its model mass is about 78% greater |
| Joint regex and pose standard deviations | KHR adds waist yaw and two shoulder-yaw axes | Define explicit KHR mappings; do not reuse Microban names |

Velocity tracking, uprightness, foot slip, air time, self-collision, joint-limit,
and action-smoothness evaluation concepts remain appropriate. Their geometry
selectors, thresholds, and weights require KHR-specific validation.
