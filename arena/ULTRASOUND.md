# Ultrasound B-mode in the liver-scan workflow

From the workflow repository root:

```bash
./run.sh ultrasound_liver_scan --rule-based --ultrasound --record ultrasound.hdf5
```

`--ultrasound` adds the `ultrasound` image stream to the runtime scene and opens
**Ultrasound B-mode (OptiX)** alongside the viewport. Without this flag the
existing RGB-only workflow is unchanged. Use `--no-sensor-view` to hide the
panel while retaining acquisition/recording. `--headless` also suppresses the
panel. `--no-cameras` cannot be combined with `--ultrasound`.

The sensor uses the real `ultrasound_simulator.cuda` ray-tracing backend from
`i4h-sensor-simulation`, executed in a disposable GPU Docker container. The
container loads the same abdominal meshes and tissue assignments as its web
demo; no web server needs to be started. The host needs Docker GPU access and
the already-built image `i4h_sim_build:ultrasound-simulator`, including mesh
assets at `/opt/ultrasound-mesh`. To use another compatible image or GPU:

```bash
./run.sh ultrasound_liver_scan --rule-based --ultrasound \
  --ultrasound-image i4h_sim_build:ultrasound-simulator --ultrasound-gpu 0
```

The GPU argument is a physical index or GPU UUID, independently of Isaac's
`--device` and any host `CUDA_VISIBLE_DEVICES` mapping. The worker mounts only
its Python entry point, has no network service, and exits with the rollout.
The default setup script does not build this Docker image.

## Data and coordinate contract

- Acquisition is configured at 10 Hz of simulation time, with 256 × 256 pixels,
  a curvilinear probe and 180 mm maximum ray distance.
- Poses come from the scene's existing `ee_to_us_transform` and
  `mesh_to_organ_transform` frame sensors after physics advances. Their target
  world poses use metres and xyzw quaternions. The adapter computes the probe
  pose relative to the mesh, converts metres to millimetres and rotation to
  extrinsic XYZ radians (the backend uses `Rz @ Ry @ Rx`).
- The frame updates with both robot and phantom motion. The implementation
  reuses the authored rigid calibration; it does not model probe-pressure
  deformation or contact-dependent acoustic coupling.
- `scene.camera("ultrasound")` exposes grayscale B-mode as RGB uint8 using the
  upstream -60..0 dB display window. `scene.sensor_signal("ultrasound",
  "bmode_db")` exposes float32 dB values. No-echo nonfinite samples and the
  native outside-sector sentinel map to -120 dB.
- `--record` stores the displayed B-mode under
  `data/demo_N/obs/ultrasound`, alongside `room` and `wrist`. This first version
  does not record the raw dB buffer or per-acquisition probe poses. Frames are
  sampled at each recorded control step, so repeated frames between sensor
  updates are expected.
- The policy's existing RGB inputs and the workflow success criterion are
  unchanged. This is ray-based B-mode, not a full acoustic wave solver.

The live image is an integration smoke test, not an independent validation of
anatomical registration or quantitative acoustic accuracy. Changing the USD
phantom or the container meshes requires checking their shared calibration.

## Verification

```bash
arena/.venv/bin/python -m pytest arena/tests/test_ultrasound.py -q
./run.sh ultrasound_liver_scan --rule-based --ultrasound --episodes 1 --record ultrasound.hdf5
```

Verify the image changes during the sweep, the episode success summary, and
the `ultrasound` dataset in the recorded HDF5. Worker startup, mesh, GPU and
protocol failures are reported in the run's `i4h_arena.log`; they do not switch
to synthetic images.

## Phantom audit

The workflow USD uses Healthcare 0.5.0/132c82d; the default ultrasound image
was built with mesh assets requested as version 0.2.0. SHA-256 checks of
Skin.obj, Liver.obj and Bone.obj nevertheless match the corresponding 0.5.0
catalog files exactly. The version label alone is not a geometry mismatch. The USD external
phantom bounds are approximately 255 × 300 × 202 mm. The container Skin.obj
bounds are approximately 327 × 219 × 344 mm before the authored mesh-frame
rotation. These assets must not be described as independently registered or
identical based only on their ABDPhantom catalog name. The current integration
uses the existing frame calibration, which still needs anatomical registration
against the rendered external phantom. Quaternions at the Isaac Lab sensor
boundary are XYZW; the earlier WXYZ interpretation was incorrect.


## Phantom geometry in recordings

For `panda_phantom`, the standard HDF5 recorder stores the following alongside
images and measured TCP observations, for env 0 at every recorded control step:

- `obs/phantom_pose`: organ root world pose, `(T, 7)`.
- `obs/mesh_pose`: calibrated `mesh_to_organ_transform` world pose, `(T, 7)`.
- `obs/ultrasound_probe_pose`: `ee_to_us_transform` world pose, `(T, 7)`.
- `obs/timestamps`: simulation seconds (`common_step_counter * step_dt`), `(T,)`.

Pose order is xyz in metres followed by quaternion wxyz; dataset attributes
record units, frame and quaternion order. `phantom_recording_schema=1` is stored
on the episode. All buffers clear on episode restart/reset, so a new phantom
randomization cannot inherit the previous episode's pose. The calibrated mesh
pose includes the acoustic offset and should be used to place Skin.obj (whose
vertices are in millimetres) in a world-space replay. It is not the root pose.

No new command-line flags are required. Existing HDF5 episodes are untouched;
new episodes appended to the file contain these fields. Sensor frame IDs retain
their existing meaning: repeated IDs indicate an older ultrasound image.
