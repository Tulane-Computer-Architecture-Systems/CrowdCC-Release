# CrowdCC corruptions

CrowdCC defines 23 corruptions across six families and three severity levels (`light` < `moderate` <
`high`). Parameters are defined in `crowdcc/families/*.py` and listed by `crowdcc list --json`.

- Severity parameters are shared across datasets.
- The default random seed is `1234` (`--seed`). The `structured_foreground_occlusion` layout is fixed for
  each image and severity level and does not use this option.
- Corruptions operate on full frames and produce lossless PNGs.
- ↓ marks parameters that decrease with severity.
- Seven GT-altering corruptions move or remove heads. Evaluation uses `n_kept` from each
  `<image_id>.gt.json`. GT-altering corruptions also drop out-of-frame heads; GT-preserving corruptions
  retain the original annotations.

## 1. Overview

Primary severity parameters are listed below. Full parameter sets follow in Section 2.

| corruption | family | GT | needs | primary parameter | light | moderate | high |
|---|---|---|---|---|---:|---:|---:|
| `camera_motion_blur` | optics_motion | preserving | gt_points | `length_spacing_frac` | 0.05 | 0.15 | 0.25 |
| `far_field_defocus` | optics_motion | preserving | depth, gt_points | `max_blur_spacing_frac` | 0.05 | 0.10 | 0.15 |
| `lens_contamination` | optics_motion | **altering** | gt_points | `radius_frac` | 0.01 | 0.02 | 0.03 |
| `lens_flare_veiling` | optics_motion | preserving | none | `veil_strength` | 0.07 | 0.13 | 0.24 |
| `gaussian_noise` | sensor_exposure | preserving | none | `sigma` | 0.010 | 0.025 | 0.050 |
| `low_light` | sensor_exposure | preserving | none | `exposure` ↓ | 0.6 | 0.4 | 0.2 |
| `highlight_clipping_bloom` | sensor_exposure | preserving | none | `exposure` | 2.4 | 3.2 | 3.8 |
| `fixed_pattern_sensor_noise` | sensor_exposure | preserving | none | `column_sigma` | 0.002 | 0.008 | 0.012 |
| `vignetting_lens_shading` | sensor_exposure | preserving | none | `k` | 1.25 | 1.80 | 2.70 |
| `h264_crf_compression` | encoding_resolution | preserving | ffmpeg | `crf` | 30 | 35 | 40 |
| `moire_aliasing` | encoding_resolution | preserving | none | `sample_factor` ↓ | 0.70 | 0.60 | 0.50 |
| `spacing_resolution_loss` | encoding_resolution | preserving | gt_points | `spacing_fraction` ↓ | 0.8 | 0.7 | 0.6 |
| `packet_loss_concealment` | encoding_resolution | **altering** | ffmpeg, gt_points | `loss_rate` | 0.0625 | 0.125 | 0.25 |
| `rain_streaks` | environment | preserving | gt_points | `density` (streaks per square frame) | 300 | 600 | 900 |
| `snow_particles` | environment | preserving | gt_points | `density` (flakes per square frame) | 600 | 800 | 950 |
| `depth_aware_haze` | environment | preserving | depth | `beta` | 0.18 | 0.32 | 0.55 |
| `cast_shadow_illumination` | environment | preserving | none | `coverage` | 0.30 | 0.45 | 0.60 |
| `crowd_like_distractors` | scene_clutter | **altering** | gt_points | `count` | 5 | 15 | 30 |
| `object_distractors` | scene_clutter | **altering** | gt_points | `count_gt_frac` | 0.2 | 0.4 | 0.6 |
| `texture_distractors` | scene_clutter | preserving | gt_points | `coverage` | 0.15 | 0.28 | 0.45 |
| `structured_foreground_occlusion` | visibility_loss | **altering** | gt_points | `coverage` | 0.06 | 0.15 | 0.28 |
| `fov_loss_camera_reaim` | visibility_loss | **altering** | gt_points | `crop_fraction` | 0.15 | 0.30 | 0.45 |
| `viewpoint_perspective_shift` | visibility_loss | **altering** | gt_points | `top_compression` | 0.15 | 0.30 | 0.45 |

## 2. Parameter sets by family

### 2.1 `optics_motion`: lens and motion degradations

#### `camera_motion_blur`: Directional camera-motion blur

| parameter | light | moderate | high | meaning |
|---|---:|---:|---:|---|
| `length_spacing_frac` | 0.05 | 0.15 | 0.25 | streak length relative to the image's median head spacing |
| `angle_deg` | 90.0 | 90.0 | 90.0 | mean blur direction |
| `angle_jitter_deg` | 8.0 | 8.0 | 8.0 | per-image seeded deviation from that direction |
| `angle_clip_sigmas` | 2.5 | 2.5 | 2.5 | maximum angular deviation, in standard deviations |

#### `far_field_defocus`: Depth-dependent defocus

| parameter | light | moderate | high | meaning |
|---|---:|---:|---:|---|
| `max_blur_spacing_frac` | 0.05 | 0.10 | 0.15 | nominal σ at the most-defocused depth, in median-head-spacing units; bins use their centre depth, so the largest applied σ is 0.73–0.86 of it |
| `focus_depth` | 0.08 | 0.08 | 0.08 | focal plane, 0 = near, 1 = far |
| `layers` | 2 | 3 | 4 | number of composited depth bins |

#### `lens_contamination`: Lens droplets

| parameter | light | moderate | high | meaning |
|---|---:|---:|---:|---|
| `opacity_range` | [0.4, 0.95] | [0.4, 0.95] | [0.4, 0.95] | per-drop opacity range |
| `density` ↓ | 150.0 | 65.0 | 38.0 | drops per square frame |
| `radius_frac` | 0.01 | 0.02 | 0.03 | drop radius as a fraction of the frame |
| `radius_lognormal_sigma` | 0.3 | 0.4 | 0.5 | spread of the size distribution |
| `refraction_strength` | 0.3 | 0.4 | 0.5 | refraction magnitude |
| `blur_radius_frac` | 0.15 | 0.20 | 0.25 | defocus radius relative to drop radius |
| `brighten` | 0.03 | 0.05 | 0.07 | forward-scattered light |
| `edge_feather_frac` ↓ | 0.20 | 0.15 | 0.10 | drop-boundary softness |

GT: Remove heads under drops with opacity above 0.8.

#### `lens_flare_veiling`: Glare, veil and reflection ghosts

| parameter | light | moderate | high | meaning |
|---|---:|---:|---:|---|
| `core_gain` | 0.45 | 0.70 | 1.20 | brightness of the source bloom |
| `glow_radius_frac` | 0.12 | 0.17 | 0.24 | radius of the source bloom |
| `veil_strength` | 0.07 | 0.13 | 0.24 | strength of the inverse-square scatter tail |
| `veil_radius_frac` | 0.55 | 0.62 | 0.72 | spatial extent of the veil |
| `ghost_count` | 2 | 3 | 3 | reflection ghosts along the optical axis |
| `ghost_strength` | 0.010 | 0.018 | 0.028 | ghost brightness |
| `ghost_radius_frac` | 0.07 | 0.08 | 0.09 | ghost size |

### 2.2 `sensor_exposure`: sensor and exposure degradations

#### `gaussian_noise`: Gaussian read noise in linear light

| parameter | light | moderate | high | meaning |
|---|---:|---:|---:|---|
| `sigma` | 0.010 | 0.025 | 0.050 | noise standard deviation in linear-light units |

#### `low_light`: Reduced exposure with shot and read noise

| parameter | light | moderate | high | meaning |
|---|---:|---:|---:|---|
| `exposure` ↓ | 0.6 | 0.4 | 0.2 | exposure multiplier, adjusted for scene brightness |
| `photons` ↓ | 1200.0 | 800.0 | 400.0 | full-scale photon count (shot-noise level) |
| `read_sigma` | 0.0015 | 0.0030 | 0.0060 | read-noise standard deviation, linear units |

#### `highlight_clipping_bloom`: Highlight clipping and bloom

| parameter | light | moderate | high | meaning |
|---|---:|---:|---:|---|
| `exposure` | 2.4 | 3.2 | 3.8 | linear gain (> 1) |
| `bloom_sigma_frac` | 0.010 | 0.018 | 0.026 | halo radius as a fraction of the short side |
| `bloom_gain` | 0.35 | 0.55 | 0.72 | fraction of the clipped excess scattered back |

#### `fixed_pattern_sensor_noise`: Persistent sensor defects and row/column noise

| parameter | light | moderate | high | meaning |
|---|---:|---:|---:|---|
| `stuck_fraction` | 0.00005 | 0.0002 | 0.0003 | fraction of photosites forced to 0 or 255 |
| `column_sigma` | 0.002 | 0.008 | 0.012 | per-column offset standard deviation |
| `row_sigma` | 0.002 | 0.008 | 0.012 | per-row offset standard deviation |
| `temporal_sigma` | 0.002 | 0.008 | 0.012 | per-image temporal noise |

Sensor patterns use `rng_sensor`; temporal noise uses `rng_image`. Stuck pixels are set to 0 or 255 after
sRGB encoding.

#### `vignetting_lens_shading`: Radial illumination falloff

| parameter | light | moderate | high | meaning |
|---|---:|---:|---:|---|
| `k` | 1.25 | 1.80 | 2.70 | falloff coefficient in `max(1 − k·r², 0.05)` |

Minimum transmission is 0.05. These strengths exceed typical real-lens shading.

### 2.3 `encoding_resolution`: encoding and resolution degradations

#### `h264_crf_compression`: H.264 compression

| parameter | light | moderate | high | meaning |
|---|---:|---:|---:|---|
| `codec` | libx264 | libx264 | libx264 | encoder |
| `crf` | 30 | 35 | 40 | rate factor; higher is lower quality |

Achieved bits per pixel are recorded per image.

#### `moire_aliasing`: Sampling aliasing

| parameter | light | moderate | high | meaning |
|---|---:|---:|---:|---|
| `sample_factor` ↓ | 0.70 | 0.60 | 0.50 | resampling ratio |
| `cfa` | true | true | true | colour-filter-array sampling |

Mean pixel change does not always increase with severity; the generator reports this as a note.

#### `spacing_resolution_loss`: Resolution loss based on head spacing

| parameter | light | moderate | high | meaning |
|---|---:|---:|---:|---|
| `spacing_fraction` ↓ | 0.8 | 0.7 | 0.6 | target spacing relative to the dataset's original median head spacing |

Median head spacing (`NATIVE_MEDIAN_SPACING_PX`, measured on the training splits): SHA 17.23 px, SHB
33.11, JHU 32.99, QNRF 34.78. For other datasets the median is measured on the corrupted split. Images
already denser than the target receive the relative reduction without upsampling.

#### `packet_loss_concealment`: H.264 slice loss and concealment

| parameter | light | moderate | high | meaning |
|---|---:|---:|---:|---|
| `slices` | 32 | 32 | 32 | VCL slices per frame |
| `crf` | 28 | 28 | 28 | encoding quality before slice loss |
| `loss_rate` | 0.0625 | 0.125 | 0.25 | fraction of slices dropped |
| `max_retries` | 6 | 6 | 6 | retries with a new selection of dropped slices if decoding fails |

GT: Remove heads at pixels changed by more than 6 gray levels relative to the loss-free decode.

### 2.4 `environment`: weather and illumination

#### `rain_streaks`: Rain streaks and scattering veil

| parameter | light | moderate | high | meaning |
|---|---:|---:|---:|---|
| `density` | 300.0 | 600.0 | 900.0 | streaks per square frame, scaled by the aspect ratio |
| `length_frac` | 0.010 | 0.015 | 0.020 | streak length as a fraction of the short side |
| `width_frac` | 0.0013 | 0.0013 | 0.0013 | streak width |
| `angle_deg` | 14.0 | 14.0 | 14.0 | fall direction |
| `angle_jitter_deg` | 5.0 | 6.0 | 7.0 | angular spread |
| `opacity` | 0.25 | 0.30 | 0.35 | per-streak opacity |
| `veil` | 0.03 | 0.06 | 0.09 | frame-wide atmospheric veil |

#### `snow_particles`: Near and far snowflakes

| parameter | light | moderate | high | meaning |
|---|---:|---:|---:|---|
| `density` | 600.0 | 800.0 | 950.0 | flakes per square frame, scaled by the aspect ratio; the layer is combined with its 180° rotation |
| `size_frac` | 0.00175 | 0.00220 | 0.00260 | flake size |
| `streak_ratio` | 1.5 | 1.6 | 1.7 | motion streak relative to flake size |
| `angle_deg` | 10.0 | 10.0 | 10.0 | fall direction |
| `angle_jitter_deg` | 14.0 | 15.0 | 16.0 | angular spread |
| `near_fraction` | 0.030 | 0.045 | 0.055 | fraction of near flakes |
| `near_size_ratio` | 3.5 | 4.0 | 4.5 | near-to-far flake size ratio |
| `near_blur_frac` | 0.004 | 0.007 | 0.010 | near-flake defocus |
| `veil` | 0.08 | 0.13 | 0.17 | frame-wide atmospheric veil |

#### `depth_aware_haze`: Depth-dependent atmospheric haze

| parameter | light | moderate | high | meaning |
|---|---:|---:|---:|---|
| `beta` | 0.18 | 0.32 | 0.55 | extinction per unit normalized depth |
| `airlight` | 0.80 | 0.85 | 0.90 | linear radiance of the airlight |

#### `cast_shadow_illumination`: Cast shadows

| parameter | light | moderate | high | meaning |
|---|---:|---:|---:|---|
| `coverage` | 0.30 | 0.45 | 0.60 | fraction of the frame in shadow |
| `umbra_transmission` ↓ | 0.62 | 0.38 | 0.20 | fraction of light retained in the fully shaded region |
| `penumbra_frac` ↓ | 0.10 | 0.07 | 0.05 | width of the soft shadow edge |
| `boundary_wave_amp` | 0.10 | 0.13 | 0.16 | boundary irregularity |

### 2.5 `scene_clutter`: people, objects and background textures

#### `crowd_like_distractors`: Depicted people

| parameter | light | moderate | high | meaning |
|---|---:|---:|---:|---|
| `count` | 5 | 15 | 30 | base decoy count, scaled by image size |
| `height_spacing_frac` ↓ | 3.5 | 3.0 | 2.5 | decoy height relative to head spacing |
| `cue_mix` | poster, led_screen, reflection, pictogram | (same) | (same) | decoy types |

The requested count is `round(count * clip(sqrt(megapixels / 0.7), 1, 2.5))`. Placement constraints can
reduce the number actually added.

GT: Do not count decoys; remove original heads covered by a decoy.

#### `object_distractors`: Head-sized non-human objects

| parameter | light | moderate | high | meaning |
|---|---:|---:|---:|---|
| `count_gt_frac` | 0.2 | 0.4 | 0.6 | decoy count as a fraction of the annotated head count, limited to 20–600 |
| `size_head_frac` | [0.8, 2.0] | [0.8, 2.0] | [0.8, 2.0] | decoy diameter relative to head diameter, drawn per decoy |
| `cue_mix` | balloon, produce, helmet, ball | (same) | (same) | decoy types |

GT: Do not count objects; remove original heads covered by an object.

#### `texture_distractors`: Head-scale background texture

| parameter | light | moderate | high | meaning |
|---|---:|---:|---:|---|
| `coverage` | 0.15 | 0.28 | 0.45 | fraction of the frame covered |
| `pitch_head_frac` | 1.0 | 1.0 | 1.0 | blob pitch relative to head diameter |
| `cue_mix` | gravel, foliage, window_grid | (same) | (same) | texture types |

### 2.6 `visibility_loss`: partial scene visibility

#### `structured_foreground_occlusion`: Foreground occluders

| parameter | light | moderate | high | meaning |
|---|---:|---:|---:|---|
| `coverage` | 0.06 | 0.15 | 0.28 | achieved occluded fraction of the frame |
| `occluder_darkness` ↓ | 0.18 | 0.16 | 0.14 | occluder albedo relative to the frame's median luminance |
| `occluder_mix` | pole, railing, banner | + umbrella | + umbrella, spectator | occluder shapes |

GT: Remove covered heads.

#### `fov_loss_camera_reaim`: Field-of-view loss

| parameter | light | moderate | high | meaning |
|---|---:|---:|---:|---|
| `crop_fraction` | 0.15 | 0.30 | 0.45 | fraction lost from one edge; `corner` splits it between left and top (0.28 and 0.40 of the area) |
| `direction` | right | corner | corner | re-aiming direction |

#### `viewpoint_perspective_shift`: Camera-pitch keystone warp

| parameter | light | moderate | high | meaning |
|---|---:|---:|---:|---|
| `top_compression` | 0.15 | 0.30 | 0.45 | inward displacement of the top edge |

GT: Warp all points using the keystone homography.
