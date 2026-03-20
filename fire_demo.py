#!/usr/bin/env python3
"""
Echtzeit-Feuer-Simulation mit NVIDIA Warp (optimiert)
Standalone-Demo — kein Blender noetig.

Installation:
    pip install warp-lang numpy pygame

Steuerung:
    ESC        — Beenden
    R          — Reset
    Up/Down    — Grid-Aufloesung aendern
    H          — Half-Res Rendering toggle
"""

import warp as wp
import numpy as np
import time
import sys

wp.init()

if not wp.is_cuda_available():
    print("FEHLER: Keine CUDA-GPU gefunden. Warp braucht eine NVIDIA GPU.")
    sys.exit(1)


# ─── Warp Kernels ──────────────────────────────────────────────


@wp.func
def curl_noise_3d(pos: wp.vec4, eps: float):
    """Curl of a 3D noise potential field. Divergence-free turbulence."""
    dx = wp.vec4(eps, 0.0, 0.0, 0.0)
    dy = wp.vec4(0.0, eps, 0.0, 0.0)
    dz = wp.vec4(0.0, 0.0, eps, 0.0)

    # Curl = nabla x Psi, Psi sampled with different seeds
    cx = (wp.noise(wp.uint32(2), pos + dy) - wp.noise(wp.uint32(2), pos - dy)
        - wp.noise(wp.uint32(1), pos + dz) + wp.noise(wp.uint32(1), pos - dz)) / (2.0 * eps)
    cy = (wp.noise(wp.uint32(0), pos + dz) - wp.noise(wp.uint32(0), pos - dz)
        - wp.noise(wp.uint32(2), pos + dx) + wp.noise(wp.uint32(2), pos - dx)) / (2.0 * eps)
    cz = (wp.noise(wp.uint32(1), pos + dx) - wp.noise(wp.uint32(1), pos - dx)
        - wp.noise(wp.uint32(0), pos + dy) + wp.noise(wp.uint32(0), pos - dy)) / (2.0 * eps)

    return wp.vec3(cx, cy, cz)


@wp.kernel
def sim_step_fused(
    temperature: wp.array(dtype=float),
    density: wp.array(dtype=float),
    vel_x: wp.array(dtype=float),
    vel_y: wp.array(dtype=float),
    vel_z: wp.array(dtype=float),
    sim_occupancy: wp.array(dtype=int),
    n: int,
    frame: int,
    buoyancy: float,
    turbulence: float,
    dt: float,
    temp_decay: float,
    dens_decay: float,
    vel_decay: float,
    block_size: int,
):
    """Fused: emit + forces + dissipate in one kernel (saves launch overhead)."""
    tid = wp.tid()
    n2 = n * n
    i = tid // n2
    j = (tid // n) % n
    k = tid % n

    # Sparse: skip inactive blocks (except emission zone)
    if j >= 8 and block_is_active(sim_occupancy, i, j, k, n, block_size) == 0:
        return

    t = temperature[tid]
    d = density[tid]
    vx = vel_x[tid]
    vy = vel_y[tid]
    vz = vel_z[tid]

    # --- Emit fire at source (bottom center) ---
    if j < 8:
        cx = float(n) / 2.0
        cz = float(n) / 2.0
        radius = float(n) * 0.18
        offset_x = wp.sin(float(frame) * 0.05) * float(n) * 0.05
        offset_z = wp.cos(float(frame) * 0.07) * float(n) * 0.05
        dx = float(i) - (cx + offset_x)
        dz = float(k) - (cz + offset_z)
        dist = wp.sqrt(dx * dx + dz * dz)
        if dist < radius:
            falloff = 1.0 - dist / radius
            falloff = falloff * falloff
            t = t + 0.8 * falloff
            d = d + 0.4 * falloff
            vy = vy + 0.3 * falloff

    # --- Forces: buoyancy + turbulence ---
    vy = vy + t * buoyancy * dt

    if t > 0.05:
        # Curl noise: divergence-free turbulence from Perlin noise potential
        noise_scale = 0.06
        time_val = float(frame) * 0.015
        pos = wp.vec4(
            float(i) * noise_scale,
            float(j) * noise_scale,
            float(k) * noise_scale,
            time_val,
        )
        curl = curl_noise_3d(pos, 0.5)
        strength = turbulence * t * dt
        vx = vx + curl[0] * strength
        vy = vy + curl[1] * strength * 0.3  # less vertical turbulence
        vz = vz + curl[2] * strength

    # --- Dissipate ---
    temperature[tid] = wp.max(t * temp_decay, 0.0)
    density[tid] = wp.max(d * dens_decay, 0.0)
    vel_x[tid] = vx * vel_decay
    vel_y[tid] = vy * vel_decay
    vel_z[tid] = vz * vel_decay


@wp.kernel
def advect_field(
    field_src: wp.array(dtype=float),
    field_dst: wp.array(dtype=float),
    vel_x: wp.array(dtype=float),
    vel_y: wp.array(dtype=float),
    vel_z: wp.array(dtype=float),
    n: int,
    dt: float,
):
    tid = wp.tid()
    n2 = n * n
    i = tid // n2
    j = (tid // n) % n
    k = tid % n

    if i < 1 or i >= n - 1 or j < 1 or j >= n - 1 or k < 1 or k >= n - 1:
        field_dst[tid] = 0.0
        return

    px = float(i) - vel_x[tid] * dt
    py = float(j) - vel_y[tid] * dt
    pz = float(k) - vel_z[tid] * dt

    px = wp.clamp(px, 1.0, float(n - 2))
    py = wp.clamp(py, 1.0, float(n - 2))
    pz = wp.clamp(pz, 1.0, float(n - 2))

    i0 = int(px)
    j0 = int(py)
    k0 = int(pz)
    i1 = i0 + 1
    j1 = j0 + 1
    k1 = k0 + 1

    fx = px - float(i0)
    fy = py - float(j0)
    fz = pz - float(k0)

    v = (
        field_src[i0 * n2 + j0 * n + k0] * (1.0 - fx) * (1.0 - fy) * (1.0 - fz)
        + field_src[i0 * n2 + j0 * n + k1] * (1.0 - fx) * (1.0 - fy) * fz
        + field_src[i0 * n2 + j1 * n + k0] * (1.0 - fx) * fy * (1.0 - fz)
        + field_src[i0 * n2 + j1 * n + k1] * (1.0 - fx) * fy * fz
        + field_src[i1 * n2 + j0 * n + k0] * fx * (1.0 - fy) * (1.0 - fz)
        + field_src[i1 * n2 + j0 * n + k1] * fx * (1.0 - fy) * fz
        + field_src[i1 * n2 + j1 * n + k0] * fx * fy * (1.0 - fz)
        + field_src[i1 * n2 + j1 * n + k1] * fx * fy * fz
    )

    field_dst[tid] = v


@wp.kernel
def advect_velocity_fused(
    vel_x_src: wp.array(dtype=float),
    vel_y_src: wp.array(dtype=float),
    vel_z_src: wp.array(dtype=float),
    vel_x_dst: wp.array(dtype=float),
    vel_y_dst: wp.array(dtype=float),
    vel_z_dst: wp.array(dtype=float),
    vel_x: wp.array(dtype=float),
    vel_y: wp.array(dtype=float),
    vel_z: wp.array(dtype=float),
    n: int,
    dt: float,
):
    tid = wp.tid()
    n2 = n * n
    i = tid // n2
    j = (tid // n) % n
    k = tid % n

    if i < 1 or i >= n - 1 or j < 1 or j >= n - 1 or k < 1 or k >= n - 1:
        vel_x_dst[tid] = 0.0
        vel_y_dst[tid] = 0.0
        vel_z_dst[tid] = 0.0
        return

    # Single backtrace computation shared by all three components
    px = float(i) - vel_x[tid] * dt
    py = float(j) - vel_y[tid] * dt
    pz = float(k) - vel_z[tid] * dt

    px = wp.clamp(px, 1.0, float(n - 2))
    py = wp.clamp(py, 1.0, float(n - 2))
    pz = wp.clamp(pz, 1.0, float(n - 2))

    i0 = int(px)
    j0 = int(py)
    k0 = int(pz)
    i1 = i0 + 1
    j1 = j0 + 1
    k1 = k0 + 1

    fx = px - float(i0)
    fy = py - float(j0)
    fz = pz - float(k0)

    # Precompute weight combinations
    w000 = (1.0 - fx) * (1.0 - fy) * (1.0 - fz)
    w001 = (1.0 - fx) * (1.0 - fy) * fz
    w010 = (1.0 - fx) * fy * (1.0 - fz)
    w011 = (1.0 - fx) * fy * fz
    w100 = fx * (1.0 - fy) * (1.0 - fz)
    w101 = fx * (1.0 - fy) * fz
    w110 = fx * fy * (1.0 - fz)
    w111 = fx * fy * fz

    # Precompute index offsets
    idx000 = i0 * n2 + j0 * n + k0
    idx001 = i0 * n2 + j0 * n + k1
    idx010 = i0 * n2 + j1 * n + k0
    idx011 = i0 * n2 + j1 * n + k1
    idx100 = i1 * n2 + j0 * n + k0
    idx101 = i1 * n2 + j0 * n + k1
    idx110 = i1 * n2 + j1 * n + k0
    idx111 = i1 * n2 + j1 * n + k1

    # Interpolate all three velocity components with shared weights and indices
    vel_x_dst[tid] = (
        vel_x_src[idx000] * w000 + vel_x_src[idx001] * w001
        + vel_x_src[idx010] * w010 + vel_x_src[idx011] * w011
        + vel_x_src[idx100] * w100 + vel_x_src[idx101] * w101
        + vel_x_src[idx110] * w110 + vel_x_src[idx111] * w111
    )
    vel_y_dst[tid] = (
        vel_y_src[idx000] * w000 + vel_y_src[idx001] * w001
        + vel_y_src[idx010] * w010 + vel_y_src[idx011] * w011
        + vel_y_src[idx100] * w100 + vel_y_src[idx101] * w101
        + vel_y_src[idx110] * w110 + vel_y_src[idx111] * w111
    )
    vel_z_dst[tid] = (
        vel_z_src[idx000] * w000 + vel_z_src[idx001] * w001
        + vel_z_src[idx010] * w010 + vel_z_src[idx011] * w011
        + vel_z_src[idx100] * w100 + vel_z_src[idx101] * w101
        + vel_z_src[idx110] * w110 + vel_z_src[idx111] * w111
    )


@wp.kernel
def compute_vorticity(
    vel_x: wp.array(dtype=float),
    vel_y: wp.array(dtype=float),
    vel_z: wp.array(dtype=float),
    omega_x: wp.array(dtype=float),
    omega_y: wp.array(dtype=float),
    omega_z: wp.array(dtype=float),
    omega_mag: wp.array(dtype=float),
    n: int,
):
    """Pass 1: compute curl of velocity and its magnitude."""
    tid = wp.tid()
    n2 = n * n
    i = tid // n2
    j = (tid // n) % n
    k = tid % n

    if i < 1 or i >= n - 1 or j < 1 or j >= n - 1 or k < 1 or k >= n - 1:
        omega_x[tid] = 0.0
        omega_y[tid] = 0.0
        omega_z[tid] = 0.0
        omega_mag[tid] = 0.0
        return

    wx = (vel_z[i*n2+(j+1)*n+k] - vel_z[i*n2+(j-1)*n+k]
        - vel_y[i*n2+j*n+(k+1)] + vel_y[i*n2+j*n+(k-1)]) * 0.5
    wy = (vel_x[i*n2+j*n+(k+1)] - vel_x[i*n2+j*n+(k-1)]
        - vel_z[(i+1)*n2+j*n+k] + vel_z[(i-1)*n2+j*n+k]) * 0.5
    wz = (vel_y[(i+1)*n2+j*n+k] - vel_y[(i-1)*n2+j*n+k]
        - vel_x[i*n2+(j+1)*n+k] + vel_x[i*n2+(j-1)*n+k]) * 0.5

    omega_x[tid] = wx
    omega_y[tid] = wy
    omega_z[tid] = wz
    omega_mag[tid] = wp.sqrt(wx * wx + wy * wy + wz * wz)


@wp.kernel
def apply_vorticity_confinement(
    vel_x: wp.array(dtype=float),
    vel_y: wp.array(dtype=float),
    vel_z: wp.array(dtype=float),
    omega_x: wp.array(dtype=float),
    omega_y: wp.array(dtype=float),
    omega_z: wp.array(dtype=float),
    omega_mag: wp.array(dtype=float),
    n: int,
    epsilon: float,
    dt: float,
):
    """Pass 2: gradient of |omega|, then cross product for confinement force."""
    tid = wp.tid()
    n2 = n * n
    i = tid // n2
    j = (tid // n) % n
    k = tid % n

    if i < 2 or i >= n - 2 or j < 2 or j >= n - 2 or k < 2 or k >= n - 2:
        return

    # Gradient of vorticity magnitude
    gx = (omega_mag[(i+1)*n2+j*n+k] - omega_mag[(i-1)*n2+j*n+k]) * 0.5
    gy = (omega_mag[i*n2+(j+1)*n+k] - omega_mag[i*n2+(j-1)*n+k]) * 0.5
    gz = (omega_mag[i*n2+j*n+(k+1)] - omega_mag[i*n2+j*n+(k-1)]) * 0.5

    glen = wp.sqrt(gx * gx + gy * gy + gz * gz) + 1.0e-6
    nx = gx / glen
    ny = gy / glen
    nz = gz / glen

    wx = omega_x[tid]
    wy = omega_y[tid]
    wz = omega_z[tid]

    # Force = epsilon * (N x omega)
    vel_x[tid] = vel_x[tid] + epsilon * (ny * wz - nz * wy) * dt
    vel_y[tid] = vel_y[tid] + epsilon * (nz * wx - nx * wz) * dt
    vel_z[tid] = vel_z[tid] + epsilon * (nx * wy - ny * wx) * dt


@wp.kernel
def advect_maccormack_correct(
    field_orig: wp.array(dtype=float),
    field_fwd: wp.array(dtype=float),
    field_bwd: wp.array(dtype=float),
    field_out: wp.array(dtype=float),
    vel_x: wp.array(dtype=float),
    vel_y: wp.array(dtype=float),
    vel_z: wp.array(dtype=float),
    n: int,
    dt: float,
):
    """MacCormack correction: orig + 0.5*(orig - backward(forward(orig))), clamped."""
    tid = wp.tid()
    n2 = n * n
    i = tid // n2
    j = (tid // n) % n
    k = tid % n

    if i < 1 or i >= n - 1 or j < 1 or j >= n - 1 or k < 1 or k >= n - 1:
        field_out[tid] = 0.0
        return

    corrected = field_fwd[tid] + 0.5 * (field_orig[tid] - field_bwd[tid])

    # Clamp to min/max of neighbors to prevent oscillation
    px = float(i) - vel_x[tid] * dt
    py = float(j) - vel_y[tid] * dt
    pz = float(k) - vel_z[tid] * dt
    i0 = wp.clamp(int(px), 1, n - 2)
    j0 = wp.clamp(int(py), 1, n - 2)
    k0 = wp.clamp(int(pz), 1, n - 2)

    # Min/max of the 8 corners used in trilinear interpolation
    v000 = field_orig[i0 * n2 + j0 * n + k0]
    v001 = field_orig[i0 * n2 + j0 * n + (k0 + 1)]
    v010 = field_orig[i0 * n2 + (j0 + 1) * n + k0]
    v011 = field_orig[i0 * n2 + (j0 + 1) * n + (k0 + 1)]
    v100 = field_orig[(i0 + 1) * n2 + j0 * n + k0]
    v101 = field_orig[(i0 + 1) * n2 + j0 * n + (k0 + 1)]
    v110 = field_orig[(i0 + 1) * n2 + (j0 + 1) * n + k0]
    v111 = field_orig[(i0 + 1) * n2 + (j0 + 1) * n + (k0 + 1)]

    lo = wp.min(wp.min(wp.min(v000, v001), wp.min(v010, v011)),
                wp.min(wp.min(v100, v101), wp.min(v110, v111)))
    hi = wp.max(wp.max(wp.max(v000, v001), wp.max(v010, v011)),
                wp.max(wp.max(v100, v101), wp.max(v110, v111)))

    field_out[tid] = wp.clamp(corrected, lo, hi)


@wp.kernel
def advect_backward(
    field_src: wp.array(dtype=float),
    field_dst: wp.array(dtype=float),
    vel_x: wp.array(dtype=float),
    vel_y: wp.array(dtype=float),
    vel_z: wp.array(dtype=float),
    n: int,
    dt: float,
):
    """Backward advection (trace forward in time) for MacCormack."""
    tid = wp.tid()
    n2 = n * n
    i = tid // n2
    j = (tid // n) % n
    k = tid % n

    if i < 1 or i >= n - 1 or j < 1 or j >= n - 1 or k < 1 or k >= n - 1:
        field_dst[tid] = 0.0
        return

    # Trace forward (positive dt)
    px = float(i) + vel_x[tid] * dt
    py = float(j) + vel_y[tid] * dt
    pz = float(k) + vel_z[tid] * dt

    px = wp.clamp(px, 1.0, float(n - 2))
    py = wp.clamp(py, 1.0, float(n - 2))
    pz = wp.clamp(pz, 1.0, float(n - 2))

    i0 = int(px)
    j0 = int(py)
    k0 = int(pz)
    i1 = i0 + 1
    j1 = j0 + 1
    k1 = k0 + 1

    fx = px - float(i0)
    fy = py - float(j0)
    fz = pz - float(k0)

    v = (
        field_src[i0 * n2 + j0 * n + k0] * (1.0 - fx) * (1.0 - fy) * (1.0 - fz)
        + field_src[i0 * n2 + j0 * n + k1] * (1.0 - fx) * (1.0 - fy) * fz
        + field_src[i0 * n2 + j1 * n + k0] * (1.0 - fx) * fy * (1.0 - fz)
        + field_src[i0 * n2 + j1 * n + k1] * (1.0 - fx) * fy * fz
        + field_src[i1 * n2 + j0 * n + k0] * fx * (1.0 - fy) * (1.0 - fz)
        + field_src[i1 * n2 + j0 * n + k1] * fx * (1.0 - fy) * fz
        + field_src[i1 * n2 + j1 * n + k0] * fx * fy * (1.0 - fz)
        + field_src[i1 * n2 + j1 * n + k1] * fx * fy * fz
    )

    field_dst[tid] = v


@wp.kernel
def compute_divergence(
    vel_x: wp.array(dtype=float),
    vel_y: wp.array(dtype=float),
    vel_z: wp.array(dtype=float),
    div: wp.array(dtype=float),
    n: int,
):
    tid = wp.tid()
    n2 = n * n
    i = tid // n2
    j = (tid // n) % n
    k = tid % n

    if i < 1 or i >= n - 1 or j < 1 or j >= n - 1 or k < 1 or k >= n - 1:
        div[tid] = 0.0
        return

    div[tid] = -0.5 * (
        vel_x[(i+1)*n2+j*n+k] - vel_x[(i-1)*n2+j*n+k]
        + vel_y[i*n2+(j+1)*n+k] - vel_y[i*n2+(j-1)*n+k]
        + vel_z[i*n2+j*n+(k+1)] - vel_z[i*n2+j*n+(k-1)]
    )


@wp.kernel
def jacobi_step(
    pressure_in: wp.array(dtype=float),
    pressure_out: wp.array(dtype=float),
    div: wp.array(dtype=float),
    n: int,
):
    tid = wp.tid()
    n2 = n * n
    i = tid // n2
    j = (tid // n) % n
    k = tid % n

    if i < 1 or i >= n - 1 or j < 1 or j >= n - 1 or k < 1 or k >= n - 1:
        pressure_out[tid] = 0.0
        return

    pressure_out[tid] = (
        pressure_in[(i-1)*n2+j*n+k] + pressure_in[(i+1)*n2+j*n+k]
        + pressure_in[i*n2+(j-1)*n+k] + pressure_in[i*n2+(j+1)*n+k]
        + pressure_in[i*n2+j*n+(k-1)] + pressure_in[i*n2+j*n+(k+1)]
        + div[tid]
    ) / 6.0


@wp.kernel
def subtract_pressure_gradient(
    vel_x: wp.array(dtype=float),
    vel_y: wp.array(dtype=float),
    vel_z: wp.array(dtype=float),
    pressure: wp.array(dtype=float),
    n: int,
):
    tid = wp.tid()
    n2 = n * n
    i = tid // n2
    j = (tid // n) % n
    k = tid % n

    if i < 1 or i >= n - 1 or j < 1 or j >= n - 1 or k < 1 or k >= n - 1:
        return

    vel_x[tid] = vel_x[tid] - 0.5 * (pressure[(i+1)*n2+j*n+k] - pressure[(i-1)*n2+j*n+k])
    vel_y[tid] = vel_y[tid] - 0.5 * (pressure[i*n2+(j+1)*n+k] - pressure[i*n2+(j-1)*n+k])
    vel_z[tid] = vel_z[tid] - 0.5 * (pressure[i*n2+j*n+(k+1)] - pressure[i*n2+j*n+(k-1)])


@wp.kernel
def diffuse_field(
    field_in: wp.array(dtype=float),
    field_out: wp.array(dtype=float),
    n: int,
    rate: float,
):
    tid = wp.tid()
    n2 = n * n
    i = tid // n2
    j = (tid // n) % n
    k = tid % n

    if i < 1 or i >= n - 1 or j < 1 or j >= n - 1 or k < 1 or k >= n - 1:
        field_out[tid] = 0.0
        return

    center = field_in[tid]
    avg = (
        field_in[(i - 1) * n2 + j * n + k]
        + field_in[(i + 1) * n2 + j * n + k]
        + field_in[i * n2 + (j - 1) * n + k]
        + field_in[i * n2 + (j + 1) * n + k]
        + field_in[i * n2 + j * n + (k - 1)]
        + field_in[i * n2 + j * n + (k + 1)]
    ) / 6.0

    field_out[tid] = center + rate * (avg - center)


@wp.kernel
def compute_light_volume(
    density: wp.array(dtype=float),
    light_vol: wp.array(dtype=wp.float16),
    n: int,
    absorption: float,
):
    """Precompute light transmittance with multi-scattering approximation.
    Uses Frostbite-inspired approach: scattered light adds back energy."""
    tid = wp.tid()
    i = tid // n
    k = tid % n
    n2 = n * n

    accumulated = float(0.0)
    scatter_accum = float(0.0)
    for j_step in range(n):
        j = n - 1 - j_step  # top to bottom
        idx = i * n2 + j * n + k
        d = density[idx]
        transmittance = wp.exp(-accumulated * absorption)
        # Multi-scattering: each scattering event redirects some light
        # back into the medium (approximated as exponential bounce)
        scatter_contrib = (1.0 - wp.exp(-d * absorption * 0.5)) * transmittance * 0.45
        scatter_accum = scatter_accum + scatter_contrib
        # Combined: direct transmittance + accumulated scattered light
        light_vol[idx] = wp.float16(wp.clamp(transmittance + scatter_accum * 0.8, 0.0, 1.0))
        accumulated = accumulated + d


@wp.kernel
def render_fire(
    temperature: wp.array(dtype=float),
    density: wp.array(dtype=float),
    light_vol: wp.array(dtype=wp.float16),
    occupancy: wp.array(dtype=int),
    image: wp.array(dtype=wp.vec3),
    n: int,
    img_w: int,
    img_h: int,
    block_size: int,
):
    pid = wp.tid()
    px = pid % img_w
    py = pid // img_w

    if py >= img_h:
        return

    gi = int(float(px) * float(n) / float(img_w))
    gj = int(float(img_h - 1 - py) * float(n) / float(img_h))
    gi = wp.clamp(gi, 0, n - 1)
    gj = wp.clamp(gj, 0, n - 1)

    n2 = n * n
    r = float(0.0)
    g = float(0.0)
    b = float(0.0)
    alpha = float(0.0)
    step = float(n) * 0.008

    # Henyey-Greenstein phase function (loop-invariant — compute once)
    g_param = 0.35  # forward scattering bias
    cos_theta = 0.36  # Z-component of light direction (fixed for Z-view)
    g2 = g_param * g_param
    denom = 1.0 + g2 - 2.0 * g_param * cos_theta
    phase = (1.0 - g2) / (4.0 * 3.14159 * denom * wp.sqrt(denom))
    phase_norm = phase * 4.0  # normalize so backlit smoke glows

    occ_dim = n // block_size
    occ_bi = gi // block_size
    occ_bj = gj // block_size

    k = int(0)
    while k < n:
        if alpha > 0.98:
            break

        # Block-level occupancy skip
        bk = k // block_size
        occ_idx = occ_bi * occ_dim * occ_dim + occ_bj * occ_dim + bk
        if occupancy[occ_idx] == 0:
            # Skip to next block boundary
            k = (bk + 1) * block_size
            continue

        idx = gi * n2 + gj * n + k
        t = temperature[idx]
        d = density[idx]

        if t < 0.01 and d < 0.01:
            k = k + 1
            continue

        # Precomputed light transmittance (replaces 16-step shadow loop)
        light_atten = float(light_vol[idx])

        # Fire color: blackbody (self-emitting)
        cr = wp.clamp(t * 5.0, 0.0, 1.0)
        cg = wp.clamp(t * 2.5 - 0.2, 0.0, 1.0)
        cb = wp.clamp(t * 1.2 - 0.5, 0.0, 1.0)
        emit = wp.clamp(t * 2.0, 0.0, 1.0)

        # Smoke: lit by directional light + ambient + fire glow (phase_norm precomputed above)
        ambient = 0.025
        smoke_bright = ambient + 0.1 * light_atten * phase_norm
        fire_illum = wp.clamp(t * 0.3, 0.0, 0.15)
        smoke_r = smoke_bright + fire_illum * 1.0
        smoke_g = smoke_bright + fire_illum * 0.4
        smoke_b = smoke_bright + fire_illum * 0.15

        fire_a = wp.clamp(t * 0.6, 0.0, 1.0) * step
        smoke_a = wp.clamp(d * 0.2, 0.0, 0.4) * step

        sa = fire_a + smoke_a
        if sa > 0.0:
            mix = fire_a / (fire_a + smoke_a + 0.001)
            fr = cr * emit * mix + smoke_r * (1.0 - mix)
            fg = cg * emit * mix + smoke_g * (1.0 - mix)
            fb = cb * emit * mix + smoke_b * (1.0 - mix)

            contrib = sa * (1.0 - alpha)
            r = r + fr * contrib
            g = g + fg * contrib
            b = b + fb * contrib
            alpha = alpha + contrib

        k = k + 1

    image[pid] = wp.vec3(
        wp.clamp(r, 0.0, 1.0),
        wp.clamp(g, 0.0, 1.0),
        wp.clamp(b, 0.0, 1.0),
    )


@wp.kernel
def build_occupancy(
    temperature: wp.array(dtype=float),
    density: wp.array(dtype=float),
    occupancy: wp.array(dtype=int),
    n: int,
    block_size: int,
):
    """Mark 8x8x8 blocks as occupied if any voxel has content."""
    tid = wp.tid()
    blocks_per_dim = n // block_size
    b2 = blocks_per_dim * blocks_per_dim
    bi = tid // b2
    bj = (tid // blocks_per_dim) % blocks_per_dim
    bk = tid % blocks_per_dim

    n2 = n * n
    occupied = int(0)
    for li in range(block_size):
        if occupied == 1:
            break
        for lj in range(block_size):
            if occupied == 1:
                break
            for lk in range(block_size):
                gi = bi * block_size + li
                gj = bj * block_size + lj
                gk = bk * block_size + lk
                if gi < n and gj < n and gk < n:
                    idx = gi * n2 + gj * n + gk
                    if temperature[idx] > 0.01 or density[idx] > 0.01:
                        occupied = 1

    occupancy[tid] = occupied


@wp.kernel
def build_occupancy_dilated(
    temperature: wp.array(dtype=float),
    density: wp.array(dtype=float),
    occupancy: wp.array(dtype=int),
    n: int,
    block_size: int,
):
    """Build occupancy with 1-block dilation for sim kernels (neighbor access)."""
    tid = wp.tid()
    blocks_per_dim = n // block_size
    b2 = blocks_per_dim * blocks_per_dim
    bi = tid // b2
    bj = (tid // blocks_per_dim) % blocks_per_dim
    bk = tid % blocks_per_dim

    n2 = n * n
    # Check this block AND its 26 neighbors (3x3x3 neighborhood)
    found = int(0)
    for di in range(3):
        if found == 1:
            break
        for dj in range(3):
            if found == 1:
                break
            for dk in range(3):
                if found == 1:
                    break
                ni = bi + di - 1
                nj = bj + dj - 1
                nk = bk + dk - 1
                if ni < 0 or ni >= blocks_per_dim or nj < 0 or nj >= blocks_per_dim or nk < 0 or nk >= blocks_per_dim:
                    continue
                # Iterate all voxels in neighbor block, early-exit on first hit
                for li in range(block_size):
                    if found == 1:
                        break
                    for lj in range(block_size):
                        if found == 1:
                            break
                        for lk in range(block_size):
                            if found == 1:
                                break
                            gi = ni * block_size + li
                            gj = nj * block_size + lj
                            gk = nk * block_size + lk
                            if gi < n and gj < n and gk < n:
                                idx = gi * n2 + gj * n + gk
                                if temperature[idx] > 0.02 or density[idx] > 0.02:
                                    found = 1

    occupancy[tid] = found


@wp.func
def block_is_active(
    occupancy: wp.array(dtype=int),
    i: int, j: int, k: int,
    n: int, block_size: int,
):
    """Check if the block containing voxel (i,j,k) is active."""
    bpd = n // block_size
    bi = i // block_size
    bj = j // block_size
    bk = k // block_size
    if bi >= bpd or bj >= bpd or bk >= bpd:
        return int(0)
    return occupancy[bi * bpd * bpd + bj * bpd + bk]


@wp.kernel
def bloom_threshold(
    image: wp.array(dtype=wp.vec3),
    bright: wp.array(dtype=wp.vec3),
    threshold: float,
):
    """Extract bright pixels for bloom."""
    tid = wp.tid()
    c = image[tid]
    lum = c[0] * 0.299 + c[1] * 0.587 + c[2] * 0.114
    if lum > threshold:
        scale = (lum - threshold) / (1.0 - threshold + 0.001)
        bright[tid] = wp.vec3(c[0] * scale, c[1] * scale, c[2] * scale)
    else:
        bright[tid] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def bloom_blur_h(
    src: wp.array(dtype=wp.vec3),
    dst: wp.array(dtype=wp.vec3),
    w: int,
    h: int,
):
    """Horizontal Gaussian blur (9-tap)."""
    tid = wp.tid()
    px = tid % w
    py = tid // w
    if py >= h:
        return

    weights_0 = 0.227
    weights_1 = 0.194
    weights_2 = 0.122
    weights_3 = 0.054
    weights_4 = 0.016

    r = float(0.0)
    g = float(0.0)
    b = float(0.0)

    for offset in range(9):
        oo = offset - 4
        sx = wp.clamp(px + oo, 0, w - 1)
        idx = py * w + sx
        c = src[idx]
        wt = weights_0
        ao = oo
        if ao < 0:
            ao = -ao
        if ao == 1:
            wt = weights_1
        elif ao == 2:
            wt = weights_2
        elif ao == 3:
            wt = weights_3
        elif ao == 4:
            wt = weights_4
        r = r + c[0] * wt
        g = g + c[1] * wt
        b = b + c[2] * wt

    dst[tid] = wp.vec3(r, g, b)


@wp.kernel
def bloom_blur_v(
    src: wp.array(dtype=wp.vec3),
    dst: wp.array(dtype=wp.vec3),
    w: int,
    h: int,
):
    """Vertical Gaussian blur (9-tap)."""
    tid = wp.tid()
    px = tid % w
    py = tid // w
    if py >= h:
        return

    weights_0 = 0.227
    weights_1 = 0.194
    weights_2 = 0.122
    weights_3 = 0.054
    weights_4 = 0.016

    r = float(0.0)
    g = float(0.0)
    b = float(0.0)

    for offset in range(9):
        oo = offset - 4
        sy = wp.clamp(py + oo, 0, h - 1)
        idx = sy * w + px
        c = src[idx]
        wt = weights_0
        ao = oo
        if ao < 0:
            ao = -ao
        if ao == 1:
            wt = weights_1
        elif ao == 2:
            wt = weights_2
        elif ao == 3:
            wt = weights_3
        elif ao == 4:
            wt = weights_4
        r = r + c[0] * wt
        g = g + c[1] * wt
        b = b + c[2] * wt

    dst[tid] = wp.vec3(r, g, b)


@wp.kernel
def bloom_combine(
    image: wp.array(dtype=wp.vec3),
    bloom: wp.array(dtype=wp.vec3),
    intensity: float,
):
    """Add bloom to original image."""
    tid = wp.tid()
    c = image[tid]
    b = bloom[tid]
    image[tid] = wp.vec3(
        wp.clamp(c[0] + b[0] * intensity, 0.0, 1.0),
        wp.clamp(c[1] + b[1] * intensity, 0.0, 1.0),
        wp.clamp(c[2] + b[2] * intensity, 0.0, 1.0),
    )


# ─── Simulation ────────────────────────────────────────────────


class FireSim:
    def __init__(self, n=128):
        self.n = n
        self.total = n * n * n
        self.frame = 0
        self._alloc()

    def _alloc(self):
        total = self.total
        z = np.zeros(total, dtype=np.float32)
        self.temperature = wp.array(z, dtype=float, device="cuda")
        self.temp_buf = wp.array(z, dtype=float, device="cuda")
        self.density = wp.array(z, dtype=float, device="cuda")
        self.dens_buf = wp.array(z, dtype=float, device="cuda")
        self.vel_x = wp.array(z, dtype=float, device="cuda")
        self.vel_y = wp.array(z, dtype=float, device="cuda")
        self.vel_z = wp.array(z, dtype=float, device="cuda")
        self.vx_buf = wp.array(z, dtype=float, device="cuda")
        self.vy_buf = wp.array(z, dtype=float, device="cuda")
        self.vz_buf = wp.array(z, dtype=float, device="cuda")
        self.light_vol = wp.zeros(total, dtype=wp.float16, device="cuda")
        self.mc_buf = wp.array(z, dtype=float, device="cuda")  # MacCormack extra buffer
        self.omega_x = wp.array(z, dtype=float, device="cuda")
        self.omega_y = wp.array(z, dtype=float, device="cuda")
        self.omega_z = wp.array(z, dtype=float, device="cuda")
        self.omega_mag = wp.array(z, dtype=float, device="cuda")
        self.pressure = wp.array(z, dtype=float, device="cuda")
        self.pressure_buf = wp.array(z, dtype=float, device="cuda")
        self.divergence = wp.array(z, dtype=float, device="cuda")
        self.block_size = 8
        occ_dim = max(self.n // self.block_size, 1)
        self.occ_total = occ_dim * occ_dim * occ_dim
        self.occupancy = wp.zeros(self.occ_total, dtype=int, device="cuda")
        self.sim_occupancy = wp.zeros(self.occ_total, dtype=int, device="cuda")
        self._jacobi_graph = None

    def reset(self):
        self.frame = 0
        self._alloc()

    def _advect_mc(self, field, buf, dt):
        """MacCormack advection: forward, backward, correct with clamping."""
        n = self.n
        total = self.total
        # 1. Forward advect: field → buf
        wp.launch(advect_field, dim=total,
                  inputs=[field, buf, self.vel_x, self.vel_y, self.vel_z, n, dt],
                  device="cuda")
        # 2. Backward advect: buf → mc_buf
        wp.launch(advect_backward, dim=total,
                  inputs=[buf, self.mc_buf, self.vel_x, self.vel_y, self.vel_z, n, dt],
                  device="cuda")
        # 3. Correct: read field(orig)+buf(fwd)+mc_buf(bwd) → write buf
        wp.launch(advect_maccormack_correct, dim=total,
                  inputs=[field, buf, self.mc_buf, buf,
                          self.vel_x, self.vel_y, self.vel_z, n, dt],
                  device="cuda")
        return buf, field  # buf has corrected result

    def _diffuse(self, field, buf, rate):
        wp.launch(
            diffuse_field, dim=self.total,
            inputs=[field, buf, self.n, rate], device="cuda",
        )
        return buf, field

    def step(self):
        n = self.n
        total = self.total
        dt = 0.25

        # 0. Build dilated occupancy for sparse sim
        self.sim_occupancy.zero_()
        wp.launch(build_occupancy_dilated, dim=self.occ_total,
                  inputs=[self.temperature, self.density, self.sim_occupancy,
                          n, self.block_size],
                  device="cuda")

        # 1. Fused: emit + forces + dissipate (1 launch instead of 3)
        wp.launch(
            sim_step_fused,
            dim=total,
            inputs=[
                self.temperature, self.density,
                self.vel_x, self.vel_y, self.vel_z,
                self.sim_occupancy,
                n, self.frame,
                0.8, 1.5, dt,     # buoyancy, turbulence, dt
                0.96, 0.95, 0.94,  # temp_decay, dens_decay, vel_decay
                self.block_size,
            ],
            device="cuda",
        )

        # 2. Vorticity Confinement (every 2nd frame)
        if self.frame % 2 == 0:
            wp.launch(
                compute_vorticity, dim=total,
                inputs=[self.vel_x, self.vel_y, self.vel_z,
                        self.omega_x, self.omega_y, self.omega_z, self.omega_mag, n],
                device="cuda",
            )
            wp.launch(
                apply_vorticity_confinement, dim=total,
                inputs=[self.vel_x, self.vel_y, self.vel_z,
                        self.omega_x, self.omega_y, self.omega_z, self.omega_mag,
                        n, 0.5, dt],  # stronger epsilon to compensate for skipping
                device="cuda",
            )

        # 3. Diffuse velocity
        self.vel_x, self.vx_buf = self._diffuse(self.vel_x, self.vx_buf, 0.2)
        self.vel_y, self.vy_buf = self._diffuse(self.vel_y, self.vy_buf, 0.2)
        self.vel_z, self.vz_buf = self._diffuse(self.vel_z, self.vz_buf, 0.2)

        # 4. Pressure projection (every 3rd frame — most expensive step)
        if self.frame % 3 == 0:
            wp.launch(compute_divergence, dim=total,
                      inputs=[self.vel_x, self.vel_y, self.vel_z, self.divergence, n],
                      device="cuda")
            self.pressure.zero_()
            self.pressure_buf.zero_()
            if self._jacobi_graph is None:
                try:
                    wp.capture_begin(device="cuda")
                    wp.launch(jacobi_step, dim=total,
                              inputs=[self.pressure, self.pressure_buf, self.divergence, n],
                              device="cuda")
                    wp.launch(jacobi_step, dim=total,
                              inputs=[self.pressure_buf, self.pressure, self.divergence, n],
                              device="cuda")
                    self._jacobi_graph = wp.capture_end(device="cuda")
                except Exception:
                    self._jacobi_graph = None

            if self._jacobi_graph is not None:
                for _ in range(5):  # 5 replays x 2 iters = 10 Jacobi iterations
                    wp.capture_launch(self._jacobi_graph)
            else:
                for _ in range(10):
                    wp.launch(jacobi_step, dim=total,
                              inputs=[self.pressure, self.pressure_buf, self.divergence, n],
                              device="cuda")
                    self.pressure, self.pressure_buf = self.pressure_buf, self.pressure
            wp.launch(subtract_pressure_gradient, dim=total,
                      inputs=[self.vel_x, self.vel_y, self.vel_z, self.pressure, n],
                      device="cuda")

        # 5. MacCormack advect temperature, standard advect density
        self.temperature, self.temp_buf = self._advect_mc(self.temperature, self.temp_buf, dt)
        wp.launch(advect_field, dim=total,
                  inputs=[self.density, self.dens_buf,
                          self.vel_x, self.vel_y, self.vel_z, n, dt],
                  device="cuda")
        self.density, self.dens_buf = self.dens_buf, self.density

        # 5. Advect velocity (fused — single backtrace for all 3 components)
        wp.launch(advect_velocity_fused, dim=total,
                  inputs=[self.vel_x, self.vel_y, self.vel_z,
                          self.vx_buf, self.vy_buf, self.vz_buf,
                          self.vel_x, self.vel_y, self.vel_z, n, dt],
                  device="cuda")
        self.vel_x, self.vx_buf = self.vx_buf, self.vel_x
        self.vel_y, self.vy_buf = self.vy_buf, self.vel_y
        self.vel_z, self.vz_buf = self.vz_buf, self.vel_z

        self.frame += 1

    def compute_lighting(self):
        """Precompute light volume — O(N^2) threads, O(N) sweep each."""
        wp.launch(
            compute_light_volume,
            dim=self.n * self.n,
            inputs=[self.density, self.light_vol, self.n, 0.4],
            device="cuda",
        )

    def render(self, img_w, img_h, image_buf, bloom_a, bloom_b):
        self.compute_lighting()
        # Build occupancy grid for adaptive ray marching
        self.occupancy.zero_()
        wp.launch(build_occupancy, dim=self.occ_total,
                  inputs=[self.temperature, self.density, self.occupancy,
                          self.n, self.block_size],
                  device="cuda")
        pixels = img_w * img_h
        wp.launch(
            render_fire, dim=pixels,
            inputs=[self.temperature, self.density, self.light_vol,
                    self.occupancy, image_buf, self.n, img_w, img_h, self.block_size],
            device="cuda",
        )
        # Bloom post-processing
        wp.launch(bloom_threshold, dim=pixels,
                  inputs=[image_buf, bloom_a, 0.4], device="cuda")
        wp.launch(bloom_blur_h, dim=pixels,
                  inputs=[bloom_a, bloom_b, img_w, img_h], device="cuda")
        wp.launch(bloom_blur_v, dim=pixels,
                  inputs=[bloom_b, bloom_a, img_w, img_h], device="cuda")
        # Second blur pass for wider glow
        wp.launch(bloom_blur_h, dim=pixels,
                  inputs=[bloom_a, bloom_b, img_w, img_h], device="cuda")
        wp.launch(bloom_blur_v, dim=pixels,
                  inputs=[bloom_b, bloom_a, img_w, img_h], device="cuda")
        wp.launch(bloom_combine, dim=pixels,
                  inputs=[image_buf, bloom_a, 0.6], device="cuda")


# ─── Main ──────────────────────────────────────────────────────


# Minimal 5x7 bitmap font
_GLYPHS = {
    ' ': [0x00,0x00,0x00,0x00,0x00,0x00,0x00],
    '0': [0x0E,0x11,0x13,0x15,0x19,0x11,0x0E],
    '1': [0x04,0x0C,0x04,0x04,0x04,0x04,0x0E],
    '2': [0x0E,0x11,0x01,0x06,0x08,0x10,0x1F],
    '3': [0x0E,0x11,0x01,0x06,0x01,0x11,0x0E],
    '4': [0x02,0x06,0x0A,0x12,0x1F,0x02,0x02],
    '5': [0x1F,0x10,0x1E,0x01,0x01,0x11,0x0E],
    '6': [0x06,0x08,0x10,0x1E,0x11,0x11,0x0E],
    '7': [0x1F,0x01,0x02,0x04,0x08,0x08,0x08],
    '8': [0x0E,0x11,0x11,0x0E,0x11,0x11,0x0E],
    '9': [0x0E,0x11,0x11,0x0F,0x01,0x02,0x0C],
    'A': [0x0E,0x11,0x11,0x1F,0x11,0x11,0x11],
    'B': [0x1E,0x11,0x11,0x1E,0x11,0x11,0x1E],
    'C': [0x0E,0x11,0x10,0x10,0x10,0x11,0x0E],
    'D': [0x1E,0x11,0x11,0x11,0x11,0x11,0x1E],
    'E': [0x1F,0x10,0x10,0x1E,0x10,0x10,0x1F],
    'F': [0x1F,0x10,0x10,0x1E,0x10,0x10,0x10],
    'G': [0x0E,0x11,0x10,0x17,0x11,0x11,0x0E],
    'H': [0x11,0x11,0x11,0x1F,0x11,0x11,0x11],
    'I': [0x0E,0x04,0x04,0x04,0x04,0x04,0x0E],
    'K': [0x11,0x12,0x14,0x18,0x14,0x12,0x11],
    'L': [0x10,0x10,0x10,0x10,0x10,0x10,0x1F],
    'M': [0x11,0x1B,0x15,0x15,0x11,0x11,0x11],
    'N': [0x11,0x19,0x15,0x13,0x11,0x11,0x11],
    'O': [0x0E,0x11,0x11,0x11,0x11,0x11,0x0E],
    'P': [0x1E,0x11,0x11,0x1E,0x10,0x10,0x10],
    'Q': [0x0E,0x11,0x11,0x11,0x15,0x12,0x0D],
    'R': [0x1E,0x11,0x11,0x1E,0x14,0x12,0x11],
    'S': [0x0E,0x11,0x10,0x0E,0x01,0x11,0x0E],
    'T': [0x1F,0x04,0x04,0x04,0x04,0x04,0x04],
    'U': [0x11,0x11,0x11,0x11,0x11,0x11,0x0E],
    'V': [0x11,0x11,0x11,0x11,0x0A,0x0A,0x04],
    'W': [0x11,0x11,0x11,0x15,0x15,0x1B,0x11],
    'X': [0x11,0x11,0x0A,0x04,0x0A,0x11,0x11],
    'd': [0x01,0x01,0x0F,0x11,0x11,0x11,0x0F],
    'e': [0x00,0x00,0x0E,0x11,0x1F,0x10,0x0E],
    'i': [0x04,0x00,0x0C,0x04,0x04,0x04,0x0E],
    'l': [0x0C,0x04,0x04,0x04,0x04,0x04,0x0E],
    'm': [0x00,0x00,0x1A,0x15,0x15,0x15,0x11],
    'o': [0x00,0x00,0x0E,0x11,0x11,0x11,0x0E],
    'p': [0x00,0x00,0x1E,0x11,0x1E,0x10,0x10],
    'r': [0x00,0x00,0x16,0x19,0x10,0x10,0x10],
    's': [0x00,0x00,0x0F,0x10,0x0E,0x01,0x1E],
    't': [0x04,0x04,0x0E,0x04,0x04,0x04,0x06],
    'v': [0x00,0x00,0x11,0x11,0x11,0x0A,0x04],
    'x': [0x00,0x00,0x11,0x0A,0x04,0x0A,0x11],
    ':': [0x00,0x04,0x00,0x00,0x00,0x04,0x00],
    '.': [0x00,0x00,0x00,0x00,0x00,0x00,0x04],
    '=': [0x00,0x00,0x1F,0x00,0x1F,0x00,0x00],
    '/': [0x01,0x02,0x02,0x04,0x08,0x08,0x10],
    '^': [0x04,0x0A,0x11,0x00,0x00,0x00,0x00],
    '(': [0x02,0x04,0x08,0x08,0x08,0x04,0x02],
    ')': [0x08,0x04,0x02,0x02,0x02,0x04,0x08],
    '+': [0x00,0x04,0x04,0x1F,0x04,0x04,0x00],
    '-': [0x00,0x00,0x00,0x1F,0x00,0x00,0x00],
}


# ─── CUDA-OpenGL Interop ───────────────────────────────────────


class GLInterop:
    """Zero-copy CUDA→OpenGL via PBO + cudaGraphicsGLRegisterBuffer."""

    def __init__(self, width, height):
        import ctypes
        from OpenGL.GL import (
            glGenBuffers, glBindBuffer, glBufferData, glGenTextures,
            glBindTexture, glTexImage2D, glTexParameteri, glTexSubImage2D,
            glEnable, glDisable, glBegin, glEnd, glVertex2f, glTexCoord2f,
            glDeleteBuffers, glDeleteTextures,
            GL_PIXEL_UNPACK_BUFFER, GL_DYNAMIC_DRAW, GL_TEXTURE_2D,
            GL_RGB32F, GL_RGB, GL_FLOAT, GL_TEXTURE_MIN_FILTER,
            GL_TEXTURE_MAG_FILTER, GL_LINEAR, GL_QUADS,
        )
        self.gl = {
            'glBindBuffer': glBindBuffer, 'glBindTexture': glBindTexture,
            'glTexSubImage2D': glTexSubImage2D, 'glEnable': glEnable,
            'glDisable': glDisable,
            'glBegin': glBegin, 'glEnd': glEnd, 'glVertex2f': glVertex2f,
            'glTexCoord2f': glTexCoord2f,
            'glDeleteBuffers': glDeleteBuffers,
            'glDeleteTextures': glDeleteTextures,
            'GL_PIXEL_UNPACK_BUFFER': GL_PIXEL_UNPACK_BUFFER,
            'GL_TEXTURE_2D': GL_TEXTURE_2D, 'GL_RGB': GL_RGB,
            'GL_FLOAT': GL_FLOAT, 'GL_QUADS': GL_QUADS,
        }
        self.width = width
        self.height = height
        self.ctypes = ctypes
        buf_size = width * height * 3 * 4  # 3 floats x 4 bytes

        # Create PBO
        self.pbo = glGenBuffers(1)
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, self.pbo)
        glBufferData(GL_PIXEL_UNPACK_BUFFER, buf_size, None, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)

        # Create texture
        self.texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, width, height, 0,
                     GL_RGB, GL_FLOAT, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glBindTexture(GL_TEXTURE_2D, 0)

        # Register PBO with CUDA
        self._cudart = ctypes.CDLL("libcudart.so")
        self._resource = ctypes.c_void_p()
        err = self._cudart.cudaGraphicsGLRegisterBuffer(
            ctypes.byref(self._resource),
            ctypes.c_uint(int(self.pbo)),
            0,  # cudaGraphicsRegisterFlagsNone
        )
        if err != 0:
            raise RuntimeError(f"cudaGraphicsGLRegisterBuffer failed: {err}")

    def map_to_warp(self):
        """Map PBO → CUDA device pointer → wp.array(dtype=wp.vec3)."""
        ct = self.ctypes
        res_arr = (ct.c_void_p * 1)(self._resource)
        self._cudart.cudaGraphicsMapResources(1, res_arr, None)
        dev_ptr = ct.c_void_p()
        size = ct.c_size_t()
        self._cudart.cudaGraphicsResourceGetMappedPointer(
            ct.byref(dev_ptr), ct.byref(size), self._resource
        )
        return wp.array(
            ptr=dev_ptr.value, dtype=wp.vec3,
            shape=(self.width * self.height,),
            device="cuda", copy=False,
        )

    def unmap(self):
        res_arr = (self.ctypes.c_void_p * 1)(self._resource)
        self._cudart.cudaGraphicsUnmapResources(1, res_arr, None)

    def display(self):
        """Copy PBO → texture and draw fullscreen quad."""
        gl = self.gl
        gl['glBindBuffer'](gl['GL_PIXEL_UNPACK_BUFFER'], self.pbo)
        gl['glBindTexture'](gl['GL_TEXTURE_2D'], self.texture)
        gl['glTexSubImage2D'](
            gl['GL_TEXTURE_2D'], 0, 0, 0, self.width, self.height,
            gl['GL_RGB'], gl['GL_FLOAT'], None,
        )
        gl['glBindBuffer'](gl['GL_PIXEL_UNPACK_BUFFER'], 0)

        gl['glEnable'](gl['GL_TEXTURE_2D'])
        gl['glBindTexture'](gl['GL_TEXTURE_2D'], self.texture)
        gl['glBegin'](gl['GL_QUADS'])
        gl['glTexCoord2f'](0, 1); gl['glVertex2f'](-1, -1)
        gl['glTexCoord2f'](1, 1); gl['glVertex2f'](1, -1)
        gl['glTexCoord2f'](1, 0); gl['glVertex2f'](1, 1)
        gl['glTexCoord2f'](0, 0); gl['glVertex2f'](-1, 1)
        gl['glEnd']()
        gl['glBindTexture'](gl['GL_TEXTURE_2D'], 0)
        gl['glDisable'](gl['GL_TEXTURE_2D'])

    def cleanup(self):
        """Release CUDA-GL resource, PBO, and texture."""
        ct = self.ctypes
        gl = self.gl
        # Unregister CUDA graphics resource
        if self._resource:
            self._cudart.cudaGraphicsUnregisterResource(self._resource)
            self._resource = None
        # Delete PBO
        if self.pbo:
            gl['glDeleteBuffers'](1, [int(self.pbo)])
            self.pbo = None
        # Delete texture
        if self.texture:
            gl['glDeleteTextures'](1, [int(self.texture)])
            self.texture = None


def main():
    import pygame

    IMG_W, IMG_H = 512, 512
    HALF_W, HALF_H = 256, 256
    grid_size = 128
    half_res = False

    sim = FireSim(grid_size)
    image_buf = wp.zeros(IMG_W * IMG_H, dtype=wp.vec3, device="cuda")
    image_buf_half = wp.zeros(HALF_W * HALF_H, dtype=wp.vec3, device="cuda")
    bloom_a = wp.zeros(IMG_W * IMG_H, dtype=wp.vec3, device="cuda")
    bloom_b = wp.zeros(IMG_W * IMG_H, dtype=wp.vec3, device="cuda")
    bloom_a_half = wp.zeros(HALF_W * HALF_H, dtype=wp.vec3, device="cuda")
    bloom_b_half = wp.zeros(HALF_W * HALF_H, dtype=wp.vec3, device="cuda")

    pygame.init()

    # Try OpenGL interop (zero-copy), fallback to surfarray
    use_gl = False
    gl_interop = None
    try:
        from pygame.locals import OPENGL, DOUBLEBUF
        screen = pygame.display.set_mode((IMG_W, IMG_H), OPENGL | DOUBLEBUF)
        gl_interop = GLInterop(IMG_W, IMG_H)
        use_gl = True
        print("OpenGL Interop: ACTIVE (zero-copy)")
    except Exception as e:
        print(f"OpenGL Interop failed ({e}), using surfarray fallback")
        screen = pygame.display.set_mode((IMG_W, IMG_H))

    pygame.display.set_caption(f"Warp Fire | Grid {grid_size}")
    clock = pygame.time.Clock()

    def draw_text(surf, x, y, text, color=(255, 255, 255), scale=2):
        cx = x
        for ch in text.upper():
            glyph = _GLYPHS.get(ch)
            if glyph is None:
                cx += 4 * scale
                continue
            for row_i, row in enumerate(glyph):
                for col in range(5):
                    if row & (0x10 >> col):
                        surf.fill(color, (cx + col * scale, y + row_i * scale, scale, scale))
            cx += 6 * scale

    def draw_hud_bg(surf, x, y, w, h):
        bg = pygame.Surface((w, h), pygame.SRCALPHA)
        bg.fill((0, 0, 0, 140))
        surf.blit(bg, (x, y))

    grid_sizes = [32, 48, 64, 96, 128, 192, 256, 384, 512]
    grid_idx = grid_sizes.index(grid_size) if grid_size in grid_sizes else 4

    def set_grid(idx):
        nonlocal grid_size, grid_idx, sim
        grid_idx = max(0, min(idx, len(grid_sizes) - 1))
        grid_size = grid_sizes[grid_idx]
        sim = FireSim(grid_size)

    render_every = 1  # render every N-th frame (1=every frame)
    frame_counter = 0
    last_surf = None

    # Performance logger
    import csv
    log_path = "/home/michael/Dokumente/Fire_Blender/perf_log.csv"
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["timestamp", "fps", "sim_ms", "render_ms", "grid", "voxels", "res", "gl", "skip"])
    log_interval = 0.5  # log every 0.5 seconds
    last_log_time = time.perf_counter()
    sim_ms = 0.0
    render_ms = 0.0
    print(f"Performance log: {log_path}")

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    sim.reset()
                elif event.key == pygame.K_h:
                    half_res = not half_res
                elif event.key == pygame.K_t:
                    render_every = 3 if render_every == 1 else 1
                elif event.key in (pygame.K_UP, pygame.K_PLUS, pygame.K_KP_PLUS):
                    set_grid(grid_idx + 1)
                elif event.key in (pygame.K_DOWN, pygame.K_MINUS, pygame.K_KP_MINUS):
                    set_grid(grid_idx - 1)

        # Simulate
        t0 = time.perf_counter()
        sim.step()
        wp.synchronize()
        sim_ms = (time.perf_counter() - t0) * 1000.0

        # Render (skip frames if temporal reprojection active)
        frame_counter += 1
        do_render = (frame_counter % render_every == 0) or last_surf is None
        if do_render:
            t0 = time.perf_counter()

            if use_gl and not half_res:
                # === OpenGL Interop path: zero-copy ===
                gl_buf = gl_interop.map_to_warp()
                sim.render(IMG_W, IMG_H, gl_buf, bloom_a, bloom_b)
                wp.synchronize()
                gl_interop.unmap()
                render_ms = (time.perf_counter() - t0) * 1000.0
                gl_interop.display()
            else:
                # === Surfarray fallback ===
                if half_res:
                    sim.render(HALF_W, HALF_H, image_buf_half, bloom_a_half, bloom_b_half)
                else:
                    sim.render(IMG_W, IMG_H, image_buf, bloom_a, bloom_b)
                wp.synchronize()
                render_ms = (time.perf_counter() - t0) * 1000.0

                if half_res:
                    img_np = image_buf_half.numpy()
                    img_rgb = (img_np * 255.0).clip(0, 255).astype(np.uint8)
                    img_rgb = img_rgb.reshape(HALF_H, HALF_W, 3)
                    small_surf = pygame.surfarray.make_surface(img_rgb.swapaxes(0, 1))
                    last_surf = pygame.transform.scale(small_surf, (IMG_W, IMG_H))
                else:
                    img_np = image_buf.numpy()
                    img_rgb = (img_np * 255.0).clip(0, 255).astype(np.uint8)
                    img_rgb = img_rgb.reshape(IMG_H, IMG_W, 3)
                    last_surf = pygame.surfarray.make_surface(img_rgb.swapaxes(0, 1))

                screen.blit(last_surf, (0, 0))

        elif not use_gl and last_surf is not None:
            screen.blit(last_surf, (0, 0))

        # HUD (surfarray mode only — OpenGL mode shows in title)
        fps = clock.get_fps()
        voxels = grid_size ** 3
        if voxels >= 1_000_000:
            voxel_str = f"{voxels / 1_000_000:.1f}M"
        else:
            voxel_str = f"{voxels // 1000}K"
        res_str = "HALF" if half_res else "FULL"
        gl_str = " GL" if use_gl else ""
        temp_str = f"  SKIP:{render_every}" if render_every > 1 else ""
        info_top = f"FPS: {fps:.0f}  Sim: {sim_ms:.1f}ms  Render: {render_ms:.1f}ms"
        info_grid = f"Grid: {grid_size}^3 ({voxel_str})  Res: {res_str}{gl_str}{temp_str}"
        info_controls = "Up/Down=Grid  H=HalfRes  T=Temporal  R=Reset  ESC=Quit"

        if use_gl:
            # Draw HUD with OpenGL
            from OpenGL.GL import (
                glMatrixMode, glPushMatrix, glPopMatrix, glLoadIdentity,
                glOrtho, glEnable, glDisable, glBlendFunc, glColor4f, glColor3f,
                glBegin, glEnd, glVertex2f,
                GL_PROJECTION, GL_MODELVIEW, GL_BLEND, GL_TEXTURE_2D,
                GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_QUADS,
            )
            # Switch to 2D ortho projection
            glMatrixMode(GL_PROJECTION)
            glPushMatrix()
            glLoadIdentity()
            glOrtho(0, IMG_W, IMG_H, 0, -1, 1)
            glMatrixMode(GL_MODELVIEW)
            glPushMatrix()
            glLoadIdentity()
            glDisable(GL_TEXTURE_2D)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

            def gl_draw_rect(x, y, w, h, r, g, b, a):
                glColor4f(r, g, b, a)
                glBegin(GL_QUADS)
                glVertex2f(x, y); glVertex2f(x+w, y)
                glVertex2f(x+w, y+h); glVertex2f(x, y+h)
                glEnd()

            def gl_draw_text(x, y, text, cr, cg, cb, scale=2):
                cx = x
                for ch in text.upper():
                    glyph = _GLYPHS.get(ch)
                    if glyph is None:
                        cx += 4 * scale
                        continue
                    for row_i, row in enumerate(glyph):
                        for col in range(5):
                            if row & (0x10 >> col):
                                gl_draw_rect(cx + col*scale, y + row_i*scale,
                                             scale, scale, cr, cg, cb, 1.0)
                    cx += 6 * scale

            gl_draw_rect(4, 4, 420, 40, 0, 0, 0, 0.55)
            gl_draw_text(8, 6, info_top, 1.0, 1.0, 1.0)
            gl_draw_text(8, 22, info_grid, 0.7, 0.7, 0.7)
            gl_draw_rect(4, IMG_H - 22, 440, 18, 0, 0, 0, 0.55)
            gl_draw_text(8, IMG_H - 20, info_controls, 0.55, 0.55, 0.55)

            glDisable(GL_BLEND)
            glMatrixMode(GL_PROJECTION)
            glPopMatrix()
            glMatrixMode(GL_MODELVIEW)
            glPopMatrix()
        else:
            draw_hud_bg(screen, 4, 4, 420, 40)
            draw_text(screen, 8, 6, info_top, (255, 255, 255))
            draw_text(screen, 8, 22, info_grid, (180, 180, 180))
            draw_hud_bg(screen, 4, IMG_H - 22, 440, 18)
            draw_text(screen, 8, IMG_H - 20, info_controls, (140, 140, 140))

        # Performance logging
        now = time.perf_counter()
        if now - last_log_time >= log_interval:
            log_writer.writerow([
                f"{now:.2f}", f"{fps:.1f}", f"{sim_ms:.2f}", f"{render_ms:.2f}",
                grid_size, grid_size**3,
                "HALF" if half_res else "FULL",
                "GL" if use_gl else "SW",
                render_every,
            ])
            log_file.flush()
            last_log_time = now

        pygame.display.flip()
        clock.tick(60)

    log_file.close()
    print(f"Log saved: {log_path}")
    if gl_interop is not None:
        gl_interop.cleanup()
    pygame.quit()


if __name__ == "__main__":
    main()
