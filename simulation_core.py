import jax
import jax.numpy as jnp
from jax import jit, lax
import numpy as np
import time
import pandas as pd
import sys
import matplotlib.pyplot as plt

# --- 1. CONFIGURATION (COLD FLOW) ---
class Config:
    N_PARTICLES = 12000  
    
    # DENSITY: 50% (Jamming is essential for this theory)
    # R_Arena = 9.3 for 12k particles
    R_ARENA = 9.3      
    R_PARTICLE = 0.06
    
    # PHYSICS (The "Cold" Adjustment)
    V_ACTIVE = 1.0       # LOW ENERGY (Was 3.0)
    GAMMA = 5.0          # High Friction (Sticky)
    DT = 0.01
    
    # CRITICAL: Almost zero noise to let Bias win
    NOISE_ROT = 0.01     
    K_SPRING = 800.0     
    
    # RAMP SETTINGS (Super Strong Bias Search)
    MAX_BIAS = 2.0       # Sweeping 0.0 -> 2.0
    TOTAL_STEPS = 50000 
    BATCH_SIZE = 1000   
    DTYPE = jnp.float16

print(f"--- CRITICAL POINT SEARCH (COLD FLOW) ---")
print(f"Protocol: Low Energy, High Density, Zero Noise")
print(f"Target: Find the trigger point for N={Config.N_PARTICLES}")

# --- 2. JAX ENGINE ---

@jit
def compute_forces(pos, vel, thetas, current_bias):
    pos = pos.astype(Config.DTYPE)
    vel = vel.astype(Config.DTYPE)
    
    F_active = jnp.stack([
        jnp.cos(thetas) * Config.V_ACTIVE * Config.GAMMA,
        jnp.sin(thetas) * Config.V_ACTIVE * Config.GAMMA
    ], axis=1).astype(Config.DTYPE)

    F_drag = (-Config.GAMMA * vel).astype(Config.DTYPE)

    # Dynamic Bias
    radii = jnp.linalg.norm(pos, axis=1, keepdims=True) + 1e-3
    tangent_vecs = jnp.hstack([-pos[:, 1:2], pos[:, 0:1]]) / radii
    F_bias = (tangent_vecs * current_bias * Config.GAMMA).astype(Config.DTYPE)

    # Collisions
    diff = pos[:, None, :] - pos[None, :, :]
    dist_sq = jnp.sum(diff**2, axis=-1)
    dist = jnp.sqrt(dist_sq + 1e-3)
    overlap = jnp.maximum(0.0, 2 * Config.R_PARTICLE - dist)
    mask = (dist < 2 * Config.R_PARTICLE) & (dist > 1e-4)
    force_mag = Config.K_SPRING * overlap * mask
    F_col = jnp.sum((diff / (dist[..., None] + 1e-4)) * force_mag[..., None], axis=1)

    # Boundary
    dist_center = jnp.linalg.norm(pos, axis=1)
    wall_overlap = jnp.maximum(0.0, dist_center - (Config.R_ARENA - Config.R_PARTICLE))
    norm_pos = pos / (dist_center[:, None] + 1e-3)
    F_wall = -norm_pos * Config.K_SPRING * 5.0 * wall_overlap[:, None]

    return (F_active + F_drag + F_bias + F_col + F_wall).astype(jnp.float32)

@jit
def run_ramp_batch(pos, vel, thetas, key, start_step):
    def step_fn(carry, step_idx):
        p, v, th, k = carry
        global_step = start_step + step_idx
        progress = global_step / Config.TOTAL_STEPS
        current_bias = progress * Config.MAX_BIAS
        
        k, subk = jax.random.split(k)
        noise = jax.random.normal(subk, shape=(Config.N_PARTICLES,)) * Config.NOISE_ROT * jnp.sqrt(Config.DT)
        th += noise
        
        forces = compute_forces(p, v, th, current_bias)
        v += (forces / 1.0) * Config.DT
        p += v * Config.DT
        return (p, v, th, k), current_bias

    (pos, vel, thetas, key), biases = lax.scan(step_fn, (pos, vel, thetas, key), jnp.arange(Config.BATCH_SIZE))
    
    # Metrics
    radii = jnp.linalg.norm(pos, axis=1, keepdims=True) + 1e-6
    tangent = jnp.hstack([-pos[:, 1:2], pos[:, 0:1]]) / radii
    v_norm = vel / (jnp.linalg.norm(vel, axis=1, keepdims=True) + 1e-6)
    order = jnp.mean(jnp.sum(v_norm * tangent, axis=1))
    
    return pos, vel, thetas, key, order, biases[-1]

# --- 3. EXECUTION ---

key = jax.random.PRNGKey(42)
pos = jax.random.uniform(key, (Config.N_PARTICLES, 2), minval=-Config.R_ARENA, maxval=Config.R_ARENA)
r = jnp.linalg.norm(pos, axis=1, keepdims=True)
pos = pos * jnp.where(r > (Config.R_ARENA-1), (Config.R_ARENA-1)/r, 1.0)
vel = jax.random.normal(key, (Config.N_PARTICLES, 2)) * 0.1
thetas = jax.random.uniform(key, (Config.N_PARTICLES,)) * 2 * jnp.pi

print("Compiling...")
run_ramp_batch(pos, vel, thetas, key, 0)
print("Ready.")

history_bias = []
history_order = []

print(f"\nSTARTING RAMP: 0.0 -> {Config.MAX_BIAS}")
print("="*75)
print(f"{'STEP':<10} | {'PROGRESS':<20} | {'BIAS':<10} | {'ORDER':<10} | {'STATUS':<10}")
print("="*75)

start_run = time.time()
num_batches = Config.TOTAL_STEPS // Config.BATCH_SIZE

for i in range(num_batches):
    global_step_start = i * Config.BATCH_SIZE
    pos, vel, thetas, key, order, current_bias = run_ramp_batch(pos, vel, thetas, key, global_step_start)
    
    ord_val = float(order)
    bias_val = float(current_bias)
    history_order.append(ord_val)
    history_bias.append(bias_val)
    
    pct = (i+1) / num_batches
    bar = "█" * int(20*pct) + "-" * (20-int(20*pct))
    
    status = "CHAOS"
    if ord_val > 0.3: status = "WAKE"
    if ord_val > 0.6: status = "VORTEX"
    if ord_val > 0.8: status = "LOCKED"
    
    sys.stdout.write(f"\r{global_step_start:<10} | [{bar}] {int(pct*100)}% | {bias_val:.3f}      | {ord_val:.4f}     | {status:<10}")
    sys.stdout.flush()

total_time = time.time() - start_run
print(f"\n\nSEARCH COMPLETE in {total_time:.2f}s")

# --- 4. ANALYSIS ---
df = pd.DataFrame({'Bias': history_bias, 'Order': history_order})
df.to_csv('cold_flow_results.csv', index=False)

critical_point = df[df['Order'] > 0.5]
if not critical_point.empty:
    bc = critical_point.iloc[0]['Bias']
    print(f"\n*** CRITICAL BIAS DETECTED: Bc = {bc:.3f} ***")
else:
    print("\n*** NO TRANSITION FOUND ***")

plt.figure(figsize=(10, 6))
plt.plot(df['Bias'], df['Order'], color='blue', linewidth=2)
plt.axhline(0.5, color='red', linestyle='--', label='Threshold')
if not critical_point.empty:
    plt.axvline(bc, color='green', linestyle=':', label=f'Critical Bias ({bc:.3f})')
plt.title(f"Transition Search (Cold Flow, N={Config.N_PARTICLES})")
plt.xlabel("Applied Bias")
plt.ylabel("Order")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('cold_flow_plot.png')
print("Graph saved.")
