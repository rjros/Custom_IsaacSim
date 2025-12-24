# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""
Cantilever Beam Validation - Reverse Release Test
Based on: "Sim-to-Real for Soft Robots using Differentiable FEM"

Experimental Setup:
1. Let beam settle under gravity for 3s (reach equilibrium)
2. Kinematically raise beam to horizontal position
3. Release beam and observe oscillation back down

This matches the experimental condition where the beam was "kept at 
horizontal position then released" - simulating the mechanical hold/release.
"""

from isaacsim import SimulationApp

simulation_app = SimulationApp({
    "headless": False, 
    "extra_args": ["--/app/useFabricSceneDelegate=0"]
})

import argparse
import numpy as np
import torch

# Check for matplotlib and scipy availability
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Plots will not be generated.")

try:
    from scipy import signal
    from scipy.optimize import curve_fit
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available. Frequency analysis will be limited.")

from isaacsim.core.api import World
from isaacsim.core.api.materials.deformable_material import DeformableMaterial
from isaacsim.core.prims import DeformablePrim, SingleDeformablePrim
from omni.physx.scripts import physicsUtils

from pxr import Gf, UsdGeom, PhysxSchema, UsdPhysics, Sdf, UsdShade, UsdLux
import isaacsim.core.utils.deformable_mesh_utils as deformableMeshUtils

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

parser = argparse.ArgumentParser(
    description='Dragon Skin 10 Cantilever Reverse Release Validation',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument("--beam_length", type=float, default=0.10, 
                    help="Beam length in meters (10cm)")
parser.add_argument("--beam_width", type=float, default=0.03, 
                    help="Beam width in meters (3cm)")
parser.add_argument("--beam_height", type=float, default=0.03, 
                    help="Beam height in meters (3cm)")
parser.add_argument("--resolution", type=int, default=20, 
                    help="Simulation mesh resolution")
parser.add_argument("--solver_iterations", type=int, default=40,
                    help="Position solver iterations per timestep")
parser.add_argument("--settling_time", type=float, default=3.0, 
                    help="Time to let beam settle under gravity (seconds)")
parser.add_argument("--raise_duration", type=float, default=1.0, 
                    help="Time to raise beam to horizontal (seconds)")
parser.add_argument("--observation_time", type=float, default=3.0, 
                    help="Time to observe after release (seconds)")
parser.add_argument("--mass_damping", type=float, default=9.11, 
                    help="Mass-proportional damping coefficient α (s^-1)")
args, unknown = parser.parse_known_args()


class DragonSkinReverseReleaseValidation:
    """
    Three-phase validation test:
    1. SETTLING (0 to 3s): Beam drops under gravity, reaches equilibrium
    2. RAISING (3s to 4s): Kinematically raise beam back to horizontal
    3. RELEASE (4s+): Release and observe oscillation
    
    This simulates the experimental condition where the beam is mechanically
    held horizontal then released.
    """
    
    def __init__(self):
        # Phase definitions
        self.PHASE_SETTLING = 0
        self.PHASE_RAISING = 1
        self.PHASE_OBSERVING = 2
        self.current_phase = self.PHASE_SETTLING
        
        # Timing
        self.time_step = 0.01  # 100 Hz
        self.settling_time = args.settling_time
        self.raise_duration = args.raise_duration
        self.observation_time = args.observation_time
        
        self.raise_start_time = self.settling_time
        self.release_time = self.settling_time + self.raise_duration
        self.total_time = self.release_time + self.observation_time
        
        print(f"\n{'='*80}")
        print(f"Dragon Skin 10 Cantilever - Reverse Release Validation")
        print(f"Based on: 'Sim-to-Real for Soft Robots using Differentiable FEM'")
        print(f"{'='*80}")
        print(f"\nThree-Phase Experimental Setup:")
        print(f"  Phase 1 (0 to {self.settling_time}s): SETTLING")
        print(f"    - Beam drops under gravity")
        print(f"    - Reaches equilibrium position")
        print(f"  Phase 2 ({self.settling_time}s to {self.release_time}s): RAISING")
        print(f"    - Kinematically raise beam to horizontal")
        print(f"    - Simulates mechanical hold")
        print(f"  Phase 3 ({self.release_time}s to {self.total_time}s): RELEASE")
        print(f"    - Release beam from horizontal")
        print(f"    - Observe oscillation/settling")
        print(f"\nMaterial Parameters (Optimized from paper):")
        print(f"  Young's Modulus:   E = 234,900 Pa")
        print(f"  Poisson's Ratio:   ν = 0.439")
        print(f"  Density:           ρ = 1,210 kg/m³")
        print(f"  Mass Damping:      α = {args.mass_damping} s⁻¹")
        print(f"{'='*80}\n")
        
        self.my_world = World(
            stage_units_in_meters=1.0, 
            backend="torch", 
            device="cuda",
            physics_dt=self.time_step,
            rendering_dt=self.time_step * 2
        )
        self.stage = simulation_app.context.get_stage()
        
        # Beam geometry
        self.beam_length = args.beam_length
        self.beam_width = args.beam_width
        self.beam_height = args.beam_height
        self.resolution = args.resolution
        
        # Data recording
        self.time_history = []
        self.phase_history = []
        self.tip_z_history = []
        self.left_z_history = []
        self.right_z_history = []
        
        self.initial_positions = None
        self.settled_positions = None  # Positions at end of settling
        self.horizontal_positions = None  # Target horizontal positions
        
        self.tip_node_idx = None
        self.left_node_idx = None
        self.right_node_idx = None
        
        # Setup environment
        self.my_world.scene.add_default_ground_plane()
        self.setup_environment()
        self.makeEnvs()
        self.setup_camera()

    def setup_environment(self):
        """Setup lighting"""
        dome_light_path = "/World/DomeLight"
        dome_light = UsdLux.DomeLight.Define(self.stage, dome_light_path)
        dome_light.CreateIntensityAttr().Set(1000.0)
        dome_light.CreateColorAttr().Set(Gf.Vec3f(1.0, 1.0, 1.0))

    def makeEnvs(self):
        print("Creating reverse release validation environment...")
        
        env_path = "/World/Env0"
        env = UsdGeom.Xform.Define(self.stage, env_path)

        # Create beam mesh
        mesh_path = env.GetPrim().GetPath().AppendChild("deformable")
        skin_mesh = UsdGeom.Mesh.Define(self.stage, mesh_path)
        
        # Visual mesh
        tri_points, tri_indices = deformableMeshUtils.createTriangleMeshCube(16)
        skin_mesh.GetPointsAttr().Set(tri_points)
        skin_mesh.GetFaceVertexIndicesAttr().Set(tri_indices)
        skin_mesh.GetFaceVertexCountsAttr().Set([3] * (len(tri_indices) // 3))
        
        # Setup transforms - HORIZONTAL starting position
        physicsUtils.setup_transform_as_scale_orient_translate(skin_mesh)
        beam_size = Gf.Vec3f(self.beam_length, self.beam_width, self.beam_height)
        physicsUtils.set_or_add_scale_op(skin_mesh, beam_size)
        
        # Position beam horizontally at height 2.0m
        beam_center = Gf.Vec3f(0.0, 0.0, 2.0)
        physicsUtils.set_or_add_translate_op(skin_mesh, beam_center)
        
        # Visual material
        material_path = env.GetPrim().GetPath().AppendChild("beamMaterial")
        material = UsdShade.Material.Define(self.stage, material_path)
        shader = UsdShade.Shader.Define(self.stage, material_path.AppendChild("Shader"))
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1.0, 0.7, 0.7))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        binding_api = UsdShade.MaterialBindingAPI.Apply(skin_mesh.GetPrim())
        binding_api.Bind(material)
        
        # Dragon Skin 10 material
        deformable_material_path = env.GetPrim().GetPath().AppendChild("dragonSkin10").pathString
        self.deformable_material = DeformableMaterial(
            prim_path=deformable_material_path,
            dynamic_friction=0.5,
            youngs_modulus=234900.0,
            poissons_ratio=0.439,
            damping_scale=0.0,
            elasticity_damping=0.0,
            # density=1210.0
        )

        print(f"\nDeformable Configuration:")
        print(f"  Beam: {self.beam_length*100:.1f}cm × {self.beam_width*100:.1f}cm × {self.beam_height*100:.1f}cm")
        print(f"  Starting position: Horizontal at z = {beam_center[2]}m")
        print(f"  Resolution: {self.resolution} elements")
        print(f"  Solver iterations: {args.solver_iterations}")
        
        self.deformable = SingleDeformablePrim(
            name="dragonSkinBeam",
            prim_path=str(mesh_path),
            deformable_material=self.deformable_material,
            vertex_velocity_damping=args.mass_damping,
            sleep_damping=0.0,
            sleep_threshold=0.0,
            settling_threshold=0.01,
            self_collision=False,
            solver_position_iteration_count=args.solver_iterations,
            kinematic_enabled=False,  # Start with physics
            simulation_hexahedral_resolution=self.resolution,
            collision_simplification=False,
        )
        self.my_world.scene.add(self.deformable)
        
        # Fixed boundary (anchor/clamp)
        anchor_path = env.GetPrim().GetPath().AppendChild("anchor")
        anchor_size = (0.01, self.beam_width * 1.2, self.beam_height * 1.2)
        anchor_pos = (beam_center[0] - self.beam_length * 0.5, beam_center[1], beam_center[2])

        anchor_mesh = physicsUtils.create_mesh_cube(self.stage, str(anchor_path), 0.5)
        physicsUtils.set_or_add_scale_op(anchor_mesh, anchor_size)
        physicsUtils.setup_transform_as_scale_orient_translate(anchor_mesh)
        physicsUtils.set_or_add_translate_op(anchor_mesh, anchor_pos)

        anchor_prim = self.stage.GetPrimAtPath(str(anchor_path))
        rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(anchor_prim)
        rigid_body_api.CreateKinematicEnabledAttr().Set(True)
        UsdPhysics.CollisionAPI.Apply(anchor_prim)
        
        # Anchor material
        anchor_material_path = env.GetPrim().GetPath().AppendChild("anchorMaterial")
        anchor_material = UsdShade.Material.Define(self.stage, anchor_material_path)
        anchor_shader = UsdShade.Shader.Define(self.stage, anchor_material_path.AppendChild("Shader"))
        anchor_shader.CreateIdAttr("UsdPreviewSurface")
        anchor_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.2, 0.2, 0.2))
        anchor_material.CreateSurfaceOutput().ConnectToSource(anchor_shader.ConnectableAPI(), "surface")
        anchor_binding = UsdShade.MaterialBindingAPI.Apply(anchor_mesh.GetPrim())
        anchor_binding.Bind(anchor_material)

        # Create attachment
        attachment_path = mesh_path.AppendElementString("attachment")
        attachment = PhysxSchema.PhysxPhysicsAttachment.Define(self.stage, attachment_path)
        attachment.GetActor0Rel().SetTargets([mesh_path])
        attachment.GetActor1Rel().SetTargets([anchor_path])
        PhysxSchema.PhysxAutoAttachmentAPI.Apply(attachment.GetPrim())
        
        print(f"✓ Environment created\n")

        # Create view for accessing deformable data
        self.deformableView = DeformablePrim(
            prim_paths_expr="/World/Env0/deformable", 
            name="deformableView"
        )
        self.my_world.scene.add(self.deformableView)
        
        # Initialize
        self.my_world.reset(soft=False)
        self.initial_positions = self.deformableView.get_simulation_mesh_nodal_positions().clone()
        self._find_tracking_points()

    def _find_tracking_points(self):
        """Find tip, left, and right tracking points"""
        if self.initial_positions is None or len(self.initial_positions) == 0:
            print("Warning: No positions available")
            return
        
        env_positions = self.initial_positions[0]
        
        # Find tip (maximum x)
        x_coords = [(idx, pos[0].item(), pos[1].item()) for idx, pos in enumerate(env_positions)]
        x_coords.sort(key=lambda x: x[1], reverse=True)
        
        # Get 3 furthest points in X (tip region)
        tip_candidates = x_coords[:3]
        
        # Among tip candidates, find left (min y) and right (max y)
        tip_indices = [t[0] for t in tip_candidates]
        tip_y_values = [env_positions[idx][1].item() for idx in tip_indices]
        
        left_local_idx = np.argmin(tip_y_values)
        right_local_idx = np.argmax(tip_y_values)
        
        self.left_node_idx = tip_indices[left_local_idx]
        self.right_node_idx = tip_indices[right_local_idx]
        self.tip_node_idx = tip_indices[1]  # Middle one
        
        print(f"Tracking points identified:")
        print(f"  Tip:   node {self.tip_node_idx:4d}, pos {env_positions[self.tip_node_idx].cpu().numpy()}")
        print(f"  Left:  node {self.left_node_idx:4d}, pos {env_positions[self.left_node_idx].cpu().numpy()}")
        print(f"  Right: node {self.right_node_idx:4d}, pos {env_positions[self.right_node_idx].cpu().numpy()}")
        print()

    def setup_camera(self):
        """Position camera"""
        from omni.isaac.core.utils.viewports import set_camera_view
        eye = Gf.Vec3d(0.20, 0.20, 2.0)
        target = Gf.Vec3d(0.0, 0.0, 1.95)
        set_camera_view(eye=eye, target=target, camera_prim_path="/OmniverseKit_Persp")

    def raise_beam_kinematically(self, current_time):
        """Kinematically interpolate beam positions from settled to horizontal"""
        if self.settled_positions is None or self.horizontal_positions is None:
            return
        
        # Calculate interpolation factor (0 at start, 1 at end)
        t = (current_time - self.raise_start_time) / self.raise_duration
        t = np.clip(t, 0.0, 1.0)
        
        # Smooth interpolation (ease-in-out)
        t_smooth = 3*t**2 - 2*t**3  # Smoothstep function
        
        # Interpolate positions
        current_positions = self.settled_positions + t_smooth * (self.horizontal_positions - self.settled_positions)
        
        # Set positions
        self.deformableView.set_simulation_mesh_nodal_positions(current_positions)

    def record_data(self, current_time):
        """Record z-positions of tracking points"""
        positions = self.deformableView.get_simulation_mesh_nodal_positions()
        
        if positions is None or len(positions) == 0:
            return
        
        env_positions = positions[0]
        
        # Get z-positions
        tip_z = env_positions[self.tip_node_idx, 2].cpu().item()
        left_z = env_positions[self.left_node_idx, 2].cpu().item()
        right_z = env_positions[self.right_node_idx, 2].cpu().item()
        
        # Store data
        self.time_history.append(current_time)
        self.phase_history.append(self.current_phase)
        self.tip_z_history.append(tip_z * 1000)  # Convert to mm
        self.left_z_history.append(left_z * 1000)
        self.right_z_history.append(right_z * 1000)

    def analyze_results(self):
        """Analyze motion focusing on release phase"""
        if len(self.time_history) < 10:
            print("Not enough data")
            return None
        
        times = np.array(self.time_history)
        phases = np.array(self.phase_history)
        tip_z = np.array(self.tip_z_history)
        left_z = np.array(self.left_z_history)
        right_z = np.array(self.right_z_history)
        
        # Find release phase data
        release_mask = phases == self.PHASE_OBSERVING
        release_times = times[release_mask] - self.release_time
        release_tip_z = tip_z[release_mask]
        
        # Calculate z-displacement from horizontal (reference at release)
        horizontal_z = tip_z[np.where(times >= self.release_time)[0][0]]
        tip_displacement = horizontal_z - tip_z
        
        results = {
            'times': times,
            'phases': phases,
            'tip_z': tip_z,
            'left_z': left_z,
            'right_z': right_z,
            'tip_displacement': tip_displacement,
            'release_times': release_times,
            'release_tip_z': release_tip_z,
            'horizontal_z': horizontal_z,
        }
        
        # Settling phase statistics
        settling_mask = phases == self.PHASE_SETTLING
        if np.any(settling_mask):
            settled_z = tip_z[settling_mask][-1]
            settling_displacement = horizontal_z - settled_z
            results['settled_z'] = settled_z
            results['settling_displacement'] = settling_displacement
            print(f"\nPhase 1 - Settling:")
            print(f"  Final settled z: {settled_z:.3f} mm")
            print(f"  Total drop: {settling_displacement:.3f} mm")
        
        # Release phase statistics
        if len(release_tip_z) > 0:
            results['max_displacement_after_release'] = np.max(horizontal_z - release_tip_z)
            results['final_z_after_release'] = release_tip_z[-1]
            
            print(f"\nPhase 3 - After Release:")
            print(f"  Initial (horizontal): {horizontal_z:.3f} mm")
            print(f"  Final: {release_tip_z[-1]:.3f} mm")
            print(f"  Max displacement: {results['max_displacement_after_release']:.3f} mm")
            
            # Check for oscillations
            if SCIPY_AVAILABLE and len(release_tip_z) > 20:
                peaks, _ = signal.find_peaks(release_tip_z, distance=10)
                if len(peaks) >= 2:
                    peak_times = release_times[peaks]
                    periods = np.diff(peak_times)
                    avg_period = np.mean(periods)
                    frequency = 1.0 / avg_period
                    
                    results['frequency_hz'] = frequency
                    results['period_s'] = avg_period
                    print(f"  Oscillation frequency: {frequency:.3f} Hz")
                    print(f"  Period: {avg_period:.4f} s")
        
        return results

    def plot_results(self, analysis):
        """Generate plots"""
        if not MATPLOTLIB_AVAILABLE:
            print("\nSkipping plot generation")
            return None
        
        print("\nGenerating plots...")
        
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        times = analysis['times']
        tip_z = analysis['tip_z']
        left_z = analysis['left_z']
        right_z = analysis['right_z']
        
        # Plot 1: Complete timeline with phases
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(times, tip_z, 'b-', linewidth=2, label='Tip Z')
        ax1.axvline(x=self.raise_start_time, color='orange', linestyle='--', linewidth=2, label='Start Raise')
        ax1.axvline(x=self.release_time, color='red', linestyle='--', linewidth=2, label='Release')
        
        # Shade phases
        ax1.axvspan(0, self.raise_start_time, alpha=0.2, color='gray', label='Phase 1: Settling')
        ax1.axvspan(self.raise_start_time, self.release_time, alpha=0.2, color='orange', label='Phase 2: Raising')
        ax1.axvspan(self.release_time, times[-1], alpha=0.2, color='green', label='Phase 3: Observing')
        
        ax1.set_xlabel('Time (s)', fontsize=11)
        ax1.set_ylabel('Tip Z Position (mm)', fontsize=11)
        ax1.set_title('Complete Timeline: Three-Phase Test', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='best', fontsize=9)
        
        # Plot 2: Settling phase detail
        ax2 = fig.add_subplot(gs[1, 0])
        settling_mask = analysis['phases'] == self.PHASE_SETTLING
        settling_times = times[settling_mask]
        settling_tip = tip_z[settling_mask]
        if len(settling_times) > 0:
            ax2.plot(settling_times, settling_tip, 'b-', linewidth=2)
            ax2.set_xlabel('Time (s)', fontsize=11)
            ax2.set_ylabel('Tip Z Position (mm)', fontsize=11)
            ax2.set_title('Phase 1: Settling Under Gravity', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Release phase detail (z-displacement from horizontal)
        ax3 = fig.add_subplot(gs[1, 1])
        release_mask = analysis['phases'] == self.PHASE_OBSERVING
        if np.any(release_mask):
            release_times = analysis['release_times']
            horizontal_z = analysis['horizontal_z']
            release_disp = horizontal_z - tip_z[release_mask]
            
            ax3.plot(release_times, release_disp, 'g-', linewidth=2)
            ax3.axhline(y=0, color='k', linestyle=':', alpha=0.5)
            ax3.set_xlabel('Time Since Release (s)', fontsize=11)
            ax3.set_ylabel('Displacement from Horizontal (mm)', fontsize=11)
            ax3.set_title('Phase 3: Motion After Release', fontsize=12, fontweight='bold')
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Left vs Right comparison (release phase)
        ax4 = fig.add_subplot(gs[2, 0])
        if np.any(release_mask):
            ax4.plot(release_times, left_z[release_mask], 'b-', linewidth=2, label='Left')
            ax4.plot(release_times, right_z[release_mask], 'r-', linewidth=2, label='Right')
            ax4.set_xlabel('Time Since Release (s)', fontsize=11)
            ax4.set_ylabel('Z Position (mm)', fontsize=11)
            ax4.set_title('Left & Right Markers After Release', fontsize=12, fontweight='bold')
            ax4.grid(True, alpha=0.3)
            ax4.legend()
        
        # Plot 5: Summary
        ax5 = fig.add_subplot(gs[2, 1])
        
        summary_text = f"""
THREE-PHASE VALIDATION SUMMARY

Material: Dragon Skin 10
  E = 234,900 Pa, ν = 0.439
  ρ = 1,210 kg/m³, α = {args.mass_damping} s⁻¹

Beam: {self.beam_length*100:.1f} × {self.beam_width*100:.1f} × {self.beam_height*100:.1f} cm

PHASE TIMING:
  Settling:  0 to {self.settling_time}s
  Raising:   {self.settling_time}s to {self.release_time}s
  Observing: {self.release_time}s to {self.total_time}s

RESULTS:
"""
        
        if 'settled_z' in analysis:
            summary_text += f"  Settled position: {analysis['settled_z']:.3f} mm\n"
            summary_text += f"  Settling drop: {analysis['settling_displacement']:.3f} mm\n"
        
        if 'horizontal_z' in analysis:
            summary_text += f"\n  Horizontal (release): {analysis['horizontal_z']:.3f} mm\n"
        
        if 'max_displacement_after_release' in analysis:
            summary_text += f"  Max drop after release: {analysis['max_displacement_after_release']:.3f} mm\n"
        
        if 'frequency_hz' in analysis:
            summary_text += f"\n  Oscillation freq: {analysis['frequency_hz']:.3f} Hz\n"
        
        ax5.text(0.1, 0.5, summary_text, transform=ax5.transAxes,
                fontsize=9, verticalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax5.axis('off')
        
        fig.suptitle('Dragon Skin 10 - Reverse Release Validation (Settle → Raise → Release)', 
                    fontsize=14, fontweight='bold')
        
        # Save
        import os
        output_dir = '/home/claude/experiments/'
        os.makedirs(output_dir, exist_ok=True)
        
        filename = os.path.join(output_dir, 'cantilever_reverse_release.png')
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"✓ Plot saved: {filename}")
        
        return fig

    def save_data_csv(self, output_dir, analysis):
        """Save data to CSV"""
        import csv
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Full timeline data
        csv_filename = os.path.join(output_dir, 'reverse_release_full.csv')
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['time_s', 'phase', 'tip_z_mm', 'left_z_mm', 'right_z_mm'])
            
            for i in range(len(self.time_history)):
                phase_name = ['SETTLING', 'RAISING', 'OBSERVING'][self.phase_history[i]]
                writer.writerow([
                    self.time_history[i],
                    phase_name,
                    self.tip_z_history[i],
                    self.left_z_history[i],
                    self.right_z_history[i],
                ])
        
        print(f"✓ Full timeline saved: {csv_filename}")
        
        # Release phase only (for comparison)
        release_mask = analysis['phases'] == self.PHASE_OBSERVING
        if np.any(release_mask):
            release_filename = os.path.join(output_dir, 'reverse_release_observation.csv')
            with open(release_filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['time_since_release_s', 'tip_z_mm', 'left_z_mm', 'right_z_mm'])
                
                release_times = analysis['release_times']
                tip_z = analysis['tip_z'][release_mask]
                left_z = analysis['left_z'][release_mask]
                right_z = analysis['right_z'][release_mask]
                
                for i in range(len(release_times)):
                    writer.writerow([
                        release_times[i],
                        tip_z[i],
                        left_z[i],
                        right_z[i],
                    ])
            
            print(f"✓ Release phase saved: {release_filename}")

    def play(self):
        """Run simulation"""
        print(f"\nStarting three-phase simulation...\n")
        
        step_count = 0
        
        while simulation_app.is_running():
            if self.my_world.is_playing():
                current_time = self.my_world.current_time
                
                # Phase 1: SETTLING
                if current_time < self.raise_start_time:
                    if self.current_phase != self.PHASE_SETTLING:
                        print(f"\n{'='*60}")
                        print(f"PHASE 1: SETTLING (letting beam drop under gravity)")
                        print(f"{'='*60}\n")
                        self.current_phase = self.PHASE_SETTLING
                    
                    if step_count > 5:
                        self.record_data(current_time)
                        
                        if len(self.time_history) % 30 == 0:
                            print(f"[SETTLING] t={current_time:.3f}s | Tip z: {self.tip_z_history[-1]:.3f} mm")
                
                # Transition to Phase 2: RAISING
                elif current_time >= self.raise_start_time and current_time < self.release_time:
                    if self.current_phase != self.PHASE_RAISING:
                        print(f"\n{'='*60}")
                        print(f"PHASE 2: RAISING (kinematically moving to horizontal)")
                        print(f"{'='*60}\n")
                        
                        # Store settled positions
                        self.settled_positions = self.deformableView.get_simulation_mesh_nodal_positions().clone()
                        self.horizontal_positions = self.initial_positions.clone()
                        
                        # Enable kinematic mode
                        self.deformable.set_kinematic_enabled(True)
                        
                        self.current_phase = self.PHASE_RAISING
                    
                    # Kinematically interpolate
                    self.raise_beam_kinematically(current_time)
                    self.record_data(current_time)
                    
                    if len(self.time_history) % 10 == 0:
                        print(f"[RAISING] t={current_time:.3f}s | Progress: {((current_time - self.raise_start_time) / self.raise_duration * 100):.1f}%")
                
                # Transition to Phase 3: OBSERVING
                elif current_time >= self.release_time and current_time < self.total_time:
                    if self.current_phase != self.PHASE_OBSERVING:
                        print(f"\n{'='*60}")
                        print(f"PHASE 3: RELEASE (observing motion from horizontal)")
                        print(f"{'='*60}\n")
                        
                        # Disable kinematic mode (release!)
                        self.deformable.set_kinematic_enabled(False)
                        
                        self.current_phase = self.PHASE_OBSERVING
                    
                    self.record_data(current_time)
                    
                    if len(self.time_history) % 20 == 0:
                        print(f"[OBSERVING] t={current_time - self.release_time:.3f}s | Tip z: {self.tip_z_history[-1]:.3f} mm")
                
                # End simulation
                elif current_time >= self.total_time:
                    print(f"\n{'='*60}")
                    print(f"Simulation complete - analyzing results...")
                    print(f"{'='*60}\n")
                    
                    analysis = self.analyze_results()
                    
                    output_dir = '/home/claude/experiments/'
                    self.save_data_csv(output_dir, analysis)
                    
                    if MATPLOTLIB_AVAILABLE:
                        self.plot_results(analysis)
                    
                    print(f"\n✓ All results saved to {output_dir}")
                    print("\nSimulation complete. Close window to exit.")
                    
                    while simulation_app.is_running():
                        self.my_world.step(render=True)
                    break
            
            self.my_world.step(render=True)
            step_count += 1

        simulation_app.close()


if __name__ == "__main__":
    DragonSkinReverseReleaseValidation().play()