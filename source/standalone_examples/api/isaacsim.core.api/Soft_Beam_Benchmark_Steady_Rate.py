# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""
Cantilever Beam Validation Test - Dragon Skin 10
Based on: "Sim-to-Real for Soft Robots using Differentiable FEM"

Experimental Setup (from paper):
"We keep the beam at its horizontal starting position and then release it 
so it bends downwards under gravity"

Two scenarios:
1) Gravity only - beam released from horizontal position
2) Gravity + external force F applied along tip edge
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
    description='Dragon Skin 10 Cantilever Release Validation',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument("--beam_length", type=float, default=0.10, 
                    help="Beam length in meters (10cm in paper)")
parser.add_argument("--beam_width", type=float, default=0.03, 
                    help="Beam width in meters (3cm in paper)")
parser.add_argument("--beam_height", type=float, default=0.03, 
                    help="Beam height in meters (3cm in paper)")
parser.add_argument("--resolution", type=int, default=30, 
                    help="Simulation mesh resolution")
parser.add_argument("--solver_iterations", type=int, default=130,
                    help="Position solver iterations per timestep")
parser.add_argument("--simulation_time", type=float, default=3.0, 
                    help="Total simulation time (seconds)")
parser.add_argument("--external_force", type=float, default=0.02, 
                    help="External downward force at tip (N). Use 0 for gravity-only test")
parser.add_argument("--mass_damping", type=float, default=9.11, 
                    help="Mass-proportional damping coefficient α (s^-1)")
parser.add_argument("--start_frozen", action='store_true',
                    help="Start with beam held horizontal (kinematic) then release")
args, unknown = parser.parse_known_args()


class DragonSkinReleaseValidation:
    """
    Validation matching paper's experimental setup:
    "We keep the beam at its horizontal starting position and then release it 
    so it bends downwards under gravity"
    
    The beam starts perfectly horizontal and is released at t=0.
    We track Left and Right reference points as in the paper.
    """
    
    def __init__(self):
        # Timing
        self.time_step = 0.01  # 100 Hz (paper captures at 100Hz)
        self.simulation_time = args.simulation_time
        self.start_frozen = args.start_frozen
        self.release_time = 0.1 if self.start_frozen else 0.0  # Small delay if starting frozen
        
        print(f"\n{'='*80}")
        print(f"Dragon Skin 10 Cantilever Release - Validation Test")
        print(f"Based on: 'Sim-to-Real for Soft Robots using Differentiable FEM'")
        print(f"{'='*80}")
        print(f"\nExperimental Setup:")
        print(f"  Beam starts: {'Held horizontal (kinematic)' if self.start_frozen else 'At rest horizontal'}")
        print(f"  Released at: t = {self.release_time}s")
        print(f"  External force: {args.external_force} N {'(gravity only)' if args.external_force == 0 else ''}")
        print(f"  Recording: 0 to {self.simulation_time}s at 100 Hz")
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
        
        # External force
        self.external_force = args.external_force
        self.force_active = (self.external_force > 0.0)
        
        # Data recording
        self.time_history = []
        self.left_point_history = []   # Left reference point (paper) - z displacement only
        self.right_point_history = []  # Right reference point (paper) - z displacement only
        self.tip_deflection_history = []
        
        self.initial_positions = None
        self.left_point_idx = None
        self.right_point_idx = None
        self.tip_node_idx = None
        
        self.released = not self.start_frozen  # Track if beam has been released
        
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
        print("Creating validation environment...")
        
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
        
        # Visual material (salmon pink for silicone)
        material_path = env.GetPrim().GetPath().AppendChild("beamMaterial")
        material = UsdShade.Material.Define(self.stage, material_path)
        shader = UsdShade.Shader.Define(self.stage, material_path.AppendChild("Shader"))
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1.0, 0.7, 0.7))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        binding_api = UsdShade.MaterialBindingAPI.Apply(skin_mesh.GetPrim())
        binding_api.Bind(material)
        
        # Dragon Skin 10 material - optimized parameters from paper
        deformable_material_path = env.GetPrim().GetPath().AppendChild("dragonSkin10").pathString
        self.deformable_material = DeformableMaterial(
            prim_path=deformable_material_path,
            dynamic_friction=0.0,
            youngs_modulus=234900.0,    # E = 234.9 kPa from paper
            poissons_ratio=0.439,       # ν = 0.439 from paper
            damping_scale=0.0,
            elasticity_damping=0.0,
            # density=1210.0              # ρ = 1,210 kg/m³ from paper
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
            vertex_velocity_damping=args.mass_damping,  # Mass-proportional damping α
            sleep_damping=0.0,
            sleep_threshold=0.0,
            settling_threshold=0.01,
            self_collision=False,
            solver_position_iteration_count=args.solver_iterations,
            kinematic_enabled=self.start_frozen,  # Start kinematic if frozen
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
        self.initial_positions = self.deformableView.get_simulation_mesh_nodal_positions().cpu()
        self._find_tracking_points()

    def _find_tracking_points(self):
        """
        Find tracking points matching paper's setup:
        - "Left" and "Right" reference points
        - Tip node for deflection measurement
        """
        if self.initial_positions is None or len(self.initial_positions) == 0:
            print("Warning: No positions available")
            return
        
        env_positions = self.initial_positions[0]
        
        # Sort by x-coordinate to find key points
        x_coords = [(idx, pos[0].item(), pos[1].item()) for idx, pos in enumerate(env_positions)]
        x_coords.sort(key=lambda x: x[1])  # Sort by x position
        
        num_points = len(x_coords)
        
        # Tip is at maximum x (free end)
        self.tip_node_idx = x_coords[-1][0]
        
        # "Left" and "Right" reference points - paper mentions tracking these
        # Assuming they are on opposite sides of the beam near the tip
        # Find points at tip x-position but different y-coordinates
        tip_x = x_coords[-1][1]
        
        # Find all points near the tip
        tip_points = [(idx, x, y) for idx, x, y in x_coords if abs(x - tip_x) < 0.001]
        tip_points.sort(key=lambda p: p[2])  # Sort by y coordinate
        
        if len(tip_points) >= 2:
            self.left_point_idx = tip_points[0][0]   # Minimum y (left)
            self.right_point_idx = tip_points[-1][0]  # Maximum y (right)
        else:
            # Fallback: use quarter and three-quarter points
            self.left_point_idx = x_coords[3 * num_points // 4][0]
            self.right_point_idx = x_coords[-1][0]
        
        print(f"Tracking points identified:")
        print(f"  Left point:  node {self.left_point_idx:4d}, pos {env_positions[self.left_point_idx].numpy()}")
        print(f"  Right point: node {self.right_point_idx:4d}, pos {env_positions[self.right_point_idx].numpy()}")
        print(f"  Tip point:   node {self.tip_node_idx:4d}, pos {env_positions[self.tip_node_idx].numpy()}")
        print()

    def setup_camera(self):
        """Position camera to view release"""
        from omni.isaac.core.utils.viewports import set_camera_view
        eye = Gf.Vec3d(0.20, 0.20, 2.0)
        target = Gf.Vec3d(0.0, 0.0, 1.95)
        set_camera_view(eye=eye, target=target, camera_prim_path="/OmniverseKit_Persp")

    def apply_external_force(self):
        """Apply external force to tip if configured"""
        if not self.force_active or self.tip_node_idx is None:
            return
        
        positions = self.deformableView.get_simulation_mesh_nodal_positions()
        if positions is None or len(positions) == 0:
            return
        
        # Create force array
        num_nodes = positions.shape[1]
        forces = torch.zeros((1, num_nodes, 3), device=positions.device)
        
        # Apply downward force to tip
        forces[0, self.tip_node_idx, 2] = -self.external_force
        
        self.deformableView.apply_forces(forces, indices=None)

    def release_beam(self):
        """Release beam from kinematic hold"""
        if not self.released:
            print(f"\n{'='*60}")
            print(f"RELEASING BEAM at t={self.my_world.current_time:.3f}s")
            print(f"{'='*60}\n")
            
            # Disable kinematic mode
            self.deformable.set_kinematic_enabled(False)
            self.released = True

    def record_data(self, current_time):
        """Record z-displacement of tracking points"""
        positions = self.deformableView.get_simulation_mesh_nodal_positions()
        
        if positions is None or len(positions) == 0:
            return
        
        env_positions = positions[0]
        
        # Get z-positions of tracked points
        left_z = env_positions[self.left_point_idx, 2].cpu().item()
        right_z = env_positions[self.right_point_idx, 2].cpu().item()
        tip_z = env_positions[self.tip_node_idx, 2].cpu().item()
        
        # Calculate z-displacements from initial horizontal position (mm)
        initial_left_z = self.initial_positions[0, self.left_point_idx, 2].item()
        initial_right_z = self.initial_positions[0, self.right_point_idx, 2].item()
        initial_tip_z = self.initial_positions[0, self.tip_node_idx, 2].item()
        
        left_displacement = (initial_left_z - left_z) * 1000  # mm, positive = downward
        right_displacement = (initial_right_z - right_z) * 1000
        tip_displacement = tip_z*1000 #(initial_tip_z - tip_z) * 1000
        
        # Store data
        self.time_history.append(current_time)
        self.left_point_history.append(left_displacement)
        self.right_point_history.append(right_displacement)
        self.tip_deflection_history.append(tip_displacement)

    def analyze_results(self):
        """Analyze beam displacement"""
        if len(self.time_history) < 10:
            print("Not enough data for analysis")
            return None
        
        times = np.array(self.time_history)
        tip_displacements = np.array(self.tip_deflection_history)
        left_displacements = np.array(self.left_point_history)
        right_displacements = np.array(self.right_point_history)
        
        results = {
            'times': times,
            'tip_displacements': tip_displacements,
            'left_displacements': left_displacements,
            'right_displacements': right_displacements,
            'max_tip_displacement': np.max(tip_displacements),
            'final_tip_displacement': tip_displacements[-1],
        }
        
        # Find if there are oscillations
        if SCIPY_AVAILABLE:
            peaks, _ = signal.find_peaks(tip_displacements, distance=10)
            
            if len(peaks) >= 2:
                peak_times = times[peaks]
                periods = np.diff(peak_times)
                avg_period = np.mean(periods)
                frequency = 1.0 / avg_period
                
                results['num_peaks'] = len(peaks)
                results['frequency_hz'] = frequency
                results['period_s'] = avg_period
                
                print(f"\nMotion Analysis:")
                print(f"  Detected {len(peaks)} oscillation peaks")
                print(f"  Frequency: {frequency:.3f} Hz")
                print(f"  Period: {avg_period:.4f} s")
        
        print(f"\nDisplacement Summary:")
        print(f"  Maximum tip displacement: {results['max_tip_displacement']:.3f} mm")
        print(f"  Final tip displacement: {results['final_tip_displacement']:.3f} mm")
        print(f"  Final left displacement: {left_displacements[-1]:.3f} mm")
        print(f"  Final right displacement: {right_displacements[-1]:.3f} mm")
        
        return results

    def plot_results(self, analysis):
        """Generate validation plots showing z-displacement"""
        if not MATPLOTLIB_AVAILABLE:
            print("\nSkipping plot generation (matplotlib not available)")
            return None
        
        print("\nGenerating plots...")
        
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        times = analysis['times']
        tip_disp = analysis['tip_displacements']
        left_disp = analysis['left_displacements']
        right_disp = analysis['right_displacements']
        
        # Plot 1: Tip z-displacement vs time
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(times, tip_disp, 'b-', linewidth=2, label='Tip Z-Displacement')
        if self.start_frozen:
            ax1.axvline(x=self.release_time, color='r', linestyle='--', 
                       linewidth=2, label='Release')
        ax1.set_xlabel('Time (s)', fontsize=11)
        ax1.set_ylabel('Z-Displacement (mm)', fontsize=11)
        ax1.set_title('Tip Z-Displacement vs Time', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Left & Right point comparison
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(times, left_disp, 'b-', linewidth=2, label='Left Point')
        ax2.plot(times, right_disp, 'r-', linewidth=2, label='Right Point')
        if self.start_frozen:
            ax2.axvline(x=self.release_time, color='gray', linestyle='--', linewidth=1)
        ax2.set_xlabel('Time (s)', fontsize=11)
        ax2.set_ylabel('Z-Displacement (mm)', fontsize=11)
        ax2.set_title('Left & Right Reference Points (Paper Tracking)', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Plot 3: All three points together
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(times, tip_disp, 'purple', linewidth=2, label='Tip', alpha=0.8)
        ax3.plot(times, left_disp, 'blue', linewidth=2, label='Left', alpha=0.8)
        ax3.plot(times, right_disp, 'red', linewidth=2, label='Right', alpha=0.8)
        if self.start_frozen:
            ax3.axvline(x=self.release_time, color='gray', linestyle='--', linewidth=1)
        ax3.set_xlabel('Time (s)', fontsize=11)
        ax3.set_ylabel('Z-Displacement (mm)', fontsize=11)
        ax3.set_title('All Tracking Points Z-Displacement', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Plot 4: Summary
        ax4 = fig.add_subplot(gs[1, 1])
        
        summary_text = f"""
VALIDATION SUMMARY

Setup: {'Held then released' if self.start_frozen else 'Released from rest'}
External Force: {self.external_force} N

Material: Dragon Skin 10
  E = 234,900 Pa
  ν = 0.439
  ρ = 1,210 kg/m³
  α = {args.mass_damping} s⁻¹

Beam: {self.beam_length*100:.1f} × {self.beam_width*100:.1f} × {self.beam_height*100:.1f} cm
Resolution: {self.resolution} elements

RESULTS (Z-Displacement):
  Max Tip: {analysis['max_tip_displacement']:.3f} mm
  Final Tip: {analysis['final_tip_displacement']:.3f} mm
  Final Left: {left_disp[-1]:.3f} mm
  Final Right: {right_disp[-1]:.3f} mm
"""
        
        if 'frequency_hz' in analysis and analysis['frequency_hz'] is not None:
            summary_text += f"\n  Oscillation Freq: {analysis['frequency_hz']:.3f} Hz"
            summary_text += f"\n  Period: {analysis['period_s']:.4f} s"
        
        ax4.text(0.1, 0.5, summary_text, transform=ax4.transAxes,
                fontsize=10, verticalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax4.axis('off')
        
        title = f'Dragon Skin 10 Cantilever Release - Z-Displacement Validation'
        if self.external_force > 0:
            title += f' (with {self.external_force}N force)'
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        # Save
        import os
        output_dir = '/home/rjrosales/Simulation/Custom_IsaacSim/experiments/'
        os.makedirs(output_dir, exist_ok=True)
        
        filename = os.path.join(output_dir, 'cantilever_z_displacement.png')
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"✓ Plot saved: {filename}")
        
        return fig

    def save_data_csv(self, output_dir, analysis):
        """Save z-displacement tracking data to CSV"""
        import csv
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Main tracking data - z-displacement only
        csv_filename = os.path.join(output_dir, 'beam_z_displacement.csv')
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['time_s', 'tip_z_displacement_mm', 'left_z_displacement_mm', 'right_z_displacement_mm'])
            
            for i in range(len(self.time_history)):
                writer.writerow([
                    self.time_history[i],
                    self.tip_deflection_history[i],
                    self.left_point_history[i],
                    self.right_point_history[i],
                ])
        
        print(f"✓ Z-displacement data saved: {csv_filename}")
        
        # Summary
        summary_filename = os.path.join(output_dir, 'validation_summary.csv')
        with open(summary_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Parameter', 'Value'])
            writer.writerow(['Beam Length (m)', self.beam_length])
            writer.writerow(['Beam Width (m)', self.beam_width])
            writer.writerow(['Beam Height (m)', self.beam_height])
            writer.writerow(['External Force (N)', self.external_force])
            writer.writerow(['Start Frozen', self.start_frozen])
            writer.writerow(['Release Time (s)', self.release_time])
            writer.writerow(['Young\'s Modulus (Pa)', 234900.0])
            writer.writerow(['Poisson\'s Ratio', 0.439])
            writer.writerow(['Density (kg/m³)', 1210.0])
            writer.writerow(['Mass Damping (s^-1)', args.mass_damping])
            writer.writerow(['Max Tip Displacement (mm)', analysis['max_tip_displacement']])
            writer.writerow(['Final Tip Displacement (mm)', analysis['final_tip_displacement']])
            writer.writerow(['Final Left Displacement (mm)', analysis['left_displacements'][-1]])
            writer.writerow(['Final Right Displacement (mm)', analysis['right_displacements'][-1]])
            
            if 'frequency_hz' in analysis and analysis['frequency_hz'] is not None:
                writer.writerow(['Oscillation Frequency (Hz)', analysis['frequency_hz']])
                writer.writerow(['Period (s)', analysis['period_s']])
        
        print(f"✓ Summary saved: {summary_filename}")

    def play(self):
        """Run simulation"""
        print(f"\nStarting simulation (0 to {self.simulation_time}s)...\n")
        
        step_count = 0
        
        while simulation_app.is_running():
            if self.my_world.is_playing():
                current_time = self.my_world.current_time
                
                # Release beam if frozen and time reached
                if self.start_frozen and current_time >= self.release_time and not self.released:
                    self.release_beam()
                
                # Apply external force if configured
                if self.released and self.force_active:
                    self.apply_external_force()
                
                # Record data after initial steps
                if step_count > 5 and current_time <= self.simulation_time:
                    self.record_data(current_time)
                    
                    if len(self.time_history) % 20 == 0:
                        print(f"t={current_time:.3f}s | Tip displacement: {self.tip_deflection_history[-1]:.3f} mm")
                
                # End simulation
                if current_time >= self.simulation_time:
                    print(f"\n{'='*60}")
                    print(f"Simulation complete - analyzing results...")
                    print(f"{'='*60}\n")
                    
                    analysis = self.analyze_results()
                    
                    output_dir = '/home/rjrosales/Simulation/Custom_IsaacSim/experiments/'
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
    DragonSkinReleaseValidation().play()