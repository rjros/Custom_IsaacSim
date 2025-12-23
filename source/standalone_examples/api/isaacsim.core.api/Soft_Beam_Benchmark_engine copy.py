# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""
Cantilever Beam Validation Test - Dragon Skin 10
Based on: "Sim-to-Real for Soft Robots using Differentiable FEM"
Material: Smooth-On Dragon Skin 10 (Shore Hardness 10A)
Time window: 0 to 1.5 seconds
"""

from isaacsim import SimulationApp

simulation_app = SimulationApp({
    "headless": False, 
    "extra_args": ["--/app/useFabricSceneDelegate=0"]
})

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from isaacsim.core.api import World
from isaacsim.core.api.materials.deformable_material import DeformableMaterial
from isaacsim.core.prims import DeformablePrim, SingleDeformablePrim
from omni.physx.scripts import physicsUtils
from pxr import Gf, UsdGeom, PhysxSchema, UsdPhysics, Sdf, UsdShade, UsdLux
import isaacsim.core.utils.deformable_mesh_utils as deformableMeshUtils

parser = argparse.ArgumentParser()
parser.add_argument("--beam_length", type=float, default=0.10, help="Beam length in meters")
parser.add_argument("--beam_width", type=float, default=0.03, help="Beam width in meters")
parser.add_argument("--beam_height", type=float, default=0.03, help="Beam height in meters")
parser.add_argument("--resolution", type=int, default=20, help="Simulation mesh resolution")
parser.add_argument("--max_time", type=float, default=1.5, help="Simulation time in seconds")
args, unknown = parser.parse_known_args()


class DragonSkinCantileverValidation:
    """
    Validation test matching paper parameters:
    - Material: Dragon Skin 10 (Shore 10A)
    - Poisson's ratio: ν = 0.4999 (incompressible)
    - Density: ρ = 1070 kg/m³
    - Time step: h = 0.01s
    - Recording window: 0 to 1.5s
    """
    
    def __init__(self):
        # Paper parameters
        self.time_step = 0.01  # h = 0.01s from paper
        self.max_simulation_time = args.max_time
        
        print(f"\n{'='*80}")
        print(f"Dragon Skin 10 Cantilever Beam - Validation Test")
        print(f"Based on: 'Sim-to-Real for Soft Robots using Differentiable FEM'")
        print(f"{'='*80}")
        print(f"\nPaper Parameters:")
        print(f"  Material:          Smooth-On Dragon Skin 10 (Shore Hardness 10A)")
        print(f"  Poisson's Ratio:   ν = 0.4999 (nearly incompressible)")
        print(f"  Density:           ρ = 1,070 kg/m³")
        print(f"  Time Step:         h = 0.01s")
        print(f"  Recording Window:  0 to {self.max_simulation_time}s")
        print(f"{'='*80}\n")
        
        self.my_world = World(
            stage_units_in_meters=1.0, 
            backend="torch", 
            device="cuda",
            physics_dt=self.time_step,      # Match paper exactly
            rendering_dt=self.time_step * 2 # Render every 2 physics steps
        )
        self.stage = simulation_app.context.get_stage()
        
        # Beam geometry
        self.beam_length = args.beam_length
        self.beam_width = args.beam_width
        self.beam_height = args.beam_height
        self.resolution = args.resolution
        
        # Data recording
        self.time_history = []
        self.deflection_history = {
            'fixed_end': [],
            'quarter': [],
            'midpoint': [],
            'three_quarter': [],
            'free_end': []
        }
        self.position_history = {
            'fixed_end': [],
            'quarter': [],
            'midpoint': [],
            'three_quarter': [],
            'free_end': []
        }
        
        self.initial_positions = None
        self.tracked_point_indices = {}
        
        # Setup environment
        self.my_world.scene.add_default_ground_plane()
        self.setup_environment()
        self.makeEnvs()
        self.setup_camera()
        
        # Calculate theoretical deflection
        self.calculate_theoretical()

    def calculate_theoretical(self):
        """Calculate theoretical tip deflection using beam theory"""
        # Material properties
        E = 263824.0  # Young's modulus (Pa)
        rho = 1070.0  # Density (kg/m³)
        g = 9.81      # Gravity (m/s²)
        
        # Beam geometry
        L = self.beam_length
        w = self.beam_width
        h = self.beam_height
        
        # Second moment of area (rectangular cross-section)
        I = (w * h**3) / 12
        
        # Weight per unit length
        weight_per_length = rho * w * h * g
        
        # Theoretical tip deflection for distributed load (beam's own weight)
        # δ = (w*L^4)/(8*E*I) where w is weight per length
        self.theoretical_tip_deflection = (weight_per_length * L**4) / (8 * E * I)
        
        print(f"Theoretical Analysis:")
        print(f"  Second moment of area: I = {I:.6e} m⁴")
        print(f"  Weight per length:     w = {weight_per_length:.4f} N/m")
        print(f"  Expected tip deflection: δ = {self.theoretical_tip_deflection*1000:.4f} mm")
        print(f"{'='*80}\n")

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
        
        # Visual mesh (high resolution for smooth rendering)
        tri_points, tri_indices = deformableMeshUtils.createTriangleMeshCube(16)
        skin_mesh.GetPointsAttr().Set(tri_points)
        skin_mesh.GetFaceVertexIndicesAttr().Set(tri_indices)
        skin_mesh.GetFaceVertexCountsAttr().Set([3] * (len(tri_indices) // 3))
        
        # Setup transforms
        physicsUtils.setup_transform_as_scale_orient_translate(skin_mesh)
        beam_size = Gf.Vec3f(self.beam_length, self.beam_width, self.beam_height)
        physicsUtils.set_or_add_scale_op(skin_mesh, beam_size)
        beam_center = Gf.Vec3f(0.0, 0.0, 1.0)
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
        
        # Dragon Skin 10 material properties (from paper)
        deformable_material_path = env.GetPrim().GetPath().AppendChild("dragonSkin10").pathString
        self.deformable_material = DeformableMaterial(
            prim_path=deformable_material_path,
            dynamic_friction=0.5,
            youngs_modulus=263824.0,    # E (will be calibrated in paper)
            poissons_ratio=0.4999,      # ν = 0.4999 (nearly incompressible)
            damping_scale=0.0,          # No artificial damping
            elasticity_damping=0.0,
            # density=1070.0              # ρ = 1,070 kg/m³
        )

        print(f"\nDeformable Configuration:")
        print(f"  Beam: {self.beam_length*100:.1f}cm × {self.beam_width*100:.1f}cm × {self.beam_height*100:.1f}cm")
        print(f"  Resolution: {self.resolution} elements (longest dimension)")
        
        self.deformable = SingleDeformablePrim(
            name="dragonSkinBeam",
            prim_path=str(mesh_path),
            deformable_material=self.deformable_material,
            vertex_velocity_damping=0.0,
            sleep_damping=0.0,
            sleep_threshold=0.0,
            settling_threshold=0.0,
            self_collision=False,
            solver_position_iteration_count=30,
            kinematic_enabled=False,
            simulation_hexahedral_resolution=self.resolution,
            collision_simplification=False,
        )
        self.my_world.scene.add(self.deformable)
        
        # Fixed boundary (anchor)
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
        self._setup_tracking_points()

    def _setup_tracking_points(self):
        """Setup tracking points along beam"""
        if self.initial_positions is None or len(self.initial_positions) == 0:
            print("Warning: No positions available")
            return
        
        print(f"Setting up tracking points...")
        env_positions = self.initial_positions[0]
        
        # Sort by x-coordinate
        x_coords = [(idx, pos[0].item()) for idx, pos in enumerate(env_positions)]
        x_coords.sort(key=lambda x: x[1])
        
        num_points = len(x_coords)
        positions = {
            'fixed_end': 0,
            'quarter': num_points // 4,
            'midpoint': num_points // 2,
            'three_quarter': 3 * num_points // 4,
            'free_end': num_points - 1
        }
        
        print(f"Tracking {len(positions)} points:")
        for label, pos_idx in positions.items():
            idx, x_val = x_coords[pos_idx]
            self.tracked_point_indices[label] = idx
            node_pos = env_positions[idx]
            print(f"  {label:15s}: node {idx:4d}, pos ({node_pos[0]:7.4f}, {node_pos[1]:7.4f}, {node_pos[2]:7.4f})")
        
        print()

    def setup_camera(self):
        """Position camera"""
        from omni.isaac.core.utils.viewports import set_camera_view
        eye = Gf.Vec3d(0.15, 0.15, 1.0)
        target = Gf.Vec3d(0.0, 0.0, 1.0)
        set_camera_view(eye=eye, target=target, camera_prim_path="/OmniverseKit_Persp")

    def record_data(self, current_time):
        """Record deflection data at current timestep"""
        current_positions = self.deformableView.get_simulation_mesh_nodal_positions()
        
        if current_positions is None or len(current_positions) == 0:
            return
        
        env_positions = current_positions[0]
        initial_positions = self.initial_positions[0]
        
        self.time_history.append(current_time)
        
        for label, idx in self.tracked_point_indices.items():
            current_z = env_positions[idx, 2].item()
            initial_z = initial_positions[idx, 2].item()
            deflection = initial_z - current_z  # Positive = downward
            
            self.deflection_history[label].append(deflection * 1000)  # Convert to mm
            self.position_history[label].append([
                env_positions[idx, 0].item(),
                env_positions[idx, 1].item(),
                current_z
            ])

    def plot_results(self):
        """Generate plots comparing simulation to theory"""
        print("\nGenerating plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Dragon Skin 10 Cantilever Beam - Validation Results', fontsize=14, fontweight='bold')
        
        # Convert lists to arrays
        times = np.array(self.time_history)
        
        # Plot 1: Tip deflection vs time
        ax1 = axes[0, 0]
        tip_deflections = np.array(self.deflection_history['free_end'])
        ax1.plot(times, tip_deflections, 'b-', linewidth=2, label='Isaac Sim')
        ax1.axhline(y=self.theoretical_tip_deflection * 1000, color='r', linestyle='--', 
                    linewidth=2, label=f'Theory ({self.theoretical_tip_deflection*1000:.3f} mm)')
        ax1.set_xlabel('Time (s)', fontsize=11)
        ax1.set_ylabel('Tip Deflection (mm)', fontsize=11)
        ax1.set_title('Free End Deflection vs Time', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10)
        ax1.set_xlim([0, self.max_simulation_time])
        
        # Plot 2: Deflection profile at final time
        ax2 = axes[0, 1]
        final_idx = -1
        x_positions = []
        deflections_final = []
        labels_list = ['fixed_end', 'quarter', 'midpoint', 'three_quarter', 'free_end']
        
        for label in labels_list:
            if len(self.position_history[label]) > 0:
                x_positions.append(self.position_history[label][final_idx][0])
                deflections_final.append(self.deflection_history[label][final_idx])
        
        # Theoretical deflection curve
        x_theory = np.linspace(min(x_positions), max(x_positions), 100)
        L = self.beam_length
        x_from_fixed = x_theory - min(x_positions)  # Distance from fixed end
        
        # Theoretical deflection: δ(x) = (w*x²)/(24*E*I) * (6*L² - 4*L*x + x²)
        E = 263824.0
        rho = 1070.0
        g = 9.81
        w_beam = self.beam_width
        h_beam = self.beam_height
        I = (w_beam * h_beam**3) / 12
        w = rho * w_beam * h_beam * g
        
        deflection_theory = (w * x_from_fixed**2 / (24 * E * I)) * (6 * L**2 - 4 * L * x_from_fixed + x_from_fixed**2)
        
        ax2.plot(x_theory, deflection_theory * 1000, 'r--', linewidth=2, label='Theory')
        ax2.plot(x_positions, deflections_final, 'bo-', markersize=8, linewidth=2, label='Isaac Sim')
        ax2.set_xlabel('X Position (m)', fontsize=11)
        ax2.set_ylabel('Deflection (mm)', fontsize=11)
        ax2.set_title(f'Deflection Profile at t={times[final_idx]:.3f}s', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=10)
        ax2.invert_yaxis()  # Make downward deflection go down
        
        # Plot 3: All tracking points deflection
        ax3 = axes[1, 0]
        colors = ['green', 'blue', 'orange', 'purple', 'red']
        for (label, color) in zip(labels_list, colors):
            deflections = np.array(self.deflection_history[label])
            ax3.plot(times, deflections, color=color, linewidth=1.5, label=label.replace('_', ' ').title())
        
        ax3.set_xlabel('Time (s)', fontsize=11)
        ax3.set_ylabel('Deflection (mm)', fontsize=11)
        ax3.set_title('Deflection at Multiple Points', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=9, loc='best')
        ax3.set_xlim([0, self.max_simulation_time])
        
        # Plot 4: Error analysis
        ax4 = axes[1, 1]
        if len(tip_deflections) > 0:
            final_deflection = tip_deflections[-1]
            theoretical = self.theoretical_tip_deflection * 1000
            error_percent = abs(final_deflection - theoretical) / theoretical * 100
            
            # Text summary
            summary_text = f"""
VALIDATION SUMMARY

Material: Dragon Skin 10 (Shore 10A)
E = 263,824 Pa
ν = 0.4999
ρ = 1,070 kg/m³

Beam: {self.beam_length*100:.1f} × {self.beam_width*100:.1f} × {self.beam_height*100:.1f} cm
Resolution: {self.resolution} elements

RESULTS (at t={times[-1]:.3f}s):
Theoretical:  {theoretical:.4f} mm
Simulated:    {final_deflection:.4f} mm
Error:        {error_percent:.2f}%

Status: {"✓ PASS" if error_percent < 15 else "✗ FAIL"}
(Threshold: ±15%)
"""
            ax4.text(0.1, 0.5, summary_text, transform=ax4.transAxes,
                    fontsize=10, verticalalignment='center', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            ax4.axis('off')
        
        plt.tight_layout()
        
        # Save plot
        filename = f'/home/claude/cantilever_validation_results.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"✓ Plot saved: {filename}")
        
        return fig

    def play(self):
        """Run simulation and record data"""
        print(f"\nStarting simulation (recording 0 to {self.max_simulation_time}s)...\n")
        
        step_count = 0
        
        while simulation_app.is_running():
            if self.my_world.is_playing():
                current_time = self.my_world.current_time
                
                # Record data
                if current_time <= self.max_simulation_time and step_count > 5:
                    self.record_data(current_time)
                    
                    # Progress update
                    if len(self.time_history) % 10 == 0:
                        print(f"t = {current_time:.3f}s | Steps: {step_count:4d} | "
                              f"Tip deflection: {self.deflection_history['free_end'][-1]:.4f} mm")
                
                # Stop after max time
                if current_time > self.max_simulation_time:
                    print(f"\n✓ Reached {self.max_simulation_time}s - generating plots...")
                    self.plot_results()
                    
                    # Copy plot to outputs
                    import shutil
                    shutil.copy('/home/rjrosales/cantilever_validation_results.png', 
                               '/mnt/user-data/outputs/cantilever_validation_results.png')
                    print("✓ Plot saved to outputs")
                    
                    print("\nSimulation complete. Close window to exit.")
                    # Keep window open for viewing
                    while simulation_app.is_running():
                        self.my_world.step(render=True)
                    break
            
            self.my_world.step(render=True)
            step_count += 1

        simulation_app.close()


if __name__ == "__main__":
    DragonSkinCantileverValidation().play()