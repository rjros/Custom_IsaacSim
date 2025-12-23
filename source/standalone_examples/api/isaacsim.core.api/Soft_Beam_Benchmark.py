# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

## Cantilever Beam with Fixed Boundary - Using DeformablePrim API ##

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False, "extra_args": ["--/app/useFabricSceneDelegate=0"]})
import argparse
import sys

import carb
import isaacsim.core.utils.deformable_mesh_utils as deformableMeshUtils
import numpy as np
import torch
from isaacsim.core.api import World
from isaacsim.core.api.materials.deformable_material import DeformableMaterial
from isaacsim.core.prims import DeformablePrim, SingleDeformablePrim
from omni.physx.scripts import physicsUtils, deformableUtils
from pxr import Gf, UsdGeom, PhysxSchema, UsdPhysics, Sdf, UsdShade

parser = argparse.ArgumentParser()
parser.add_argument("--test", default=False, action="store_true", help="Run in test mode")
args, unknown = parser.parse_known_args()


class CantileverBeamExample:
    def __init__(self):
        self._array_container = torch.Tensor
        self.my_world = World(stage_units_in_meters=1.0, backend="torch", device="cuda")
        self.stage = simulation_app.context.get_stage()
        self.num_envs = 1
        
        # Add ground plane and lighting
        self.my_world.scene.add_default_ground_plane()
        self.setup_environment()
        
        self.initial_positions = None
        self.tracked_point_indices = []
        self.tracked_point_labels = []
        
        self.makeEnvs()
        self.setup_camera()

    def setup_environment(self):
        """Setup lighting for better visibility"""
        from pxr import UsdLux
        
        # Create a dome light
        dome_light_path = "/World/DomeLight"
        dome_light = UsdLux.DomeLight.Define(self.stage, dome_light_path)
        dome_light.CreateIntensityAttr().Set(800.0)
        dome_light.CreateColorAttr().Set(Gf.Vec3f(0.9, 0.9, 0.95))

    def makeEnvs(self):
        print("Creating cantilever beam environments...")
        
        for i in range(self.num_envs):
            init_loc = Gf.Vec3f(i * 2 - self.num_envs, 0.0, 0.0)
            
            env_scope = UsdGeom.Scope.Define(self.stage, "/World/Envs")
            env_path = f"/World/Envs/Env{i}"
            env = UsdGeom.Xform.Define(self.stage, env_path)
            physicsUtils.set_or_add_translate_op(UsdGeom.Xformable(env), init_loc)

            # Create beam mesh (rectangular beam, not cube)
            mesh_path = env.GetPrim().GetPath().AppendChild("deformable")
            skin_mesh = UsdGeom.Mesh.Define(self.stage, mesh_path)
            
            # Create a finer mesh for the beam (more points for better tracking)
            tri_points, tri_indices = deformableMeshUtils.createTriangleMeshCube(16)
            skin_mesh.GetPointsAttr().Set(tri_points)
            skin_mesh.GetFaceVertexIndicesAttr().Set(tri_indices)
            skin_mesh.GetFaceVertexCountsAttr().Set([3] * (len(tri_indices) // 3))
            
            # Setup transforms
            physicsUtils.setup_transform_as_scale_orient_translate(skin_mesh)
            
            # Beam dimensions: 10cm x 3cm x 3cm (length x width x height)
            beam_size = Gf.Vec3f(0.10, 0.03, 0.03)
            physicsUtils.set_or_add_scale_op(skin_mesh, beam_size)
            
            # Position beam at height 1.0m
            beam_center = Gf.Vec3f(0.0, 0.0, 1.0)
            physicsUtils.set_or_add_translate_op(skin_mesh, beam_center)
            
            # Add red color material
            material_path = env.GetPrim().GetPath().AppendChild("beamMaterial")
            material = UsdShade.Material.Define(self.stage, material_path)
            shader = UsdShade.Shader.Define(self.stage, material_path.AppendChild("Shader"))
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1.0, 0.2, 0.2))
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.4)
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            binding_api = UsdShade.MaterialBindingAPI.Apply(skin_mesh.GetPrim())
            binding_api.Bind(material)
            
            # Create deformable material
            deformable_material_path = env.GetPrim().GetPath().AppendChild("deformableMaterial").pathString
            self.deformable_material = DeformableMaterial(
                prim_path=deformable_material_path,
                dynamic_friction=0.5,
                youngs_modulus=5e5,
                poissons_ratio=0.49,
                damping_scale=0.0,
                elasticity_damping=0.0,
            )

            # Create deformable prim
            self.deformable = SingleDeformablePrim(
                name=f"deformablePrim{i}",
                prim_path=str(mesh_path),
                deformable_material=self.deformable_material,
                vertex_velocity_damping=0.0,
                sleep_damping=1.0,
                sleep_threshold=0.05,
                settling_threshold=0.1,
                self_collision=False,
                solver_position_iteration_count=50,
                kinematic_enabled=False,
                simulation_hexahedral_resolution=0.001,
                collision_simplification=True,
            )
            self.my_world.scene.add(self.deformable)
            
            # ==========================================================
            # Create anchor (kinematic rigid body at left end)
            # ==========================================================
            anchor_path = env.GetPrim().GetPath().AppendChild("anchor")
            anchor_size = (0.01, 0.05, 0.05)
            anchor_pos = (beam_center[0] - beam_size[0] * 0.5, beam_center[1], beam_center[2])

            # Create anchor cube
            anchor_mesh = physicsUtils.create_mesh_cube(self.stage, str(anchor_path), 0.5)
            physicsUtils.set_or_add_scale_op(anchor_mesh, anchor_size)
            physicsUtils.setup_transform_as_scale_orient_translate(anchor_mesh)
            physicsUtils.set_or_add_translate_op(anchor_mesh, anchor_pos)

            # Make anchor kinematic rigid body
            anchor_prim = self.stage.GetPrimAtPath(str(anchor_path))
            rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(anchor_prim)
            rigid_body_api.CreateKinematicEnabledAttr().Set(True)
            UsdPhysics.CollisionAPI.Apply(anchor_prim)

            # Create attachment
            attachment_path = mesh_path.AppendElementString("attachment")
            attachment = PhysxSchema.PhysxPhysicsAttachment.Define(self.stage, attachment_path)
            attachment.GetActor0Rel().SetTargets([mesh_path])
            attachment.GetActor1Rel().SetTargets([anchor_path])
            PhysxSchema.PhysxAutoAttachmentAPI.Apply(attachment.GetPrim())
            
            print(f"Environment {i}: Cantilever beam created with fixed boundary")

        # Create a view to access all deformables
        self.deformableView = DeformablePrim(
            prim_paths_expr="/World/Envs/Env*/deformable", 
            name="deformableView1"
        )
        self.my_world.scene.add(self.deformableView)
        
        # Reset to initialize physics
        self.my_world.reset(soft=False)
        
        # Get initial positions (these are in world coordinates)
        self.initial_positions = self.deformableView.get_simulation_mesh_nodal_positions().cpu()
        self.initial_velocities = self.deformableView.get_simulation_mesh_nodal_velocities().cpu()
        
        # Setup tracking points
        self._setup_tracking_points()

    def _setup_tracking_points(self):
        """Identify key points along the beam to track"""
        if self.initial_positions is None or len(self.initial_positions) == 0:
            print("Warning: No initial positions available")
            return
        
        print(f"\n=== Setting up tracking points ===")
        
        # Get positions for first environment
        env_positions = self.initial_positions[0]  # Shape: [num_nodes, 3]
        
        print(f"Total nodes in simulation mesh: {len(env_positions)}")
        
        # Sort by x-coordinate to find points along beam length
        x_coords = [(idx, pos[0].item()) for idx, pos in enumerate(env_positions)]
        x_coords.sort(key=lambda x: x[1])
        
        # Select 5 points: fixed end, 25%, 50%, 75%, free end
        num_points = len(x_coords)
        tracking_positions = [0, num_points//4, num_points//2, 3*num_points//4, num_points-1]
        
        self.tracked_point_indices = []
        self.tracked_point_labels = []
        
        for i, pos in enumerate(tracking_positions):
            idx, x_val = x_coords[pos]
            self.tracked_point_indices.append(idx)
            
            if i == 0:
                label = "Fixed End"
            elif i == len(tracking_positions) - 1:
                label = "Free End"
            else:
                percentage = (i / (len(tracking_positions) - 1)) * 100
                label = f"{percentage:.0f}% along"
            
            self.tracked_point_labels.append(label)
            node_pos = env_positions[idx]
            print(f"  {label:12s}: node {idx:3d}, position ({node_pos[0]:.4f}, {node_pos[1]:.4f}, {node_pos[2]:.4f})")
        
        print(f"✓ Tracking {len(self.tracked_point_indices)} points along the beam\n")

    def setup_camera(self):
        """Position camera to view the beam"""
        from omni.isaac.core.utils.viewports import set_camera_view
        
        eye = Gf.Vec3d(0.2, 0.2, 1.0)
        target = Gf.Vec3d(0.0, 0.0, 1.0)
        
        set_camera_view(eye=eye, target=target, camera_prim_path="/OmniverseKit_Persp")

    def play(self):
        print("\nStarting simulation...")
        print("The beam should sag under gravity with the left end fixed.")
        print("Tracking z-positions of points along the beam...\n")
        
        initial_z_positions = None
        previous_free_end_pos = None
        velocity_history = []
        settled = False
        
        while simulation_app.is_running():
            if self.my_world.is_playing():
                if self.my_world.current_time_step_index == 1:
                    self.my_world.reset(soft=False)
                    initial_z_positions = None
                    previous_free_end_pos = None
                    velocity_history = []
                    settled = False

            self.my_world.step(render=True)
            
            # Track positions every frame
            if self.my_world.current_time_step_index > 10 and len(self.tracked_point_indices) > 0:
                # Get current positions from the view
                current_positions = self.deformableView.get_simulation_mesh_nodal_positions()
                
                if current_positions is not None and len(current_positions) > 0:
                    # Get positions for first environment
                    env_positions = current_positions[0]  # [num_nodes, 3]
                    
                    # Store initial z-positions
                    if initial_z_positions is None:
                        initial_z_positions = [env_positions[idx, 2].item() for idx in self.tracked_point_indices]
                        print("Initial z-positions recorded:")
                        for label, z in zip(self.tracked_point_labels, initial_z_positions):
                            print(f"  {label:12s}: z = {z:.5f}m")
                        print()
                    
                    # Track velocity of free end
                    free_end_idx = self.tracked_point_indices[-1]
                    free_end_pos = env_positions[free_end_idx]
                    
                    if previous_free_end_pos is not None:
                        velocity = torch.norm(free_end_pos - previous_free_end_pos).item()
                        velocity_history.append(velocity)
                        if len(velocity_history) > 30:
                            velocity_history.pop(0)
                        
                        # Check settlement
                        if len(velocity_history) >= 30:
                            avg_velocity = sum(velocity_history) / len(velocity_history)
                            if avg_velocity < 0.00001 and not settled:
                                settled = True
                                print(f"\n{'='*70}")
                                print(f"*** BEAM SETTLED at step {self.my_world.current_time_step_index} ***")
                                print(f"{'='*70}")
                                print("\nFinal Positions and Deflections:")
                                for label, idx, init_z in zip(self.tracked_point_labels, self.tracked_point_indices, initial_z_positions):
                                    pos = env_positions[idx]
                                    deflection = init_z - pos[2].item()
                                    print(f"  {label:12s}: z = {pos[2].item():.5f}m, deflection = {deflection:.5f}m ({deflection*1000:.2f}mm)")
                                print(f"\nAverage velocity: {avg_velocity:.8f} m/step")
                                print(f"{'='*70}\n")
                    
                    previous_free_end_pos = free_end_pos.clone()
                    
                    # Periodic updates every 100 steps
                    if self.my_world.current_time_step_index % 100 == 0 and initial_z_positions:
                        print(f"\n--- Step {self.my_world.current_time_step_index} ---")
                        for label, idx, init_z in zip(self.tracked_point_labels, self.tracked_point_indices, initial_z_positions):
                            pos = env_positions[idx]
                            deflection = init_z - pos[2].item()
                            print(f"  {label:12s}: z = {pos[2].item():.5f}m, Δz = {deflection:.5f}m")
                        if len(velocity_history) > 0:
                            print(f"  Free end velocity: {velocity_history[-1]:.6f} m/step")

        simulation_app.close()


if __name__ == "__main__":
    CantileverBeamExample().play()