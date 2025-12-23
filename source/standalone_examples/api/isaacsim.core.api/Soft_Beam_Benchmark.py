# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

## Cantilever Beam with Fixed Boundary - Isaac Sim 5.1 Compatible ##

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})
import argparse
import sys

import carb
import numpy as np
import torch
from isaacsim.core.api import World
from pxr import Gf, UsdGeom, PhysxSchema, UsdPhysics, Sdf
from omni.physx.scripts import physicsUtils, deformableUtils

parser = argparse.ArgumentParser()
parser.add_argument("--test", default=False, action="store_true", help="Run in test mode")
args, unknown = parser.parse_known_args()


class CantileverBeamExample:
    def __init__(self):
        self.my_world = World(stage_units_in_meters=1.0, backend="torch", device="cuda")
        self.stage = simulation_app.context.get_stage()
        self.num_envs = 1
        
        # Change background color for better visibility
        self.setup_environment()
        
        # Add ground plane
        self.my_world.scene.add_default_ground_plane()
        
        self.makeEnvs()
        
        # Set camera to focus on the beam
        self.setup_camera()
        
        # Store marker info for tracking
        self.marker_sphere_path = None
        self.tracked_node_index = None
        self.deformable_prim_path = None

    def makeEnvs(self):
        print("Creating cantilever beam environments...")
        
        for i in range(self.num_envs):
            init_loc = Gf.Vec3f(i * 2 - self.num_envs, 0.0, 0.0)

            # Create environment scope
            env_path = f"/World/Envs/Env{i}"
            env = UsdGeom.Xform.Define(self.stage, env_path)
            physicsUtils.set_or_add_translate_op(UsdGeom.Xformable(env), init_loc)

            # -----------------------------
            # Create beam mesh (triangle mesh)
            # -----------------------------
            mesh_path = env.GetPrim().GetPath().AppendChild("deformable")
            skin_mesh = UsdGeom.Mesh.Define(self.stage, mesh_path)

            # Create a simple beam mesh (you can customize this)
            self._create_beam_mesh(skin_mesh)

            # Beam dimensions: 10cm x 3cm x 3cm (length x width x height)
            beam_size = (0.10, 0.03, 0.03)
            physicsUtils.set_or_add_scale_op(skin_mesh, beam_size)
            physicsUtils.setup_transform_as_scale_orient_translate(skin_mesh)

            # Position beam (centered, raised above ground)
            beam_center = (0.0, 0.0, 1.0)
            physicsUtils.set_or_add_translate_op(skin_mesh, beam_center)
            
            # Add color to the beam (bright red for visibility)
            from pxr import UsdShade
            material_path = env.GetPrim().GetPath().AppendChild("beamMaterial")
            material = UsdShade.Material.Define(self.stage, material_path)
            shader = UsdShade.Shader.Define(self.stage, material_path.AppendChild("Shader"))
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1.0, 0.2, 0.2))  # Bright red
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.4)
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            
            # Bind material to mesh
            binding_api = UsdShade.MaterialBindingAPI.Apply(skin_mesh.GetPrim())
            binding_api.Bind(material)

            # -----------------------------
            # Add deformable body physics using PhysX schemas
            # -----------------------------
            deformable_prim = self.stage.GetPrimAtPath(str(mesh_path))
            
            # Apply deformable body API with all properties
            # These parameters handle vertex damping, sleep settings, etc.
            deformableUtils.add_physx_deformable_body(
                self.stage,
                str(mesh_path),
                simulation_hexahedral_resolution=6,
                collision_simplification=True,
                self_collision=False,
                solver_position_iteration_count=20,
                # Optional: vertex_velocity_damping is set via kinematic_enabled=False
                kinematic_enabled=False,
            )
            
            # Set deformable body material properties
            # Damping is controlled through the material, not the body API
            mat_path = env.GetPrim().GetPath().AppendChild("deformableMaterial")
            deformableUtils.add_deformable_body_material(
                self.stage,
                str(mat_path),
                youngs_modulus=5e4,
                poissons_ratio=0.4,
                dynamic_friction=0.5,
                density=1000.0,
                # Note: damping_scale and elasticity_damping may be available
                # depending on your exact Isaac Sim 5.1 version
            )
            
            # Apply material to deformable
            physicsUtils.add_physics_material_to_prim(
                self.stage,
                deformable_prim,
                str(mat_path)
            )
            
            # Set collision offsets (these ARE available)
            collision_api = PhysxSchema.PhysxCollisionAPI.Apply(deformable_prim)
            collision_api.CreateRestOffsetAttr().Set(0.0)
            collision_api.CreateContactOffsetAttr().Set(0.001)

            # ==========================================================
            # Create anchor (kinematic rigid body at left end)
            # ==========================================================
            anchor_path = env.GetPrim().GetPath().AppendChild("anchor")
            anchor_size = (0.01, 0.05, 0.05)
            anchor_pos = (beam_center[0] - beam_size[0] * 0.5, beam_center[1], beam_center[2])

            # Create anchor cube mesh with uniform size, then scale it
            # create_mesh_cube takes a single float for uniform halfSize
            anchor_mesh = physicsUtils.create_mesh_cube(self.stage, str(anchor_path), 0.5)
            physicsUtils.set_or_add_scale_op(anchor_mesh, anchor_size)
            physicsUtils.setup_transform_as_scale_orient_translate(anchor_mesh)
            physicsUtils.set_or_add_translate_op(anchor_mesh, anchor_pos)

            # Make anchor kinematic rigid body using USD Physics API
            anchor_prim = self.stage.GetPrimAtPath(str(anchor_path))
            rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(anchor_prim)
            rigid_body_api.CreateKinematicEnabledAttr().Set(True)
            
            # Add collision API
            UsdPhysics.CollisionAPI.Apply(anchor_prim)

            # ==========================================================
            # Create attachment using PhysX schema (Isaac Sim 5.1 method)
            # ==========================================================
            attachment_path = mesh_path.AppendElementString("attachment")
            attachment = PhysxSchema.PhysxPhysicsAttachment.Define(self.stage, attachment_path)
            
            # Set the two actors (deformable and rigid anchor)
            attachment.GetActor0Rel().SetTargets([mesh_path])
            attachment.GetActor1Rel().SetTargets([anchor_path])
            
            # Apply auto-attachment API for automatic tetrahedral attachment
            PhysxSchema.PhysxAutoAttachmentAPI.Apply(attachment.GetPrim())
            
            # Store deformable prim path for later node tracking
            self.deformable_prim_path = str(mesh_path)
            
            print(f"Environment {i}: Cantilever beam created with fixed boundary at left end")

        # Reset simulation to initialize physics
        self.my_world.reset(soft=False)
        
        # ==========================================================
        # After physics initialization, find the node at the free end
        # and create a visual marker sphere there
        # ==========================================================
        self._setup_marker_at_free_end()

    def _create_beam_mesh(self, mesh_prim):
        """Create a simple rectangular beam mesh"""
        # Create vertices for a rectangular beam (1x1x1 unit cube, will be scaled)
        points = [
            (-0.5, -0.5, -0.5), (0.5, -0.5, -0.5), (0.5, 0.5, -0.5), (-0.5, 0.5, -0.5),
            (-0.5, -0.5, 0.5), (0.5, -0.5, 0.5), (0.5, 0.5, 0.5), (-0.5, 0.5, 0.5),
        ]
        
        # Triangle indices (2 triangles per face, 6 faces)
        indices = [
            0, 1, 2, 0, 2, 3,  # front
            4, 6, 5, 4, 7, 6,  # back
            0, 4, 5, 0, 5, 1,  # bottom
            2, 6, 7, 2, 7, 3,  # top
            0, 3, 7, 0, 7, 4,  # left
            1, 5, 6, 1, 6, 2,  # right
        ]
        
        mesh_prim.GetPointsAttr().Set(points)
        mesh_prim.GetFaceVertexIndicesAttr().Set(indices)
        mesh_prim.GetFaceVertexCountsAttr().Set([3] * (len(indices) // 3))

    def setup_environment(self):
        """Setup lighting and background for better visibility"""
        import carb.settings
        
        # Set a light gray background color using settings
        settings = carb.settings.get_settings()
        settings.set("/rtx/post/backgroundZeroAlpha/enabled", False)
        settings.set("/rtx/rendermode", "PathTracing")  # Better lighting
        
        # Add better lighting to the scene
        from pxr import UsdLux, Sdf
        
        # Create a dome light for better visibility
        # Dome light provides natural, even lighting
        dome_light_path = "/World/DomeLight"
        dome_light = UsdLux.DomeLight.Define(self.stage, dome_light_path)
        dome_light.CreateIntensityAttr().Set(800.0)
        dome_light.CreateColorAttr().Set(Gf.Vec3f(0.9, 0.9, 0.95))  # Slight blue tint
    
    def _setup_marker_at_free_end(self):
        """Create a visual marker sphere attached to a node at the free end of the beam"""
        if not self.deformable_prim_path:
            print("Warning: No deformable prim path set")
            return
        
        print(f"Setting up marker for deformable at: {self.deformable_prim_path}")
        
        # Get deformable prim - use the visual mesh initially
        deformable_prim = self.stage.GetPrimAtPath(self.deformable_prim_path)
        if not deformable_prim:
            print(f"Error: Could not find prim at {self.deformable_prim_path}")
            return
        
        # Get the visual mesh points to find the free end
        visual_mesh = UsdGeom.Mesh(deformable_prim)
        points_attr = visual_mesh.GetPointsAttr()
        points = points_attr.Get()
        
        if not points or len(points) == 0:
            print("Error: No points found in mesh")
            return
        
        print(f"Found {len(points)} points in mesh")
        
        # Find the node at the rightmost position (free end)
        # The beam is scaled, so we need to consider the scale
        # Beam center is at (0, 0, 1.0), length is 0.10m
        max_x = -float('inf')
        free_end_node_idx = 0
        
        for idx, point in enumerate(points):
            # Points are in local space before scaling
            if point[0] > max_x:
                max_x = point[0]
                free_end_node_idx = idx
        
        self.tracked_node_index = free_end_node_idx
        
        # Get the actual world position considering transforms
        xform = UsdGeom.Xformable(deformable_prim)
        world_transform = xform.ComputeLocalToWorldTransform(0)
        local_point = points[free_end_node_idx]
        initial_pos = world_transform.Transform(Gf.Vec3d(local_point[0], local_point[1], local_point[2]))
        
        print(f"Tracking node {free_end_node_idx}")
        print(f"  Local position: {local_point}")
        print(f"  World position: {initial_pos}")
        
        # Create a visual marker sphere at this node's world position
        marker_path = "/World/Envs/Env0/marker_sphere"  # Use absolute path
        marker_sphere = UsdGeom.Sphere.Define(self.stage, marker_path)
        
        marker_radius = 0.015  # Make it larger - 15mm sphere for better visibility
        marker_sphere.CreateRadiusAttr().Set(marker_radius)
        
        # Set position in world space
        marker_sphere.AddTranslateOp().Set(Gf.Vec3d(initial_pos[0], initial_pos[1], initial_pos[2]))
        
        # Make marker bright yellow with high emissive for visibility
        from pxr import UsdShade
        marker_material_path = "/World/Envs/Env0/marker_material"
        marker_material = UsdShade.Material.Define(self.stage, marker_material_path)
        marker_shader = UsdShade.Shader.Define(self.stage, marker_material_path + "/Shader")
        marker_shader.CreateIdAttr("UsdPreviewSurface")
        marker_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1.0, 1.0, 0.0))  # Bright yellow
        marker_shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1.0, 1.0, 0.0))  # Strong yellow glow
        marker_shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.1)
        marker_shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        marker_material.CreateSurfaceOutput().ConnectToSource(marker_shader.ConnectableAPI(), "surface")
        
        # Bind material to marker
        marker_binding_api = UsdShade.MaterialBindingAPI.Apply(marker_sphere.GetPrim())
        marker_binding_api.Bind(marker_material)
        
        self.marker_sphere_path = marker_path
        
        print(f"✓ Created yellow marker sphere at world position: {initial_pos}")
        print(f"  Marker path: {marker_path}")
        print(f"  Marker radius: {marker_radius}m")
        print("  Marker will follow the beam's deformation in real-time")

    def setup_camera(self):
        """Position camera to get a close view of the cantilever beam"""
        from omni.isaac.core.utils.viewports import set_camera_view
        
        # Camera positioned to view the beam from slightly below and to the side
        # Beam is at position (0, 0, 1.0) with size (0.10, 0.03, 0.03)
        eye = Gf.Vec3d(0.2, 0.2, 1.0)     # Camera position - at beam height, viewing from angle
        target = Gf.Vec3d(0.0, 0.0, 1.0)  # Look directly at beam center
        
        set_camera_view(
            eye=eye,
            target=target,
            camera_prim_path="/OmniverseKit_Persp"
        )

    def play(self):
        print("Starting simulation...")
        print("The beam should sag under gravity while the left end remains fixed.")
        print("Watch the yellow marker sphere - it's attached to a node at the free end.")
        
        # Variables to track marker motion
        previous_marker_pos = None
        velocity_history = []
        settled = False
        
        while simulation_app.is_running():
            if self.my_world.is_playing():
                # Handle simulation re-initialization after restart
                if self.my_world.current_time_step_index == 1:
                    self.my_world.reset(soft=False)
                    previous_marker_pos = None
                    velocity_history = []
                    settled = False

                # Update marker position to follow the deformable node
                if self.marker_sphere_path and self.tracked_node_index is not None:
                    # Get the deformable prim and compute world position
                    deformable_prim = self.stage.GetPrimAtPath(self.deformable_prim_path)
                    visual_mesh = UsdGeom.Mesh(deformable_prim)
                    points_attr = visual_mesh.GetPointsAttr()
                    current_points = points_attr.Get()
                    
                    if current_points and self.tracked_node_index < len(current_points):
                        # Get current position in local space
                        local_node_pos = current_points[self.tracked_node_index]
                        
                        # Transform to world space
                        xform = UsdGeom.Xformable(deformable_prim)
                        world_transform = xform.ComputeLocalToWorldTransform(0)
                        node_pos = world_transform.Transform(Gf.Vec3d(local_node_pos[0], local_node_pos[1], local_node_pos[2]))
                        
                        # Update marker sphere position
                        marker_prim = self.stage.GetPrimAtPath(self.marker_sphere_path)
                        if marker_prim:
                            marker_xform = UsdGeom.Xformable(marker_prim)
                            translate_ops = marker_xform.GetOrderedXformOps()
                            if translate_ops:
                                translate_ops[0].Set(Gf.Vec3d(node_pos[0], node_pos[1], node_pos[2]))
                        
                        # Track velocity for settlement detection
                        if self.my_world.current_time_step_index > 10:
                            if previous_marker_pos is not None:
                                # Calculate velocity (change in position)
                                velocity = [
                                    abs(node_pos[0] - previous_marker_pos[0]),
                                    abs(node_pos[1] - previous_marker_pos[1]),
                                    abs(node_pos[2] - previous_marker_pos[2])
                                ]
                                total_velocity = (velocity[0]**2 + velocity[1]**2 + velocity[2]**2)**0.5
                                
                                # Keep a rolling window of velocities
                                velocity_history.append(total_velocity)
                                if len(velocity_history) > 30:  # Track last 30 frames
                                    velocity_history.pop(0)
                                
                                # Check if beam has settled (velocity consistently low)
                                if len(velocity_history) >= 30:
                                    avg_velocity = sum(velocity_history) / len(velocity_history)
                                    if avg_velocity < 0.00001 and not settled:  # Threshold for "stopped"
                                        settled = True
                                        print(f"\n*** BEAM SETTLED at step {self.my_world.current_time_step_index} ***")
                                        print(f"Final node position: x={node_pos[0]:.5f}, y={node_pos[1]:.5f}, z={node_pos[2]:.5f}")
                                        print(f"Vertical deflection: {(1.0 - node_pos[2]):.5f}m")
                                        print(f"Average velocity: {avg_velocity:.8f} m/step\n")
                                
                                # Print periodic updates every 100 steps
                                if self.my_world.current_time_step_index % 100 == 0:
                                    deflection = 1.0 - node_pos[2]
                                    print(f"Step {self.my_world.current_time_step_index}: Node Z={node_pos[2]:.5f}m, Deflection={deflection:.5f}m, Velocity={total_velocity:.6f}m/step")
                            
                            previous_marker_pos = node_pos

            self.my_world.step(render=True)

        simulation_app.close()


if __name__ == "__main__":
    CantileverBeamExample().play()