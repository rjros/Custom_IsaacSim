# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""
Contact Force Test - Using RigidPrim Views

Based on the official contact force example, using RigidPrim with
track_contact_forces=True for reliable contact detection.
"""

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import numpy as np
import torch
import omni.usd
from pxr import Gf, UsdGeom, PhysxSchema, UsdPhysics, Sdf, UsdShade, UsdLux

from isaacsim.core.api import World
from isaacsim.core.prims import RigidPrim
from omni.physx.scripts import physicsUtils

np.random.seed(42)


class ContactForceTest:
    
    def __init__(self):
        self.time_step = 0.01
        self.box_size = 0.1  # 10cm box
        self.sphere_radius = 0.01  # 1cm sphere
        self.approach_speed = 0.005  # 5mm/s - VERY SLOW
        
        print(f"\n{'='*80}")
        print(f"Contact Force Test - RigidPrim Views")
        print(f"{'='*80}")
        print(f"  Sphere: {self.sphere_radius*100:.1f} cm radius")
        print(f"  Box: {self.box_size*100:.1f} cm cube")
        print(f"  Approach speed: {self.approach_speed*1000:.1f} mm/s (SLOW)")
        print(f"{'='*80}\n")
        
        self._array_container = torch.Tensor
        self.my_world = World(
            stage_units_in_meters=1.0,
            backend="torch",
            device="cuda",
            physics_dt=self.time_step,
            rendering_dt=self.time_step * 2
        )
        self.stage = omni.usd.get_context().get_stage()
        
        self.contact_history = []
        self.force_history = []
        self.time_history = []
        self.position_history = []
        
        self.my_world.scene.add_default_ground_plane()
        self.setup_environment()
        self.setup_scene()
        self.setup_camera()
    
    def setup_environment(self):
        dome_light_path = "/World/DomeLight"
        dome_light = UsdLux.DomeLight.Define(self.stage, dome_light_path)
        dome_light.CreateIntensityAttr().Set(1500.0)
    
    def setup_scene(self):
        print("Creating scene...")
        
        # === RIGID BOX (static, on ground) ===
        box_center_z = self.box_size / 2
        box_path = "/World/rigid_box"
        
        box_geom = UsdGeom.Cube.Define(self.stage, box_path)
        box_geom.CreateSizeAttr().Set(1.0)
        
        xform = UsdGeom.Xformable(box_geom)
        xform.ClearXformOpOrder()
        xform.AddScaleOp().Set(Gf.Vec3f(self.box_size, self.box_size, self.box_size))
        xform.AddTranslateOp().Set(Gf.Vec3f(0, 0, box_center_z))
        
        box_prim = self.stage.GetPrimAtPath(box_path)
        
        # Physics - kinematic (static)
        rigid_api = UsdPhysics.RigidBodyAPI.Apply(box_prim)
        rigid_api.CreateKinematicEnabledAttr().Set(True)
        
        UsdPhysics.CollisionAPI.Apply(box_prim)
        
        # Contact report API
        contact_report = PhysxSchema.PhysxContactReportAPI.Apply(box_prim)
        contact_report.CreateThresholdAttr(0.0)
        
        # Material
        box_mat = UsdShade.Material.Define(self.stage, "/World/boxMat")
        box_shader = UsdShade.Shader.Define(self.stage, "/World/boxMat/Shader")
        box_shader.CreateIdAttr("UsdPreviewSurface")
        box_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.3, 0.6, 0.9))
        box_mat.CreateSurfaceOutput().ConnectToSource(box_shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(box_geom.GetPrim()).Bind(box_mat)
        
        # === RIGID SPHERE (kinematic, controlled motion) ===
        self.initial_sphere_z = self.box_size + self.sphere_radius + 0.05  # Start 5cm above
        self.sphere_path = "/World/rigid_sphere"
        
        sphere_geom = UsdGeom.Sphere.Define(self.stage, self.sphere_path)
        sphere_geom.CreateRadiusAttr().Set(self.sphere_radius)
        
        xform = UsdGeom.Xformable(sphere_geom)
        xform.ClearXformOpOrder()
        self.sphere_translate_op = xform.AddTranslateOp()
        self.sphere_translate_op.Set(Gf.Vec3f(0, 0, self.initial_sphere_z))
        
        sphere_prim = self.stage.GetPrimAtPath(self.sphere_path)
        
        # Physics - kinematic (controlled)
        rigid_api = UsdPhysics.RigidBodyAPI.Apply(sphere_prim)
        rigid_api.CreateKinematicEnabledAttr().Set(True)
        
        UsdPhysics.CollisionAPI.Apply(sphere_prim)
        
        # Contact report API
        contact_report = PhysxSchema.PhysxContactReportAPI.Apply(sphere_prim)
        contact_report.CreateThresholdAttr(0.0)
        
        # Material
        sphere_mat = UsdShade.Material.Define(self.stage, "/World/sphereMat")
        sphere_shader = UsdShade.Shader.Define(self.stage, "/World/sphereMat/Shader")
        sphere_shader.CreateIdAttr("UsdPreviewSurface")
        sphere_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.8, 0.2, 0.2))
        sphere_shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.8)
        sphere_shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.2)
        sphere_mat.CreateSurfaceOutput().ConnectToSource(sphere_shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(sphere_geom.GetPrim()).Bind(sphere_mat)
        
        # === CREATE RIGID PRIM VIEWS FOR CONTACT TRACKING ===
        print("Creating RigidPrim views for contact force tracking...")
        
        # Sphere view with contact force tracking, filtering for box contacts
        self._sphere_view = RigidPrim(
            prim_paths_expr="/World/rigid_sphere",
            name="sphere_view",
            track_contact_forces=True,
            contact_filter_prim_paths_expr=["/World/rigid_box"],
        )
        
        # Box view with contact force tracking, filtering for sphere contacts
        self._box_view = RigidPrim(
            prim_paths_expr="/World/rigid_box",
            name="box_view",
            track_contact_forces=True,
            contact_filter_prim_paths_expr=["/World/rigid_sphere"],
        )
        
        self.my_world.scene.add(self._sphere_view)
        self.my_world.scene.add(self._box_view)
        
        print("✓ Scene created")
        print(f"  Box: kinematic rigid body on ground")
        print(f"  Sphere: kinematic rigid body, controlled descent")
        print(f"  RigidPrim views: tracking contact forces between sphere and box")
        print()
        
        self.my_world.reset(soft=False)
    
    def setup_camera(self):
        from omni.isaac.core.utils.viewports import set_camera_view
        set_camera_view(
            eye=Gf.Vec3d(0.3, 0.3, 0.2),
            target=Gf.Vec3d(0, 0, self.box_size/2),
            camera_prim_path="/OmniverseKit_Persp"
        )
    
    def get_contact_forces(self):
        """Get contact forces from RigidPrim views"""
        try:
            # Get net contact forces on sphere
            sphere_net_forces = self._sphere_view.get_net_contact_forces(dt=self.time_step)
            
            # Get contact force matrix (forces from box on sphere)
            sphere_force_matrix = self._sphere_view.get_contact_force_matrix(dt=self.time_step)
            
            # Get net contact forces on box
            box_net_forces = self._box_view.get_net_contact_forces(dt=self.time_step)
            
            # Calculate magnitudes
            if sphere_net_forces is not None and len(sphere_net_forces) > 0:
                sphere_force_mag = torch.linalg.norm(sphere_net_forces[0]).item()
            else:
                sphere_force_mag = 0.0
            
            if box_net_forces is not None and len(box_net_forces) > 0:
                box_force_mag = torch.linalg.norm(box_net_forces[0]).item()
            else:
                box_force_mag = 0.0
            
            return {
                'sphere_net_force': sphere_force_mag,
                'box_net_force': box_force_mag,
                'sphere_force_vector': sphere_net_forces,
                'box_force_vector': box_net_forces,
                'force_matrix': sphere_force_matrix
            }
        except Exception as e:
            print(f"Error reading forces: {e}")
            return {
                'sphere_net_force': 0.0,
                'box_net_force': 0.0,
                'sphere_force_vector': None,
                'box_force_vector': None,
                'force_matrix': None
            }
    
    def update_sphere_position(self, t):
        """Move sphere slowly downward"""
        # Slow descent
        z = self.initial_sphere_z - (self.approach_speed * t)
        
        # Stop at box surface
        min_z = self.box_size + self.sphere_radius - 0.01  # Allow 1cm penetration
        z = max(z, min_z)
        
        self.sphere_translate_op.Set(Gf.Vec3f(0, 0, z))
        return z
    
    def record_data(self, t, z_pos):
        """Record timestep data"""
        contact_data = self.get_contact_forces()
        
        self.time_history.append(t)
        self.position_history.append(z_pos)
        self.force_history.append(contact_data['sphere_net_force'])
        
        return contact_data
    
    def run(self):
        print("Starting test...\n")
        print("Sphere will slowly descend and contact the box.")
        print("Contact forces tracked via RigidPrim views.\n")
        print(f"{'Time (s)':<10} {'Z pos (mm)':<12} {'Sphere F (N)':<14} {'Box F (N)':<14}")
        print("-" * 80)
        
        step = 0
        last_print_time = 0
        print_interval = 0.5  # Print every 0.5 seconds
        test_duration = 15.0  # Run for 15 seconds
        
        while simulation_app.is_running():
            if self.my_world.is_playing():
                t = self.my_world.current_time
                
                # Update sphere position
                z_pos = self.update_sphere_position(t)
                
                # Record data
                if step > 5:  # Skip first few steps
                    contact_data = self.record_data(t, z_pos)
                    
                    # Print updates
                    if t - last_print_time >= print_interval:
                        sphere_f = contact_data['sphere_net_force']
                        box_f = contact_data['box_net_force']
                        
                        print(f"{t:<10.2f} {z_pos*1000:<12.2f} {sphere_f:<14.6f} {box_f:<14.6f}")
                        
                        # Print detailed info if contact detected
                        if sphere_f > 0.001 or box_f > 0.001:
                            print(f"  *** CONTACT DETECTED ***")
                            if contact_data['sphere_force_vector'] is not None:
                                print(f"  Sphere force vector: {contact_data['sphere_force_vector']}")
                            if contact_data['box_force_vector'] is not None:
                                print(f"  Box force vector: {contact_data['box_force_vector']}")
                        
                        last_print_time = t
                
                # Stop after test duration
                if t > test_duration:
                    print(f"\n✓ Test complete at t={t:.2f}s")
                    self.print_summary()
                    print("\nClose window to exit.")
                    
                    # Keep running for visualization
                    while simulation_app.is_running():
                        self.my_world.step(render=True)
                    break
            
            self.my_world.step(render=True)
            step += 1
        
        simulation_app.close()
    
    def print_summary(self):
        """Print summary of contact events"""
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        
        if not self.force_history:
            print("No data collected!")
            return
        
        # Contact statistics
        force_array = np.array(self.force_history)
        contact_threshold = 0.001  # 1mN threshold
        contact_detected = force_array > contact_threshold
        
        total_steps = len(self.force_history)
        contact_steps = np.sum(contact_detected)
        max_force = np.max(force_array)
        avg_force = np.mean(force_array[contact_detected]) if np.any(contact_detected) else 0
        
        print(f"Total steps: {total_steps}")
        print(f"Steps with contact: {contact_steps} ({contact_steps/total_steps*100:.1f}%)")
        print(f"Max force: {max_force:.6f} N")
        print(f"Avg force (when in contact): {avg_force:.6f} N")
        
        # First contact time
        for i, has_contact in enumerate(contact_detected):
            if has_contact:
                print(f"First contact at: t={self.time_history[i]:.3f}s, z={self.position_history[i]*1000:.2f}mm")
                break
        else:
            print("WARNING: No contact detected during test!")
        
        print("="*80)


if __name__ == "__main__":
    ContactForceTest().run()