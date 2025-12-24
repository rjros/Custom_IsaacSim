# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""
Contact Force Test - MOVING PLATFORM VERSION

Simplest approach: Sphere sits on a kinematic platform that moves down.
No nested rigid bodies, no complex joints.
"""

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import numpy as np
import matplotlib.pyplot as plt
from pxr import Gf, UsdGeom, UsdPhysics, PhysxSchema, Sdf, UsdShade, UsdLux

from isaacsim.core.api import World
from isaacsim.core.api.objects import FixedCuboid
from isaacsim.core.prims import RigidPrim
import omni.usd

np.random.seed(42)


class ContactForceTestPlatform:
    
    def __init__(self):
        self.time_step = 1/60.0
        self.box_size = 0.1  # 10cm box
        self.sphere_radius = 0.01  # 1cm sphere
        self.platform_size = 0.03  # 3cm platform
        self.descent_speed = 0.005  # 5mm/s
        self.max_descent = 0.05  # 5cm total travel
        
        print(f"\n{'='*80}")
        print(f"Contact Force Test - MOVING PLATFORM")
        print(f"{'='*80}")
        print(f"  Sphere: {self.sphere_radius*100:.1f} cm radius on moving platform")
        print(f"  Box: {self.box_size*100:.1f} cm cube")
        print(f"  Descent speed: {self.descent_speed*1000:.1f} mm/s")
        print(f"{'='*80}\n")
        
        self.world = World(stage_units_in_meters=1.0)
        self.stage = omni.usd.get_context().get_stage()
        
        self.force_log = []
        self.time_log = []
        self.position_log = []
        
        self.world.scene.add_default_ground_plane()
        self.setup_environment()
        self.setup_scene()
        self.setup_camera()
    
    def setup_environment(self):
        dome_light_path = "/World/DomeLight"
        dome_light = UsdLux.DomeLight.Define(self.stage, dome_light_path)
        dome_light.CreateIntensityAttr().Set(1500.0)
    
    def setup_scene(self):
        print("Creating scene...")
        
        # === FIXED BOX ===
        box_center_z = self.box_size / 2
        self.box = self.world.scene.add(
            FixedCuboid(
                prim_path="/World/FixedBox",
                name="fixed_box",
                position=np.array([0, 0, box_center_z]),
                size=self.box_size,
                color=np.array([0.3, 0.6, 0.9]) * 255,
            )
        )
        
        # === MOVING PLATFORM (kinematic) ===
        platform_path = "/World/MovingPlatform"
        self.initial_platform_z = self.box_size + self.sphere_radius * 2 + 0.03
        
        platform_geom = UsdGeom.Cube.Define(self.stage, platform_path)
        platform_geom.CreateSizeAttr().Set(1.0)
        
        platform_xform = UsdGeom.Xformable(platform_geom)
        platform_xform.ClearXformOpOrder()
        platform_xform.AddScaleOp().Set(Gf.Vec3f(self.platform_size, self.platform_size, 0.005))
        self.platform_translate_op = platform_xform.AddTranslateOp()
        self.platform_translate_op.Set(Gf.Vec3f(0, 0, self.initial_platform_z))
        
        platform_prim = platform_geom.GetPrim()
        
        # Kinematic rigid body
        rigid_api = UsdPhysics.RigidBodyAPI.Apply(platform_prim)
        rigid_api.CreateKinematicEnabledAttr().Set(True)
        
        UsdPhysics.CollisionAPI.Apply(platform_prim)
        
        # Platform material (semi-transparent)
        platform_mat = UsdShade.Material.Define(self.stage, "/World/platformMat")
        platform_shader = UsdShade.Shader.Define(self.stage, "/World/platformMat/Shader")
        platform_shader.CreateIdAttr("UsdPreviewSurface")
        platform_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.5, 0.5, 0.5))
        platform_shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(0.5)
        platform_mat.CreateSurfaceOutput().ConnectToSource(platform_shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(platform_prim).Bind(platform_mat)
        
        # === SPHERE ON PLATFORM ===
        sphere_path = "/World/Sphere"
        sphere_z = self.initial_platform_z + 0.0025 + self.sphere_radius  # Just above platform
        
        sphere_geom = UsdGeom.Sphere.Define(self.stage, sphere_path)
        sphere_geom.CreateRadiusAttr().Set(self.sphere_radius)
        
        sphere_xform = UsdGeom.Xformable(sphere_geom)
        sphere_xform.ClearXformOpOrder()
        sphere_xform.AddTranslateOp().Set(Gf.Vec3f(0, 0, sphere_z))
        
        sphere_prim = sphere_geom.GetPrim()
        
        # Dynamic rigid body (not kinematic - will rest on platform)
        rigid_api = UsdPhysics.RigidBodyAPI.Apply(sphere_prim)
        rigid_api.CreateKinematicEnabledAttr().Set(False)
        
        mass_api = UsdPhysics.MassAPI.Apply(sphere_prim)
        mass_api.CreateMassAttr().Set(0.1)
        
        UsdPhysics.CollisionAPI.Apply(sphere_prim)
        
        # Contact report
        contact_report = PhysxSchema.PhysxContactReportAPI.Apply(sphere_prim)
        contact_report.CreateThresholdAttr(0.0)
        
        # Sphere material
        sphere_mat = UsdShade.Material.Define(self.stage, "/World/sphereMat")
        sphere_shader = UsdShade.Shader.Define(self.stage, "/World/sphereMat/Shader")
        sphere_shader.CreateIdAttr("UsdPreviewSurface")
        sphere_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.8, 0.2, 0.2))
        sphere_shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.8)
        sphere_shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.2)
        sphere_mat.CreateSurfaceOutput().ConnectToSource(sphere_shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(sphere_prim).Bind(sphere_mat)
        
        # === CONTACT TRACKING ===
        print("Creating RigidPrim for contact tracking...")
        self.contact_view = RigidPrim(
            prim_paths_expr="/World/Sphere",
            name="contact_tracking",
            contact_filter_prim_paths_expr=["/World/FixedBox"],
            max_contact_count=32,
        )
        self.world.scene.add(self.contact_view)
        
        print("✓ Scene created")
        print(f"  Box: static at z={box_center_z*1000:.1f}mm")
        print(f"  Platform: starts at z={self.initial_platform_z*1000:.1f}mm")
        print(f"  Sphere: rests on platform, will be pushed into box")
        print()
        
        self.world.reset()
        
        self.sphere_path = sphere_path
        self.min_platform_z = box_center_z
        self.movement_active = True
    
    def setup_camera(self):
        from omni.isaac.core.utils.viewports import set_camera_view
        set_camera_view(
            eye=Gf.Vec3d(0.25, 0.25, 0.15),
            target=Gf.Vec3d(0, 0, self.box_size/2),
            camera_prim_path="/OmniverseKit_Persp"
        )
    
    def get_contact_force(self):
        """Extract contact force"""
        try:
            result = self.contact_view.get_contact_force_data(dt=self.time_step)
            
            if result is not None:
                forces, points, normals, distances, counts, starts = result
                
                if forces.shape[0] > 0:
                    normal_force = np.sum(np.sum(forces * normals, axis=1))
                    force_magnitude = np.linalg.norm(np.sum(forces, axis=0))
                    
                    return {
                        'normal_force': abs(normal_force),
                        'force_magnitude': force_magnitude,
                        'num_contacts': forces.shape[0],
                    }
            
            return {
                'normal_force': 0.0,
                'force_magnitude': 0.0,
                'num_contacts': 0,
            }
            
        except Exception as e:
            return {
                'normal_force': 0.0,
                'force_magnitude': 0.0,
                'num_contacts': 0,
            }
    
    def get_sphere_height(self):
        """Get sphere world height"""
        sphere_prim = self.stage.GetPrimAtPath(self.sphere_path)
        xform = UsdGeom.Xformable(sphere_prim)
        world_transform = xform.ComputeLocalToWorldTransform(0)
        translation = world_transform.ExtractTranslation()
        return translation[2]
    
    def move_platform(self, t):
        """Move platform downward"""
        if not self.movement_active:
            return
        
        # Calculate new position
        new_z = self.initial_platform_z - (self.descent_speed * t)
        
        # Stop at minimum
        if new_z <= self.min_platform_z:
            new_z = self.min_platform_z
            if self.movement_active:
                self.movement_active = False
                print(f"\n  *** Platform reached bottom at t={t:.2f}s ***\n")
        
        # Update platform position
        self.platform_translate_op.Set(Gf.Vec3f(0, 0, new_z))
    
    def run(self):
        print("Starting test...\n")
        print("Platform will descend, pushing sphere into box.")
        print(f"{'Time (s)':<10} {'Sphere Z (mm)':<15} {'N Contacts':<12} {'Force (N)':<12}")
        print("-" * 70)
        
        step = 0
        last_print_time = 0
        print_interval = 0.5
        test_duration = 12.0
        
        contact_started = False
        
        while simulation_app.is_running():
            if self.world.is_playing():
                t = self.world.current_time
                
                # Move platform before physics step
                self.move_platform(t)
                
                # Step simulation
                self.world.step(render=True)
                
                # Get data after step
                if step > 10:
                    z_height = self.get_sphere_height()
                    contact_data = self.get_contact_force()
                    
                    self.time_log.append(t)
                    self.position_log.append(z_height)
                    self.force_log.append(contact_data['normal_force'])
                    
                    # Print updates
                    if t - last_print_time >= print_interval:
                        n_contacts = contact_data['num_contacts']
                        normal_f = contact_data['normal_force']
                        
                        print(f"{t:<10.2f} {z_height*1000:<15.2f} {n_contacts:<12} {normal_f:<12.6f}")
                        
                        # Alert on first contact
                        if normal_f > 0.001 and not contact_started:
                            contact_started = True
                            print(f"\n  *** CONTACT at t={t:.2f}s, z={z_height*1000:.2f}mm, F={normal_f:.4f}N ***\n")
                        
                        last_print_time = t
                
                # Stop after test duration
                if t > test_duration:
                    print(f"\n✓ Test complete at t={t:.2f}s")
                    self.print_summary()
                    self.plot_results()
                    print("\nClose window to exit.")
                    
                    while simulation_app.is_running():
                        self.world.step(render=True)
                    break
            else:
                self.world.step(render=True)
            
            step += 1
        
        simulation_app.close()
    
    def print_summary(self):
        """Print summary"""
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        
        if not self.force_log:
            print("No data collected!")
            return
        
        force_array = np.array(self.force_log)
        contact_threshold = 0.001
        contact_detected = force_array > contact_threshold
        
        total_steps = len(self.force_log)
        contact_steps = np.sum(contact_detected)
        max_force = np.max(force_array)
        avg_force = np.mean(force_array[contact_detected]) if np.any(contact_detected) else 0
        
        print(f"Total steps: {total_steps}")
        print(f"Steps with contact: {contact_steps} ({contact_steps/total_steps*100:.1f}%)")
        print(f"Max force: {max_force:.4f} N")
        print(f"Avg force (when in contact): {avg_force:.4f} N")
        
        # First contact
        for i, has_contact in enumerate(contact_detected):
            if has_contact:
                print(f"First contact: t={self.time_log[i]:.3f}s, z={self.position_log[i]*1000:.2f}mm")
                print(f"First contact force: {self.force_log[i]:.4f} N")
                break
        else:
            print("WARNING: No contact detected!")
        
        print("="*70)
    
    def plot_results(self):
        """Plot results"""
        force = np.array(self.force_log)
        time = np.array(self.time_log)
        height = np.array(self.position_log) * 1000
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        ax1.plot(time, force, 'b-', linewidth=2)
        ax1.set_xlabel('Time (s)', fontsize=12)
        ax1.set_ylabel('Normal Contact Force (N)', fontsize=12)
        ax1.set_title('Contact Force vs Time (Moving Platform)', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(time, height, 'r-', linewidth=2)
        ax2.set_xlabel('Time (s)', fontsize=12)
        ax2.set_ylabel('Sphere Height (mm)', fontsize=12)
        ax2.set_title('Sphere Height vs Time', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('platform_contact_test.png', dpi=150, bbox_inches='tight')
        print(f"\n📊 Plot saved: platform_contact_test.png")


if __name__ == "__main__":
    ContactForceTestPlatform().run()