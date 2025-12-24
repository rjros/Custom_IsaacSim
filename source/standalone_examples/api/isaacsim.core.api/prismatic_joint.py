# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""
Contact Force Test - PROPER PRISMATIC JOINT

Based on official USD Physics prismatic joint example.
Proper joint setup with body0 (world) and body1 (sphere).
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


class ContactForceTestJoint:
    
    def __init__(self):
        self.time_step = 1/60.0
        self.box_size = 0.1  # 10cm box
        self.sphere_radius = 0.01  # 1cm sphere
        self.descent_speed = 0.005  # 5mm/s
        self.max_descent = 0.10  # 4cm travel
        
        print(f"\n{'='*80}")
        print(f"Contact Force Test - PROPER PRISMATIC JOINT")
        print(f"{'='*80}")
        print(f"  Sphere: {self.sphere_radius*100:.1f} cm radius")
        print(f"  Box: {self.box_size*100:.1f} cm cube")
        print(f"  Descent speed: {self.descent_speed*1000:.1f} mm/s")
        print(f"  Travel: {self.max_descent*1000:.1f} mm")
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
        
        # === FIXED ANCHOR (Body0 for joint) ===
        # This is the stationary part - joint will be attached here
        anchor_path = "/World/Anchor"
        self.initial_height = self.box_size + self.sphere_radius + 0.02
        
        anchor_xform = UsdGeom.Xform.Define(self.stage, anchor_path)
        anchor_xform.AddTranslateOp().Set(Gf.Vec3f(0, 0, self.initial_height))
        
        # Make anchor kinematic (fixed in space)
        anchor_prim = anchor_xform.GetPrim()
        rigid_api = UsdPhysics.RigidBodyAPI.Apply(anchor_prim)
        rigid_api.CreateKinematicEnabledAttr().Set(True)
        
        # === SPHERE (Body1 for joint - will slide) ===
        sphere_path = "/World/Sphere"
        
        sphere_geom = UsdGeom.Sphere.Define(self.stage, sphere_path)
        sphere_geom.CreateRadiusAttr().Set(self.sphere_radius)
        
        # Position at same location as anchor initially
        sphere_xform = UsdGeom.Xformable(sphere_geom)
        sphere_xform.ClearXformOpOrder()
        sphere_xform.AddTranslateOp().Set(Gf.Vec3f(0, 0, self.initial_height))
        
        sphere_prim = sphere_geom.GetPrim()
        
        # Dynamic rigid body (will move via joint)
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
        
        # === CREATE PRISMATIC JOINT ===
        joint_path = "/World/PrismaticJoint_Z"
        prismatic_joint = UsdPhysics.PrismaticJoint.Define(self.stage, joint_path)
        
    
        
        # Axis (Z = vertical)
        prismatic_joint.CreateAxisAttr("Z")
        
        # Limits (0 to -max_descent, moving downward)
        prismatic_joint.CreateLowerLimitAttr(-self.max_descent)
        prismatic_joint.CreateUpperLimitAttr(0.0)
        
        # Connect bodies: Body0 (anchor/fixed) -> Body1 (sphere/moving)
        prismatic_joint.CreateBody0Rel().SetTargets([anchor_path])
        prismatic_joint.CreateBody1Rel().SetTargets([sphere_path])
        
        # === ADD JOINT DRIVE ===
        joint_prim = prismatic_joint.GetPrim()
        
        # Linear drive API
        drive = UsdPhysics.DriveAPI.Apply(joint_prim, "linear")
        drive.CreateTypeAttr("force")
        drive.CreateDampingAttr(1000.0)  # High damping for smooth motion
        drive.CreateStiffnessAttr(0.0)  # No spring
        drive.CreateMaxForceAttr(100.0)
        
        # Set target velocity (negative = downward)
        drive.CreateTargetVelocityAttr(-self.descent_speed)
        
        # PhysX joint settings
        physx_joint = PhysxSchema.PhysxJointAPI.Apply(joint_prim)
        physx_joint.CreateMaxJointVelocityAttr().Set(self.descent_speed * 2)
        
        print("✓ Prismatic joint created")
        print(f"  Anchor (Body0): fixed at z={self.initial_height*1000:.1f}mm")
        print(f"  Sphere (Body1): will slide down along Z-axis")
        
        self.joint_prim = joint_prim
        self.drive_api = drive
        self.sphere_path = sphere_path
        
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
        print()
        
        self.world.reset()
        
        self.stopped = False
    
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
    
    def get_joint_position(self):
        """Get joint position (how far it has traveled)"""
        sphere_prim = self.stage.GetPrimAtPath(self.sphere_path)
        xform = UsdGeom.Xformable(sphere_prim)
        local_transform = xform.GetLocalTransformation()
        translation = local_transform.ExtractTranslation()
        
        # Joint position relative to anchor
        return translation[2] - self.initial_height
    
    def control_joint(self, t):
        """Monitor and stop joint when limit reached"""
        joint_pos = self.get_joint_position()
        
        # Stop when close to limit
        if joint_pos <= -self.max_descent + 0.001 and not self.stopped:
            self.stopped = True
            self.drive_api.GetTargetVelocityAttr().Set(0.0)
            print(f"\n  *** Joint reached limit at t={t:.2f}s, position={joint_pos*1000:.2f}mm ***\n")
    
    def run(self):
        print("Starting test...\n")
        print("Sphere will descend via prismatic joint.")
        print(f"{'Time (s)':<10} {'Sphere Z (mm)':<15} {'Joint (mm)':<12} {'N Contacts':<12} {'Force (N)':<12}")
        print("-" * 80)
        
        step = 0
        last_print_time = 0
        print_interval = 0.5
        test_duration = 10.0
        
        contact_started = False
        
        while simulation_app.is_running():
            if self.world.is_playing():
                t = self.world.current_time
                
                # Check joint position
                self.control_joint(t)
                
                # Step simulation
                self.world.step(render=True)
                
                # Get data
                if step > 10:
                    z_height = self.get_sphere_height()
                    joint_pos = self.get_joint_position()
                    contact_data = self.get_contact_force()
                    
                    self.time_log.append(t)
                    self.position_log.append(z_height)
                    self.force_log.append(contact_data['normal_force'])
                    
                    # Print updates
                    if t - last_print_time >= print_interval:
                        n_contacts = contact_data['num_contacts']
                        normal_f = contact_data['normal_force']
                        
                        print(f"{t:<10.2f} {z_height*1000:<15.2f} {joint_pos*1000:<12.2f} {n_contacts:<12} {normal_f:<12.6f}")
                        
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
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        
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
        
        print("="*80)
    
    def plot_results(self):
        """Plot results"""
        force = np.array(self.force_log)
        time = np.array(self.time_log)
        height = np.array(self.position_log) * 1000
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        ax1.plot(time, force, 'b-', linewidth=2)
        ax1.set_xlabel('Time (s)', fontsize=12)
        ax1.set_ylabel('Normal Contact Force (N)', fontsize=12)
        ax1.set_title('Contact Force vs Time (Prismatic Joint)', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(time, height, 'r-', linewidth=2)
        ax2.set_xlabel('Time (s)', fontsize=12)
        ax2.set_ylabel('Sphere Height (mm)', fontsize=12)
        ax2.set_title('Sphere Height vs Time', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('joint_contact_test.png', dpi=150, bbox_inches='tight')
        print(f"\n📊 Plot saved: joint_contact_test.png")


if __name__ == "__main__":
    ContactForceTestJoint().run()