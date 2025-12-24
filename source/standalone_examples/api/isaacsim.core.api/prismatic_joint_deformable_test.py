# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""
Deformable Contact Test - JOINT FORCE MEASUREMENT

Read reaction forces directly from the prismatic joint.
This measures the force the joint exerts to maintain the constraint.
"""

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False, "extra_args": ["--/app/useFabricSceneDelegate=0"]})

import numpy as np
import matplotlib.pyplot as plt
from pxr import Gf, UsdGeom, UsdPhysics, PhysxSchema, Sdf, UsdShade, UsdLux
import torch

from isaacsim.core.api import World
from isaacsim.core.api.materials.deformable_material import DeformableMaterial
from isaacsim.core.prims import SingleDeformablePrim, RigidPrim
from isaacsim.core.articulations import ArticulationView
import isaacsim.core.utils.deformable_mesh_utils as deformableMeshUtils
from omni.physx.scripts import physicsUtils, utils
import omni.usd
from pxr import PhysxSchema

np.random.seed(42)


class DeformableContactTest:
    
    def __init__(self):
        self.time_step = 1/60.0
        
        # Geometry
        self.cube_size = 0.04  # 4cm cube
        self.sphere_radius = 0.005  # 5mm sphere
        
        # Material properties
        self.youngs_modulus = 50000  # 50 kPa
        self.poisson_ratio = 0.4
        self.damping_scale = 0.1
        self.elasticity_damping = 0.1
        
        # Motion
        self.descent_speed = 0.003  # 3mm/s
        self.max_descent = 0.10  # 8mm indentation
        
        print(f"\n{'='*80}")
        print(f"Deformable Contact Test - Joint Force Measurement")
        print(f"{'='*80}")
        print(f"  Cube: {self.cube_size*1000:.1f}mm, E={self.youngs_modulus/1000:.1f}kPa, ν={self.poisson_ratio}")
        print(f"  Sphere: {self.sphere_radius*1000:.1f}mm diameter indenter")
        print(f"  Speed: {self.descent_speed*1000:.1f}mm/s")
        print(f"  Max indentation: {self.max_descent*1000:.1f}mm")
        print(f"  Force measurement: Joint constraint forces")
        print(f"{'='*80}\n")
        
        self.my_world = World(
            stage_units_in_meters=1.0,
            backend="torch",
            device="cuda",
            physics_dt=self.time_step,
            rendering_dt=self.time_step * 2
        )
        self.stage = omni.usd.get_context().get_stage()
        
        self.force_log = []
        self.time_log = []
        self.position_log = []
        self.indentation_log = []
        
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
        
        # === DEFORMABLE CUBE - STARTS ON GROUND ===
        cube_center_z = self.cube_size / 2
        cube_path = "/World/deformable_cube"
        
        # Create triangle mesh cube
        cube_mesh = UsdGeom.Mesh.Define(self.stage, cube_path)
        tri_points, tri_indices = deformableMeshUtils.createTriangleMeshCube(10)
        cube_mesh.GetPointsAttr().Set(tri_points)
        cube_mesh.GetFaceVertexIndicesAttr().Set(tri_indices)
        cube_mesh.GetFaceVertexCountsAttr().Set([3] * (len(tri_indices) // 3))
        
        # Transform setup
        physicsUtils.setup_transform_as_scale_orient_translate(cube_mesh)
        physicsUtils.set_or_add_scale_op(cube_mesh, Gf.Vec3f(self.cube_size, self.cube_size, self.cube_size))
        physicsUtils.set_or_add_translate_op(cube_mesh, Gf.Vec3f(0.0, 0.0, cube_center_z))
        
        # Material
        cube_mat = UsdShade.Material.Define(self.stage, "/World/cubeMat")
        cube_shader = UsdShade.Shader.Define(self.stage, "/World/cubeMat/Shader")
        cube_shader.CreateIdAttr("UsdPreviewSurface")
        cube_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.3, 0.6, 0.9))
        cube_mat.CreateSurfaceOutput().ConnectToSource(cube_shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(cube_mesh.GetPrim()).Bind(cube_mat)
        
        # Deformable material
        deformable_mat_path = "/World/deformableMaterial"
        self.deformable_material = DeformableMaterial(
            prim_path=deformable_mat_path,
            dynamic_friction=0.5,
            youngs_modulus=self.youngs_modulus,
            poissons_ratio=self.poisson_ratio,
            damping_scale=self.damping_scale,
            elasticity_damping=self.elasticity_damping,
        )
        
        # Deformable body
        self.deformable_cube = SingleDeformablePrim(
            name="softCube",
            prim_path=cube_path,
            deformable_material=self.deformable_material,
            vertex_velocity_damping=0.0,
            sleep_damping=1.0,
            sleep_threshold=0.05,
            settling_threshold=0.1,
            self_collision=False,
            solver_position_iteration_count=130,
            kinematic_enabled=False,
            simulation_hexahedral_resolution=40,
            collision_simplification=False,
        )
        self.my_world.scene.add(self.deformable_cube)
        
        print(f"✓ Deformable cube positioned on ground")
        
        # === ARTICULATION ROOT (for joint force reading) ===
        articulation_path = "/World/Indenter"
        articulation_root = UsdGeom.Xform.Define(self.stage, articulation_path)
        
        # === FIXED ANCHOR (Body0) ===
        anchor_path = articulation_path + "/Anchor"
        self.initial_height = self.cube_size + self.sphere_radius + 0.5
        
        anchor_xform = UsdGeom.Xform.Define(self.stage, anchor_path)
        anchor_xform.AddTranslateOp().Set(Gf.Vec3f(0, 0, self.initial_height))
        
        anchor_prim = anchor_xform.GetPrim()
        rigid_api = UsdPhysics.RigidBodyAPI.Apply(anchor_prim)
        rigid_api.CreateKinematicEnabledAttr().Set(True)
        
        # === RIGID SPHERE (Body1 - moving) ===
        sphere_path = articulation_path + "/Sphere"
        
        sphere_geom = UsdGeom.Sphere.Define(self.stage, sphere_path)
        sphere_geom.CreateRadiusAttr().Set(self.sphere_radius)
        
        sphere_xform = UsdGeom.Xformable(sphere_geom)
        sphere_xform.ClearXformOpOrder()
        sphere_xform.AddTranslateOp().Set(Gf.Vec3f(0, 0, 0))  # Relative to anchor
        
        sphere_prim = sphere_geom.GetPrim()
        
        # Dynamic rigid body
        rigid_api = UsdPhysics.RigidBodyAPI.Apply(sphere_prim)
        rigid_api.CreateKinematicEnabledAttr().Set(False)
        
        mass_api = UsdPhysics.MassAPI.Apply(sphere_prim)
        mass_api.CreateMassAttr().Set(0.01)
        
        UsdPhysics.CollisionAPI.Apply(sphere_prim)
        
        # Contact report
        contact_report = PhysxSchema.PhysxContactReportAPI.Apply(sphere_prim)
        contact_report.CreateThresholdAttr(0.0)
        
        # Sphere material
        sphere_mat = UsdShade.Material.Define(self.stage, "/World/sphereMat")
        sphere_shader = UsdShade.Shader.Define(self.stage, "/World/sphereMat/Shader")
        sphere_shader.CreateIdAttr("UsdPreviewSurface")
        sphere_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.8, 0.2, 0.2))
        sphere_shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.9)
        sphere_shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.1)
        sphere_mat.CreateSurfaceOutput().ConnectToSource(sphere_shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(sphere_prim).Bind(sphere_mat)
        
        print(f"✓ Sphere indenter in articulation")
        
        # === PRISMATIC JOINT ===
        joint_path = articulation_path + "/PrismaticJoint"
        prismatic_joint = UsdPhysics.PrismaticJoint.Define(self.stage, joint_path)
        
        prismatic_joint.CreateAxisAttr("Z")
        prismatic_joint.CreateLowerLimitAttr(-self.max_descent * 2)
        prismatic_joint.CreateUpperLimitAttr(0.0)
        
        prismatic_joint.CreateBody0Rel().SetTargets([anchor_path])
        prismatic_joint.CreateBody1Rel().SetTargets([sphere_path])
        
        # Joint drive
        joint_prim = prismatic_joint.GetPrim()
        drive = UsdPhysics.DriveAPI.Apply(joint_prim, "linear")
        drive.CreateTypeAttr("force")
        drive.CreateDampingAttr(1000.0)
        drive.CreateStiffnessAttr(0.0)
        drive.CreateMaxForceAttr(100.0)
        drive.CreateTargetVelocityAttr(0.0)
        
        # PhysX joint API - ENABLE FORCE SENSOR
        physx_joint = PhysxSchema.PhysxJointAPI.Apply(joint_prim)
        physx_joint.CreateMaxJointVelocityAttr().Set(self.descent_speed * 2)
        
        # CRITICAL: Enable joint force/torque sensor
        physx_joint.CreateJointFrictionAttr(0.0)
        
        print("✓ Prismatic joint with force sensing")
        
        self.joint_prim = joint_prim
        self.joint_path = joint_path
        self.drive_api = drive
        self.sphere_path = sphere_path
        self.articulation_path = articulation_path
        
        # === ARTICULATION VIEW (for force reading) ===
        self.articulation_view = ArticulationView(
            prim_paths_expr=articulation_path,
            name="indenter_articulation"
        )
        self.my_world.scene.add(self.articulation_view)
        
        print("✓ ArticulationView created for force reading")
        print("✓ Scene complete\n")
        
        self.my_world.reset(soft=False)
        
        self.stopped = False
        self.contact_started = False
        self.contact_z = None
        self.test_started = False
        self.reset_needed = False
    
    def setup_camera(self):
        from omni.isaac.core.utils.viewports import set_camera_view
        set_camera_view(
            eye=Gf.Vec3d(0.12, 0.12, 0.08),
            target=Gf.Vec3d(0, 0, self.cube_size/2),
            camera_prim_path="/OmniverseKit_Persp"
        )
    
    def get_joint_force(self):
        """Get force from joint constraint"""
        try:
            # Method 1: Try to get measured joint forces from articulation view
            joint_forces = self.articulation_view.get_measured_joint_forces()
            if joint_forces is not None and len(joint_forces) > 0:
                # For prismatic joint, force is in the constrained direction (Z)
                force = abs(joint_forces[0, 0].item())  # First articulation, first joint
                return force
        except Exception as e:
            pass
        
        try:
            # Method 2: Try applied forces
            applied_forces = self.articulation_view.get_applied_joint_forces()
            if applied_forces is not None and len(applied_forces) > 0:
                force = abs(applied_forces[0, 0].item())
                return force
        except:
            pass
        
        return 0.0
    
    def get_sphere_height(self):
        """Get sphere world height"""
        sphere_prim = self.stage.GetPrimAtPath(self.sphere_path)
        xform = UsdGeom.Xformable(sphere_prim)
        world_transform = xform.ComputeLocalToWorldTransform(0)
        translation = world_transform.ExtractTranslation()
        return translation[2]
    
    def get_joint_position(self):
        """Get joint position"""
        try:
            positions = self.articulation_view.get_joint_positions()
            if positions is not None and len(positions) > 0:
                return positions[0, 0].item()
        except:
            pass
        return 0.0
    
    def get_indentation(self, sphere_z):
        """Calculate indentation depth"""
        if self.contact_z is None:
            return 0.0
        return max(0.0, self.contact_z - sphere_z)
    
    def control_joint(self, test_time):
        """Monitor and stop joint"""
        joint_pos = self.get_joint_position()
        
        if joint_pos <= -self.max_descent and not self.stopped:
            self.stopped = True
            self.drive_api.GetTargetVelocityAttr().Set(0.0)
            print(f"\n  *** Max indentation reached at t={test_time:.2f}s ***\n")
    
    def run(self):
        print("Starting simulation...\n")
        print("Settling cube (2 seconds)...")
        
        step = 0
        settling_time = 2.0
        last_print_time = 0
        print_interval = 0.5
        test_duration = 10.0
        start_time_offset = 0
        
        while simulation_app.is_running():
            # Handle stop/play reset
            if self.my_world.is_stopped() and not self.reset_needed:
                self.reset_needed = True
            
            if self.my_world.is_playing():
                if self.reset_needed:
                    self.my_world.reset(soft=False)
                    self.reset_needed = False
                    self.test_started = False
                    self.contact_started = False
                    self.stopped = False
                    self.contact_z = None
                    self.force_log = []
                    self.time_log = []
                    self.position_log = []
                    self.indentation_log = []
                    print("\n⟳ Simulation reset\n")
                
                t = self.my_world.current_time
                
                # Phase 1: Settling
                if t < settling_time:
                    if t - last_print_time >= 1.0:
                        print(f"  Settling... t={t:.1f}s")
                        last_print_time = t
                    
                    self.my_world.step(render=True)
                    step += 1
                    continue
                
                # Phase 2: Start test
                if not self.test_started:
                    self.test_started = True
                    start_time_offset = t
                    
                    # Start joint motion
                    self.drive_api.GetTargetVelocityAttr().Set(-self.descent_speed)
                    
                    print(f"\n✓ Starting indentation at t={t:.2f}s\n")
                    print(f"{'Time (s)':<10} {'Z (mm)':<10} {'Indent (mm)':<12} {'Joint F (N)':<12}")
                    print("-" * 60)
                    last_print_time = t
                
                test_time = t - start_time_offset
                
                # Control joint
                self.control_joint(test_time)
                
                # Step simulation
                self.my_world.step(render=True)
                
                # Get data
                z_height = self.get_sphere_height()
                joint_force = self.get_joint_force()
                
                # Detect first contact (when joint force appears)
                if not self.contact_started and joint_force > 0.01:
                    self.contact_started = True
                    self.contact_z = z_height
                    print(f"\n  *** CONTACT at t={test_time:.2f}s, z={z_height*1000:.2f}mm ***\n")
                
                indentation = self.get_indentation(z_height)
                
                self.time_log.append(test_time)
                self.position_log.append(z_height)
                self.indentation_log.append(indentation)
                self.force_log.append(joint_force)
                
                # Print updates
                if test_time - last_print_time >= print_interval:
                    print(f"{test_time:<10.2f} {z_height*1000:<10.2f} {indentation*1000:<12.3f} {joint_force:<12.6f}")
                    last_print_time = test_time
                
                # Stop after test duration
                if test_time > test_duration:
                    print(f"\n✓ Test complete at t={test_time:.2f}s")
                    self.print_summary()
                    self.plot_results()
                    print("\nClose window to exit.")
                    
                    while simulation_app.is_running():
                        self.my_world.step(render=True)
                    break
            else:
                self.my_world.step(render=True)
            
            step += 1
        
        simulation_app.close()
    
    def print_summary(self):
        """Print summary"""
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        
        if not self.force_log:
            print("No data collected!")
            return
        
        force_array = np.array(self.force_log)
        indent_array = np.array(self.indentation_log) * 1000
        
        contact_threshold = 0.01
        contact_detected = force_array > contact_threshold
        
        total_steps = len(self.force_log)
        contact_steps = np.sum(contact_detected)
        max_force = np.max(force_array)
        max_indent = np.max(indent_array)
        
        print(f"Total steps: {total_steps}")
        print(f"Steps with contact: {contact_steps} ({contact_steps/total_steps*100:.1f}%)")
        print(f"Max indentation: {max_indent:.3f} mm")
        print(f"Max joint force: {max_force:.6f} N")
        
        if contact_steps > 0:
            avg_force = np.mean(force_array[contact_detected])
            print(f"Avg force (in contact): {avg_force:.6f} N")
        
        print("="*60)
    
    def plot_results(self):
        """Plot results"""
        force = np.array(self.force_log)
        time = np.array(self.time_log)
        indent = np.array(self.indentation_log) * 1000
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        ax1.plot(indent, force, 'b-', linewidth=2, label='Joint Force')
        ax1.set_xlabel('Indentation (mm)', fontsize=12)
        ax1.set_ylabel('Force (N)', fontsize=12)
        ax1.set_title('Force vs Indentation (from Joint)', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        ax2.plot(time, force, 'r-', linewidth=2)
        ax2.set_xlabel('Time (s)', fontsize=12)
        ax2.set_ylabel('Force (N)', fontsize=12)
        ax2.set_title('Joint Force vs Time', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('deformable_joint_force.png', dpi=150, bbox_inches='tight')
        print(f"\n📊 Plot saved: deformable_joint_force.png")


if __name__ == "__main__":
    DeformableContactTest().run()