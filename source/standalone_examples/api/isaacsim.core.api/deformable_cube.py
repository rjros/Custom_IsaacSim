# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""
Deformable Cube Test - BASED ON WORKING FRANKA DEMO

Using the exact deformable settings from the working Franka demo.
"""

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False, "extra_args": ["--/app/useFabricSceneDelegate=0"]})

import numpy as np
from pxr import Gf, UsdGeom, Sdf, UsdShade, UsdLux, UsdPhysics, PhysxSchema
import torch

from isaacsim.core.api import World
from omni.physx.scripts import deformableUtils, physicsUtils
import isaacsim.core.utils.deformable_mesh_utils as deformableMeshUtils
import omni.usd

np.random.seed(42)


class DeformableCubeTest:
    
    def __init__(self):
        self.time_step = 1/60.0
        self.cube_size = 0.04  # 5cm cube (same as Franka demo)
        self.sphere_radius = 0.005  # 5mm sphere
        self.descent_speed = 0.002  # 2mm/s
        self.max_descent = 0.015  # 8mm indentation
        
        print(f"\n{'='*80}")
        print(f"Deformable Indentation Test - Franka Settings + Joint")
        print(f"{'='*80}")
        print(f"  Cube: {self.cube_size*1000:.1f}mm (deformable)")
        print(f"  Sphere: {self.sphere_radius*1000:.1f}mm (rigid indenter)")
        print(f"  Speed: {self.descent_speed*1000:.1f}mm/s")
        print(f"  Settings: From working Franka grasping demo")
        print(f"{'='*80}\n")
        
        self.my_world = World(
            stage_units_in_meters=1.0,
            backend="torch",
            device="cuda",
            physics_dt=self.time_step,
            rendering_dt=self.time_step * 2
        )
        self.stage = omni.usd.get_context().get_stage()
        
        # Use default ground plane (like sample)
        self.my_world.scene.add_default_ground_plane()
        
        self.setup_environment()
        self.setup_scene()
        self.setup_camera()
    
    def setup_environment(self):
        dome_light_path = "/World/DomeLight"
        dome_light = UsdLux.DomeLight.Define(self.stage, dome_light_path)
        dome_light.CreateIntensityAttr().Set(1500.0)
        
        # Add simple visual grid
        self.create_visual_ground()
    
    def create_visual_ground(self):
        """Create a visible grid ground plane"""
        ground_path = "/World/VisualGround"
        
        # Create a large plane mesh
        plane = UsdGeom.Mesh.Define(self.stage, ground_path)
        
        # Create grid vertices (2m x 2m plane with grid)
        size = 2.0
        grid_divisions = 40
        
        points = []
        indices = []
        counts = []
        
        step = size / grid_divisions
        
        # Create grid lines in X direction
        for i in range(grid_divisions + 1):
            y = -size/2 + i * step
            points.append(Gf.Vec3f(-size/2, y, 0))
            points.append(Gf.Vec3f(size/2, y, 0))
        
        # Create grid lines in Y direction  
        for i in range(grid_divisions + 1):
            x = -size/2 + i * step
            points.append(Gf.Vec3f(x, -size/2, 0))
            points.append(Gf.Vec3f(x, size/2, 0))
        
        # For mesh, create simple quad
        quad_points = [
            Gf.Vec3f(-size/2, -size/2, -0.001),
            Gf.Vec3f(size/2, -size/2, -0.001),
            Gf.Vec3f(size/2, size/2, -0.001),
            Gf.Vec3f(-size/2, size/2, -0.001),
        ]
        
        quad_indices = [0, 1, 2, 3]
        
        plane.GetPointsAttr().Set(quad_points)
        plane.GetFaceVertexIndicesAttr().Set(quad_indices)
        plane.GetFaceVertexCountsAttr().Set([4])
        
        # Material - light gray with grid
        ground_mat = UsdShade.Material.Define(self.stage, "/World/groundMat")
        ground_shader = UsdShade.Shader.Define(self.stage, "/World/groundMat/Shader")
        ground_shader.CreateIdAttr("UsdPreviewSurface")
        ground_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.8, 0.8, 0.8))
        ground_shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
        ground_mat.CreateSurfaceOutput().ConnectToSource(ground_shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(plane.GetPrim()).Bind(ground_mat)
        
        # Add grid lines as a separate geometry
        grid_path = "/World/Grid"
        grid = UsdGeom.BasisCurves.Define(self.stage, grid_path)
        
        # Create line segments
        line_points = []
        line_counts = []
        
        # X direction lines
        for i in range(grid_divisions + 1):
            y = -size/2 + i * step
            line_points.append(Gf.Vec3f(-size/2, y, 0.001))
            line_points.append(Gf.Vec3f(size/2, y, 0.001))
            line_counts.append(2)
        
        # Y direction lines
        for i in range(grid_divisions + 1):
            x = -size/2 + i * step
            line_points.append(Gf.Vec3f(x, -size/2, 0.001))
            line_points.append(Gf.Vec3f(x, size/2, 0.001))
            line_counts.append(2)
        
        grid.GetPointsAttr().Set(line_points)
        grid.GetCurveVertexCountsAttr().Set(line_counts)
        grid.GetTypeAttr().Set("linear")
        grid.GetWidthsAttr().Set([0.5] * len(line_counts))
        
        # Grid color - darker gray
        grid.CreateDisplayColorAttr().Set([Gf.Vec3f(0.5, 0.5, 0.5)])
        
        print("✓ Visual ground plane created (2m x 2m grid)")
    
    def create_sphere_indenter(self, cube_top_z):
        """Create sphere indenter on prismatic joint"""
        
        print("\nCreating sphere indenter with prismatic joint...")
        
        # === ANCHOR (Body0 - fixed) ===
        anchor_path = "/World/Anchor"
        self.initial_height = self.cube_size + self.sphere_radius

        
        anchor_xform = UsdGeom.Xform.Define(self.stage, anchor_path)
        anchor_xform.AddTranslateOp().Set(Gf.Vec3f(0, 0, self.initial_height))
        
        anchor_prim = anchor_xform.GetPrim()
        rigid_api = UsdPhysics.RigidBodyAPI.Apply(anchor_prim)
        rigid_api.CreateKinematicEnabledAttr().Set(True)
        
        # === SPHERE (Body1 - moving) ===
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
        mass_api.CreateMassAttr().Set(0.05)  # 50g - heavier to deform soft cube
        
        UsdPhysics.CollisionAPI.Apply(sphere_prim)
        
        # Contact report
        contact_report = PhysxSchema.PhysxContactReportAPI.Apply(sphere_prim)
        contact_report.CreateThresholdAttr(0.0)
        
        # Sphere material - metallic red
        sphere_mat = UsdShade.Material.Define(self.stage, "/World/sphereMat")
        sphere_shader = UsdShade.Shader.Define(self.stage, "/World/sphereMat/Shader")
        sphere_shader.CreateIdAttr("UsdPreviewSurface")
        sphere_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.9, 0.1, 0.1))
        sphere_shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.95)
        sphere_shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.05)
        sphere_mat.CreateSurfaceOutput().ConnectToSource(sphere_shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(sphere_prim).Bind(sphere_mat)
        
        # === PRISMATIC JOINT ===
        joint_path = "/World/PrismaticJoint"
        prismatic_joint = UsdPhysics.PrismaticJoint.Define(self.stage, joint_path)
        
        prismatic_joint.CreateAxisAttr("Z")
        prismatic_joint.CreateLowerLimitAttr(-0.020)  # 20mm travel
        prismatic_joint.CreateUpperLimitAttr(0.0)
        
        prismatic_joint.CreateBody0Rel().SetTargets([anchor_path])
        prismatic_joint.CreateBody1Rel().SetTargets([sphere_path])
        
        # Joint drive - starts stopped
        joint_prim = prismatic_joint.GetPrim()
        drive = UsdPhysics.DriveAPI.Apply(joint_prim, "linear")
        drive.CreateTypeAttr("force")
        drive.CreateDampingAttr(1000.0)
        drive.CreateStiffnessAttr(0.0)
        drive.CreateMaxForceAttr(100.0)
        drive.CreateTargetVelocityAttr(0.0)  # Start stopped
        
        physx_joint = PhysxSchema.PhysxJointAPI.Apply(joint_prim)
        physx_joint.CreateMaxJointVelocityAttr().Set(self.descent_speed * 2)
        
        # === CONTACT TRACKING (without filter - not supported for deformables) ===
        from isaacsim.core.prims import RigidPrim
        
        print("\nNote: Contact filtering not supported with deformables")
        print("Using general contact tracking instead...")
        
        self.contact_view = RigidPrim(
            prim_paths_expr="/World/Sphere",
            name="contact_tracking",
            # Don't use contact_filter_prim_paths_expr with deformables!
            max_contact_count=32,
        )
        self.my_world.scene.add(self.contact_view)
        
        self.joint_prim = joint_prim
        self.drive_api = drive
        self.sphere_path = sphere_path
        
        print(f"✓ Sphere indenter created at z={self.initial_height*1000:.1f}mm")
        print(f"✓ Prismatic joint configured")
        print(f"✓ Contact tracking enabled (RigidPrim)")
        print(f"  Expected contact at z≈{(cube_top_z + self.sphere_radius)*1000:.1f}mm")
    
    def setup_scene(self):
        print("Creating deformable cube...")
        
        # === DEFORMABLE MATERIAL (exactly like Franka demo red cube) ===
        deformable_material_path = "/World/DeformableBodyMaterial"
        
        deformableUtils.add_deformable_body_material(
            self.stage,
            deformable_material_path,
            youngs_modulus=50000.0,  # 50 kPa - MUCH softer (was 10 MPa)
            poissons_ratio=0.499,  # Slightly compressible
            damping_scale=0.0,
            elasticity_damping=0.005,  # More damping for stability
            dynamic_friction=0.8,
            # density=100,
        )
        
        print(f"✓ Deformable material created")
        print(f"  Young's modulus: 50 kPa (soft)")
        print(f"  Poisson's ratio: 0.45")
        
        # === DEFORMABLE CUBE ===
        cube_path = "/World/deformable_cube"
        
        # Create triangle mesh cube (like Franka demo)
        cube_mesh = UsdGeom.Mesh.Define(self.stage, cube_path)
        tri_points, tri_indices = deformableMeshUtils.createTriangleMeshCube(10)  # Resolution 10
        cube_mesh.GetPointsAttr().Set(tri_points)
        cube_mesh.GetFaceVertexIndicesAttr().Set(tri_indices)
        cube_mesh.GetFaceVertexCountsAttr().Set([3] * (len(tri_indices) // 3))
        
        # Position: start on ground
        cube_center_z = self.cube_size / 2
        physicsUtils.setup_transform_as_scale_orient_translate(cube_mesh)
        physicsUtils.set_or_add_scale_op(cube_mesh, Gf.Vec3f(self.cube_size, self.cube_size, self.cube_size))
        physicsUtils.set_or_add_translate_op(cube_mesh, Gf.Vec3f(0.0, 0.0, cube_center_z))
        
        # Add deformable body (like Franka demo)
        deformableUtils.add_physx_deformable_body(
            self.stage,
            cube_path,
            simulation_hexahedral_resolution=30,  # Franka demo value
            collision_simplification=True,  # Franka demo value
            self_collision=False,  # Franka demo value
            solver_position_iteration_count=100,  # Franka demo value
        )
        
        # Add physics material
        physicsUtils.add_physics_material_to_prim(
            self.stage,
            cube_mesh.GetPrim(),
            deformable_material_path
        )
        
        # Collision offsets (like Franka demo)
        physxCollisionAPI = PhysxSchema.PhysxCollisionAPI.Apply(cube_mesh.GetPrim())
        physxCollisionAPI.GetContactOffsetAttr().Set(0.02)
        physxCollisionAPI.CreateRestOffsetAttr().Set(0.001)
        
        # Material appearance
        cube_mat = UsdShade.Material.Define(self.stage, "/World/cubeMat")
        cube_shader = UsdShade.Shader.Define(self.stage, "/World/cubeMat/Shader")
        cube_shader.CreateIdAttr("UsdPreviewSurface")
        cube_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.8, 0.2, 0.2))
        cube_shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(0.9)
        cube_mat.CreateSurfaceOutput().ConnectToSource(cube_shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(cube_mesh.GetPrim()).Bind(cube_mat)
        
        print(f"✓ Deformable cube created at z={cube_center_z*1000:.1f}mm")
        print(f"  Resolution: 3 (hexahedral)")
        # print(f"  Solver iterations: {self.pos_iterations}")
        print(f"  Collision: contact offset=0.02, rest offset=0.001")
        
        # === ADD SPHERE AND JOINT ===
        self.create_sphere_indenter(cube_center_z)
        
        print()
        
        self.my_world.reset(soft=False)
        
        self.cube_path = cube_path
        self.cube_center_z = cube_center_z
    
    def setup_camera(self):
        from omni.isaac.core.utils.viewports import set_camera_view
        set_camera_view(
            eye=Gf.Vec3d(0.15, 0.15, 0.10),
            target=Gf.Vec3d(0, 0, self.cube_size/2),
            camera_prim_path="/OmniverseKit_Persp"
        )
    
    def run(self):
        print("Starting simulation...\n")
        print("Phase 1: Cube settling (2 seconds)")
        print("Phase 2: Sphere indentation test\n")
        
        step = 0
        settling_time = 2.0
        last_print_time = 0
        print_interval = 0.5
        test_duration = 12.0
        test_started = False
        start_time_offset = 0
        
        contact_started = False
        contact_z = None
        
        force_log = []
        time_log = []
        indent_log = []
        
        while simulation_app.is_running():
            if self.my_world.is_playing():
                t = self.my_world.current_time
                
                # Step simulation
                self.my_world.step(render=True)
                
                # Phase 1: Settling
                if t < settling_time:
                    if step == 1:
                        print(f"{'Time (s)':<10} {'Status':<40}")
                        print("-" * 60)
                    if t - last_print_time >= 1.0:
                        print(f"{t:<10.1f} Cube settling...")
                        last_print_time = t
                    step += 1
                    continue
                
                # Phase 2: Start indentation
                if not test_started:
                    test_started = True
                    start_time_offset = t
                    
                    # Start sphere descent
                    self.drive_api.GetTargetVelocityAttr().Set(-self.descent_speed)
                    
                    print(f"\n✓ Cube settled. Starting indentation test.\n")
                    print(f"{'Time (s)':<10} {'Sphere Z (mm)':<15} {'Indent (mm)':<12} {'Force (N)':<12}")
                    print("-" * 60)
                    last_print_time = t
                
                test_time = t - start_time_offset
                
                # Get sphere position
                sphere_z = self.get_sphere_height()
                
                # Get contact force (like rigid cube example)
                contact_force = self.get_contact_force()
                
                # Detect contact
                if not contact_started and contact_force > 0.01:
                    contact_started = True
                    contact_z = sphere_z
                    print(f"\n  *** CONTACT at t={test_time:.2f}s, z={sphere_z*1000:.2f}mm ***\n")
                
                # Calculate indentation
                if contact_z is not None:
                    indentation = max(0, contact_z - sphere_z)
                else:
                    indentation = 0
                
                # Log data
                force_log.append(contact_force)
                time_log.append(test_time)
                indent_log.append(indentation)
                
                # Print updates
                if test_time - last_print_time >= print_interval:
                    print(f"{test_time:<10.2f} {sphere_z*1000:<15.2f} {indentation*1000:<12.3f} {contact_force:<12.6f}")
                    last_print_time = test_time
                
                # Stop at max indentation
                joint_pos = self.get_joint_position()
                if joint_pos <= -self.max_descent:
                    self.drive_api.GetTargetVelocityAttr().Set(0.0)
                    if step % 100 == 0:
                        print(f"{test_time:<10.2f} {sphere_z*1000:<15.2f} {indentation*1000:<12.3f} {joint_force:<12.6f} (MAX)")
                
                # Complete
                if test_time > test_duration:
                    print(f"\n✓ Test complete")
                    self.print_summary(force_log, indent_log, contact_started)
                    print("\nClose window to exit.")
                    
                    while simulation_app.is_running():
                        self.my_world.step(render=True)
                    break
            else:
                self.my_world.step(render=True)
            
            step += 1
        
        simulation_app.close()
    
    def get_sphere_height(self):
        """Get sphere center world height"""
        sphere_prim = self.stage.GetPrimAtPath(self.sphere_path)
        xform = UsdGeom.Xformable(sphere_prim)
        world_transform = xform.ComputeLocalToWorldTransform(0)
        return world_transform.ExtractTranslation()[2]
    
    def get_joint_position(self):
        """Get joint position (how far traveled from initial)"""
        sphere_prim = self.stage.GetPrimAtPath(self.sphere_path)
        xform = UsdGeom.Xformable(sphere_prim)
        local_transform = xform.GetLocalTransformation()
        translation = local_transform.ExtractTranslation()
        # Joint position relative to initial
        return translation[2] - self.initial_height
    
    def get_contact_force(self):
        """Get contact force from RigidPrim (like rigid cube example)"""
        try:
            result = self.contact_view.get_contact_force_data(dt=self.time_step)
            
            if result is not None:
                forces, points, normals, distances, counts, starts = result
                
                if forces.shape[0] > 0:
                    # Calculate normal force (force projected onto contact normal)
                    normal_force = np.sum(np.sum(forces * normals, axis=1))
                    return abs(normal_force)
            
            return 0.0
        except Exception as e:
            return 0.0
    
    def print_summary(self, force_log, indent_log, contact_started):
        """Print test summary"""
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        
        if not force_log:
            print("No data collected!")
            return
        
        force_array = np.array(force_log)
        indent_array = np.array(indent_log) * 1000  # mm
        
        if contact_started:
            max_force = np.max(force_array)
            max_indent = np.max(indent_array)
            print(f"Contact detected: YES")
            print(f"Max indentation: {max_indent:.3f} mm")
            print(f"Max force: {max_force:.6f} N")
        else:
            print("Contact detected: NO")
            print("⚠ Sphere may not have reached cube")
        
        print("="*60)


if __name__ == "__main__":
    DeformableCubeTest().run()