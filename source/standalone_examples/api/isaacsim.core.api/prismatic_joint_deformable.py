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
import omni.usd

np.random.seed(42)


class DeformableCubeTest:
    
    def __init__(self):
        self.time_step = 1/60.0
        self.cube_size = 0.05  # 5cm cube (same as Franka demo)
        
        print(f"\n{'='*80}")
        print(f"Deformable Cube Test - Based on Franka Demo")
        print(f"{'='*80}")
        print(f"  Cube: {self.cube_size*1000:.1f}mm")
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
        
        # Don't use default ground plane - create custom one like Franka demo
        # self.setup_physics_scene()
        self.world.scene.add_default_ground_plane()
        self.setup_environment()
        self.setup_scene()
        self.setup_camera()
    
    def setup_physics_scene(self):
        """Create physics scene with Franka demo settings"""
        from omni.physx.scripts import utils
        
        # Physics scene (like Franka demo)
        scene_path = "/World/physicsScene"
        scene = UsdPhysics.Scene.Define(self.stage, scene_path)
        scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
        scene.CreateGravityMagnitudeAttr().Set(9.807)
        
        physxSceneAPI = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
        
        # Critical settings from Franka demo
        physxSceneAPI.CreateFrictionOffsetThresholdAttr().Set(0.001)
        physxSceneAPI.CreateFrictionCorrelationDistanceAttr().Set(0.0005)
        
        # Iteration counts
        self.pos_iterations = 20
        self.vel_iterations = 1
        physxSceneAPI.GetMaxPositionIterationCountAttr().Set(self.pos_iterations)
        physxSceneAPI.GetMaxVelocityIterationCountAttr().Set(self.vel_iterations)
        
        UsdGeom.SetStageUpAxis(self.stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(self.stage, 1.0)
        
        # Ground plane (like Franka demo)
        utils.addPlaneCollider(self.stage, "/World/groundPlane", "Z")
        
        # Add visual grid ground plane
        self.create_visual_ground()
        
        print("✓ Physics scene created (Franka demo settings)")
    
    def setup_environment(self):
        dome_light_path = "/World/DomeLight"
        dome_light = UsdLux.DomeLight.Define(self.stage, dome_light_path)
        dome_light.CreateIntensityAttr().Set(1500.0)
    
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
        from pxr import UsdGeom
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
    
    def setup_scene(self):
        print("Creating deformable cube...")
        
        # === DEFORMABLE MATERIAL (exactly like Franka demo red cube) ===
        deformable_material_path = "/World/DeformableBodyMaterial"
        
        deformableUtils.add_deformable_body_material(
            self.stage,
            deformable_material_path,
            youngs_modulus=10000000.0,  # 10 MPa (Franka demo value)
            poissons_ratio=0.499,  # Nearly incompressible
            damping_scale=0.0,
            elasticity_damping=0.0001,  # Very small damping
            dynamic_friction=1.0,  # High friction
            density=100,  # 100 kg/m^3
        )
        
        print(f"✓ Deformable material created")
        print(f"  Young's modulus: 10 MPa")
        print(f"  Poisson's ratio: 0.499")
        print(f"  Density: 100 kg/m³")
        
        # === DEFORMABLE CUBE ===
        # Create cube mesh path (we'll use a simple cube mesh)
        cube_path = "/World/deformable_cube"
        
        # Create cube geometry
        cube_mesh = UsdGeom.Cube.Define(self.stage, cube_path)
        cube_mesh.CreateSizeAttr().Set(1.0)  # Unit cube
        
        # Position: start on ground
        cube_center_z = self.cube_size / 2
        cube_mesh.AddTranslateOp().Set(Gf.Vec3f(0.0, 0.0, cube_center_z))
        cube_mesh.AddOrientOp().Set(Gf.Quatf(1.0))
        cube_mesh.AddScaleOp().Set(Gf.Vec3f(self.cube_size, self.cube_size, self.cube_size))
        
        # Add deformable body (like Franka demo)
        deformableUtils.add_physx_deformable_body(
            self.stage,
            cube_path,
            simulation_hexahedral_resolution=3,  # Franka demo value
            collision_simplification=True,  # Franka demo value
            self_collision=False,  # Franka demo value
            solver_position_iteration_count=self.pos_iterations,  # Match scene iterations
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
        print(f"  Solver iterations: {self.pos_iterations}")
        print(f"  Collision: contact offset=0.02, rest offset=0.001")
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
        print("Cube should settle and remain stable (Franka demo settings).\n")
        print(f"{'Time (s)':<10} {'Status':<40}")
        print("-" * 60)
        
        step = 0
        last_print_time = 0
        print_interval = 1.0
        
        while simulation_app.is_running():
            if self.my_world.is_playing():
                t = self.my_world.current_time
                
                # Step simulation
                self.my_world.step(render=True)
                
                # Print status
                if step > 10 and t - last_print_time >= print_interval:
                    if t < 3:
                        status = "Settling..."
                    elif t < 5:
                        status = "Should be stable now"
                    else:
                        status = "Monitoring stability"
                    
                    print(f"{t:<10.1f} {status:<40}")
                    last_print_time = t
                
                # Run for 10 seconds
                if t > 10.0:
                    print(f"\n✓ Test complete at t={t:.2f}s")
                    print("\nRESULTS:")
                    print("  If cube is:")
                    print("    ✓ Sitting still on ground → Settings are good!")
                    print("    ✗ Floating/vibrating → Need to adjust settings")
                    print("\nClose window to exit.")
                    
                    while simulation_app.is_running():
                        self.my_world.step(render=True)
                    break
            else:
                self.my_world.step(render=True)
            
            step += 1
        
        simulation_app.close()


if __name__ == "__main__":
    DeformableCubeTest().run()