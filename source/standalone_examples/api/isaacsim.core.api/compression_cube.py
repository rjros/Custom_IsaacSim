# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""
Hertzian Contact Test - SIMPLIFIED VERSION with ContactSensor

This version avoids potential API issues by:
- Working directly with USD prims (no RigidPrim wrapper)
- Using UsdGeom for transforms
- Simpler, more robust approach

Based on the reference ContactSensor example code.
"""

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import argparse
import numpy as np
import csv
from pathlib import Path
import omni.usd
from pxr import Gf, UsdGeom, PhysxSchema, UsdPhysics, Sdf, UsdShade, UsdLux

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from isaacsim.core.api import World
from isaacsim.core.api.materials.deformable_material import DeformableMaterial
from isaacsim.core.prims import DeformablePrim, SingleDeformablePrim
from isaacsim.sensors.physics import ContactSensor
from omni.physx.scripts import physicsUtils
import isaacsim.core.utils.deformable_mesh_utils as deformableMeshUtils

# Seeds
np.random.seed(42)

parser = argparse.ArgumentParser(description='Hertzian Contact Test - Simplified')
parser.add_argument("--cube_size", type=float, default=0.04, help="Cube size (m)")
parser.add_argument("--sphere_radius", type=float, default=0.005, help="Sphere radius (m)")
parser.add_argument("--youngs_modulus", type=float, default=50000, help="Young's modulus (Pa)")
parser.add_argument("--poisson_ratio", type=float, default=0.48, help="Poisson's ratio")
parser.add_argument("--indentation_depth", type=float, default=0.006, help="Indentation depth (m)")
parser.add_argument("--indentation_speed", type=float, default=0.01, help="Indentation speed (m/s)")
parser.add_argument("--resolution", type=int, default=20, help="Mesh resolution")
parser.add_argument("--solver_iterations", type=int, default=150, help="Solver iterations")
parser.add_argument("--output_dir", type=str, default="./hertzian_results", help="Output directory")
args = parser.parse_args()


class HertzianContactTest:
    
    def __init__(self):
        self.time_step = 0.005
        self.cube_size = args.cube_size
        self.sphere_radius = args.sphere_radius
        self.indentation_depth = args.indentation_depth
        self.indentation_speed = args.indentation_speed
        self.youngs_modulus = args.youngs_modulus
        self.poisson_ratio = args.poisson_ratio
        
        self.approach_distance = self.sphere_radius + 0.002
        self.total_distance = self.approach_distance + self.indentation_depth
        self.test_duration = self.total_distance / self.indentation_speed
        
        print(f"\n{'='*80}")
        print(f"Hertzian Contact Test - Simplified with ContactSensor")
        print(f"{'='*80}")
        print(f"  Sphere: {self.sphere_radius*2000:.1f} mm diameter")
        print(f"  Cube: {self.cube_size*1000:.1f} mm")
        print(f"  E = {self.youngs_modulus/1000:.1f} kPa, ν = {self.poisson_ratio}")
        print(f"  Indentation: {self.indentation_depth*1000:.1f} mm at {self.indentation_speed*1000:.1f} mm/s")
        print(f"  Duration: {self.test_duration:.2f} s")
        print(f"{'='*80}\n")
        
        self.my_world = World(
            stage_units_in_meters=1.0,
            backend="torch",
            device="cuda",
            physics_dt=self.time_step,
            rendering_dt=self.time_step * 4
        )
        self.stage = omni.usd.get_context().get_stage()
        
        self.time_history = []
        self.position_history = []
        self.indentation_history = []
        self.force_history = []
        self.contact_started = False
        self.initial_sphere_z = None
        
        self.my_world.scene.add_default_ground_plane()
        self.setup_environment()
        self.setup_scene()
        self.setup_camera()
        self.calculate_theory()
    
    def calculate_theory(self):
        E_eff = self.youngs_modulus / (1 - self.poisson_ratio**2)
        self.theoretical_forces = []
        self.theoretical_indentations = np.linspace(0, self.indentation_depth, 100)
        
        for delta in self.theoretical_indentations:
            if delta > 0:
                F = (4/3) * E_eff * np.sqrt(self.sphere_radius) * (delta ** 1.5)
                self.theoretical_forces.append(F)
            else:
                self.theoretical_forces.append(0.0)
        
        print(f"Theory: E* = {E_eff/1000:.2f} kPa, F_max = {self.theoretical_forces[-1]:.4f} N\n")
    
    def setup_environment(self):
        dome_light_path = "/World/DomeLight"
        dome_light = UsdLux.DomeLight.Define(self.stage, dome_light_path)
        dome_light.CreateIntensityAttr().Set(1500.0)
    
    def setup_scene(self):
        print("Creating scene...")
        
        # === DEFORMABLE CUBE ===
        # Position cube so bottom is on ground plane (z=0)
        cube_center_z = self.cube_size / 2  # Center is at half height
        cube_path = "/World/deformable_cube"
        cube_mesh = UsdGeom.Mesh.Define(self.stage, cube_path)
        
        tri_points, tri_indices = deformableMeshUtils.createTriangleMeshCube(20)
        cube_mesh.GetPointsAttr().Set(tri_points)
        cube_mesh.GetFaceVertexIndicesAttr().Set(tri_indices)
        cube_mesh.GetFaceVertexCountsAttr().Set([3] * (len(tri_indices) // 3))
        
        physicsUtils.setup_transform_as_scale_orient_translate(cube_mesh)
        physicsUtils.set_or_add_scale_op(cube_mesh, Gf.Vec3f(self.cube_size, self.cube_size, self.cube_size))
        physicsUtils.set_or_add_translate_op(cube_mesh, Gf.Vec3f(0, 0, cube_center_z))
        
        # Material
        mat_path = "/World/cubeMat"
        mat = UsdShade.Material.Define(self.stage, mat_path)
        shader = UsdShade.Shader.Define(self.stage, mat_path + "/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.3, 0.6, 0.9))
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(cube_mesh.GetPrim()).Bind(mat)
        
        # Deformable properties
        def_mat_path = "/World/softMaterial"
        self.deformable_material = DeformableMaterial(
            prim_path=def_mat_path,
            dynamic_friction=0.5,
            youngs_modulus=self.youngs_modulus,
            poissons_ratio=self.poisson_ratio,
            elasticity_damping=0.01,
        )
        
        self.deformable_cube = SingleDeformablePrim(
            name="softCube",
            prim_path=cube_path,
            deformable_material=self.deformable_material,
            vertex_velocity_damping=0.01,
            solver_position_iteration_count=args.solver_iterations,
            simulation_hexahedral_resolution=args.resolution,
        )
        self.my_world.scene.add(self.deformable_cube)
        
        # === CUBE SITS ON GROUND PLANE ===
        # No explicit attachment needed - cube will rest on ground via collision
        # Increase friction to prevent sliding
        
        # The ground plane already exists from my_world.scene.add_default_ground_plane()
        # We just need to ensure the cube has good friction with it
        
        # === RIGID SPHERE ===
        self.initial_sphere_z = self.cube_size + self.sphere_radius + 0.002
        self.sphere_path = "/World/rigid_sphere"
        
        sphere_geom = UsdGeom.Sphere.Define(self.stage, self.sphere_path)
        sphere_geom.CreateRadiusAttr().Set(self.sphere_radius)
        
        xform = UsdGeom.Xformable(sphere_geom)
        xform.ClearXformOpOrder()
        self.sphere_translate_op = xform.AddTranslateOp()
        self.sphere_translate_op.Set(Gf.Vec3f(0, 0, self.initial_sphere_z))
        
        sphere_prim = self.stage.GetPrimAtPath(self.sphere_path)
        
        # Physics
        rigid_api = UsdPhysics.RigidBodyAPI.Apply(sphere_prim)
        rigid_api.CreateKinematicEnabledAttr().Set(True)
        
        UsdPhysics.CollisionAPI.Apply(sphere_prim)
        
        # CRITICAL: Contact report API
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
        
        # === CONTACT SENSOR ===
        self.contact_sensor = ContactSensor(
            prim_path=self.sphere_path + "/ContactSensor",
            name="SphereContactSensor",
            frequency=60,
            translation=np.array([0, 0, 0]),
            min_threshold=0,
            max_threshold=10000000,
            radius=self.sphere_radius * 1.5
        )
        
        print("✓ Scene created")
        print(f"  Cube rests on ground plane (bottom at z=0, top at z={self.cube_size*1000:.1f}mm)")
        print(f"  High friction prevents sliding during indentation")
        print()
        self.my_world.reset(soft=False)
    
    def setup_camera(self):
        from omni.isaac.core.utils.viewports import set_camera_view
        set_camera_view(
            eye=Gf.Vec3d(0.08, 0.08, 0.06),
            target=Gf.Vec3d(0, 0, self.cube_size/2),
            camera_prim_path="/OmniverseKit_Persp"
        )
    
    def get_force_from_sensor(self):
        try:
            data = self.contact_sensor.get_current_frame()
            if data and isinstance(data, dict) and data.get('in_contact', False):
                val = data.get('value', 0.0)
                if isinstance(val, (list, np.ndarray)):
                    return np.linalg.norm(val), data
                return float(val), data
        except:
            pass
        return 0.0, None
    
    def update_position(self, t):
        if t > self.test_duration:
            return 0.0
        
        z = self.initial_sphere_z - (self.indentation_speed * t)
        self.sphere_translate_op.Set(Gf.Vec3f(0, 0, z))
        
        if not self.contact_started and z <= (self.cube_size + self.sphere_radius):
            self.contact_started = True
            print(f"✓ Contact at t={t:.3f}s, z={z*1000:.2f}mm")
        
        if self.contact_started:
            return max(0, (self.cube_size + self.sphere_radius) - z)
        return 0.0
    
    def record(self, t):
        indent = self.update_position(t)
        pos = self.sphere_translate_op.Get()
        force, _ = self.get_force_from_sensor()
        
        self.time_history.append(t)
        self.position_history.append(pos[2])
        self.indentation_history.append(indent)
        self.force_history.append(force)
    
    def save_results(self):
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        
        with open(out / "data.csv", 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['Time_s', 'Z_m', 'Indent_mm', 'Force_N'])
            for i in range(len(self.time_history)):
                w.writerow([self.time_history[i], self.position_history[i],
                           self.indentation_history[i]*1000, self.force_history[i]])
        
        print(f"✓ Saved: {out}/data.csv")
        
        if MATPLOTLIB_AVAILABLE:
            self.plot()
    
    def plot(self):
        ind = np.array(self.indentation_history) * 1000
        force = np.array(self.force_history)
        time = np.array(self.time_history)
        
        fig, ax = plt.subplots(2, 2, figsize=(12, 9))
        fig.suptitle('Hertzian Contact Test', fontsize=14, fontweight='bold')
        
        ax[0,0].plot(ind, force, 'b-', lw=2, label='Sensor')
        ax[0,0].plot(self.theoretical_indentations*1000, self.theoretical_forces, 'r--', lw=2, label='Theory')
        ax[0,0].set_xlabel('Indentation (mm)')
        ax[0,0].set_ylabel('Force (N)')
        ax[0,0].legend()
        ax[0,0].grid(True, alpha=0.3)
        
        ax[0,1].plot(time, force, 'g-', lw=2)
        ax[0,1].set_xlabel('Time (s)')
        ax[0,1].set_ylabel('Force (N)')
        ax[0,1].grid(True, alpha=0.3)
        
        ax[1,0].plot(time, ind, 'orange', lw=2)
        ax[1,0].set_xlabel('Time (s)')
        ax[1,0].set_ylabel('Indentation (mm)')
        ax[1,0].grid(True, alpha=0.3)
        
        max_f = max(force) if len(force) > 0 else 0
        max_i = max(ind) if len(ind) > 0 else 0
        theory_f = self.theoretical_forces[-1]
        err = abs(max_f - theory_f) / theory_f * 100 if theory_f > 0 else 0
        
        txt = f"""RESULTS

Indentation: {max_i:.3f} mm
Force (meas): {max_f:.4f} N
Force (theory): {theory_f:.4f} N
Error: {err:.1f}%
"""
        ax[1,1].text(0.2, 0.5, txt, transform=ax[1,1].transAxes,
                    fontfamily='monospace', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        ax[1,1].axis('off')
        
        plt.tight_layout()
        plt.savefig(Path(args.output_dir) / "results.png", dpi=150)
        print(f"✓ Saved: {Path(args.output_dir)}/results.png")
    
    def run(self):
        print("Starting test...\n")
        print("Allowing cube to settle on ground plane (2 seconds)...")
        
        step = 0
        settling_time = 2.0  # Give cube more time to settle without attachment
        test_started = False
        start_time_offset = 0
        
        while simulation_app.is_running():
            if self.my_world.is_playing():
                t = self.my_world.current_time
                
                # Settling phase - don't move sphere yet
                if t < settling_time:
                    if step % 50 == 0 and step > 0:
                        print(f"  Settling... {t:.1f}s / {settling_time:.1f}s")
                    self.my_world.step(render=True)
                    step += 1
                    continue
                
                # Start test after settling
                if not test_started:
                    test_started = True
                    start_time_offset = t
                    print(f"\n✓ Cube settled. Starting indentation test at t={t:.2f}s\n")
                
                # Adjust time for test (subtract settling time)
                test_time = t - start_time_offset
                
                if step > 5:
                    self.record(test_time)
                    
                    if len(self.time_history) % 20 == 0:
                        i = self.indentation_history[-1]*1000 if self.indentation_history else 0
                        f = self.force_history[-1] if self.force_history else 0
                        print(f"t={test_time:.2f}s | Indent={i:.3f}mm | Force={f:.4f}N")
                
                if test_time > self.test_duration + 0.5:
                    print(f"\n✓ Complete at test_time={test_time:.2f}s")
                    self.save_results()
                    print("\nClose window to exit.")
                    while simulation_app.is_running():
                        self.my_world.step(render=True)
                    break
            
            self.my_world.step(render=True)
            step += 1
        
        simulation_app.close()


if __name__ == "__main__":
    HertzianContactTest().run()