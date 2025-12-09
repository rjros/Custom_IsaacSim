# SPDX-License-Identifier: Apache-2.0

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import numpy as np
import matplotlib.pyplot as plt

from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
from isaacsim.core.prims import RigidPrim
from pxr import UsdGeom, UsdPhysics

simulation_app.update_app_window()
simulation_app.set_setting("/app/asyncRendering", True)
simulation_app.set_setting("/physics/autoStart", False)   # ⛔ do NOT start physics automatically
simulation_app.set_setting("/app/runSim", False)          # ensure simulation is paused at launch



class CompressionTestRigid:
    def __init__(self):
        self.world = World(stage_units_in_meters=1.0)
        self.stage = self.world.stage
        self.world.scene.add_default_ground_plane()

        self._create_fixed_block()
        self._create_dynamic_block()
        self._create_prismatic_joint()
        self._create_rigid_view()

        self.force_log = []
        self.disp_log = []
        self.world.reset()

        print("Rigid compression test initialized.")

    # ------------------------------------------------------------------
    def _create_fixed_block(self):
        """Bottom rigid block, reference surface"""
        self.block = self.world.scene.add(
            FixedCuboid(
                prim_path="/FixedBlock",
                name="fixed_block",
                position=np.array([0, 0, 0.25]),  # height = 0.5 → center at 0.25
                size=0.5,
                color=np.array([255, 0, 0]),
            )
        )

    # ------------------------------------------------------------------
    def _create_dynamic_block(self):
        """Top cube to compress the block"""
        self.top = self.world.scene.add(
            DynamicCuboid(
                prim_path="/DynamicTop",
                name="dynamic_top",
                position=np.array([0, 0, 1.0]),
                scale=np.array([0.6, 0.6, 0.3]),
                size=1.0,
                color=np.array([255, 255, 0]),
            )
        )

    # ------------------------------------------------------------------
    def _create_prismatic_joint(self):
        """Joint that moves DynamicTop along Z"""
        joint = UsdPhysics.PrismaticJoint.Define(self.stage, "/World/JointZ")
        joint.CreateAxisAttr("Z")
        joint.CreateLowerLimitAttr(0.0)
        joint.CreateUpperLimitAttr(1.2)

        # Static reference
        xform = UsdGeom.Xform.Define(self.stage, "/World/Ref")
        joint.CreateBody0Rel().SetTargets(["/World/Ref"])

        # Body under displacement control
        joint.CreateBody1Rel().SetTargets(["/DynamicTop"])

        self.drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
        self.drive.CreateTypeAttr().Set("position")
        self.drive.CreateStiffnessAttr().Set(40000)
        self.drive.CreateDampingAttr().Set(2000)

    # ------------------------------------------------------------------
    def _create_rigid_view(self):
        """Attach a view to capture contact forces"""
        self.contact_view = RigidPrim(
            prim_paths_expr="/DynamicTop",                   # object of interest
            contact_filter_prim_paths_expr=["/FixedBlock"],  # MUST specify!
            name="contact_view",
            max_contact_count=32,
        )
        self.world.scene.add(self.contact_view)

    # ------------------------------------------------------------------
    def play(self):
        """Main loop — compress, record, plot"""
        target = 1.0
        direction = -1

        while simulation_app.is_running():
            if self.world.is_playing():

                target += 0.001 * direction

                if target <= 0.35:  # start contact and compress
                    direction = +1
                elif target >= 1.0:
                    direction = -1

                self.drive.GetTargetPositionAttr().Set(target)

                # --- force extraction ---
                result = self.contact_view.get_contact_force_data(dt=1/60)
                if result is not None:
                    forces, points, normals, distances, counts, starts = result
                    if forces.shape[0] > 0:
                        # Project impulse along normals for normal force
                        normal_force = np.sum(np.sum(forces * normals, axis=1))
                    else:
                        normal_force = 0
                else:
                    normal_force = 0

                self.force_log.append(normal_force)
                self.disp_log.append(target)

            self.world.step(render=True)

        self._plot()
        simulation_app.
        simulation_app.close()

    # ------------------------------------------------------------------
    def _plot(self):
        """Plot after simulation window closes"""
        disp = np.array(self.disp_log)
        force = np.array(self.force_log)

        plt.figure(figsize=(7, 4))
        plt.plot(disp, force, linewidth=2)
        plt.xlabel("Displacement (m)")
        plt.ylabel("Contact Force (N)")
        plt.title("Rigid Compression Test — Force vs Displacement")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("force_displacement2.png", dpi=300)
        print("Plot saved to force_displacement.png")

if __name__ == "__main__":
        CompressionTestRigid().play()
