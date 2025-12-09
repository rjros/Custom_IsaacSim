# SPDX-License-Identifier: Apache-2.0

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import numpy as np
import matplotlib.pyplot as plt

from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
from isaacsim.core.prims import RigidPrim


class DropCompressionTestRigid:
    def __init__(self):
        self.world = World(stage_units_in_meters=1.0)
        self.stage = self.world.stage
        self.world.scene.add_default_ground_plane()

        self.force_log = []
        self.height_log = []  # track cube Z position over time

        self._create_fixed_block()
        self._create_top_cube()
        self._create_rigid_view()

        self.world.reset()
        print("Drop compression test initialized — cube will fall onto block")

    # ------------------------------------------------------------------
    def _create_fixed_block(self):
        """Create bottom rigid block"""
        self.block = self.world.scene.add(
            FixedCuboid(
                prim_path="/FixedBlock",
                name="fixed_block",
                position=np.array([0, 0, 0.25]),
                size=0.5,
                color=np.array([255, 0, 0]),
            )
        )

    # ------------------------------------------------------------------
    def _create_top_cube(self):
        """Drop cube from above"""
        self.top = self.world.scene.add(
            DynamicCuboid(
                prim_path="/TopCube",
                name="top_cube",
                position=np.array([0, 0, 0.9]),  # 2 meters above block
                size=0.5,
                color=np.array([255, 255, 0]),
                mass=1.0
            )
        )

    # ------------------------------------------------------------------
    def _create_rigid_view(self):
        """Capture ONLY contact forces with FixedBlock"""
        self.contact_view = RigidPrim(
            prim_paths_expr="/TopCube",
            contact_filter_prim_paths_expr=["/FixedBlock"],
            name="contact_view",
            max_contact_count=32,
        )
        self.world.scene.add(self.contact_view)

    # ------------------------------------------------------------------
    def play(self):
        """Drop → collide → log forces"""
        print("Simulation running... Cube will drop.")

        while simulation_app.is_running():
            if self.world.is_playing():
                # --- Read Z position of the cube for reference ---
                z = self.top.get_world_pose()[0][2]
                self.height_log.append(z)

                # --- Contact forces extraction ---
                result = self.contact_view.get_contact_force_data(dt=1/60)
                if result is not None:
                    forces, points, normals, distances, counts, starts = result
                    if forces.shape[0] > 0:
                        normal_force = np.sum(np.sum(forces * normals, axis=1))
                    else:
                        normal_force = 0.0
                else:
                    normal_force = 0.0

                self.force_log.append(normal_force)

            self.world.step(render=True)

        self._plot_results()
        simulation_app.close()

    # ------------------------------------------------------------------
    def _plot_results(self):
        """Save force–time plot"""
        force = np.array(self.force_log)
        time = np.arange(force.size) / 60.0  # assuming 60 Hz simulation

        plt.figure(figsize=(7, 4))
        plt.plot(time, force, linewidth=2)
        plt.xlabel("Time (s)")
        plt.ylabel("Normal Contact Force (N)")
        plt.ylim(0,10)
        plt.yticks(np.linspace(0, 10, 51))

        plt.title("Drop Test — Contact Force vs Time")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("drop_force_plot.png", dpi=300)

        print("📁 Plot saved as drop_force_plot.png")
        print("Done.")


if __name__ == "__main__":
    DropCompressionTestRigid().play()
