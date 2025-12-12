from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

from isaacsim.core.api import World
import numpy as np
from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
from isaacsim.core.api.physics_context import PhysicsContext
PhysicsContext()

## Sensor libraries
from isaacsim.sensors.physics import ContactSensor
from pxr import Gf

# Creating a contact sensor can only be done on prim with a collider ap, and it depends on
# the Contact Report API.

import omni
from pxr import PhysxSchema


my_world = World(stage_units_in_meters=1.0)

# GroundPlane(prim_path="/World/groundPlane", size=10, color=np.array([0.5, 0.5, 0.5]))
cube=my_world.scene.add(
    DynamicCuboid(prim_path="/World/Cube",
    name="Cube1",
    position=np.array([-.5, -.2, 10.0]),
    scale=np.array([.5, .5, .5]),
    color=np.array([.2,.3,0.]),
    mass=2.0
    )
)

## For fixed cuboid contact sensor is not giving any information
cube=my_world.scene.add(
    FixedCuboid(prim_path="/World/Cube0",
    name="Cube0",
    position=np.array([-.5, -.2, 0.25]),
    scale=np.array([2.0, 2.0, 0.5]),
    color=np.array([.2,.3,0.])
    )
)


cube2=my_world.scene.add(
    DynamicCuboid(prim_path="/World/Cube2",
    name="Cube2",
    position=np.array([-.5, -.2, 1.0]),
    scale=np.array([.5, .5, .5]),
    color=np.array([.2,.3,0.]),
    mass=2.0
    )
)

sensor = ContactSensor(
    prim_path="/World/Cube0/Contact_Sensor",
    name="Contact_Sensor",
    frequency=60,
    translation=np.array([0, 0, 0.0]),
    min_threshold=0,
    max_threshold=10000000,
    radius=1
)

stage = omni.usd.get_context().get_stage()
parent_prim = stage.GetPrimAtPath("/World/Cube0")
contact_report = PhysxSchema.PhysxContactReportAPI.Apply(parent_prim)

# Set a minimum threshold for the contact report to zero
contact_report.CreateThresholdAttr(0.0)






my_world.scene.add_default_ground_plane()


for i in range(5):
    for i in range(500):
        my_world.step(render=True)
        value = sensor.get_current_frame()
        print(value)
        # print(cube.get_world_pose())

simulation_app.close()


