# SPDX-License-Identifier: Apache-2.0
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import omni.ui as ui
import carb

# ------------------------------------------------------------
# Button callbacks
# ------------------------------------------------------------
def on_start_clicked():
    carb.log_info("▶️ Start button pressed!")

def on_stop_clicked():
    carb.log_info("⏹ Stop button pressed!")

# ------------------------------------------------------------
# Build a simple UI window
# ------------------------------------------------------------
def build_ui():
    with ui.Window("UI Test Window", width=250, height=120):
        with ui.VStack(spacing=10, height=0):
            ui.Label("Button Test Panel", alignment=ui.Alignment.CENTER)

            ui.Button("▶️ START", clicked_fn=on_start_clicked)
            ui.Button("⏹ STOP", clicked_fn=on_stop_clicked)

build_ui()

# ------------------------------------------------------------
# Basic Isaac Sim loop
# ------------------------------------------------------------
while simulation_app.is_running():
    # Keep UI responsive
    pass

simulation_app.close()
