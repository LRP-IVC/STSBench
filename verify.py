import argparse
from pathlib import Path
import uuid
import time

from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy import select
import rerun as rr
import rerun.blueprint as rrb
from nuscenes import nuscenes
import numpy as np
import matplotlib
from prompt_toolkit.shortcuts import message_dialog
# from prompt_toolkit.formatted_text import HTML

from prompt_toolkit.application import Application, get_app
from prompt_toolkit.formatted_text import HTML, merge_formatted_text
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style
from prompt_toolkit.shortcuts import ProgressBar
from prompt_toolkit.widgets import (
    Box,
    Button,
    Dialog,
    Label,
    CheckboxList,
    HorizontalLine,
)
from prompt_toolkit.layout import (
    HSplit,
    Layout,
    VSplit,
)

from annotator.data.models import SensorType
from annotator.data.maneuvers import (
    PositiveManeuver,
    NegativeManeuver,
    Maneuver,
    ManeuverType,
)
from annotator.rerun.export_gps import derive_latlon
from annotator.mining.consts import (
    NEGATIVE_MANEUVERS,
    MANEUVER_DESCRIPTION,
    PED_NEGATIVE_MANEUVERS,
)


def show_scene_rr(maneuver, args):
    cmap = matplotlib.colormaps["turbo_r"]
    norm = matplotlib.colors.Normalize(
        vmin=3.0,
        vmax=75.0,
    )
    rr.script_setup(args, "stsbench_annotator", recording_id=uuid.uuid4())

    rr.log(
        "ego_vel",
        rr.SeriesLine(color=0xFF0000FF, width=3.0, name="Ego Velocity km/h"),
        static=True,
    )

    rr.log(
        "agnet_a_vel",
        rr.SeriesLine(color=0x1B6097FF, width=3.0, name="Agent A Velocity km/h"),
        static=True,
    )

    rr.log(
        "agnet_b_vel",
        rr.SeriesLine(color=0x318F2DFF, width=3.0, name="Agent B Velocity km/h"),
        static=True,
    )

    ego_trajectory_lat_lon = []
    agent_a_trajectory_lat_lon = []
    agent_b_trajectory_lat_lon = []
    for i, frame in enumerate(maneuver.frames):
        # log sensor calibration
        for sensor in SensorType:
            sensor = frame.get_sensor(sensor)
            rr.log(
                f"world/ego_vehicle/{sensor.type.name}",
                rr.Transform3D(
                    translation=sensor.xyz,
                    rotation=rr.Quaternion(xyzw=sensor.qxyzw),
                    from_parent=False,
                ),
                static=True,
            )

            if sensor.is_camera:
                rr.log(
                    f"world/ego_vehicle/{sensor.type.name}",
                    rr.Pinhole(
                        focal_length=sensor.focal_length,
                        principal_point=sensor.principal_point,
                        width=sensor.width,
                        height=sensor.height,
                    ),
                    static=True,
                )

        ego = frame.ego
        rr.set_time_seconds("timestamp", frame.timestamp * 1e-6)

        # velocity
        if maneuver.is_ego:
            rr.log("ego_vel", rr.Scalar(ego.velocity_kmh))
        if maneuver.is_agent:
            rr.log("agnet_a_vel", rr.Scalar(maneuver.agents[i].velocity_kmh))
        if maneuver.is_other_agent:
            rr.log("agnet_b_vel", rr.Scalar(maneuver.other_agents[i].velocity_kmh))

        rr.log(
            "world/ego_vehicle",
            rr.Transform3D(
                translation=ego.xyz,
                rotation=rr.Quaternion(xyzw=ego.qxyzw),
                axis_length=10.0,
                from_parent=False,
            ),
        )

        if maneuver.is_ego:
            position_lat_lon = derive_latlon(
                frame.scene.location, {"translation": ego.xyz}
            )
            ego_trajectory_lat_lon.append(position_lat_lon)

            rr.log(
                "world/ego_vehicle",
                rr.GeoPoints(
                    lat_lon=position_lat_lon,
                    radii=rr.Radius.ui_points(8.0),
                    colors=0xFF0000FF,
                ),
            )

            rr.log(
                "world/ego_vehicle/trajectory",
                rr.GeoLineStrings(
                    lat_lon=ego_trajectory_lat_lon,
                    radii=rr.Radius.ui_points(1.0),
                    colors=0xFF0000FF,
                ),
            )

        if maneuver.is_agent:
            agent_a_position_lat_lon = derive_latlon(
                frame.scene.location, {"translation": maneuver.agents[i].xyz}
            )
            agent_a_trajectory_lat_lon.append(agent_a_position_lat_lon)

            rr.log(
                "world/agent_a/position",
                rr.GeoPoints(
                    lat_lon=agent_a_position_lat_lon,
                    radii=rr.Radius.ui_points(8.0),
                    colors=0x1B6097FF,
                ),
            )
            rr.log(
                "world/agent_a",
                rr.Boxes3D(
                    sizes=maneuver.agents[i].lwh,
                    centers=maneuver.agents[i].xyz,
                    quaternions=rr.Quaternion(xyzw=maneuver.agents[i].qxyzw),
                    colors=0x1B6097FF,
                    radii=rr.Radius.ui_points(1.5),
                ),
            )
            rr.log(
                "world/agent_a/trajectory",
                rr.GeoLineStrings(
                    lat_lon=agent_a_trajectory_lat_lon,
                    radii=rr.Radius.ui_points(1.0),
                    colors=0x1B6097FF,
                ),
            )

        if maneuver.is_other_agent:
            agent_b_position_lat_lon = derive_latlon(
                frame.scene.location, {"translation": maneuver.other_agents[i].xyz}
            )
            agent_b_trajectory_lat_lon.append(agent_b_position_lat_lon)

            rr.log(
                "world/agent_b/position",
                rr.GeoPoints(
                    lat_lon=agent_b_position_lat_lon,
                    radii=rr.Radius.ui_points(8.0),
                    colors=0x318F2DFF,
                ),
            )
            rr.log(
                "world/agent_b",
                rr.Boxes3D(
                    sizes=maneuver.other_agents[i].lwh,
                    centers=maneuver.other_agents[i].xyz,
                    quaternions=rr.Quaternion(xyzw=maneuver.other_agents[i].qxyzw),
                    colors=0x318F2DFF,
                    radii=rr.Radius.ui_points(1.5),
                ),
            )
            rr.log(
                "world/agent_b/trajectory",
                rr.GeoLineStrings(
                    lat_lon=agent_b_trajectory_lat_lon,
                    radii=rr.Radius.ui_points(1.0),
                    colors=0x318F2DFF,
                ),
            )

        # log all sensors
        lidar = frame.get_sensor(SensorType.LIDAR_TOP)
        pointcloud = nuscenes.LidarPointCloud.from_file(str(args.dataroot / lidar.path))
        points = pointcloud.points[:3].T
        point_distances = np.linalg.norm(points, axis=1)
        point_colors = cmap(norm(point_distances))
        rr.log(
            f"world/ego_vehicle/{SensorType.LIDAR_TOP.name}",
            rr.Points3D(points, colors=point_colors),
        )

        for sensor_type in SensorType:
            if "CAM" not in sensor_type.name:
                continue
            cam = frame.get_sensor(sensor_type)
            rr.log(
                f"world/ego_vehicle/{sensor_type.name}",
                rr.EncodedImage(path=args.dataroot / cam.path),
            )
        rr.script_teardown(args)


def show_annotation_console(maneuvers, index, session, args):
    maneuver = maneuvers[index]

    start_time = time.time()

    # build pos checkbox list
    checkbox_list_pos = CheckboxList(
        values=[
            (
                m,
                HTML(
                    f"<green>{m.name}</green>:{MANEUVER_DESCRIPTION[m].format(traffic_participant=maneuver.actor.capitalize()).split(':')[1]}"
                ),
            )
            for m in maneuver.prelabeled_pos_maneuvers
        ],
        default_values=[m.type for m in maneuver.pos_maneuvers],
    )

    checkbox_list_neg = CheckboxList(
        values=[
            (
                neg,
                HTML(
                    f"<red>{neg.name}</red>:{MANEUVER_DESCRIPTION[neg].format(traffic_participant=maneuver.actor.capitalize()).split(':')[1]}"
                ),
            )
            for neg in maneuver.prelabeled_neg_maneuvers
        ],
        default_values=[nm.type for nm in maneuver.neg_maneuvers],
    )

    # helper function to create title
    def create_title(text: str, dont_extend_width: bool = False) -> Label:
        return Label(
            text,
            style="fg:ansiblue",
            dont_extend_width=dont_extend_width,
        )

    # callback functions
    def exit_clicked():
        get_app().exit(result={"exit": True})

    def back_clicked():
        get_app().exit(result={"backward": True})

    def next_clicked():
        # get_app().exit()
        get_app().exit(result={"forward": True})

    def commit_clicked():
        selected_items_pos = checkbox_list_pos.current_values
        selected_items_neg = checkbox_list_neg.current_values
        elapsed_time_ms = (time.time() - start_time) * 1000.0
        get_app().exit(
            result={
                "pos": selected_items_pos,
                "neg": selected_items_neg,
                "elapsed_time_ms": int(elapsed_time_ms),
            }
        )

    # create the dialog body (layout)
    dialog_body = HSplit([
        Box(
            HSplit([
                create_title("Positives:"),
                Box(checkbox_list_pos, padding_top=1, padding_bottom=1),
                HorizontalLine(),
                create_title(
                    "Negatives (Do not reflect entries in DB, but possible choices for positives from DB!):"
                ),
                Box(
                    checkbox_list_neg,
                    padding_top=1,
                    padding_bottom=1,
                    padding_left=0,
                    padding_right=0,
                ),
                HorizontalLine(),
            ]),
        ),
        Box(
            VSplit([
                Box(Button("Commit (c)", handler=commit_clicked)),
                Box(Button("Back (b)", handler=back_clicked)),
                Box(Button("Next (n)", handler=next_clicked)),
                Box(Button("Exit (q)", handler=exit_clicked)),
                Box(
                    create_title(
                        f"[{index + 1}/{len(maneuvers)}] #{maneuvers[index].id:04d}"
                    )
                ),
            ])
        ),
    ])

    # create the root container
    root_container = Dialog(
        title="STSBench3D Annotator",
        with_background=True,
        body=Box(
            dialog_body,
            padding=0,
            padding_left=1,
            padding_right=1,
        ),
    )

    # setup the style
    style = Style.from_dict({
        "dialog.body select-box": "bg:#cccccc",
        "dialog.body select-box": "bg:#cccccc",
        "dialog.body select-box cursor-line": "nounderline bg:ansired fg:",
        "dialog.body select-box last-line": "underline",
        "dialog.body text-area": "bg:#4444ff fg:white",
        "dialog.body text-area": "bg:#4444ff fg:white",
        "dialog.body radio-list radio": "bg:#4444ff fg:white",
        "dialog.body checkbox-list checkbox": "bg:ansiwhite fg:black",
    })

    # define key bindings
    kb = KeyBindings()

    @kb.add("q")
    def _exit(event):
        exit_clicked()

    @kb.add("n")
    def _forward(event):
        next_clicked()

    @kb.add("b")
    def _backward(event):
        back_clicked()

    @kb.add("c")
    def _continue(event):
        commit_clicked()

    # build application and run
    application: Application[None] = Application(
        layout=Layout(root_container),
        full_screen=True,
        style=style,
        key_bindings=kb,
        mouse_support=True,
    )
    selected_items = application.run()
    # print(selected_items)

    # handle results
    # step forward (next)
    if "forward" in selected_items:
        # forward to next maneuver
        index += 1
        if index < len(maneuvers):
            annotate(session, maneuvers, args, index)
        else:
            print("No more maneuvers to annotate.")
    # step backward (back)
    elif "backward" in selected_items:
        # backward to previous maneuver
        index -= 1
        if index >= 0:
            annotate(session, maneuvers, args, index)
        else:
            print("No more maneuvers to annotate.")
    # commit changes
    elif "pos" in selected_items and "neg" in selected_items:
        # warning for incorrect selection of positives
        if len(selected_items["pos"]) > 1:
            message_dialog(
                title="Error dialog window",
                text="Please select at most one maneuver at a time. Multiple selections are not supported.",
            ).run()
        else:
            index += 1

            maneuver.pos_maneuvers.clear()
            maneuver.pos_maneuvers.extend([
                PositiveManeuver(type=p) for p in selected_items["pos"]
            ])

            maneuver.neg_maneuvers.clear()
            maneuver.neg_maneuvers.extend([
                NegativeManeuver(type=p) for p in selected_items["neg"]
            ])

            # set manual labeling flag and commit
            maneuver.manually_labeled = True
            maneuver.labeling_time = selected_items.get("elapsed_time_ms", -1)
            session.commit()

        if index < len(maneuvers):
            annotate(session, maneuvers, args, index)
        else:
            print("No more maneuvers to annotate.")
    elif "exit" in selected_items:
        # exit the application
        print("Exiting the application.")
    else:
        raise NotImplementedError


def annotate(session, maneuvers, args, index=0):
    show_scene_rr(maneuvers[index], args)
    show_annotation_console(maneuvers, index, session, args)


def init_viewer(args) -> None:
    blueprint = rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(
                rrb.Spatial3DView(
                    name="3D",
                    origin="world",
                    defaults=[rr.components.ImagePlaneDistance(5.0)],
                    overrides={
                        "world/agent_a": [rr.components.FillModeBatch("solid")],
                        "world/agent_b": [rr.components.FillModeBatch("solid")],
                    },
                ),
                rrb.MapView(
                    origin="world",
                    name="MapView",
                    zoom=rrb.archetypes.MapZoom(19.0),
                    background=rrb.archetypes.MapBackground(
                        rrb.components.MapProvider.OpenStreetMap
                    ),
                ),
                rrb.TimeSeriesView(name="velocity", origin="/"),
                column_shares=[1.5, 1, 1],
            ),
            rrb.Horizontal(
                rrb.Vertical(
                    rrb.Horizontal(
                        rrb.Spatial2DView(
                            name=SensorType.CAM_FRONT_LEFT.name,
                            origin=f"world/ego_vehicle/{SensorType.CAM_FRONT_LEFT.name}",
                            contents=["$origin/**", "world/agent_a", "world/agent_b"],
                            overrides={
                                "world/agent_a": [
                                    rr.components.FillModeBatch("majorwireframe")
                                ],
                                "world/agent_b": [
                                    rr.components.FillModeBatch("majorwireframe")
                                ],
                            },
                        ),
                        rrb.Spatial2DView(
                            name=SensorType.CAM_FRONT.name,
                            origin=f"world/ego_vehicle/{SensorType.CAM_FRONT.name}",
                            contents=["$origin/**", "world/agent_a", "world/agent_b"],
                            overrides={
                                "world/agent_a": [
                                    rr.components.FillModeBatch("majorwireframe")
                                ],
                                "world/agent_b": [
                                    rr.components.FillModeBatch("majorwireframe")
                                ],
                            },
                        ),
                        rrb.Spatial2DView(
                            name=SensorType.CAM_FRONT_RIGHT.name,
                            origin=f"world/ego_vehicle/{SensorType.CAM_FRONT_RIGHT.name}",
                            contents=["$origin/**", "world/agent_a", "world/agent_b"],
                            overrides={
                                "world/agent_a": [
                                    rr.components.FillModeBatch("majorwireframe")
                                ],
                                "world/agent_b": [
                                    rr.components.FillModeBatch("majorwireframe")
                                ],
                            },
                        ),
                    ),
                    rrb.Horizontal(
                        rrb.Spatial2DView(
                            name=SensorType.CAM_BACK_LEFT.name,
                            origin=f"world/ego_vehicle/{SensorType.CAM_BACK_LEFT.name}",
                            contents=["$origin/**", "world/agent_a", "world/agent_b"],
                            overrides={
                                "world/agent_a": [
                                    rr.components.FillModeBatch("majorwireframe")
                                ],
                                "world/agent_b": [
                                    rr.components.FillModeBatch("majorwireframe")
                                ],
                            },
                        ),
                        rrb.Spatial2DView(
                            name=SensorType.CAM_BACK.name,
                            origin=f"world/ego_vehicle/{SensorType.CAM_BACK.name}",
                            contents=["$origin/**", "world/agent_a", "world/agent_b"],
                            overrides={
                                "world/agent_a": [
                                    rr.components.FillModeBatch("majorwireframe")
                                ],
                                "world/agent_b": [
                                    rr.components.FillModeBatch("majorwireframe")
                                ],
                            },
                        ),
                        rrb.Spatial2DView(
                            name=SensorType.CAM_BACK_RIGHT.name,
                            origin=f"world/ego_vehicle/{SensorType.CAM_BACK_RIGHT.name}",
                            contents=["$origin/**", "world/agent_a", "world/agent_b"],
                            overrides={
                                "world/agent_a": [
                                    rr.components.FillModeBatch("majorwireframe")
                                ],
                                "world/agent_b": [
                                    rr.components.FillModeBatch("majorwireframe")
                                ],
                            },
                        ),
                    ),
                ),
            ),
            row_shares=[1, 2],
        ),
        rrb.BlueprintPanel(state="collapsed"),
        rrb.SelectionPanel(state="collapsed"),
        rrb.TimePanel(state="collapsed"),
    )

    rr.script_setup(args, "stsbench_annotator")
    rr.send_blueprint(blueprint)


def main(args):
    init_viewer(args)
    engine = create_engine(f"sqlite:///{str(args.db_path.name)}", echo=False)
    session = Session(engine)
    stmt = (
        select(Maneuver)
        .where(Maneuver.manually_labeled == False)
        .where(Maneuver.in_use == False)
        .outerjoin(PositiveManeuver)
        .order_by(PositiveManeuver.type)
    )
    maneuvers = session.scalars(stmt).all()
    annotate(session, maneuvers, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="STSBench Annotator")
    parser.add_argument(
        "--dataroot",
        help="nuScenes data root folder",
        default=Path("./nuscenes/v1.0-trainval"),
        type=Path,
    )
    parser.add_argument(
        "--db_path",
        type=Path,
        default="nuScenes.db",
    )
    rr.script_add_args(parser)
    args = parser.parse_args()

    main(args)
