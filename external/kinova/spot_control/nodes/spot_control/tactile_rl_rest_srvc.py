import os
from flask import Flask, jsonify, request
import tactile_rl_metrics_handler

arm = None


def create_service(capture_cmd_traj, capture_exec_traj):
    app = Flask(__name__)
    joint_state_setter = tactile_rl_metrics_handler.JointStateSetter(arm, capture_cmd_traj)
    joint_state_tracker = tactile_rl_metrics_handler.JointStateTracker(arm, capture_exec_traj)

    @app.route('/')
    def hello():
        return 'Hello, World!'

    @app.route('/fetch_kinova_metrics', methods=['GET'])
    def get_kinova_metrics():
        # Fetch the KinovaMetricsR object
        kinova_metrics = tactile_rl_metrics_handler.all_km_metrics(arm, joint_state_tracker)

        # Access its attributes
        response = {
            "robot_dof_pos": kinova_metrics.robot_dof_pos,
            "robot_dof_vel": kinova_metrics.robot_dof_vel,
            "robot_dof_torque": kinova_metrics.robot_dof_torque,
            "hand_pos": kinova_metrics.hand_pos,
            "hand_rot": kinova_metrics.hand_rot
        }
        return jsonify(response)

    @app.route('/set_dof_pos', methods=['POST'])
    def set_kinova_dof_pos():
        data = request.json
        if 'robot_dof_pos' in data and isinstance(data['robot_dof_pos'], list) and len(data['robot_dof_pos']) == 6:

            resp = joint_state_setter.set_joint_angles(data['robot_dof_pos'])
            str_resp = str(resp).replace("\n", "")
            return jsonify({"message": f"Joint angles set successfully.{str_resp}"}), 200
        else:
            return jsonify({"error": "Invalid or missing 'robot_dof_pos' in request."}), 400

    @app.route('/set_dof_vel', methods=['POST'])
    def set_kinova_dof_vel():
        data = request.json
        if 'robot_dof_vel' in data and isinstance(data['robot_dof_vel'], list) and len(data['robot_dof_vel']) == 6:

            resp = joint_state_setter.set_joint_velocities(data['robot_dof_vel'])
            str_resp = str(resp).replace("\n", "")
            return jsonify({"message": f"Joint Velocities set successfully.{str_resp}"}), 200
        else:
            return jsonify({"error": "Invalid or missing 'robot_dof_vel' in request."}), 400

    @app.route('/shutdown', methods=['GET'])
    def shutdown():
        joint_state_setter.display_attempts()
        joint_state_tracker.display_attempts()
        os.kill(os.getpid(), 9)
        return 'Server shutting down...'

    return app


def stand_up_service(port, _arm, debug=True, capture_cmd_traj=False, capture_exec_traj=False):
    global arm
    arm = _arm
    assert (capture_cmd_traj and capture_exec_traj) is False, "Capture only one at a time, either commanded or executed"
    app = create_service(capture_cmd_traj, capture_exec_traj)
    app.run(debug=debug, host='0.0.0.0', port=port)  # Run the Flask app
