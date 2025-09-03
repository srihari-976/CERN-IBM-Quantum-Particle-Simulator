# app.py
from flask import Flask, render_template, request, jsonify, send_file
import json
import io
import csv
from datetime import datetime
import logging
import threading
import traceback

from simulation import ParticleSimulation
from visualization import ParticleVisualizer
import plotly
import os

app = Flask(__name__, template_folder="templates")
app.secret_key = 'particle_physics_research_2024'

# Global simulation instance
simulation = None
visualizer = ParticleVisualizer()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/create_scenario', methods=['POST'])
def create_scenario():
    """Create electron-positron collision scenario"""
    global simulation

    try:
        data = request.get_json() or {}
        use_quantum = bool(data.get('use_quantum', True))
        demo = bool(data.get('demo', True))
        ibm_token = data.get('ibm_token', None)

        # Create simulation
        simulation = ParticleSimulation(use_quantum=use_quantum, ibm_token=ibm_token)
        particles = simulation.create_electron_positron_collision_scenario(demo=demo)

        # Convert NumPy arrays to lists for JSON
        response_particles = {}
        for name, p in particles.items():
            response_particles[name] = {
                'position': p.position.tolist(),
                'velocity': p.velocity.tolist(),
                'mass': float(p.mass),
                'charge': float(p.charge),
                'name': p.name
            }

        response = {
            'status': 'success',
            'message': 'Electron-positron collision scenario created',
            'particles': response_particles
        }

        logger.info("Created electron-positron collision scenario (demo=%s)", demo)
        return jsonify(response)

    except Exception as e:
        logger.error("Error creating scenario: %s\n%s", e, traceback.format_exc())
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/run_simulation', methods=['POST'])
def run_simulation():
    """Run the particle physics simulation"""
    global simulation

    if simulation is None:
        return jsonify({'status': 'error', 'message': 'No simulation scenario created'}), 400

    try:
        data = request.get_json() or {}
        duration = float(data.get('duration', 1e-9))  # Default 1 ps
        fresh_detector = bool(data.get('fresh_detector', True))

        # Run simulation in background thread
        result_holder = {}

        def run_sim():
            try:
                # Optionally clear previous detector events so plots reflect this run only
                if fresh_detector:
                    simulation.detector.reset_detector()
                event = simulation.run_simulation(duration)
                result_holder['event'] = event
            except Exception as e:
                result_holder['error'] = str(e)
                result_holder['trace'] = traceback.format_exc()

        sim_thread = threading.Thread(target=run_sim)
        sim_thread.start()
        sim_thread.join(timeout=30)

        if sim_thread.is_alive():
            return jsonify({'status': 'error', 'message': 'Simulation timeout (increase server timeout for longer runs)'}), 500

        if 'error' in result_holder:
            logger.error("Simulation error: %s\n%s", result_holder['error'], result_holder.get('trace', ''))
            return jsonify({'status': 'error', 'message': result_holder['error']}), 500

        summary = simulation.get_simulation_summary()

        # 🔧 Recursively convert DataFrames & NumPy arrays to JSON-safe objects
        def make_json_safe(obj):
            import numpy as np
            import pandas as pd

            if isinstance(obj, pd.DataFrame):
                return obj.to_dict(orient="records")
            elif isinstance(obj, pd.Series):
                return obj.to_dict()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: make_json_safe(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_json_safe(v) for v in obj]
            else:
                return obj

        summary = simulation.get_simulation_summary()

# 🔧 Fix DataFrame serialization
        for key, value in summary.items():
            if hasattr(value, "to_dict"):  # if it's a DataFrame
                summary[key] = value.to_dict(orient="records")
        
        response = {
            'status': 'success',
            'message': 'Simulation completed successfully',
            'summary': summary,
            'total_time': float(simulation.time),
            'detector_events': len(simulation.detector.events)
        }


        return jsonify(response)

    except Exception as e:
        logger.error("Error running simulation: %s\n%s", e, traceback.format_exc())
        return jsonify({'status': 'error', 'message': str(e)}), 500



@app.route('/api/get_visualization/<plot_type>')
def get_visualization(plot_type):
    """Return a Plotly figure JSON for a given plot type"""
    global simulation

    if simulation is None:
        return jsonify({'status': 'error', 'message': 'No simulation data available'}), 400

    try:
        if plot_type == 'trajectories':
            fig = visualizer.create_3d_trajectory_plot(simulation)
        elif plot_type == 'energy':
            fig = visualizer.create_energy_evolution_plot(simulation)
        elif plot_type == 'quantum':
            quantum_analysis = simulation.analyze_quantum_effects()
            fig = visualizer.create_quantum_analysis_plot(quantum_analysis)
        elif plot_type == 'detector':
            detector_summary = simulation.detector.get_event_summary()
            fig = visualizer.create_detector_response_plot(detector_summary)
        else:
            return jsonify({'status': 'error', 'message': 'Invalid plot type'}), 400

        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        return jsonify({'status': 'success', 'plot': graphJSON})

    except Exception as e:
        logger.error("Error creating visualization: %s\n%s", e, traceback.format_exc())
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/get_quantum_analysis')
def get_quantum_analysis():
    global simulation
    if simulation is None:
        return jsonify({'status': 'error', 'message': 'No simulation data available'}), 400

    try:
        quantum_analysis = simulation.analyze_quantum_effects()
        serializable = {}

        for pname, analysis in quantum_analysis.items():
            serializable[pname] = {
                'de_broglie_wavelength': float(analysis['de_broglie_wavelength']),
                'compton_wavelength': float(analysis['compton_wavelength']),
                'uncertainty_analysis': {
                    k: float(v) if isinstance(v, (int, float)) else v
                    for k, v in analysis['uncertainty_analysis'].items()
                },
                'spin_measurement': analysis.get('spin_measurement', {}),
                'wave_packet': analysis.get('wave_packet', {})
            }

        return jsonify({'status': 'success', 'quantum_analysis': serializable})

    except Exception as e:
        logger.error("Error getting quantum analysis: %s\n%s", e, traceback.format_exc())
        return jsonify({'status': 'error', 'message': str(e)}), 500
@app.route('/api/run_qasm', methods=['POST'])
def run_qasm():
    global simulation
    if simulation is None:
        return jsonify({'status': 'error', 'message': 'No simulation data available'}), 400
    try:
        data = request.get_json() or {}
        filename = data.get('filename', '')
        shots = int(data.get('shots', 1024))

        # Sanitize filename to restrict to qasm directory
        basename = os.path.basename(filename)
        if not basename.endswith('.qasm'):
            return jsonify({'status': 'error', 'message': 'Only .qasm files are allowed'}), 400

        result = simulation.quantum_effects.run_qasm_file(basename, shots=shots)
        if 'error' in result and result.get('counts', {} ) == {}:
            return jsonify({'status': 'error', 'message': result['error']}), 500

        return jsonify({'status': 'success', 'filename': basename, 'shots': result.get('shots', shots), 'counts': result.get('counts', {})})
    except Exception as e:
        logger.error("Error running QASM: %s\n%s", e, traceback.format_exc())
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/get_detector_data')
def get_detector_data():
    global simulation
    if simulation is None:
        return jsonify({'status': 'error', 'message': 'No simulation data available'}), 400
    try:
        detector_summary = simulation.detector.get_event_summary()
        if hasattr(detector_summary, "to_dict"):
            detector_data = detector_summary.to_dict(orient="records")
        else:
            detector_data = []

        return jsonify({
            'status': 'success',
            'detector_data': detector_data,
            'total_hits': len(detector_data),
            'total_energy_deposit': float(detector_summary['energy_deposit'].sum()) if not detector_summary.empty else 0
        })

    except Exception as e:
        logger.error("Error getting detector data: %s\n%s", e, traceback.format_exc())
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/download_report')
def download_report():
    global simulation
    if simulation is None:
        return jsonify({'status': 'error', 'message': 'No simulation data available'}), 400
    try:
        output = io.StringIO()
        writer = csv.writer(output)

        # Header
        writer.writerow(['Particle Physics Simulation Report'])
        writer.writerow(['Generated:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
        writer.writerow([])

        summary = simulation.get_simulation_summary()
        writer.writerow(['Simulation Summary'])
        writer.writerow(['Total Time (s):', f"{summary.get('total_time', 0):.2e}"])
        writer.writerow(['Total Steps:', summary.get('total_steps', 0)])
        writer.writerow(['Detector Events:', summary.get('detector_events', 0)])
        writer.writerow([])

        writer.writerow(['Particle Analysis'])
        writer.writerow(['Particle', 'Final Energy (J)', 'Max Energy (J)', 'de Broglie λ (m)', 'Compton λ (m)'])
        for p in simulation.particles:
            final_energy = getattr(p, 'total_energy', 0.0)
            max_energy = max(p.energy_history) if p.energy_history else 0.0
            db = getattr(p, 'de_broglie_wavelength', 0.0)
            comp = getattr(p, 'compton_wavelength', 0.0)
            writer.writerow([p.name, f"{final_energy:.2e}", f"{max_energy:.2e}", f"{db:.2e}", f"{comp:.2e}"])

        # Detector hits
        detector_summary = simulation.detector.get_event_summary()
        if not detector_summary.empty:
            writer.writerow([])
            writer.writerow(['Detector Hits'])
            writer.writerow(['Event ID', 'Particle', 'Layer Material', 'X (m)', 'Y (m)', 'Z (m)', 'Energy Deposit (J)'])
            for _, row in detector_summary.iterrows():
                writer.writerow([
                    int(row.get('event_id', -1)),
                    row.get('particle_name', ''),
                    row.get('layer_material', ''),
                    f"{row.get('x', 0):.2e}",
                    f"{row.get('y', 0):.2e}",
                    f"{row.get('z', 0):.2e}",
                    f"{row.get('energy_deposit', 0):.2e}"
                ])

        output.seek(0)
        file_data = io.BytesIO()
        file_data.write(output.getvalue().encode('utf-8'))
        file_data.seek(0)

        filename = f'particle_simulation_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        return send_file(file_data, mimetype='text/csv', as_attachment=True, download_name=filename)

    except Exception as e:
        logger.error("Error generating report: %s\n%s", e, traceback.format_exc())
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/reset_simulation', methods=['POST'])
def reset_simulation():
    global simulation
    try:
        if simulation:
            simulation.reset_simulation()
        return jsonify({'status': 'success', 'message': 'Simulation reset successfully'})
    except Exception as e:
        logger.error("Error resetting simulation: %s\n%s", e, traceback.format_exc())
        return jsonify({'status': 'error', 'message': str(e)}), 500


if __name__ == '__main__':
    logger.info("Starting Particle Physics Simulation Server")
    app.run(debug=True, host='0.0.0.0', port=5000)
