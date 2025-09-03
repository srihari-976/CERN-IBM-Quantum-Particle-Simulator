# visualization.py
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

class ParticleVisualizer:
    def __init__(self):
        self.colors = {'electron': 'blue', 'positron': 'red', 'muon': 'green', 'default': 'gray'}

    def create_3d_trajectory_plot(self, simulation):
        fig = go.Figure()
        for particle in simulation.particles:
            if len(particle.trajectory) > 1:
                trajectory = np.array(particle.trajectory)
                fig.add_trace(go.Scatter3d(
                    x=trajectory[:, 0] * 1e9,
                    y=trajectory[:, 1] * 1e9,
                    z=trajectory[:, 2] * 1e9,
                    mode='lines+markers',
                    name=f'{particle.name} trajectory',
                    line=dict(color=self.colors.get(particle.name, self.colors['default']), width=12),
                    marker=dict(size=6)
                ))
        # detector layers as cylinders (barrel + caps)
        for layer in simulation.detector.layers:
            R = layer.radius * 1e9
            z_min = (layer.z_position - 0.5 * layer.thickness) * 1e9
            z_max = (layer.z_position + 0.5 * layer.thickness) * 1e9
            theta = np.linspace(0, 2*np.pi, 80)
            # barrel edges at z_min and z_max
            x_edge = R * np.cos(theta)
            y_edge = R * np.sin(theta)
            fig.add_trace(go.Scatter3d(x=x_edge, y=y_edge, z=np.full_like(theta, z_min), mode='lines', showlegend=False, line=dict(color='black', width=2, dash='dash')))
            fig.add_trace(go.Scatter3d(x=x_edge, y=y_edge, z=np.full_like(theta, z_max), mode='lines', showlegend=False, line=dict(color='black', width=2, dash='dash')))
            # a few vertical ribs to suggest the barrel surface
            for t in np.linspace(0, 2*np.pi, 8, endpoint=False):
                x_r = R * np.cos(t)
                y_r = R * np.sin(t)
                fig.add_trace(go.Scatter3d(x=[x_r, x_r], y=[y_r, y_r], z=[z_min, z_max], mode='lines', showlegend=False, line=dict(color='gray', width=1, dash='dot')))

        detector_summary = simulation.detector.get_event_summary()
        if not detector_summary.empty:
            fig.add_trace(go.Scatter3d(
                x=detector_summary['x'] * 1e9,
                y=detector_summary['y'] * 1e9,
                z=detector_summary['z'] * 1e9,
                mode='markers',
                name='Detector Hits',
                marker=dict(size=6, color=detector_summary['energy_deposit'], colorscale='Viridis', colorbar=dict(title="Energy Deposit (J)"), symbol='diamond')
            ))

        fig.update_layout(
            title='3D Particle Trajectories and Detector Response',
            scene=dict(
                xaxis_title='X (nm)',
                yaxis_title='Y (nm)',
                zaxis_title='Z (nm)',
                aspectmode='cube'
            ),
            width=1200,   # full width
            height=800,   # taller for better visibility
            margin=dict(l=50, r=50, t=80, b=50))

        return fig

    def create_energy_evolution_plot(self, simulation):
        fig = make_subplots(rows=2, cols=2, subplot_titles=('Total Energy vs Time', 'Momentum vs Time', 'Gamma Factor vs Time', 'Speed vs Time'))
        for particle in simulation.particles:
            if len(particle.energy_history) > 0:
                time_points = np.linspace(0, simulation.time if simulation.time > 0 else 1e-15, len(particle.energy_history))
                fig.add_trace(go.Scatter(x=time_points * 1e15, y=[float(e) for e in particle.energy_history], name=f'{particle.name} Energy'), row=1, col=1)
                fig.add_trace(go.Scatter(x=time_points * 1e15, y=[float(m) for m in particle.momentum_history], name=f'{particle.name} Momentum'), row=1, col=2)
                # gamma & speed approximate from simulation.simulation_data if available
                fig.add_trace(go.Scatter(x=time_points * 1e15, y=[float(particle.gamma) for _ in range(len(time_points))], name=f'{particle.name} γ'), row=2, col=1)
                fig.add_trace(go.Scatter(x=time_points * 1e15, y=[float(np.linalg.norm(particle.velocity) / 3e8) for _ in range(len(time_points))], name=f'{particle.name} v/c'), row=2, col=2)
        fig.update_xaxes(title_text="Time (fs)", row=1, col=1)
        fig.update_layout(
            title='Particle Dynamics Evolution',
            scene=dict(
                xaxis_title='X (nm)',
                yaxis_title='Y (nm)',
                zaxis_title='Z (nm)',
                aspectmode='cube'
            ),
            width=1200,   # full width
            height=800,   # taller for better visibility
            margin=dict(l=50, r=50, t=80, b=50))
        return fig

    def create_quantum_analysis_plot(self, quantum_analysis):
        fig = make_subplots(rows=2, cols=2, subplot_titles=('de Broglie Wavelengths', 'Uncertainty Analysis', 'Spin Measurements', 'Wave Packet Distribution'))
        particles = list(quantum_analysis.keys())
        if particles:
            wavelengths = [float(quantum_analysis[p]['de_broglie_wavelength']) for p in particles]
            fig.add_trace(go.Bar(x=particles, y=wavelengths, name='de Broglie λ'), row=1, col=1)
        # uncertainty & spin
        for i, p in enumerate(particles):
            ua = quantum_analysis[p]['uncertainty_analysis']
            fig.add_trace(go.Scatter(x=[ua['delta_x']], y=[ua['delta_p']], mode='markers', name=f'{p} ΔxΔp'), row=1, col=2)
            spin = quantum_analysis[p].get('spin_measurement', {})
            fig.add_trace(go.Bar(x=['Spin Up', 'Spin Down'], y=[spin.get('spin_up_probability', 0.5), spin.get('spin_down_probability', 0.5)], name=f'{p} Spin'), row=2, col=1)
        # wave packet
        if particles:
            wave = quantum_analysis[particles[0]].get('wave_packet', {}).get('wave_packet_distribution', {})
            if wave:
                states = list(wave.keys())
                counts = list(wave.values())
                fig.add_trace(go.Bar(x=states, y=counts, name='Wave Packet States'), row=2, col=2)
        fig.update_layout(
            title='Quantum Effects Analysis',
            scene=dict(
                xaxis_title='X (nm)',
                yaxis_title='Y (nm)',
                zaxis_title='Z (nm)',
                aspectmode='cube'
            ),
            width=1200,   # full width
            height=800,   # taller for better visibility
            margin=dict(l=50, r=50, t=80, b=50))
        return fig

    def create_detector_response_plot(self, detector_summary):
        if detector_summary.empty:
            fig = go.Figure()
            fig.add_annotation(text="No detector hits recorded", xref="paper", yref="paper", x=0.5, y=0.5)
            return fig
        
        # Ensure numeric columns for plotting
        df = detector_summary.copy()
        for col in ['energy_deposit']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Energy by material/layer (single bar plot)
        layer_energy = df.groupby('layer_material')['energy_deposit'].sum().sort_values(ascending=False)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=layer_energy.index, y=layer_energy.values, name='Energy Deposit'))

        fig.update_layout(
            title='Detector Response: Energy Deposition by Layer',
            xaxis_title='Material',
            yaxis_title='Energy (J)',
            width=1200,
            height=500,
            margin=dict(l=50, r=50, t=80, b=50),
            showlegend=False
        )
        return fig
