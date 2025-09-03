# simulation.py
import numpy as np
import logging
from particle import ElectronParticle, PositronParticle, MuonParticle
from electromagnetic_field import ElectromagneticField, CoulombForce
from quantum_effects import QuantumEffects
from detector import ParticleDetector, DetectorLayer
from scipy.constants import c
import multiprocessing as mp

class ParticleSimulation:
    def __init__(self, dt=1e-15, use_quantum=False, ibm_token=None):
        self.dt = float(dt)
        self.particles = []
        self.em_field = ElectromagneticField()
        self.detector = ParticleDetector()
        self.quantum_enabled = bool(use_quantum)
        self.quantum_effects = QuantumEffects(use_real_quantum=self.quantum_enabled, ibm_token=ibm_token)
        self.time = 0.0
        self.simulation_data = []
        self.use_multiprocessing = False  # safe default for small particle counts
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def add_particle(self, particle):
        self.particles.append(particle)
        self.logger.info("Added %s at %s", particle.name, particle.position.tolist())

    def set_electromagnetic_field(self, E_field=None, B_field=None):
        self.em_field = ElectromagneticField(E_field, B_field)
        self.logger.info("Set EM field: E=%s, B=%s", E_field, B_field)

    def calculate_forces_on_particle(self, particle_idx):
        particle = self.particles[particle_idx]
        total_force = np.zeros(3)
        total_force += self.em_field.lorentz_force(particle)
        for i, other in enumerate(self.particles):
            if i != particle_idx:
                total_force += CoulombForce.force_between_particles(particle, other)
        return total_force

    def step_simulation(self):
        if not self.particles:
            # nothing to simulate
            self.time += self.dt
            return

        if self.use_multiprocessing and len(self.particles) > 1:
            with mp.Pool() as pool:
                forces = pool.map(self.calculate_forces_on_particle, range(len(self.particles)))
        else:
            forces = [self.calculate_forces_on_particle(i) for i in range(len(self.particles))]

        for p, f in zip(self.particles, forces):
            p.apply_force(f, self.dt)
            p.update_position(self.dt)
            # lightly smear to increase chance of layer crossing in demo
            if np.random.rand() < 0.05:
                p.position += np.array([0.0, 0.0, 1e-12])

        self.time += self.dt

        step_data = {'time': self.time, 'particles': []}
        for p in self.particles:
            step_data['particles'].append({
                'name': p.name,
                'position': p.position.copy().tolist(),
                'velocity': p.velocity.copy().tolist(),
                'energy': float(p.total_energy),
                'momentum': float(np.linalg.norm(p.momentum)),
                'gamma': float(p.gamma)
            })
        self.simulation_data.append(step_data)

    def run_simulation(self, duration, progress_callback=None):
        # ensure positive dt and duration
        duration = max(float(duration), 1e-18)
        steps = max(1, int(duration / self.dt))
        self.logger.info("Running simulation for %.2e s (%d steps)", duration, steps)

        # seed initial energy/momentum histories so plots have starting points
        for p in self.particles:
            try:
                # ensure lists exist
                if not hasattr(p, 'energy_history') or p.energy_history is None:
                    p.energy_history = []
                if not hasattr(p, 'momentum_history') or p.momentum_history is None:
                    p.momentum_history = []
                # append initial values
                p.energy_history.append(float(p.total_energy))
                p.momentum_history.append(float(np.linalg.norm(p.momentum)))
            except Exception:
                # ignore if particle missing attributes
                pass

        for step in range(steps):
            self.step_simulation()
            # append energy/momentum each step for plotting
            for p in self.particles:
                try:
                    p.energy_history.append(float(p.total_energy))
                    p.momentum_history.append(float(np.linalg.norm(p.momentum)))
                except Exception:
                    pass

            if progress_callback and step % max(1, steps // 100) == 0:
                progress_callback((step + 1) / steps * 100)

        # create a detector event after the dynamics
        event = self.detector.create_event(self.particles)
        self.logger.info("Simulation completed. Final time: %.2e s", self.time)
        return event

    def analyze_quantum_effects(self):
        # Always compute analytic/wave properties; only run circuits when enabled
        q = {}
        for p in self.particles:
            base = {
                'de_broglie_wavelength': self.quantum_effects.de_broglie_wavelength(p),
                'compton_wavelength': self.quantum_effects.compton_wavelength(p),
                'uncertainty_analysis': self.quantum_effects.heisenberg_uncertainty_position_momentum(p)
            }
            if self.quantum_enabled:
                base['spin_measurement'] = self.quantum_effects.quantum_spin_measurement(p)
                base['wave_packet'] = self.quantum_effects.wave_packet_evolution(p)
            else:
                base['spin_measurement'] = {'spin_up_probability': 0.5, 'spin_down_probability': 0.5, 'counts': {}}
                base['wave_packet'] = {'wave_packet_distribution': {}, 'evolution_steps': 0}
            q[p.name] = base
        return q

    def create_electron_positron_collision_scenario(self, demo: bool = False):
        self.logger.info("Setting up electron-positron collision scenario (demo=%s)", demo)
        self.particles = []
        self.detector.reset_detector()
    
        if demo:
            # Replace detector layers with guaranteed-hit geometry near z≈0
            self.detector.layers = []
            demo_layers = [
                DetectorLayer(z_position=0.0,    radius=6e-9,  material="silicon",        thickness=6e-9),
                DetectorLayer(z_position=2e-9,   radius=1.2e-8, material="scintillator",   thickness=1.2e-8),
                DetectorLayer(z_position=5e-9,   radius=2.5e-8, material="lead_tungstate", thickness=2.0e-8),
            ]
            for layer in demo_layers:
                self.detector.add_layer(layer)

            # High-energy beams with z-components to cross end-caps/barrels
            electron = ElectronParticle(position=[-2e-8, 0.0, -1e-9], velocity=[0.8 * c, 0.0, 0.25 * c])
            positron = PositronParticle(position=[2e-8, 0.0, 1e-9], velocity=[-0.8 * c, 0.0, -0.25 * c])
            muon = MuonParticle(position=[0.0, -3e-8, -1e-9], velocity=[0.0, 0.7 * c, 0.15 * c])
        else:
            electron = ElectronParticle(position=[-1e-9, 0, 0], velocity=[0.2 * c, 0, 0])
            positron = PositronParticle(position=[1e-9, 0, 0], velocity=[-0.2 * c, 0, 0])
            muon = MuonParticle(position=[0, 2e-9, 0], velocity=[0.1 * c, -0.05 * c, 0.01 * c])
    
        self.add_particle(electron)
        self.add_particle(positron)
        self.add_particle(muon)
    
        self.set_electromagnetic_field(E_field=[1e6, 0, 0], B_field=[0, 0, 0.1])
        return {'electron': electron, 'positron': positron, 'muon': muon}



    def get_simulation_summary(self):
        if not self.simulation_data:
            return {}
        summary = {
            'total_time': self.time,
            'total_steps': len(self.simulation_data),
            'particles': {},
            'detector_events': len(self.detector.events),
            'detector_summary': self.detector.get_event_summary()
        }
        for p in self.particles:
            avg_speed = 0.0
            try:
                speeds = [np.linalg.norm(step['velocity']) for step in self.simulation_data if step.get('particles')]
                avg_speed = float(np.mean(speeds)) if speeds else 0.0
            except Exception:
                avg_speed = 0.0
            summary['particles'][p.name] = {
                'initial_position': p.trajectory[0].tolist() if p.trajectory else None,
                'final_position': p.position.tolist(),
                'initial_energy': p.energy_history[0] if getattr(p, 'energy_history', None) else None,
                'final_energy': p.total_energy,
                'max_energy': max(p.energy_history) if getattr(p, 'energy_history', None) else None,
                'trajectory_length': len(p.trajectory),
                'average_speed': avg_speed,
                'de_broglie_wavelength': self.quantum_effects.de_broglie_wavelength(p),
                'compton_wavelength': self.quantum_effects.compton_wavelength(p)
            }
        return summary

    def reset_simulation(self):
        self.particles = []
        self.detector.reset_detector()
        self.time = 0.0
        self.simulation_data = []
        self.logger.info("Simulation reset")
