# detector.py
import numpy as np
import pandas as pd
import logging

class DetectorLayer:
    def __init__(self, z_position, radius, material="silicon", thickness=0.001):
        self.z_position = float(z_position)
        self.radius = float(radius)
        self.material = material
        self.thickness = float(thickness)
        self.hits = []

    def _z_bounds(self):
        half = 0.5 * self.thickness
        return self.z_position - half, self.z_position + half

    def check_intersection(self, particle_trajectory):
        """
        Compute intersections of a polyline trajectory with a finite-thickness
        cylindrical layer (barrel of radius R, spanning z_min..z_max) and its end-caps.

        Returns list of hit dicts with keys: position, radial_distance, trajectory_index, type, normal.
        """
        hits = []
        if len(particle_trajectory) < 2:
            return hits

        z_min, z_max = self._z_bounds()
        R = self.radius
        eps = 1e-15

        for i in range(len(particle_trajectory) - 1):
            p0 = np.asarray(particle_trajectory[i], dtype=float)
            p1 = np.asarray(particle_trajectory[i + 1], dtype=float)
            dp = p1 - p0

            # 1) Barrel (side) intersections: solve for |(x,y)| = R along the segment
            dx, dy = dp[0], dp[1]
            x0, y0 = p0[0], p0[1]
            a = dx*dx + dy*dy
            b = 2.0 * (x0*dx + y0*dy)
            c = x0*x0 + y0*y0 - R*R
            if a > eps:
                disc = b*b - 4*a*c
                if disc >= 0:
                    sqrt_disc = np.sqrt(disc)
                    for root in [(-b - sqrt_disc) / (2*a), (-b + sqrt_disc) / (2*a)]:
                        if 0.0 <= root <= 1.0:
                            hit_pos = p0 + root * dp
                            if z_min - eps <= hit_pos[2] <= z_max + eps:
                                rdist = np.sqrt(hit_pos[0]**2 + hit_pos[1]**2)
                                # normal is radial unit vector at intersection
                                n = np.array([hit_pos[0], hit_pos[1], 0.0])
                                n_norm = np.linalg.norm(n)
                                if n_norm > eps:
                                    n = n / n_norm
                                else:
                                    n = np.array([1.0, 0.0, 0.0])
                                hits.append({
                                    'position': hit_pos,
                                    'radial_distance': float(rdist),
                                    'trajectory_index': i,
                                    'type': 'barrel',
                                    'normal': n
                                })

            # 2) End-cap intersections: z = z_min and z = z_max with radial <= R
            if abs(dp[2]) > eps:
                for z_plane, cap_name in [(z_min, 'cap_min'), (z_max, 'cap_max')]:
                    t = (z_plane - p0[2]) / dp[2]
                    if 0.0 <= t <= 1.0:
                        hit_pos = p0 + t * dp
                        rdist = np.sqrt(hit_pos[0]**2 + hit_pos[1]**2)
                        if rdist <= R + eps:
                            normal = np.array([0.0, 0.0, -1.0 if cap_name == 'cap_min' else 1.0])
                            hits.append({
                                'position': hit_pos,
                                'radial_distance': float(rdist),
                                'trajectory_index': i,
                                'type': cap_name,
                                'normal': normal
                            })

        return hits


class ParticleDetector:
    def __init__(self):
        self.layers = []
        self.events = []
        self.setup_default_detector()

    def setup_default_detector(self):
        # Nano-scale layers near origin (nm–µm)
        self.add_layer(DetectorLayer(1e-9, 5e-9, "silicon", 5e-10))   # 1 nm
        self.add_layer(DetectorLayer(5e-9, 1e-8, "silicon", 1e-9))    # 5 nm
        self.add_layer(DetectorLayer(1e-8, 2e-8, "silicon", 2e-9))    # 10 nm
        self.add_layer(DetectorLayer(5e-8, 5e-8, "silicon", 5e-9))    # 50 nm
        self.add_layer(DetectorLayer(1e-7, 1e-7, "silicon", 1e-8))    # 100 nm
    
        # Macro-scale calorimeters (will only be hit in long runs)
        self.add_layer(DetectorLayer(1e-3, 5e-4, "lead_tungstate", 1e-5))
        self.add_layer(DetectorLayer(1e-2, 1e-3, "gas_chamber", 1e-4))




    def add_layer(self, layer):
        self.layers.append(layer)

    def calculate_energy_deposition(self, particle, layer, hit):
        """
        Compute energy deposition using a stable Bethe–Bloch-like model.
        - Units: density [kg/m^3], thickness [m], result [J]
        - Uses K (SI) ≈ 0.307075 MeV g^-1 cm^2 converted to J m^2 kg^-1
        - Effective path length accounts for incidence angle through the layer.
        """
        try:
            v_mag = float(np.linalg.norm(particle.velocity))
            c0 = 299792458.0
            beta = max(1e-6, v_mag / c0)
            gamma = float(particle.gamma)
        except Exception:
            beta = 0.1
            gamma = 1.0

        # Material properties: density [kg/m^3], Z, A [g/mol], I [eV]
        material_properties = {
            'silicon': {'density': 2330.0, 'Z': 14.0, 'A': 28.0855, 'I_eV': 173.0},
            'lead_tungstate': {'density': 8280.0, 'Z': 74.0, 'A': 183.84, 'I_eV': 823.0},
            'iron': {'density': 7874.0, 'Z': 26.0, 'A': 55.845, 'I_eV': 286.0},
            'gas_chamber': {'density': 1.2, 'Z': 7.0, 'A': 14.01, 'I_eV': 85.0},
            'scintillator': {'density': 1030.0, 'Z': 6.5, 'A': 13.0, 'I_eV': 64.0}
        }
        if layer.material not in material_properties:
            return 0.0
        mat = material_properties[layer.material]

        # Effective path length through the layer accounting for incidence angle
        # Use local segment direction from trajectory index
        idx = int(hit.get('trajectory_index', 0))
        traj = getattr(particle, 'trajectory', [])
        if 0 <= idx < len(traj) - 1:
            seg_dir = np.asarray(traj[idx + 1], dtype=float) - np.asarray(traj[idx], dtype=float)
        else:
            # fallback to velocity direction
            seg_dir = np.asarray(getattr(particle, 'velocity', np.array([0.0, 0.0, 1.0])), dtype=float)
        seg_norm = np.linalg.norm(seg_dir)
        if seg_norm < 1e-20:
            seg_dir = np.array([0.0, 0.0, 1.0])
            seg_norm = 1.0
        v_hat = seg_dir / seg_norm

        n = np.asarray(hit.get('normal', np.array([0.0, 0.0, 1.0])), dtype=float)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-20:
            n = np.array([0.0, 0.0, 1.0])
            n_norm = 1.0
        n_hat = n / n_norm
        cos_incidence = abs(float(np.dot(v_hat, n_hat)))
        cos_incidence = max(1e-4, cos_incidence)  # avoid blow-up
        path_length = layer.thickness / cos_incidence

        # Bethe–Bloch-like dE/dx in SI units
        # K_SI ≈ 0.307075 MeV g^-1 cm^2 → J m^2 kg^-1
        K_SI = 0.307075 * 1.602176634e-13 * 1e4  # ≈ 4.919e-10 J m^2 / kg
        z_particle = abs(float(particle.charge)) / 1.602176634e-19  # in units of |e|
        Z, A = mat['Z'], mat['A']  # A in g/mol
        density = mat['density']

        # Mean excitation energy I in Joules
        I_J = mat['I_eV'] * 1.602176634e-19
        m_e_c2 = 9.10938356e-31 * (299792458.0**2)  # J
        W_max = 2.0 * m_e_c2 * (beta**2) * (gamma**2)
        argument = max(1.0 + 1e-9, (2.0 * m_e_c2 * (beta**2) * (gamma**2) / (I_J**2)))
        log_term = 0.5 * np.log(argument)
        bb_bracket = max(0.0, float(log_term - (beta**2)))

        dE_dx = (K_SI * (z_particle**2) * (Z / A) * (density / (beta**2)) * bb_bracket)
        energy_deposit = float(abs(dE_dx) * path_length)
        return energy_deposit
    
    def detect_particle(self, particle, threshold=0.0):
        particle_hits = []
        for layer in self.layers:
            hits = layer.check_intersection(particle.trajectory)
            for hit in hits:
                energy_deposit = self.calculate_energy_deposition(particle, layer, hit)

                # Only count significant deposits
                if energy_deposit > threshold:
                    hit_data = {
                        'particle_name': particle.name,
                        'layer_z': layer.z_position,
                        'layer_material': layer.material,
                        'hit_position': hit['position'],
                        'radial_distance': hit['radial_distance'],
                        'energy_deposit': energy_deposit,
                        'particle_momentum': np.linalg.norm(particle.momentum),
                        'particle_energy': particle.total_energy
                    }
                    particle_hits.append(hit_data)
                    layer.hits.append(hit_data)
        return particle_hits



    def create_event(self, particles):
        event_data = {'event_id': len(self.events), 'particles': [], 'total_energy_deposit': 0.0, 'hit_count': 0}
        logging.info("Creating detector event for %d particles", len(particles))
        for particle in particles:
            logging.info("Processing particle %s with %d trajectory points", particle.name, len(particle.trajectory))
            hits = self.detect_particle(particle, threshold=0.0)
            logging.info("Particle %s generated %d hits", particle.name, len(hits))
            if hits:
                particle_data = {'particle_name': particle.name, 'hits': hits, 'total_energy_deposit': sum(h['energy_deposit'] for h in hits)}
                event_data['particles'].append(particle_data)
                event_data['total_energy_deposit'] += particle_data['total_energy_deposit']
                event_data['hit_count'] += len(hits)
        logging.info("Event %d: total hits=%d, total energy=%.2e J", event_data['event_id'], event_data['hit_count'], event_data['total_energy_deposit'])
        self.events.append(event_data)
        return event_data

    def get_event_summary(self):
        if not self.events:
            logging.warning("No detector events found")
            return pd.DataFrame()
        logging.info("Creating event summary from %d events", len(self.events))
        summary = []
        for event in self.events:
            for p in event['particles']:
                for hit in p['hits']:
                    logging.info("Adding hit: %s at (%.2e, %.2e, %.2e) with energy %.2e J", 
                               hit['particle_name'], hit['hit_position'][0], hit['hit_position'][1], hit['hit_position'][2], hit['energy_deposit'])
                    summary.append({
                        'event_id': event['event_id'],
                        'particle_name': hit['particle_name'],
                        'layer_z': hit['layer_z'],
                        'layer_material': hit['layer_material'],
                        'x': float(hit['hit_position'][0]),
                        'y': float(hit['hit_position'][1]),
                        'z': float(hit['hit_position'][2]),
                        'radial_distance': float(hit['radial_distance']),
                        'energy_deposit': float(hit['energy_deposit']),
                        'particle_momentum': float(hit['particle_momentum']),
                        'particle_energy': float(hit['particle_energy'])
                    })
        logging.info("Event summary created with %d hit records", len(summary))
        return pd.DataFrame(summary)

    def reset_detector(self):
        for layer in self.layers:
            layer.hits = []
        self.events = []
