import numpy as np
from scipy.constants import c, e, m_e

class Particle:
    """Relativistic particle base class"""
    def __init__(self, mass, charge, position, velocity, name="particle", dt=1e-15):
        self.mass = float(mass)
        self.charge = float(charge)
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.name = name
        self.dt = float(dt)  # ✅ store default timestep
        self.trajectory = [self.position.copy()]
        self.energy_history = []
        self.momentum_history = []

    @property
    def speed(self):
        return np.linalg.norm(self.velocity)

    @property
    def gamma(self):
        v = self.speed
        if v >= c:
            return 1e6  # avoid infinity
        return 1.0 / np.sqrt(1 - (v / c) ** 2)

    @property
    def momentum(self):
        # relativistic momentum gamma*m*v (vector)
        return self.gamma * self.mass * self.velocity

    @property
    def total_energy(self):
        return self.gamma * self.mass * c**2

    @property
    def kinetic_energy(self):
        return (self.gamma - 1.0) * self.mass * c**2

    def apply_force(self, force, dt=None):
        """Update momentum & velocity using dp = F dt (relativistic approx)"""
        if dt is None:
            dt = self.dt

        p = self.momentum
        p_new = p + np.array(force, dtype=float) * dt

        p_mag = np.linalg.norm(p_new)
        if p_mag == 0:
            self.velocity = np.zeros(3)
        else:
            v_mag = p_mag * c / np.sqrt(p_mag**2 + (self.mass * c)**2)
            self.velocity = p_new * (v_mag / p_mag)

        self.momentum_history.append(np.linalg.norm(p_new))
        self.energy_history.append(self.total_energy)

    def update_position(self, dt=None):
        """Advance particle position and record trajectory + energy"""
        if dt is None:
            dt = self.dt

        self.position = self.position + self.velocity * dt
        self.trajectory.append(self.position.copy())
        self.energy_history.append(self.total_energy)


class ElectronParticle(Particle):
    def __init__(self, position, velocity, dt=1e-15):
        super().__init__(m_e, -e, position, velocity, "electron", dt)


class PositronParticle(Particle):
    def __init__(self, position, velocity, dt=1e-15):
        super().__init__(m_e, e, position, velocity, "positron", dt)


class MuonParticle(Particle):
    def __init__(self, position, velocity, dt=1e-15):
        muon_mass = 206.8 * m_e
        super().__init__(muon_mass, -e, position, velocity, "muon", dt)
