# electromagnetic_field.py
import numpy as np
from scipy.constants import c, epsilon_0, mu_0
import logging

class ElectromagneticField:
    def __init__(self, E_field=None, B_field=None):
        self.E_field = np.array(E_field, dtype=float) if E_field is not None else np.zeros(3)
        self.B_field = np.array(B_field, dtype=float) if B_field is not None else np.zeros(3)

    def electric_field_at(self, position):
        return self.E_field

    def magnetic_field_at(self, position):
        return self.B_field

    def lorentz_force(self, particle):
        E = self.electric_field_at(particle.position)
        B = self.magnetic_field_at(particle.position)
        return particle.charge * (E + np.cross(particle.velocity, B))


class CoulombForce:
    @staticmethod
    def force_between_particles(p1, p2):
        r_vec = p2.position - p1.position
        r = np.linalg.norm(r_vec)
        if r < 1e-15:
            return np.zeros(3)
        k = 1.0 / (4 * np.pi * epsilon_0)
        force_mag = k * p1.charge * p2.charge / (r**2)
        return force_mag * (r_vec / r)
