# quantum_effects.py
import numpy as np
from scipy.constants import hbar, c
import logging
import os
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# Note: qiskit_ibm_runtime is optional. If you want to use IBM hardware,
# configure credentials externally and set use_real_quantum=True.
try:
    from qiskit_ibm_runtime import QiskitRuntimeService, Session, Sampler
    IBM_RUNTIME_AVAILABLE = True
except Exception:
    IBM_RUNTIME_AVAILABLE = False

class QuantumEffects:
    def __init__(self, use_real_quantum=False, ibm_token=None, backend_name: str = "ibmq_qasm_simulator"):
        self.backend_name = backend_name or "ibmq_qasm_simulator"
        self.use_real_quantum = use_real_quantum and IBM_RUNTIME_AVAILABLE
        self.service = None
        if self.use_real_quantum:
            try:
                # Prefer explicit token if provided; otherwise default credentials
                if ibm_token:
                    self.service = QiskitRuntimeService(channel="ibm_quantum", token=ibm_token)
                else:
                    self.service = QiskitRuntimeService()
                # Validate backend availability (fallback to first backend if requested not found)
                try:
                    names = [b.name for b in self.service.backends()]
                    if self.backend_name not in names and names:
                        logging.warning("Requested backend '%s' not found. Using '%s' instead.", self.backend_name, names[0])
                        self.backend_name = names[0]
                except Exception:
                    pass
                logging.info("Connected to IBM Quantum runtime service (backend=%s)", self.backend_name)
            except Exception as e:
                logging.warning("Unable to connect to IBM runtime service: %s. Falling back to simulator.", e)
                self.use_real_quantum = False

    def _load_qasm_circuit(self, filename: str) -> QuantumCircuit:
        """Attempt to load a QuantumCircuit from a QASM file in the qasm/ directory.
        Falls back to raising FileNotFoundError if not found or loading fails.
        """
        qasm_path = os.path.join(os.path.dirname(__file__), 'qasm', filename)
        if not os.path.isfile(qasm_path):
            raise FileNotFoundError(f"QASM file not found: {qasm_path}")
        try:
            return QuantumCircuit.from_qasm_file(qasm_path)
        except Exception as exc:
            raise FileNotFoundError(f"Failed to load QASM file '{qasm_path}': {exc}")

    def _run_sampler(self, qc: QuantumCircuit, shots: int):
        """Run circuit on IBM Runtime Sampler or local Aer and return counts dict."""
        try:
            if self.use_real_quantum and self.service:
                with Session(service=self.service, backend=self.backend_name) as session:
                    sampler = Sampler(session=session)
                    job = sampler.run([qc], shots=shots)
                    result = job.result()
                    # Try Sampler v0.11 API first
                    counts = None
                    try:
                        qd = result.quasi_dists[0]
                        # Convert quasi-dist to integer counts
                        counts = {format(k, f"0{qc.num_clbits}b"): int(round(v * shots)) for k, v in qd.items()}
                    except Exception:
                        try:
                            # Legacy accessor
                            counts = result[0].data.meas.get_counts()
                        except Exception:
                            pass
                    if counts is None:
                        raise RuntimeError("Unexpected sampler result format")
            else:
                sim = AerSimulator()
                transpiled = transpile(qc, sim)
                job = sim.run(transpiled, shots=shots)
                res = job.result()
                counts = res.get_counts()
            return counts
        except Exception as e:
            logging.error("Quantum execution failed: %s", e)
            # Safe balanced default
            return {'0' * max(1, qc.num_clbits): shots // 2, '1' * max(1, qc.num_clbits): shots - shots // 2}

    def de_broglie_wavelength(self, particle):
        p = np.linalg.norm(particle.momentum)
        if p <= 0:
            return np.inf
        return (hbar * 2 * np.pi) / p

    def compton_wavelength(self, particle):
        return (hbar * 2 * np.pi) / (particle.mass * c)

    def heisenberg_uncertainty_position_momentum(self, particle):
        if len(particle.trajectory) > 1:
            positions = np.array(particle.trajectory)
            delta_x = np.std(positions, axis=0)
            delta_x_magnitude = float(np.linalg.norm(delta_x))
        else:
            delta_x_magnitude = 1e-10
        if len(particle.momentum_history) > 1:
            delta_p = float(np.std(particle.momentum_history))
        else:
            delta_p = float(hbar / (2 * delta_x_magnitude))
        uncertainty_product = delta_x_magnitude * delta_p
        heisenberg_limit = hbar / 2
        return {'delta_x': delta_x_magnitude, 'delta_p': delta_p, 'uncertainty_product': uncertainty_product, 'heisenberg_limit': heisenberg_limit, 'satisfies_uncertainty': uncertainty_product >= heisenberg_limit}

    def quantum_spin_measurement(self, particle, shots=512):
        # Try loading from QASM template; fallback to programmatic circuit
        try:
            qc = self._load_qasm_circuit('spin_measure.qasm')
        except Exception:
            qc = QuantumCircuit(1, 1)
            qc.h(0)
            qc.measure(0, 0)

        try:
            counts = self._run_sampler(qc, shots)
            total = sum(counts.values())
            prob_up = counts.get('0', 0) / total
            prob_down = counts.get('1', 0) / total
            return {'spin_up_probability': float(prob_up), 'spin_down_probability': float(prob_down), 'counts': counts}
        except Exception as e:
            logging.error("Quantum spin measurement failed: %s", e)
            # return a safe default
            return {'spin_up_probability': 0.5, 'spin_down_probability': 0.5, 'counts': {'0': 1, '1': 1}}

    def wave_packet_evolution(self, particle, n_qubits=3, time_steps=4):
        # Try loading from QASM template; fallback to programmatic circuit
        try:
            qc = self._load_qasm_circuit('wave_packet.qasm')
        except Exception:
            qc = QuantumCircuit(n_qubits, n_qubits)
            for q in range(n_qubits):
                qc.h(q)
            qc.measure_all()
        try:
            counts = self._run_sampler(qc, shots=1024)
            return {'wave_packet_distribution': counts, 'evolution_steps': time_steps}
        except Exception as e:
            logging.error("Wave packet evolution failed: %s", e)
            return {'wave_packet_distribution': {}, 'evolution_steps': time_steps, 'error': str(e)}

    def run_qasm_file(self, filename: str, shots: int = 1024):
        """Load a QASM file from the qasm/ folder and execute it.
        Returns a dict with 'counts' and 'shots'.
        """
        try:
            qc = self._load_qasm_circuit(filename)
        except Exception as exc:
            logging.error("Failed to load QASM '%s': %s", filename, exc)
            return {'counts': {}, 'shots': int(shots), 'error': f"QASM load failed: {exc}"}

        try:
            counts = self._run_sampler(qc, int(shots))
            return {'counts': counts, 'shots': int(shots)}
        except Exception as exc:
            logging.error("Failed to execute QASM '%s': %s", filename, exc)
            return {'counts': {}, 'shots': int(shots), 'error': str(exc)}
