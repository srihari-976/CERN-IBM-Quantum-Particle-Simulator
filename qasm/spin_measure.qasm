OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
creg c[1];
// Prepare superposition and measure spin along Z
h q[0];
// Optional phase rotation placeholder; parameterized externally if needed
// rz(0.0) q[0];
measure q[0] -> c[0];