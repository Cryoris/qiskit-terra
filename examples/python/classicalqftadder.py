# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals

"""Module-level docstring describing what the file content is."""

#from qiskit.circuit import QuantumCircuit
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import BasicAer
from qiskit import execute


backend = BasicAer.get_backend("qasm_simulator")

# Make a quantum program for n bit QFT adder
#n=3
n=4
a = QuantumRegister(n, "a")
b = QuantumRegister(n, "b")
cin = QuantumRegister(n, "cin")
cout = QuantumRegister(1, "cout")
ans = ClassicalRegister(n + 1, "ans")
qc = QuantumCircuit(a, b, cin, cout, ans, name="qftadd")


    
def carry(p, a, b, c,d):
    """CARRY gate."""
    p.ccx(b,c, d)
    p.cx(b, c)
    p.ccx(a, c, d)


def sum(p, a, b, c):
    """SUM gate."""
    p.cx(b, c)
    p.cx(a, c)

# Build a temporary subcircuit that adds a to b,
# storing the result in b
adder_subcircuit = QuantumCircuit(cin, a, b, cout)

for j in range(n-1):
	carry(adder_subcircuit, cin[j], a[j], b[j], cin[j+1])

carry(adder_subcircuit, cin[n-1], a[n-1], b[n-1], cout)
adder_subcircuit.cx(a[n - 1], b[n-1])
sum(adder_subcircuit,cin[n-1],a[n-1],b[n-1])

for j in reversed(range(n - 1)):
    carry(adder_subcircuit, cin[j], a[j], b[j], cin[j+1])
    sum(adder_subcircuit, cin[j], a[j], b[j])
	
# Set the inputs to the adder
#qc.x(a[0])  # Set input a = 0...0001
#qc.x(b)  # Set input b = 1...1111

#qc.x(a[0])  # Set input a = 101
#qc.x(a[2])
#qc.x(b[0])  # Set input b = 001

#n=4
qc.x(a[1])
qc.x(a[3])
qc.x(b[1])
# Apply the adder
qc += adder_subcircuit
# Measure the output register in the computational basis
for j in range(n):
    qc.measure(b[j], ans[j])
qc.measure(cout[0], ans[n])
print(qc.draw())

job = execute(qc, backend=backend, coupling_map=None, shots=1024)
result = job.result()
print(result.get_counts(qc))

