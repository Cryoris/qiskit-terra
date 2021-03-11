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


from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import BasicAer
from qiskit import execute



class ReversibleAdder(QuantumCircuit):
    """To implement QFT Adder.
    
    Based on the information given in [1].
    
    **References**
    
    [1] Thomas G.Draper, 2000. "Addition on a Quantum Computer"
    `Journal https://arxiv.org/pdf/quant-ph/0008033.pdf`_
    """
    
    def __init__(self, num_qubits_a: int, num_qubits_b: int, name='qft_adder') -> None:
        """
        Args:
            num_qubits_a: The size of the first register.
            num_qubits_b: The size of the second register.
        """
        # define the registers 
        qr_a = QuantumRegister(num_qubits_a, name='a') 
        qr_b = QuantumRegister(num_qubits_b, name='b') 
        qr_cin = QuantumRegister(num_qubits_a,name='cin')
        qr_cout = QuantumRegister(1, name='cout')
        #qr_ans=ClassicalRegister(n+1,name='ans')
	 
        
        # initialize the circuit
        super().__init__(qr_a, qr_b,qr_cin,qr_cout, name=name)

        qc_carry=QuantumCircuit(4,name='CARRY')
        qc_carry.ccx(1,2,3)
        qc_carry.cx(1,2)
        qc_carry.ccx(0,2,3)
        qc_instruction_carry=qc_carry.to_instruction()

        qc_sum=QuantumCircuit(3,name='SUM')
        qc_sum.cx(1,2)
        qc_sum.cx(0,2)
        qc_instruction_sum=qc_sum.to_instruction()
	
	
    
        # build the circuit here
        # either directly use the gate methods, e.g. self.cx(qr_a[0], qr_b[0])
        # or construct circuits and instructions and use append/compose
	# Build a temporary subcircuit that adds a to b,
	# storing the result in b

        for j in range(n-1):
        	self.append(qc_instruction_carry,[qr_cin[j], qr_a[j], qr_b[j],qr_cin[j+1]])

        self.append(qc_instruction_carry,[qr_cin[n-1], qr_a[n-1], qr_b[n-1],qr_cout])
        self.cnot(qr_a[n - 1], qr_b[n-1])
        self.append(qc_instruction_sum,[qr_cin[n-1],qr_a[n-1],qr_b[n-1]])

        for j in reversed(range(num_qubits_a - 1)):
             self.append(qc_instruction_carry,[qr_cin[j], qr_a[j], qr_b[j], qr_cin[j+1]])
             self.append(qc_instruction_sum, [qr_cin[j], qr_a[j], qr_b[j]])
      #  print(self)
        
  
   
