package net

import "math"

type synapse struct {
	source      *neuron
	destination *neuron
	weight      float64
	memory      *signal // when a signal passes through a synapse, it's id is stored in last signal to avoid loops
}

type neuron struct {
	id   string
	bias float64
	out  []synapse // outgoing connections
	in   []synapse // in connections

}

type signal struct {
	v  float64
	id int
}

func (n *neuron) eval(id int) signal {
	//note that the signal passed here
	for _, syn := range n.out {
		syn.memory = &signal{id: id}
	}

	// get signal from input recursivly
	var signals []signal
	for _, syn := range n.in {
		// we've hit an edge
		if syn.memory != nil {
			//check if its a loop or an input
			if syn.memory.id == id {
				sig := *syn.memory
				syn.memory
				return sig
			} else { // its a memory
				sig := *n.in[0].memory
				return sig
			}
		}
		sig := syn.source.eval(id)
		//manage memory here
		sig.v *= syn.weight
		signals = append(signals, sig)
	}

	var sum float64
	for _, sig := range signals {
		sum += sig.v
	}
	return signal{v: sigmoid(sum * n.bias), id: id}

}

type net struct {
	in       []neuron
	out      []neuron
	hidden   []neuron
	signalID int
}

func (n net) eval(input []float64) (output []float64) {
	n.signalID += 1
	for i, neuron := range n.in {
		neuron.in[0].memory = &signal{v: input[i], id: n.signalID}
	}

	var signals []signal
	for _, neuron := range n.out {
		signals = append(signals, neuron.eval(n.signalID))
	}

	for _, s := range signals {
		output = append(output, s.v)
	}

	return output
}

func sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(x*(-1)))
}
