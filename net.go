package net

import (
	"fmt"
	"math"
)

type synapse struct {
	source      *neuron
	destination *neuron
	weight      float64
	memory      *signal // when a signal passes through a synapse, it's id is stored in last signal to avoid loops
	visited     bool
}

type neuron struct {
	id   int
	bias float64
	out  []synapse // outgoing connections
	in   []synapse // in connections

}

type signal struct {
	v  float64
	id int
}

func (n *neuron) eval(id int) signal {
	fmt.Println("entering neuron", n.id)
	//note that the signal passed here
	for _, syn := range n.out {
		syn.visited = true
	}

	// get signal from input recursivly
	var signals []signal
	for i := range n.in {
		syn := n.in[i]
		// we've hit a loop
		if syn.visited {
			if syn.memory == nil {
				syn.memory = &signal{id: id}
			}
			sig := *syn.memory
			fmt.Printf("mem %+v\n", sig)
			return sig
		}
		// get input recursivly
		fmt.Println("calling eval for", syn.source.id)
		sig := syn.source.eval(id)
		fmt.Printf("got signal from %d, sig %+v\n", syn.source.id, sig)

		// apply weight
		sig.v *= syn.weight
		fmt.Printf("sig after weigth %+v\n", sig)

		// save signal
		signals = append(signals, sig)

		// undo visit
		syn.visited = false

		// leave memory if synapse ends in loop
		if syn.memory != nil {
			syn.memory = &sig
		}
	}

	var sum float64
	for _, sig := range signals {
		sum += sig.v
	}
	sig := signal{v: sigmoid(sum * n.bias), id: id}
	fmt.Printf("returning sig %+v\n", sig)
	return sig

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
		neuron.in[0].visited = true
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

func NewNet(in, out int) net {
	var hidden int = 1

	n := net{}
	neurons := make([]neuron, in+hidden+out, in+hidden+out)
	for i := range neurons {
		neurons[i].id = i
	}
	n.in = neurons[0:in]
	n.hidden = neurons[in : in+1]
	n.out = neurons[in+1:]

	// init input synapses
	for i := range n.in {

		// environment input
		n.in[i].in = []synapse{{destination: &n.in[i]}}

		// to hidden layer
		n.in[i].out = []synapse{{
			source:      &n.in[i],
			destination: &n.hidden[0],
		}}

		// hidden layer input
		n.hidden[0].in = append(n.hidden[0].in, n.in[i].out...)
	}

	// init output synapses
	for i := range n.out {

		// output to env
		n.out[i].out = []synapse{{source: &n.out[i]}}

		// hidden to output
		n.out[i].in = []synapse{{
			source:      &n.hidden[0],
			destination: &n.out[i],
		}}

		// hidden layer output
		n.hidden[0].out = append(n.hidden[0].out, n.out[i].in...)
	}
	return n
}

func sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(x*(-1)))
}
