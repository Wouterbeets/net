package net

import (
	"fmt"
)

const (
	inputLayer = iota
	hiddenLayer
	outputLayer
)

type neuron struct {
	id               int
	layer            byte
	shouldSaveMemory bool
	bias             float64
	visited          bool
	memory           *signal
	calculated       *signal
	activationFunc   func(float64) float64

	in  []*synapse // incoming connections
	out []*synapse // outgoing connections
}

func (n *neuron) eval2(id int) signal {
	fmt.Println("entering neuron:", n.id)
	if n.calculated != nil {
		fmt.Println("taking calculated:", n.id)
		return *n.calculated
	}
	if n.layer == inputLayer {
		fmt.Println("reached input layer:", n.id)
		if n.memory == nil {
			n.memory = &signal{id: id}
		}
		return *n.memory
	}

	var sigs []signal
	for i := range n.in {
		syn := n.in[i]
		//fmt.Println("calling synapse:", n.in[i].source.id)
		sig := syn.eval(id)
		fmt.Println("synapse source", n.in[i].source.id, "dest:", n.in[i].destination.id, "returned:", sig)
		sigs = append(sigs, sig)
	}

	var sum float64
	for _, sig := range sigs {
		sum += sig.v
	}

	//fmt.Println("synapses returned:", sum)
	if n.activationFunc == nil {
		n.activationFunc = sigmoid
	}
	sig := signal{v: n.activationFunc(sum * n.bias), id: id}
	if n.shouldSaveMemory {
		n.memory = &sig
		fmt.Println("savin mem for neur:", n.id, "mem:", sig)
		n.shouldSaveMemory = false
	}
	n.calculated = &sig
	fmt.Println("after sigmoid neuron:", n.id, "returns", sig)
	return sig
}
