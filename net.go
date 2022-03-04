package net

import (
	"fmt"
	"math"
)

type signal struct {
	v      float64
	id     int
	loop   bool
	memory *signal
}

type Net struct {
	store          map[int]*neuron
	in             []*neuron
	out            []*neuron
	hidden         []*neuron
	signalID       int
	activationFunc func(float64) float64
	weightFunc     func() float64
}

func newNet(in, hidden, out int, activationFunc func(float64) float64, weightFunc func() float64) *Net {
	n := new(Net)

	{ // set config
		n.activationFunc = activationFunc
		n.weightFunc = weightFunc
	}

	{ // init neurons
		neurons := make([]*neuron, in+hidden+out, in+hidden+out)
		for i := range neurons {
			neurons[i] = &neuron{id: i, bias: 1}
			neurons[i].activationFunc = n.activationFunc
			if i < in {
				neurons[i].layer = inputLayer
			} else if i < in+hidden {
				neurons[i].layer = hiddenLayer
			} else {
				neurons[i].layer = outputLayer
			}
			n.addNeuron(neurons[i])
		}
	}

	{ // add synapses
		// connect in layer to hidden layer
		for i := 0; i < in; i++ {
			for j := in; j < in+hidden; j++ {
				n.addSynapse(i, j, n.weightFunc())
			}
		}

		// connect hidden layer to out layer
		for i := in; i < in+hidden; i++ {
			for j := in + hidden; j < in+hidden+out; j++ {
				n.addSynapse(i, j, n.weightFunc())
			}
		}
	}
	return n
}

func (n *Net) Eval(input []float64) (output []float64, err error) {
	n.signalID += 1
	if len(input) != len(n.in) {
		return nil, fmt.Errorf("len of input data does not match net size")
	}
	for i := range n.in {
		n.in[i].memory = &signal{v: input[i], id: n.signalID}
	}

	var signals []signal
	for _, neuron := range n.out {
		signals = append(signals, neuron.eval2(n.signalID))
	}

	for _, s := range signals {
		output = append(output, s.v)
	}

	for _, neur := range n.store {
		neur.calculated = nil
	}

	return output, nil
}

func (n *Net) Synapses() (syns []*synapse) {
	for _, neur := range n.in {
		syns = append(syns, neur.in...)
		syns = append(syns, neur.out...)
	}
	for _, neur := range n.hidden {
		syns = append(syns, neur.out...)
	}
	for _, neur := range n.out {
		syns = append(syns, neur.out...)
	}
	return syns
}

func toDot(n *Net) {
	var inputNeuronIDs string
	for _, inputNeuron := range n.in {
		inputNeuronIDs += fmt.Sprintf("%d [label=%d]\n", inputNeuron.id, inputNeuron.id)
	}

	var hiddenNeuronIDs string
	for _, hiddenNeuron := range n.hidden {
		hiddenNeuronIDs += fmt.Sprintf("%d [label=%d]\n", hiddenNeuron.id, hiddenNeuron.id)
	}

	var outNeuronIDs string
	for _, outNeuron := range n.out {
		outNeuronIDs += fmt.Sprintf("%d [label=%d]\n", outNeuron.id, outNeuron.id)
	}

	var syns string
	for _, syn := range n.Synapses() {
		if syn.source != nil && syn.destination != nil {
			syns += fmt.Sprintf("%d -> %d\n", syn.source.id, syn.destination.id)
		}
	}

	fmt.Printf(`
digraph G {

    rankdir=LR;
	splines=line;
    nodesep=.2;
    ranksep=3;

    node [label=""];

    subgraph cluster_0 {
		color=white;
        node [style=solid,color=blue4, shape=circle];
		%s
		label = "input";
	}

	subgraph cluster_2 {
		color=white;
		node [style=solid,color=red2, shape=circle];
		%s
		label = "hiden";
	}

	subgraph cluster_3 {
		color=white;
		node [style=solid,color=seagreen2, shape=circle];
		%s
		label="output";
	}
	%s
}`, inputNeuronIDs, hiddenNeuronIDs, outNeuronIDs, syns)
}

func sigmoid(x float64) float64 {
	return (1/(1+math.Exp(x*(-1))) - 0.5) * 2
}

func (n *Net) addSynapse(inID, outID int, weight float64) error {
	inNeur, outNeur := n.store[inID], n.store[outID]
	s := &synapse{
		source:      inNeur,
		destination: outNeur,
		weight:      weight,
	}
	inNeur.out = append(inNeur.out, s)
	outNeur.in = append(outNeur.in, s)
	return nil
}

func (n *Net) addNeuron(neur *neuron) error {
	if n.store == nil {
		n.store = make(map[int]*neuron)
	}
	switch neur.layer {
	case inputLayer:
		n.in = append(n.in, neur)
	case hiddenLayer:
		n.hidden = append(n.hidden, neur)
	case outputLayer:
		n.out = append(n.out, neur)
	default:
		return fmt.Errorf("unknown layertype for neuron %d", neur.id)
	}
	n.store[neur.id] = neur
	return nil
}
