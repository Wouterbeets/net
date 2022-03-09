package net

import (
	"fmt"
	"sort"
)

// DNA encodes a network
type DNA []float64

type gene struct {
	sourceID int
	destID   int
	weight   float64
	destBias float64
}

// NetToDna encodes the network to dna
func NetToDna(n *Net) (dna DNA) {
	for _, s := range n.synapses() {
		dna = append(dna, s.DNA()...)
	}
	return dna
}

// String for fmt.Stringer
func (dna DNA) String() string {
	i := 0
	ret := ""
	for i < len(dna) {
		ret += "source: "
		ret += fmt.Sprintf("%f ", dna[i])
		ret += "dest: "
		ret += fmt.Sprintf("%f ", dna[i+1])
		ret += "weight: "
		ret += fmt.Sprintf("%f ", dna[i+2])
		ret += "dest bias: "
		ret += fmt.Sprintf("%f\n", dna[i+3])
		i += 4
	}
	return ret
}

func DNAToNet(dna DNA) (*Net, error) {
	n := new(Net)
	n.store = make(map[int]*neuron)
	var i int
	var g gene
	for i < len(dna) {
		g = readGene(dna, i)
		var source *neuron
		var ok bool
		if source, ok = n.store[g.sourceID]; !ok {
			source = &neuron{id: g.sourceID}
			n.store[source.id] = source
		}

		var dest *neuron
		if dest, ok = n.store[g.destID]; !ok {
			dest = &neuron{id: g.destID}
			n.store[dest.id] = dest
		}
		s := &synapse{
			source:      source,
			destination: dest,
			weight:      g.weight,
		}
		dest.bias = g.destBias
		source.out = append(source.out, s)
		dest.in = append(dest.in, s)
		i += 4
	}

	{ // assign layers
		for _, neur := range n.store {
			if len(neur.in) == 0 {
				neur.layer = inputLayer
				n.in = append(n.in, neur)
				continue
			}
			if len(neur.out) == 0 {
				neur.layer = outputLayer
				n.out = append(n.out, neur)
				continue
			}
			neur.layer = hiddenLayer
			n.hidden = append(n.hidden, neur)
		}
	}
	{ // sort layers
		sort.Slice(n.in, func(i, j int) bool { return n.in[i].id < n.in[j].id })
		sort.Slice(n.out, func(i, j int) bool { return n.out[i].id < n.out[j].id })
		sort.Slice(n.hidden, func(i, j int) bool { return n.hidden[i].id < n.hidden[j].id })
	}
	return n, nil
}

func readGene(dna DNA, i int) gene {
	g := gene{
		sourceID: int(dna[i]),
		destID:   int(dna[i+1]),
		weight:   dna[i+2],
		destBias: dna[i+3],
	}
	return g
}
