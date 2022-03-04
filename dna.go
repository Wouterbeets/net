package net

import "fmt"

// DNA encodes a network
type DNA []float64

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
