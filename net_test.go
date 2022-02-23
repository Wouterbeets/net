package net

import (
	"fmt"
	"testing"
)

func TestCreateNet(t *testing.T) {
	input := []*neuron{{id: "0"}, {id: "1"}}
	output := []*neuron{{id: "2"}}
	hidden := []*neuron{{id: "3"}, {id: "4"}}
	synapses := []synapse{
		{
			source:      input[0],
			destination: hidden[0],
		},
		{
			source:      input[0],
			destination: hidden[1],
		},
		{
			source:      input[1],
			destination: hidden[0],
		},
		{
			source:      input[1],
			destination: hidden[1],
		},
		{
			source:      hidden[0],
			destination: output[0],
		},
		{
			source:      hidden[1],
			destination: output[0],
		},
	}
	input[0].out = synapses[0:2]
	input[1].out = synapses[2:4]
	hidden[0].in = synapses[0:2]
	hidden[1].in = synapses[2:4]
	hidden[0].out = synapses[4:5]
	hidden[1].out = synapses[5:6]

	for _, v := range synapses {
		fmt.Println(v)
	}
}
