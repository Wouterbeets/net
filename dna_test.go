package net

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNetToDNA(t *testing.T) {
	in, hidden, out := 2, 2, 2
	n, err := NewBuilder().Size(in, hidden, out).Build()
	if err != nil {
		t.Error(err)
	}
	dna := NetToDna(n)
	expected := in * hidden * out * 4
	if len(dna.Synapes) != expected {
		t.Error("len dna unexpected", len(dna.Synapes), expected)
	}
}

func TestDNAToNet(t *testing.T) {
	inSize, hiddenSize, outSize := 2, 20, 20
	n, err := NewBuilder().Size(inSize, hiddenSize, outSize).Build()
	assert.NoError(t, err)

	dna := NetToDna(n)
	n2, err := DNAToNet(dna)
	assert.NoError(t, err)

	assert.Equal(t, len(n.in), len(n2.in))
	assert.Equal(t, len(n.hidden), len(n2.hidden))
	assert.Equal(t, len(n.out), len(n2.out))
	assert.Equal(t, len(n.neuronStore), len(n2.neuronStore))
	assert.Equal(t, len(n.synapses()), len(n2.synapses()))

	out, err := n.Eval([]float64{1, 1})
	assert.NoError(t, err)
	out2, err2 := n2.Eval([]float64{1, 1})
	require.NoError(t, err2)

	assert.Equal(t, out, out2)
}

func TestDNAToNet_with_dangling_neuron_connects(t *testing.T) {
	inSize, hiddenSize, outSize := 2, 2, 2
	n, err := NewBuilder().Size(inSize, hiddenSize, outSize).Build()
	assert.NoError(t, err)

	dna := NetToDna(n)
	dna.Synapes = append(dna.Synapes, SynapseGene{
		SourceID: 10,
		DestID:   11,
		Weight:   0,
		DestBias: 0,
	})
	n2, err := DNAToNet(dna)
	assert.NoError(t, err)
	out, err := n2.Eval([]float64{1, 1})
	assert.NoError(t, err)
	assert.Equal(t, len(out), 2)
}

func TestDNAToNet_with_negative_neuron_ids(t *testing.T) {
	inSize, hiddenSize, outSize := 2, 2, 2
	n, err := NewBuilder().Size(inSize, hiddenSize, outSize).Build()
	assert.NoError(t, err)

	dna := NetToDna(n)
	dna.Synapes = append(dna.Synapes, SynapseGene{
		SourceID: -1,
		DestID:   -2,
		Weight:   0,
		DestBias: 0,
	})
	n2, err := DNAToNet(dna)
	assert.NoError(t, err)
	out, err := n2.Eval([]float64{1, 1})
	assert.NoError(t, err)
	assert.Equal(t, len(out), 2)
}

func TestDNAToNet_with_single_synapse_loop(t *testing.T) {
	inSize, hiddenSize, outSize := 2, 2, 2
	n, err := NewBuilder().Size(inSize, hiddenSize, outSize).Build()
	assert.NoError(t, err)

	dna := NetToDna(n)
	dna.Synapes = append(dna.Synapes, SynapseGene{
		SourceID: 10,
		DestID:   10,
		Weight:   0,
		DestBias: 0,
	})
	n2, err := DNAToNet(dna)
	toDot(n2)
	assert.NoError(t, err)
	out, err := n2.Eval([]float64{1, 1})
	assert.NoError(t, err)
	assert.Equal(t, len(out), 2)
}
