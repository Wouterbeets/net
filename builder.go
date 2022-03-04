package net

import (
	"fmt"
	"math/rand"
	"time"
)

// Builder builds a neural net
type Builder struct {
	activationFunc func(float64) float64
	weightFunc     func() float64
	biasFunc       func() float64

	inSize     int
	outSize    int
	hiddenSize int

	dna []float64
}

var defaultWeightFunc func() float64
var defaultBiasFunc func() float64
var defaultActivationFunc func(float64) float64

func random() float64 {
	return rand.Float64()*2 - 1
}

func init() {
	defaultBiasFunc = random
	defaultWeightFunc = random
	defaultActivationFunc = sigmoid
	rand.Seed(time.Now().Unix())
}

// NewBuilder returns a network builder with default values.
// To customize; call its methods.
// Build() returns the configured network
func NewBuilder() *Builder { return defaultBuilder() }

func defaultBuilder() *Builder {
	return &Builder{
		inSize:         2,
		outSize:        2,
		hiddenSize:     2,
		activationFunc: defaultActivationFunc,
		weightFunc:     defaultWeightFunc,
		biasFunc:       defaultBiasFunc,
	}
}

// Build builds the network
func (b *Builder) Build() (*Net, error) {
	if b == nil {
		b = defaultBuilder()
	}
	if b.inSize < 1 {
		return nil, fmt.Errorf("inputSize should be > 1")
	}
	if b.outSize < 1 {
		return nil, fmt.Errorf("outputSize should be > 1")
	}
	if b.hiddenSize < 1 {
		return nil, fmt.Errorf("amountOfHiddenNeurons should be > 1")
	}
	if b.activationFunc == nil {
		b.activationFunc = defaultActivationFunc
	}
	if b.weightFunc == nil {
		b.weightFunc = defaultWeightFunc
	}
	if b.biasFunc == nil {
		b.biasFunc = defaultBiasFunc
	}
	n := newNet(b)
	return n, nil
}

// Builds the network from DNA
func (b *Builder) BuildFromDNA(dna DNA) (*Net, error) {
	if b == nil {
		b = defaultBuilder()
	}
	b.dna = dna
	if b.activationFunc == nil {
		b.activationFunc = defaultActivationFunc
	}
	if b.weightFunc == nil {
		b.weightFunc = defaultWeightFunc
	}
	if b.biasFunc == nil {
		b.biasFunc = defaultBiasFunc
	}
	//_ := newNet(b)
	return nil, fmt.Errorf("not implemented")
}

// ActivationFunc sets the activation func the neurons will use
func (b *Builder) ActivationFunc(f func(float64) float64) *Builder {
	b.activationFunc = f
	return b
}

// WeightFunc sets the func used to initialise the synapse's weight
func (b *Builder) WeightFunc(f func() float64) *Builder {
	b.weightFunc = f
	return b
}

// WeightFunc sets the func used to initialise the neuron's bias
func (b *Builder) BiasFunc(f func() float64) *Builder {
	b.biasFunc = f
	return b

}

// Size sets the net's input, output, and hidden layer size
func (b *Builder) Size(inputSize, hiddenSize, outputSize int) *Builder {
	b.inSize = inputSize
	b.hiddenSize = hiddenSize
	b.outSize = outputSize
	return b
}
