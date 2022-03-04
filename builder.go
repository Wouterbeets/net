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
	inSize         int
	outSize        int
	hiddenSize     int
}

// ActivationFunc sets with which activation func the net will be built
// the activation func is used when a neuron is activated to calculate its signal
// if this function is not called before Build() the net will be initialed with
// the default sigmoid function
func (b *Builder) ActivationFunc(f func(float64) float64) *Builder {
	b.activationFunc = f
	return b
}

// NewBuilder returns a network builder
// to customize; call its methods
// Build returns the configured network
func NewBuilder(inputSize, amountOfHiddenNeurons, outputSize int) *Builder {
	return &Builder{
		inSize:         inputSize,
		outSize:        outputSize,
		hiddenSize:     amountOfHiddenNeurons,
		activationFunc: sigmoid,
		weightFunc:     defaultWeightFunc,
	}
}

func (b *Builder) WeightFunc(f func() float64) *Builder {
	b.weightFunc = f
	return b
}

func (b *Builder) Build() (*Net, error) {
	if b == nil {
		b = &Builder{
			inSize:     2,
			hiddenSize: 2,
			outSize:    2,
		}
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
		b.activationFunc = sigmoid
	}
	if b.weightFunc == nil {
		rand.Seed(time.Now().Unix())
		b.weightFunc = defaultWeightFunc
	}
	n := newNet(b.inSize, b.hiddenSize, b.outSize, b.activationFunc, b.weightFunc)
	return n, nil
}

func defaultWeightFunc() float64 {
	return rand.Float64()*2 - 1
}
