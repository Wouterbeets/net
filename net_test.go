package net

import (
	"testing"
)

func TestNewNet(t *testing.T) {
	inSize := 2
	outSize := 20
	hiddenSize := 20
	net := NewNet(inSize, hiddenSize, outSize, simple)
	if len(net.in) != inSize {
		t.Error("len of in not", inSize)
	}
	if len(net.out) != outSize {
		t.Error("len of out not", outSize)
	}
	out, err := net.Eval([]float64{1, 2})
	if err != nil {
		t.Error(err)
	}
	if len(out) != outSize {
		t.FailNow()
	}
	//	toDot(net)
}

func TestAddNeuron(t *testing.T) {
	n := NewNet(2, 2, 2, simple)
	before := n.in[0].bias
	n.store[0].bias = 200
	after := n.in[0].bias
	if before == after {
		t.Errorf("modifying store does not modify slice")
	}
}

func simple(f float64) float64 {
	return f
}

func TestMem(t *testing.T) {
	n := NewNet(2, 2, 2, simple)
	n.addNeuron(&neuron{
		id:             6,
		layer:          hiddenLayer,
		bias:           1,
		activationFunc: simple,
	})
	n.addNeuron(&neuron{
		id:             7,
		layer:          hiddenLayer,
		bias:           1,
		activationFunc: simple,
	})
	n.addSynapse(3, 6, 1)
	n.addSynapse(6, 5, 1)
	n.addSynapse(6, 7, 1)
	n.addSynapse(7, 3, 1)
	out, err := n.Eval([]float64{1, 1})
	if err != nil {
		t.Error(err)
	}
	if out == nil {
		t.FailNow()
	}
	if out[0] != 4 {
		t.FailNow()
	}
	if out[1] != 4 {
		t.FailNow()
	}
	out, err = n.Eval([]float64{1, 1})
	if err != nil {
		t.Error(err)
	}
	if out == nil {
		t.FailNow()
	}
	if out[0] != 6 {
		t.FailNow()
	}
	if out[1] != 8 {
		t.FailNow()
	}
}

func TestSimpleMem(t *testing.T) {
	n := NewNet(2, 2, 2, simple)
	n.addNeuron(&neuron{
		id:             6,
		layer:          hiddenLayer,
		bias:           1,
		activationFunc: simple,
	})
	n.addSynapse(3, 6, 1)
	n.addSynapse(6, 3, 1)
	out, err := n.Eval([]float64{1, 1})
	if err != nil {
		t.Error(err)
	}
	if out == nil {
		t.FailNow()
	}
	if out[0] != 4 {
		t.Errorf("out should be 4")
	}
	if out[1] != 4 {
		t.Errorf("out should be 4")
	}
	out, err = n.Eval([]float64{1, 1})
	if err != nil {
		t.Error(err)
	}
	if out == nil {
		t.Errorf("out is nil")
	}
	if out[0] != 6 {
		t.Errorf("out should be 6")
	}
	if out[1] != 6 {
		t.Errorf("out should be 6")
	}
}
