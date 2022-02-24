package net

import (
	"testing"
)

func TestNewNet(t *testing.T) {
	inSize := 2
	outSize := 20
	net := NewNet(inSize, outSize)
	if len(net.in) != inSize {
		t.Error("len of in not", inSize)
	}
	if len(net.out) != outSize {
		t.Error("len of out not", inSize)
	}
	out := net.eval([]float64{1, 2})
	if out[0] != 0 {
		t.Error("net not returning default value")
	}
	if out[1] != 0 {
		t.Error("net not returning default value")
	}
	toDot(net)
}
