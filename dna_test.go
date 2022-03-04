package net

import (
	"testing"
)

func TestNetToDNA(t *testing.T) {
	in, hidden, out := 2, 2, 2
	n, err := NewBuilder().Build()
	if err != nil {
		t.Error(err)
	}
	dna := NetToDna(n)
	expected := in * hidden * out * 4
	if len(dna) != expected {
		t.Error("len dna unexpected", len(dna), expected)
	}
}
