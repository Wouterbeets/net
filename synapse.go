package net

import "fmt"

type synapse struct {
	source      *neuron
	destination *neuron
	weight      float64
	memory      *signal
}

func (s *synapse) eval(id int) signal {
	if s.source != nil &&
		s.source.visited &&
		s.source.memory != nil &&
		s.source.memory.id != id {
		fmt.Println("neuron source has memory, returning:", s.source.id, s.source.memory)
		return signal{v: s.source.memory.v * s.weight, id: s.source.memory.id}
	}
	if s.source.visited {
		fmt.Println("loop detected", s.source.id)
		s.source.shouldSaveMemory = true
		return signal{id: id}
	}
	fmt.Println("no loop, calling next neuron eval", s.source.id)
	s.source.visited = true
	sig := s.source.eval2(id)
	s.source.visited = false
	return signal{v: sig.v * s.weight, id: id}
}
