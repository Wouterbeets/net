package main

import (
	"fmt"
	"math/rand"

	"github.com/MaxHalford/gago"
	"github.com/Wouterbeets/net"
)

type Predictor struct {
	*net.Net
}

func (p *Predictor) Evaluate() float64 {
	return 1
}
func (p *Predictor) Mutate(rng *rand.Rand)                        { gago.MutNormalFloat64(p.Net.DNA(), 0.8, rng) }
func (p *Predictor) Crossover(genome gago.Genome, rng *rand.Rand) {}
func (p *Predictor) Clone() gago.Genome                           { return nil }

func main() {
	fmt.Println("test")
}
