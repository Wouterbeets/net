package net

import (
	"math/rand"
	"sort"

	"github.com/MaxHalford/eaopt"
)

// DNA encodes a network
type DNA struct {
	SynapseMap map[SynapseGene]struct{}
	Neurons    []*NeuronGene
}

type NeuronGene struct {
	ID    int
	Bias  float64
	Layer int
}

type SynapseGene struct {
	SourceID int
	DestID   int
	Weight   float64
}

// NetToDna encodes the network to dna
func NetToDna(n *Net) (dna DNA) {
	dna.SynapseMap = make(map[SynapseGene]struct{})
	for _, neur := range n.neuronStore {
		dna.Neurons = append(dna.Neurons, neur.DNA())
	}
	for s := range n.synapses() {
		dna.SynapseMap[*s.DNA()] = struct{}{}
	}
	return dna
}

func DNAToNet(dna DNA) (*Net, error) {
	n := new(Net)
	n.neuronStore = make(map[int]*neuron)

	{ // Create neurons and add to layers
		for _, neurGene := range dna.Neurons {
			neur := &neuron{
				id:    neurGene.ID,
				layer: byte(neurGene.Layer),
				bias:  neurGene.Bias,
			}
			n.neuronStore[neurGene.ID] = neur
			switch neur.layer {
			case inputLayer:
				n.in = append(n.in, neur)
			case hiddenLayer:
				n.hidden = append(n.hidden, neur)
			case outputLayer:
				n.out = append(n.out, neur)
			}
		}
	}

	{ // sort layers
		sort.Slice(n.in, func(i, j int) bool { return n.in[i].id < n.in[j].id })
		sort.Slice(n.out, func(i, j int) bool { return n.out[i].id < n.out[j].id })
		sort.Slice(n.hidden, func(i, j int) bool { return n.hidden[i].id < n.hidden[j].id })
	}

	{ // add synapes to neurons
		for synGene := range dna.SynapseMap {
			var source *neuron
			var dest *neuron
			var ok bool

			// if source neuron from synapse doesn't exist; create it
			if source, ok = n.neuronStore[synGene.SourceID]; !ok {
				source = &neuron{
					id:    synGene.SourceID,
					layer: hiddenLayer,
					bias:  defaultBiasFunc(),
				}
				n.neuronStore[source.id] = source
			}

			// if dest neuron from synapse doesn't exist; create it
			if dest, ok = n.neuronStore[synGene.DestID]; !ok {
				dest = &neuron{
					id:    synGene.DestID,
					layer: hiddenLayer,
					bias:  defaultBiasFunc(),
				}
				n.neuronStore[dest.id] = dest

			}

			s := &synapse{
				source:      source,
				destination: dest,
				weight:      synGene.Weight,
			}

			source.out = append(source.out, s)
			dest.in = append(dest.in, s)
		}
	}

	return n, nil
}

type Syns []*SynapseGene

func (g Syns) At(i int) interface{} {
	return g[i]
}

func (g Syns) Set(i int, v interface{}) {
	g[i] = v.(*SynapseGene)
}

func (g Syns) Len() int {
	return len(g)
}

func (g Syns) Swap(i, j int) {
	g[i], g[j] = g[j], g[i]
}

func (g Syns) Slice(a, b int) eaopt.Slice {
	return g[a:b]
}

func (g Syns) Split(k int) (eaopt.Slice, eaopt.Slice) {
	return g[:k], g[k:]
}

func (g Syns) Append(t eaopt.Slice) eaopt.Slice {
	return append(g, t.(Syns)...)
}

func (g Syns) Replace(t eaopt.Slice) {
	copy(g, t.(Syns))
}

func (g Syns) Copy() eaopt.Slice {
	var t = make(Syns, len(g))
	copy(t, g)
	return t
}

func (dna DNA) Mutate(rng *rand.Rand) {
	var weights []float64
	var sources []int
	var destinations []int
	var synapses []*SynapseGene
	for syn := range dna.SynapseMap {
		synapses = append(synapses, &SynapseGene{
			SourceID: syn.SourceID,
			DestID:   syn.DestID,
			Weight:   syn.Weight,
		})
	}

	for _, syn := range synapses {
		weights = append(weights, syn.Weight)
		sources = append(sources, syn.SourceID)
		destinations = append(destinations, syn.DestID)
	}

	var biases []float64
	for _, neur := range dna.Neurons {
		biases = append(biases, neur.Bias)
	}

	eaopt.MutNormalFloat64(weights, 0.1, rng)
	eaopt.MutNormalFloat64(biases, 0.1, rng)

	// Switch connections
	if rng.Float64() < 0.01 {
		eaopt.MutPermuteInt(sources, 1, rng)
		eaopt.MutPermuteInt(destinations, 1, rng)
	}

	for i, syn := range synapses {
		syn.Weight = weights[i]
		syn.SourceID = sources[i]
		syn.DestID = destinations[i]
	}

	for i, neur := range dna.Neurons {
		neur.Bias = biases[i]
	}

	// Add synapses
	if rng.Float64() < 0.9 {
		sourceID := rand.Intn(len(dna.Neurons) + 1)
		destID := rand.Intn(len(dna.Neurons) + 1)

		sg := SynapseGene{
			SourceID: sourceID,
			DestID:   destID,
			Weight:   rand.Float64(),
		}
		dna.SynapseMap[sg] = struct{}{}
	}

	// remove synapses
	if rng.Float64() < 0.01 && len(dna.SynapseMap) > 2 {
		for key := range dna.SynapseMap {
			delete(dna.SynapseMap, key)
			break
		}
	}
}

func (dna DNA) Crossover(dna2 DNA, rng *rand.Rand) {
	if len(dna2.SynapseMap) != len(dna.SynapseMap) {
		return
	}

	var pSynapses Syns
	for syn := range dna.SynapseMap {
		pSynapses = append(pSynapses, &SynapseGene{
			SourceID: syn.SourceID,
			DestID:   syn.DestID,
			Weight:   syn.Weight,
		})
	}
	var p2Synapses Syns
	for syn := range dna2.SynapseMap {
		p2Synapses = append(p2Synapses, &SynapseGene{
			SourceID: syn.SourceID,
			DestID:   syn.DestID,
			Weight:   syn.Weight,
		})
	}
	sort.Slice(pSynapses, func(i, j int) bool {
		if pSynapses[i].SourceID == pSynapses[j].SourceID {
			if pSynapses[i].DestID == pSynapses[j].DestID {
				return pSynapses[i].Weight < pSynapses[j].Weight
			} else {
				return pSynapses[i].DestID < pSynapses[j].DestID
			}
		} else {
			return pSynapses[i].SourceID < pSynapses[j].SourceID
		}
	})
	sort.Slice(p2Synapses, func(i, j int) bool {
		if p2Synapses[i].SourceID == p2Synapses[j].SourceID {
			if p2Synapses[i].DestID == p2Synapses[j].DestID {
				return p2Synapses[i].Weight < p2Synapses[j].Weight
			} else {
				return p2Synapses[i].DestID < p2Synapses[j].DestID
			}
		} else {
			return p2Synapses[i].SourceID < p2Synapses[j].SourceID
		}
	})

	eaopt.CrossGNX(pSynapses, p2Synapses, 1, rng)

	dna.SynapseMap = make(map[SynapseGene]struct{}, len(pSynapses))
	for _, g := range pSynapses {
		dna.SynapseMap[*g] = struct{}{}
	}

	dna2.SynapseMap = make(map[SynapseGene]struct{}, len(p2Synapses))
	for _, g := range p2Synapses {
		dna2.SynapseMap[*g] = struct{}{}
	}
}

func (dna DNA) Clone() DNA {
	var dna2 DNA
	dna2.SynapseMap = make(map[SynapseGene]struct{}, len(dna.SynapseMap))
	for g := range dna.SynapseMap {
		dna2.SynapseMap[g] = struct{}{}
	}

	dna2.Neurons = make([]*NeuronGene, 0, len(dna.Neurons))
	for _, ng := range dna.Neurons {
		dna2.Neurons = append(dna2.Neurons, &NeuronGene{
			ID:    ng.ID,
			Bias:  ng.Bias,
			Layer: ng.Layer,
		})
	}

	return dna2
}
