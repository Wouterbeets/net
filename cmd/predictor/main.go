package main

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/MaxHalford/eaopt"
	"github.com/Wouterbeets/net"
)

type Predictor struct {
	*net.Net
}

func (p Predictor) Size() float64 {
	dna := net.NetToDna(p.Net)
	l := len(dna.SynapseMap)
	l2 := len(dna.Neurons)
	return float64(l+l2) / 200
}

func scoreXOR(scores [][]float64) (score float64) {
	s := scores[0][0] - scores[0][1]
	if s > 0 {
		score -= 1
	}
	score -= s

	s = scores[1][0] - scores[1][1]
	if s < 0 {
		score -= 1
	}
	score += s

	s = scores[2][0] - scores[2][1]
	if s < 0 {
		score -= 1
	}
	score += s

	s = scores[3][0] - scores[3][1]
	if s > 0 {
		score -= 1
	}
	score -= s
	return
}

func (p *Predictor) Evaluate() (float64, error) {
	var scores [][]float64
	out, err := p.Net.Eval([]float64{0, 0})
	if err != nil {
		return math.MaxFloat64, nil
	}
	if len(out) != 2 {
		return math.MaxFloat64, nil
	}
	scores = append(scores, out)

	out, err = p.Net.Eval([]float64{0, 1})
	if err != nil {
		return math.MaxFloat64, nil
	}
	if len(out) != 2 {
		return math.MaxFloat64, nil
	}
	scores = append(scores, out)

	out, err = p.Net.Eval([]float64{1, 0})
	if err != nil {
		return math.MaxFloat64, nil
	}
	if len(out) != 2 {
		return math.MaxFloat64, nil
	}
	scores = append(scores, out)

	out, err = p.Net.Eval([]float64{1, 1})
	if err != nil {
		return math.MaxFloat64, nil
	}
	if len(out) != 2 {
		return math.MaxFloat64, nil
	}
	scores = append(scores, out)

	return scoreXOR(scores) + p.Size(), nil
}

func (p *Predictor) Mutate(rng *rand.Rand) {

	dna := net.NetToDna(p.Net)
	var weights []float64
	var sources []int
	var destinations []int
	var synapses []*net.SynapseGene
	for syn := range dna.SynapseMap {
		synapses = append(synapses, &net.SynapseGene{
			SourceID: syn.SourceID,
			DestID:   syn.DestID,
			DestBias: syn.DestBias,
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

	// Add synapses rareley
	if rng.Float64() < 0.01 {
		sourceID := rand.Intn(20)
		destID := rand.Intn(20)

		sg := net.SynapseGene{
			SourceID: sourceID,
			DestID:   destID,
			Weight:   rand.Float64(),
		}
		dna.SynapseMap[sg] = struct{}{}
	}

	// remove synapses rareley
	if rng.Float64() < 0.01 && len(dna.SynapseMap) > 2 {
		for key := range dna.SynapseMap {
			delete(dna.SynapseMap, key)
			break
		}
	}

	n, err := net.DNAToNet(dna)
	if err != nil {
		fmt.Println("unable to construct net, keeping old net")
		n = p.Net
	}
	p.Net = n
}

func (p *Predictor) Crossover(genome eaopt.Genome, rng *rand.Rand) {
	// Get dna
	pDNA := net.NetToDna(p.Net)
	p2 := genome.(*Predictor)
	p2DNA := net.NetToDna(p2.Net)
	if len(p2DNA.SynapseMap) != len(pDNA.SynapseMap) {
		return
	}

	var pSynapses net.Syns
	for syn := range pDNA.SynapseMap {
		pSynapses = append(pSynapses, &net.SynapseGene{
			SourceID: syn.SourceID,
			DestID:   syn.DestID,
			DestBias: syn.DestBias,
			Weight:   syn.Weight,
		})
	}
	var p2Synapses net.Syns
	for syn := range p2DNA.SynapseMap {
		p2Synapses = append(p2Synapses, &net.SynapseGene{
			SourceID: syn.SourceID,
			DestID:   syn.DestID,
			DestBias: syn.DestBias,
			Weight:   syn.Weight,
		})
	}

	eaopt.CrossGNX(pSynapses, p2Synapses, 1, rng)

	pDNA.SynapseMap = make(map[net.SynapseGene]struct{}, len(pSynapses))
	for _, g := range pSynapses {
		pDNA.SynapseMap[*g] = struct{}{}
	}

	p2DNA.SynapseMap = make(map[net.SynapseGene]struct{}, len(p2Synapses))
	for _, g := range p2Synapses {
		p2DNA.SynapseMap[*g] = struct{}{}
	}

	n, err := net.DNAToNet(pDNA)
	if err != nil {
		fmt.Println(err)
		return
	}
	p.Net = n
	n, err = net.DNAToNet(p2DNA)
	if err != nil {
		fmt.Println(err)
		return
	}
	p2.Net = n
	genome = p2
}

func (p *Predictor) Clone() eaopt.Genome {
	dna := net.NetToDna(p.Net)
	n, _ := net.DNAToNet(dna)
	return &Predictor{Net: n}
}

func NewPredictor(rng *rand.Rand) eaopt.Genome {
	n, err := net.NewBuilder().Size(2, 1, 2).Build()
	if err != nil {
		fmt.Println("error making new nn")
	}
	return &Predictor{Net: n}
}

func main() {

	var ga, err = eaopt.NewDefaultGAConfig().NewGA()
	if err != nil {
		fmt.Println(err)
		return
	}

	ga.PopSize = 50
	ga.NPops = 50
	ga.NGenerations = 3500
	ga.ParallelEval = true
	ga.HofSize = 10
	// Add a custom print function to track progress
	ga.Callback = func(ga *eaopt.GA) {
		fmt.Printf("Best fitness at generation %d: %.30f\n", ga.Generations, ga.HallOfFame[0].Fitness)
	}
	ga.EarlyStop = func(ga *eaopt.GA) bool {
		if ga.HallOfFame[0].Fitness < -42 {
			return true
		}
		return false
	}

	err = ga.Minimize(NewPredictor)
	if err != nil {
		fmt.Println(err)
	}
	n := ga.HallOfFame[0].Genome.(*Predictor).Net
	net.ToDot(n)
	dna := net.NetToDna(n)
	n, _ = net.DNAToNet(dna)
	fmt.Println(n.Eval([]float64{0, 0}))
	fmt.Println(n.Eval([]float64{0, 1}))
	fmt.Println(n.Eval([]float64{1, 0}))
	fmt.Println(n.Eval([]float64{1, 1}))

}
