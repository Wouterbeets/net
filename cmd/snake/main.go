package main

import (
	"fmt"
	"log"
	"math/rand"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/MaxHalford/eaopt"
	"github.com/Wouterbeets/net"
	"github.com/Wouterbeets/snake"
	"github.com/wouterbeets/term"
)

type Snake struct {
	*net.Net
	ID     snake.ID
	maxLen int
}

func (p *Snake) Size() float64 {
	dna := net.NetToDna(p.Net)
	l := len(dna.SynapseMap)
	l2 := len(dna.Neurons)
	return float64(l+l2) / 200
}

func (s *Snake) Play(g snake.GameState) snake.Move {
	vis := g.Vision(s.ID)
	life := g.Life(s.ID)
	in := make([]float64, len(vis))
	in = append(in, life)
	for i := range vis {
		in[i] = float64(vis[i])
	}
	out, err := s.Net.Eval(in)
	if err != nil {
		fmt.Println(err.Error())
	}
	return snake.Move{Move: out, ID: s.ID}
}

func (s *Snake) SetID(id snake.ID) {
	s.ID = id
}

func (s *Snake) Evaluate() (float64, error) {

	gameRounds := 3
	rounds := 1000
	var score float64
	for gr := 0; gr < gameRounds; gr++ {
		g, err := snake.NewGame(20, 20, []snake.Player{
			s,
		}, 1)
		if err != nil {
			fmt.Println(err.Error())
		}
		var snakeLen int
		var maxLen int
		for i := 0; i < rounds; i++ {
			snakeLen = g.PlayerLen(s.ID)
			if snakeLen > maxLen {
				maxLen = snakeLen
			}
			gameOver, _ := g.PlayRound()
			if gameOver || !g.Alive(s.ID) {
				score += (float64(i)/float64(rounds) + float64(maxLen))
				break
			}
			if i == rounds-1 {
				score += (float64(i)/float64(rounds) + float64(maxLen))
				score += float64(gr)
			}
		}

	}
	return -score, nil
}

func (s *Snake) Mutate(rng *rand.Rand) {
	dna := net.NetToDna(s.Net)

	dna.Mutate(rng)

	n, err := net.DNAToNet(dna)
	if err != nil {
		fmt.Println("unable to construct net, keeping old net")
		n = s.Net
	}
	s.Net = n
}

func (s *Snake) Crossover(genome eaopt.Genome, rng *rand.Rand) {
	// Get dna
	pDNA := net.NetToDna(s.Net)

	p2 := genome.(*Snake)
	p2DNA := net.NetToDna(p2.Net)

	pDNA.Crossover(p2DNA, rng)

	n, err := net.DNAToNet(pDNA)
	if err != nil {
		fmt.Println(err)
		return
	}
	s.Net = n
	n, err = net.DNAToNet(p2DNA)
	if err != nil {
		fmt.Println(err)
		return
	}
	p2.Net = n
	genome = p2
}

func (p *Snake) Clone() eaopt.Genome {
	dna := net.NetToDna(p.Net)
	dna2 := dna.Clone()
	n, _ := net.DNAToNet(dna2)
	return &Snake{Net: n}
}

func NewSnake(rng *rand.Rand) eaopt.Genome {
	n, err := net.NewBuilder().Size(22, 5, 3).Build()
	if err != nil {
		fmt.Println("error making new nn")
	}
	return &Snake{Net: n}
}

func sig(sigs chan os.Signal, ga *eaopt.GA) {
	<-sigs
	net.ToDot(ga.HallOfFame[0].Genome.(*Snake).Net)

	framerate := 50 * time.Millisecond
	sc := term.Screen{Input: make(chan [][]rune), UserInput: make(chan rune)}
	players := []snake.Player{
		&snake.Human{Input: sc.UserInput, Framerate: framerate},
	}
	for _, n := range ga.HallOfFame {
		players = append(players, n.Genome.(*Snake))
	}
	g, err := snake.NewGame(20, 20, players, 1)
	if err != nil {
		panic(err)
	}
	go sc.Run(framerate)

	runes := map[int8]rune{
		-1: 'M',
		0:  ' ',
		1:  '█',
		2:  'X',
	}
	for i := range players {
		runes[int8(i)+3] = '█'
	}
	for i := 0; i < 10000; i++ {
		gameOver, state := g.PlayRound()
		sc.Input <- stateToRune(state, runes)
		if gameOver {
			go sig(sigs, ga)
			return
		}
	}
	go sig(sigs, ga)
}

func main() {
	var ga, err = eaopt.NewDefaultGAConfig().NewGA()
	if err != nil {
		fmt.Println(err)
		return
	}

	f, err := os.Create("log.txt")
	if err != nil {
		fmt.Printf("unable to open file: %s", err.Error())
	}
	l := log.Default()
	l.SetOutput(f)
	ga.Logger = l

	sigs := make(chan os.Signal, 1)

	signal.Notify(sigs, syscall.SIGINT, syscall.SIGTERM)

	ga.PopSize = 1000
	ga.NPops = 10
	ga.NGenerations = 1000000
	ga.ParallelEval = true
	ga.HofSize = 3
	ga.Model = eaopt.ModDownToSize{
		NOffsprings: 2,
		SelectorA: eaopt.SelTournament{
			NContestants: 3,
		},
		SelectorB: eaopt.SelElitism{},
		MutRate:   1,
		CrossRate: 1,
	}
	ga.Migrator = eaopt.MigRing{NMigrants: 3}
	ga.MigFrequency = 10
	// Add a custom print function to track progress
	ga.Callback = func(ga *eaopt.GA) {
		var (
			distMin   float64
			distMax   float64
			distSum   float64
			popSum    float64
			popAvgSum float64
		)
		for _, p := range ga.Populations {
			pDist := p.Individuals.FitStd()
			distSum += pDist
			popSum += p.Individuals.FitMin()
			popAvgSum += p.Individuals.FitAvg()
			if pDist < distMin {
				distMin = pDist
			}
			if pDist > distMax {
				distMax = pDist
			}
		}
		log.Printf("gen: %d\thof: %.6f\tdistAvg: %.6f\tavgScore: %.6f",
			ga.Generations,
			ga.HallOfFame[0].Fitness,
			distSum/float64(len(ga.Populations)),
			popAvgSum/float64(len(ga.Populations)),
		)
	}
	ga.EarlyStop = func(ga *eaopt.GA) bool {
		if ga.HallOfFame[0].Fitness < -100000 {
			return true
		}
		return false
	}

	go sig(sigs, ga)
	err = ga.Minimize(NewSnake)
	if err != nil {
		fmt.Println(err)
	}
	net.ToDot(ga.HallOfFame[0].Genome.(*Snake).Net)

	framerate := 50 * time.Millisecond
	sc := term.Screen{Input: make(chan [][]rune), UserInput: make(chan rune)}
	players := []snake.Player{
		&snake.Human{Input: sc.UserInput, Framerate: framerate},
	}
	for _, n := range ga.HallOfFame {
		players = append(players, n.Genome.(*Snake))
	}
	g, err := snake.NewGame(20, 20, players, 1)
	if err != nil {
		panic(err)
	}
	go sc.Run(framerate)

	runes := map[int8]rune{
		-1: 'M',
		0:  ' ',
		1:  '█',
		2:  'X',
	}
	for i := range players {
		runes[int8(i)+3] = '█'
	}
	for i := 0; i < 100000; i++ {
		gameOver, state := g.PlayRound()
		sc.Input <- stateToRune(state, runes)
		if gameOver {
			return
		}
	}
}

func stateToRune(state snake.Board, runes map[int8]rune) (disp [][]rune) {
	disp = make([][]rune, len(state))
	for i := range disp {
		disp[i] = make([]rune, len(state[i]))
	}

	for y, row := range state {
		for x := range row {
			disp[y][x] = runes[state[y][x]]
		}
	}
	return disp
}
