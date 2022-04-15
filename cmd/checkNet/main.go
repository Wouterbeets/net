package main

import (
	"encoding/json"
	"fmt"

	"github.com/Wouterbeets/net"
)

func main() {
	dna := `{"Synapes":[{"SourceID":0,"DestID":2,"Weight":4.65408944715481,"DestBias":-2.0882396631131885},{"SourceID":0,"DestID":3,"Weight":-0.1994736181883809,"DestBias":-0.17295770298143875},{"SourceID":1,"DestID":2,"Weight":-8.100940898862731,"DestBias":-2.0882396631131885},{"SourceID":1,"DestID":3,"Weight":0.06677530695882042,"DestBias":-0.17295770298143875},{"SourceID":2,"DestID":4,"Weight":35.09185850180299,"DestBias":1.127770825189975},{"SourceID":2,"DestID":5,"Weight":-103.74968948396369,"DestBias":0.5805751237601424},{"SourceID":3,"DestID":4,"Weight":-5.363911198527256,"DestBias":1.127770825189975},{"SourceID":3,"DestID":5,"Weight":-1.7299536332932783,"DestBias":0.5805751237601424}],"Neurons":[{"ID":1,"Bias":1.2840821563901317,"Layer":0},{"ID":2,"Bias":-2.0882396631131885,"Layer":1},{"ID":3,"Bias":-0.17295770298143875,"Layer":1},{"ID":4,"Bias":1.127770825189975,"Layer":2},{"ID":5,"Bias":0.5805751237601424,"Layer":2},{"ID":19,"Bias":-1.4009924470659767,"Layer":1},{"ID":9,"Bias":0.004012870957972425,"Layer":1},{"ID":0,"Bias":0.00011322294782590787,"Layer":0}]}`
	var testNetDna net.DNA

	json.Unmarshal([]byte(dna), &testNetDna)
	n, err := net.DNAToNet(testNetDna)
	if err != nil {
		panic("aaaa")
	}
	out, err := n.Eval([]float64{1, 1})
	if err != nil {
		panic("aaaa")
	}
	fmt.Println(out)
	out, err = n.Eval([]float64{1, 0})
	if err != nil {
		panic("aaaa")
	}
	fmt.Println(out)
	out, err = n.Eval([]float64{0, 1})
	if err != nil {
		panic("aaaa")
	}
	fmt.Println(out)
	out, err = n.Eval([]float64{0, 0})
	if err != nil {
		panic("aaaa")
	}
	fmt.Println(out)
	net.ToDot(n)
}
