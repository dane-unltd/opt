package multi

import (
	"fmt"
	"github.com/gonum/blas/dbw"
)

type Display struct {
	Period int
}

func NewDisplay(p int) *Display {
	return &Display{Period: p}
}

func (dsp *Display) Update(sol *Solution, stats *Stats) Status {
	gradNorm := nan
	if sol.Grad != nil {
		gradNorm = dbw.Nrm2(dbw.NewVector(sol.Grad))
	}
	if stats.Iter == 0 {
		fmt.Println("------------------------------------------------------")
		fmt.Println("Iter     Fun. Eval.  Time    Objective   Gradient-Norm")
		fmt.Println("------------------------------------------------------")
	}
	if dsp.Period <= 0 {
		dsp.Period = 1
	}
	if stats.Iter%dsp.Period == 0 {
		fmt.Printf("%6d   %6d      %3.2f    %.2E    %.2E\n", stats.Iter,
			stats.FunEvals, stats.Time.Seconds(), sol.Obj, gradNorm)
	}
	return 0
}
