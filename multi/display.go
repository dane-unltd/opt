package multi

import (
	"fmt"
	"github.com/gonum/blas/dblas"
)

type Display struct {
	Period int
}

func NewDisplay(p int) *Display {
	return &Display{Period: p}
}

func (dsp *Display) Update(r *Result) Status {
	gradNorm := nan
	if r.Grad != nil {
		gradNorm = dblas.Nrm2(dblas.NewVector(r.Grad))
	}
	if r.Iter == 0 {
		fmt.Println("------------------------------------------------------")
		fmt.Println("Iter     Fun. Eval.  Time    Objective   Gradient-Norm")
		fmt.Println("------------------------------------------------------")
	}
	if dsp.Period <= 0 {
		dsp.Period = 1
	}
	if r.Iter%dsp.Period == 0 {
		fmt.Printf("%6d   %6d      %3.2f    %.2E    %.2E\n", r.Iter,
			r.FunEvals, r.Time.Seconds(), r.Obj, gradNorm)
	}
	return 0
}
