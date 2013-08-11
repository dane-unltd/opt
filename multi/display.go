package multi

import "fmt"
import "math"

type Display struct {
	Period int
}

func NewDisplay(p int) *Display {
	return &Display{Period: p}
}

func (dsp *Display) Update(r *Result) Status {
	gradNorm := math.NaN()
	if r.GradX != nil {
		gradNorm = r.GradX.Nrm2()
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
			r.FunEvals, r.Time.Seconds(), r.ObjX, gradNorm)
	}
	return 0
}
