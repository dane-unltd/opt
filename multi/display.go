package multi

import "fmt"

type Display struct {
	Period int
}

func NewDisplay(p int) *Display {
	return &Display{Period: p}
}

func (dsp *Display) Update(m *Model) Status {
	if m.Iter == 0 {
		fmt.Println("------------------------------------------------------")
		fmt.Println("Iter     Fun. Eval.  Time    Objective   Gradient-Norm")
		fmt.Println("------------------------------------------------------")
	}
	if dsp.Period <= 0 {
		dsp.Period = 1
	}
	if m.Iter%dsp.Period == 0 {
		fmt.Printf("%6d   %6d      %3.2f    %.2E    %.2E\n", m.Iter,
			m.FunEvals, m.Time.Seconds(), m.ObjX, m.gradNorm)
	}
	return 0
}
