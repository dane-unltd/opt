package multi

import "fmt"

type Display struct {
	Period int
}

func NewDisplay(f int) *Display {
	return &Display{Period: f}
}

func (dsp *Display) Update(m *Model) {
	if m.Iter == 0 {
		fmt.Println("------------------------------------------")
		fmt.Println("Iter     Time    Objective   Gradient-Norm")
		fmt.Println("------------------------------------------")
	}
	if dsp.Period <= 0 {
		dsp.Period = 1
	}
	if m.Iter%dsp.Period == 0 {
		fmt.Printf("%6d   %3.2f    %.2E    %.2E\n", m.Iter,
			m.Time.Seconds(), m.ObjX, m.GradX.Nrm2())
	}
}
