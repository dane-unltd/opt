package linprog

import (
	"fmt"
)

type Display struct {
	Period int
}

func NewDisplay(f int) *Display {
	return &Display{
		Period: f,
	}
}

func (dsp *Display) Update(mdl *Model) Status {
	if mdl.Iter == 0 {
		fmt.Println("-----------------------------------------------------------------")
		fmt.Println("Iter  Time    Dual-Feasibility Primal-Feasiblility Duality-Gap")
		fmt.Println("-----------------------------------------------------------------")
	}
	if dsp.Period <= 0 {
		dsp.Period = 1
	}
	if mdl.Iter%dsp.Period == 0 {
		fmt.Printf("%3d   %3.2f    %.2E         %.2E            %.2E\n", mdl.Iter,
			mdl.Time.Seconds(), mdl.Rd.Asum(), mdl.Rp.Asum(), mdl.Rs.Asum())
	}

	return NotTerminated
}
