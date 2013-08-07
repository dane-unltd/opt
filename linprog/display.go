package linprog

import (
	"fmt"
	"github.com/dane-unltd/linalg/mat"
)

type Display struct {
	Period     int
	rd, rp, rs mat.Vec
}

func NewDisplay(f int) *Display {
	return &Display{
		Period: f,
	}
}

func (dsp *Display) Update(mdl *Model) Status {
	if dsp.rd == nil {
		m, n := mdl.A.Dims()
		dsp.rd = mat.NewVec(n)
		dsp.rp = mat.NewVec(m)
		dsp.rs = mat.NewVec(n)
	}
	if mdl.Iter == 0 {
		fmt.Println("-----------------------------------------------------------------")
		fmt.Println("Iter  Time    Dual-Feasibility Primal-Feasiblility Duality-Gap")
		fmt.Println("-----------------------------------------------------------------")
	}
	if dsp.Period <= 0 {
		dsp.Period = 1
	}
	if mdl.Iter%dsp.Period == 0 {
		At := mdl.A.TrView()

		dsp.rd.Sub(mdl.C, mdl.S)
		dsp.rd.AddMul(At, mdl.Y, -1)
		dsp.rp.Apply(mdl.A, mdl.X)
		dsp.rp.Sub(mdl.B, dsp.rp)
		dsp.rs.Mul(mdl.X, mdl.S)
		dsp.rs.Neg(dsp.rs)
		fmt.Printf("%3d   %3.2f    %.2E         %.2E            %.2E\n", mdl.Iter,
			mdl.Time.Seconds(), dsp.rd.Asum(), dsp.rp.Asum(), dsp.rs.Asum())
	}

	return NotTerminated
}
