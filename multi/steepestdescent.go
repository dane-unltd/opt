package multi

import (
	"github.com/dane-unltd/linalg/mat"
	"github.com/dane-unltd/opt/uni"
)

type SteepestDescent struct {
	LineSearch uni.Solver
}

func NewSteepestDescent() *SteepestDescent {
	s := &SteepestDescent{
		LineSearch: uni.NewArmijo(),
	}
	return s
}

func (sol *SteepestDescent) Solve(m *Model) {

	m.init(true, false)

	s := 1.0 //initial step size
	gLin := -m.GradX.Nrm2Sq()

	d := mat.NewVec(m.N)
	d.Copy(m.GradX)
	d.Scal(-1)

	mls := uni.NewModel(NewLineFuncDeriv(m.grad, m.X, d))

	for {
		if m.Status = m.update(); m.Status != 0 {
			break
		}

		mls.SetX(s)
		mls.SetLB(0, m.ObjX, gLin)
		mls.SetUB()
		sol.LineSearch.Solve(mls)
		if mls.Status < 0 {
			m.Status = Status(mls.Status)
			break
		}
		s, m.ObjX = mls.X, mls.ObjX

		m.X.Axpy(s, d)

		m.grad.ValGrad(m.X, m.GradX)
		d.Copy(m.GradX)
		d.Scal(-1)

		gLin = -d.Nrm2Sq()
	}
}
