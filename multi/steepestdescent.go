package multi

import (
	"github.com/dane-unltd/linalg/mat"
	"github.com/dane-unltd/opt/uni"
)

type SteepestDescent struct {
	LineSearch uni.DerivSolver
}

func NewSteepestDescent() *SteepestDescent {
	s := &SteepestDescent{
		LineSearch: uni.DerivWrapper{uni.NewArmijo()},
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

	lineFunc := NewLineFuncDeriv(m.grad, m.X, d)
	lsInit := uni.NewSolution()
	lsParams := uni.NewParams()

	for ; m.Status == NotTerminated; m.update() {
		lsInit.SetX(s)
		lsInit.SetLB(0, m.ObjX, gLin)
		lsRes := sol.LineSearch.Solve(lineFunc, lsInit, lsParams)
		if lsRes.Status < 0 {
			m.Status = Status(lsRes.Status)
			break
		}
		s, m.ObjX = lsRes.X, lsRes.ObjX
		m.FunEvals += lsRes.FunEvals
		m.GradEvals += lsRes.DerivEvals

		m.X.Axpy(s, d)

		m.grad.ValGrad(m.X, m.GradX)
		m.FunEvals++
		m.GradEvals++
		d.Copy(m.GradX)
		d.Scal(-1)

		gLin = -d.Nrm2Sq()
	}
}
