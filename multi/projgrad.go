package multi

import (
	"github.com/dane-unltd/linalg/mat"
	"github.com/dane-unltd/opt/uni"
)

type ProjGrad struct {
	LineSearch uni.Solver
}

func NewProjGrad() *ProjGrad {
	s := &ProjGrad{
		LineSearch: uni.NewArmijo(),
	}
	return s
}

func (sol *ProjGrad) Solve(m *Model) {

	m.init(true, false)

	s := 1.0 //initial step size

	d := mat.NewVec(m.N)
	d.Copy(m.GradX)
	d.Scal(-1)

	xTemp := mat.NewVec(m.N)

	xTemp.Copy(m.X)
	xTemp.Axpy(s/2, d)
	m.Proj.Project(xTemp)
	xTemp.Sub(xTemp, m.X)
	xTemp.Scal(2 / s)

	gLin := -xTemp.Nrm2Sq()

	mls := uni.NewModel(NewLineFuncProj(m.grad, m.Proj, m.X, d))

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
		m.FunEvals += mls.FunEvals
		m.GradEvals += mls.DerivEvals

		m.X.Axpy(s, d)
		m.Proj.Project(m.X)

		m.grad.ValGrad(m.X, m.GradX)
		m.FunEvals++
		m.GradEvals++
		d.Copy(m.GradX)
		d.Scal(-1)

		xTemp.Copy(m.X)
		xTemp.Axpy(s/2, d)
		m.Proj.Project(xTemp)
		xTemp.Sub(xTemp, m.X)
		xTemp.Scal(2 / s)

		gLin = -xTemp.Nrm2Sq()
	}
}
