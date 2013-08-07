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

	lineFun := func(s float64) float64 {
		xTemp.Copy(m.X)
		xTemp.Axpy(s, d)
		m.Proj.Project(xTemp)
		return m.Obj.Val(xTemp)
	}

	mls := uni.NewModel(lineFun, nil)

	for {
		if m.Status = m.update(); m.Status != 0 {
			break
		}

		mls.SetX(s)
		mls.SetLB(0, m.ObjX, gLin)
		mls.SetUB()
		lsStatus := sol.LineSearch.Solve(mls)
		if lsStatus < 0 {
			m.Status = Status(lsStatus)
			break
		}
		s, m.ObjX = mls.X, mls.ObjX

		m.X.Axpy(s, d)
		m.Proj.Project(m.X)

		m.grad.ValGrad(m.X, m.GradX)
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
