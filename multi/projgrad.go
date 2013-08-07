package multi

import (
	"github.com/dane-unltd/linalg/mat"
	"github.com/dane-unltd/opt/uni"
	"math"
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

func (sol *ProjGrad) Solve(m *Model) Status {
	var status Status

	s := 1.0 //initial step size

	if m.X == nil {
		m.X = mat.NewVec(m.N)
	}
	m.Proj(m.X)

	if math.IsNaN(m.ObjX) {
		m.ObjX = m.Obj(m.X)
	}
	if m.GradX == nil {
		m.GradX = mat.NewVec(m.N)
	}
	m.Grad(m.X, m.GradX)

	d := mat.NewVec(m.N)
	d.Copy(m.GradX)
	d.Scal(-1)

	xTemp := mat.NewVec(m.N)

	xTemp.Copy(m.X)
	xTemp.Axpy(s/2, d)
	m.Proj(xTemp)
	xTemp.Sub(xTemp, m.X)
	xTemp.Scal(2 / s)

	gLin := -xTemp.Nrm2Sq()

	lineFun := func(s float64) float64 {
		xTemp.Copy(m.X)
		xTemp.Axpy(s, d)
		m.Proj(xTemp)
		return m.Obj(xTemp)
	}

	mls := uni.NewModel(lineFun, nil)

	m.init()

	for {
		if status = m.update(); status != 0 {
			break
		}

		mls.SetX(s)
		mls.SetLB(0, m.ObjX, gLin)
		mls.SetUB()
		lsStatus := sol.LineSearch.Solve(mls)
		if lsStatus < 0 {
			status = Status(lsStatus)
			break
		}
		s, m.ObjX = mls.X, mls.ObjX

		m.X.Axpy(s, d)
		m.Proj(m.X)

		m.Grad(m.X, m.GradX)
		d.Copy(m.GradX)
		d.Scal(-1)

		xTemp.Copy(m.X)
		xTemp.Axpy(s/2, d)
		m.Proj(xTemp)
		xTemp.Sub(xTemp, m.X)
		xTemp.Scal(2 / s)

		gLin = -xTemp.Nrm2Sq()
	}

	return status
}
