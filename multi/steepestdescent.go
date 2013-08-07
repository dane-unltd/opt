package multi

import (
	"github.com/dane-unltd/linalg/mat"
	"github.com/dane-unltd/opt/uni"
	"math"
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

func (sol *SteepestDescent) Solve(m *Model) Status {
	var status Status

	//for timing

	s := 1.0 //initial step size

	if m.X == nil {
		m.X = mat.NewVec(m.N)
	}
	if math.IsNaN(m.ObjX) {
		m.ObjX = m.Obj(m.X)
	}
	if m.GradX == nil {
		m.GradX = mat.NewVec(m.N)
	}
	m.Grad(m.X, m.GradX)

	gLin := -m.GradX.Nrm2Sq()

	d := mat.NewVec(m.N)
	d.Copy(m.GradX)
	d.Scal(-1)

	xTemp := mat.NewVec(m.N)

	lineFun := func(s float64) float64 {
		xTemp.Copy(m.X)
		xTemp.Axpy(s, d)
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

		m.Grad(m.X, m.GradX)
		d.Copy(m.GradX)
		d.Scal(-1)

		gLin = -d.Nrm2Sq()
	}

	return status
}
