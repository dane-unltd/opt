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

	xTemp := mat.NewVec(m.N)

	lineFun := func(s float64) float64 {
		xTemp.Copy(m.X)
		xTemp.Axpy(s, d)
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

		m.grad.ValGrad(m.X, m.GradX)
		d.Copy(m.GradX)
		d.Scal(-1)

		gLin = -d.Nrm2Sq()
	}
}
