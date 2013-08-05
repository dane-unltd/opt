package multi

import (
	"errors"
	"github.com/dane-unltd/linalg/mat"
	"github.com/dane-unltd/opt/uni"
	"math"
	"time"
)

type SteepestDescent struct {
	TolAbs, TolRel float64
	IterMax        int
	TimeMax        time.Duration
	LineSearch     uni.Solver
	Disp           bool
}

func NewSteepestDescent() *SteepestDescent {
	s := &SteepestDescent{
		TolAbs:     1e-10,
		TolRel:     1e-10,
		IterMax:    10000,
		TimeMax:    10 * time.Second,
		LineSearch: uni.NewArmijo(),
		Disp:       true,
	}
	return s
}

func (sol *SteepestDescent) Solve(m *Model) error {
	var err error

	//for timing
	tStart := time.Now()

	s := 1.0 //initial step size

	if m.x == nil {
		m.x = mat.NewVec(m.n)
	}
	if math.IsNaN(m.objX) {
		m.objX = m.obj(m.x)
	}
	if m.gradX == nil {
		m.gradX = mat.NewVec(m.n)
	}
	m.grad(m.x, m.gradX)

	gLin := -m.gradX.Nrm2Sq()
	gLin0 := gLin

	d := mat.NewVec(m.n)
	d.Copy(m.gradX)
	d.Scal(-1)

	xTemp := mat.NewVec(m.n)

	lineFun := func(s float64) float64 {
		xTemp.Copy(m.x)
		xTemp.Axpy(s, d)
		return m.obj(xTemp)
	}
	mls := uni.NewModel(lineFun, nil)

	for ; m.iter < sol.IterMax; m.iter++ {
		m.time = time.Since(tStart)
		m.DoCallbacks()

		if m.time > sol.TimeMax {
			err = errors.New("Time limit reached")
			break
		}

		if math.Abs(gLin) < sol.TolAbs ||
			math.Abs(gLin/gLin0) < sol.TolRel {
			break
		}

		mls.SetX(s)
		mls.SetLB(0, m.objX, gLin)
		mls.SetUB()
		_ = sol.LineSearch.Solve(mls)
		s, m.objX = mls.X(), mls.ObjX()

		m.x.Axpy(s, d)

		m.grad(m.x, m.gradX)
		d.Copy(m.gradX)
		d.Scal(-1)

		gLin = -d.Nrm2Sq()
	}

	if m.iter == sol.IterMax {
		err = errors.New("Maximum number of iterations reached")
	}
	return err
}
