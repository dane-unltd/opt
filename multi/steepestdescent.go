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

	x := m.X
	n := len(x)

	if math.IsNaN(m.ObjX) {
		m.ObjX = m.Obj(x)
	}
	if m.GradX == nil {
		m.GradX = mat.NewVec(n)
	}
	g := m.GradX
	m.Grad(x, g)

	gLin := -g.Nrm2Sq()
	gLin0 := gLin

	d := mat.NewVec(n)
	d.Copy(g)
	d.Scal(-1)

	xTemp := mat.NewVec(n)

	lineFun := func(s float64) float64 {
		xTemp.Copy(x)
		xTemp.Axpy(s, d)
		return m.Obj(xTemp)
	}

	for ; m.Iter < sol.IterMax; m.Iter++ {

		m.Time = time.Since(tStart)

		if m.callback != nil {
			m.callback(m)
		}

		if m.Time > sol.TimeMax {
			err = errors.New("Time limit reached")
			break
		}

		if math.Abs(gLin) < sol.TolAbs ||
			math.Abs(gLin/gLin0) < sol.TolRel {
			break
		}

		mls := uni.NewModel(lineFun, nil)
		mls.ObjLB, mls.DerivLB, mls.X = m.ObjX, gLin, s
		_ = sol.LineSearch.Solve(mls)
		s, m.ObjX = mls.X, mls.ObjX

		x.Axpy(s, d)

		m.Grad(x, g)
		d.Copy(g)
		d.Scal(-1)

		gLin = -d.Nrm2Sq()
	}

	if m.Iter == sol.IterMax {
		err = errors.New("Maximum number of iterations reached")
	}
	return err
}
