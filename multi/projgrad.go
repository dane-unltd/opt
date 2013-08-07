package multi

import (
	"errors"
	"fmt"
	"github.com/dane-unltd/linalg/mat"
	"github.com/dane-unltd/opt/uni"
	"math"
	"time"
)

type ProjGrad struct {
	TolAbs, TolRel float64
	IterMax        int
	TimeMax        time.Duration
	LineSearch     uni.Solver
}

func NewProjGrad() *ProjGrad {
	s := &ProjGrad{
		TolAbs:     1e-3,
		TolRel:     1e-3,
		IterMax:    10000,
		TimeMax:    10 * time.Second,
		LineSearch: uni.NewArmijo(),
	}
	return s
}

func (sol *ProjGrad) Solve(m *Model) error {
	var err error

	tStart := time.Now()

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
	gLin0 := gLin

	lineFun := func(s float64) float64 {
		xTemp.Copy(m.X)
		xTemp.Axpy(s, d)
		m.Proj(xTemp)
		return m.Obj(xTemp)
	}

	mls := uni.NewModel(lineFun, nil)

	for ; m.Iter < sol.IterMax; m.Iter++ {
		m.Time = time.Since(tStart)
		m.DoCallbacks()

		if m.Time > sol.TimeMax {
			err = errors.New("Time limit reached")
			break
		}

		if math.Abs(gLin) < sol.TolAbs ||
			math.Abs(gLin/gLin0) < sol.TolRel {
			break
		}

		mls.SetX(s)
		mls.SetLB(0, m.ObjX, gLin)
		mls.SetUB()
		status := sol.LineSearch.Solve(mls)
		if status < 0 {
			fmt.Println("Linesearch:", status)
			err = errors.New("Bad status in line search")
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

	if m.Iter == sol.IterMax {
		err = errors.New("Maximum number of iterations reached")
	}
	return err
}
