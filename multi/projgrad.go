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
		TolAbs:     1e-6,
		TolRel:     1e-6,
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

	if m.x == nil {
		m.x = mat.NewVec(m.n)
	}
	m.proj(m.x)

	if math.IsNaN(m.objX) {
		m.objX = m.obj(m.x)
	}
	if m.gradX == nil {
		m.gradX = mat.NewVec(m.n)
	}
	m.grad(m.x, m.gradX)

	d := mat.NewVec(m.n)
	d.Copy(m.gradX)
	d.Scal(-1)

	xTemp := mat.NewVec(m.n)

	xTemp.Copy(m.x)
	xTemp.Axpy(s/2, d)
	m.proj(xTemp)
	xTemp.Sub(xTemp, m.x)
	xTemp.Scal(2 / s)

	gLin := -xTemp.Nrm2Sq()
	gLin0 := gLin

	lineFun := func(s float64) float64 {
		xTemp.Copy(m.x)
		xTemp.Axpy(s, d)
		m.proj(xTemp)
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
		err := sol.LineSearch.Solve(mls)
		if err != nil {
			fmt.Println(err)
		}
		s, m.objX = mls.X(), mls.ObjX()

		m.x.Axpy(s, d)
		m.proj(m.x)

		m.grad(m.x, m.gradX)
		d.Copy(m.gradX)
		d.Scal(-1)

		xTemp.Copy(m.x)
		xTemp.Axpy(s/2, d)
		m.proj(xTemp)
		xTemp.Sub(xTemp, m.x)
		xTemp.Scal(2 / s)

		gLin = -xTemp.Nrm2Sq()
	}

	if m.iter == sol.IterMax {
		err = errors.New("Maximum number of iterations reached")
	}
	return err
}
