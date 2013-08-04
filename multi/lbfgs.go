package multi

import (
	"errors"
	"github.com/dane-unltd/linalg/mat"
	"github.com/dane-unltd/opt/uni"
	"math"
	"time"
)

type LBFGS struct {
	Tol        float64
	IterMax    int
	TimeMax    time.Duration
	Mem        int
	LineSearch uni.Solver
}

func NewLBFGS() *LBFGS {
	s := &LBFGS{
		Tol:        1e-6,
		IterMax:    1000,
		TimeMax:    time.Minute,
		Mem:        5,
		LineSearch: uni.NewQuadratic(),
	}
	return s
}

func (sol LBFGS) Solve(m *Model) error {
	var err error

	startT := time.Now()

	stepSize := 1.0
	x := m.X
	n := len(x)

	if math.IsNaN(m.ObjX) {
		m.ObjX = m.Obj(x)
	}

	if m.GradX == nil {
		m.GradX = mat.NewVec(n)
		m.Grad(x, m.GradX)
	}
	g := m.GradX

	gLin := 0.0

	S := make([]mat.Vec, sol.Mem)
	Y := make([]mat.Vec, sol.Mem)
	for i := 0; i < sol.Mem; i++ {
		S[i] = mat.NewVec(n)
		Y[i] = mat.NewVec(n)
	}

	d := mat.NewVec(n)

	xOld := mat.NewVec(n)
	gOld := mat.NewVec(n)
	sNew := mat.NewVec(n)
	yNew := mat.NewVec(n)

	alphas := mat.NewVec(sol.Mem)
	betas := mat.NewVec(sol.Mem)
	rhos := mat.NewVec(sol.Mem)

	xTemp := mat.NewVec(n)
	lineFun := func(step float64) float64 {
		xTemp.Copy(x)
		xTemp.Axpy(step, d)
		return m.Obj(xTemp)
	}

	for ; m.Iter < sol.IterMax; m.Iter++ {

		d.Copy(g)
		if m.Iter > 0 {
			yNew.Sub(g, gOld)
			sNew.Sub(x, xOld)

			temp := S[len(S)-1]
			copy(S[1:], S)
			S[0] = temp
			S[0].Copy(sNew)

			temp = Y[len(S)-1]
			copy(Y[1:], Y)
			Y[0] = temp
			Y[0].Copy(yNew)

			copy(rhos[1:], rhos)
			rhos[0] = 1 / mat.Dot(sNew, yNew)
			for i := 0; i < sol.Mem; i++ {
				alphas[i] = rhos[i] * mat.Dot(S[i], d)
				d.Axpy(-alphas[i], Y[i])
			}
			for i := sol.Mem - 1; i >= 0; i-- {
				betas[i] = rhos[i] * mat.Dot(Y[i], d)
				d.Axpy(alphas[i]-betas[i], S[i])
			}
		}

		d.Scal(-1)

		gLin = mat.Dot(d, g)

		m.Time = time.Since(startT)
		if m.callback != nil {
			m.callback(m)
		}

		if m.Time > sol.TimeMax {
			err = errors.New("Time limit reached")
		}
		if gLin > -sol.Tol {
			break
		}

		mls := uni.NewModel(lineFun, nil)
		mls.ObjLB, mls.DerivLB, mls.X = m.ObjX, gLin, stepSize
		_ = sol.LineSearch.Solve(mls)
		stepSize, m.ObjX = mls.X, mls.ObjX

		xOld.Copy(x)
		gOld.Copy(g)

		x.Axpy(stepSize, d)
		m.Grad(x, g)
	}

	if m.Iter == sol.IterMax {
		err = errors.New("Maximum number of iterations reached")
	}
	return err
}
