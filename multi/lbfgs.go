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

	if m.x == nil {
		m.x = mat.NewVec(m.n)
	}
	if math.IsNaN(m.objX) {
		m.objX = m.obj(m.x)
	}
	if m.gradX == nil {
		m.gradX = mat.NewVec(m.n)
		m.grad(m.x, m.gradX)
	}

	gLin := 0.0

	S := make([]mat.Vec, sol.Mem)
	Y := make([]mat.Vec, sol.Mem)
	for i := 0; i < sol.Mem; i++ {
		S[i] = mat.NewVec(m.n)
		Y[i] = mat.NewVec(m.n)
	}

	d := mat.NewVec(m.n)

	xOld := mat.NewVec(m.n)
	gOld := mat.NewVec(m.n)
	sNew := mat.NewVec(m.n)
	yNew := mat.NewVec(m.n)

	alphas := mat.NewVec(sol.Mem)
	betas := mat.NewVec(sol.Mem)
	rhos := mat.NewVec(sol.Mem)

	xTemp := mat.NewVec(m.n)

	lineFun := func(step float64) float64 {
		xTemp.Copy(m.x)
		xTemp.Axpy(step, d)
		return m.obj(xTemp)
	}
	mls := uni.NewModel(lineFun, nil)

	for ; m.iter < sol.IterMax; m.iter++ {

		d.Copy(m.gradX)
		if m.iter > 0 {
			yNew.Sub(m.gradX, gOld)
			sNew.Sub(m.x, xOld)

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

		gLin = mat.Dot(d, m.gradX)

		m.time = time.Since(startT)
		m.DoCallbacks()

		if m.time > sol.TimeMax {
			err = errors.New("Time limit reached")
		}
		if gLin > -sol.Tol {
			break
		}

		mls.SetX(stepSize)
		mls.SetLB(0, m.objX, gLin)
		mls.SetUB()
		_ = sol.LineSearch.Solve(mls)
		stepSize, m.objX = mls.X(), mls.ObjX()

		xOld.Copy(m.x)
		gOld.Copy(m.gradX)

		m.x.Axpy(stepSize, d)
		m.grad(m.x, m.gradX)
	}

	if m.iter == sol.IterMax {
		err = errors.New("Maximum number of iterations reached")
	}
	return err
}
