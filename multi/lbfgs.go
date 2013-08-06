package multi

import (
	"errors"
	"fmt"
	"github.com/dane-unltd/linalg/mat"
	"github.com/dane-unltd/opt/uni"
	"math"
	"time"
)

type LBFGS struct {
	TolAbs, TolRel float64
	IterMax        int
	TimeMax        time.Duration
	Mem            int
	LineSearch     uni.Solver
}

func NewLBFGS() *LBFGS {
	s := &LBFGS{
		TolRel:     1e-3,
		TolAbs:     1e-3,
		IterMax:    1000,
		TimeMax:    time.Minute,
		Mem:        5,
		LineSearch: uni.NewArmijo(),
	}
	return s
}

func (sol LBFGS) Solve(m *Model) error {
	var err error

	startT := time.Now()

	stepSize := 1.0

	if m.X == nil {
		m.X = mat.NewVec(m.N)
	}
	if math.IsNaN(m.ObjX) {
		m.ObjX = m.Obj(m.X)
	}
	if m.GradX == nil {
		m.GradX = mat.NewVec(m.N)
		m.Grad(m.X, m.GradX)
	}

	gLin := 0.0

	normG0 := m.GradX.Nrm2()
	normG := m.GradX.Nrm2()

	S := make([]mat.Vec, sol.Mem)
	Y := make([]mat.Vec, sol.Mem)
	for i := 0; i < sol.Mem; i++ {
		S[i] = mat.NewVec(m.N)
		Y[i] = mat.NewVec(m.N)
	}

	d := mat.NewVec(m.N)

	xOld := mat.NewVec(m.N)
	gOld := mat.NewVec(m.N)
	sNew := mat.NewVec(m.N)
	yNew := mat.NewVec(m.N)

	alphas := mat.NewVec(sol.Mem)
	betas := mat.NewVec(sol.Mem)
	rhos := mat.NewVec(sol.Mem)

	xTemp := mat.NewVec(m.N)

	lineFun := func(step float64) float64 {
		xTemp.Copy(m.X)
		xTemp.Axpy(step, d)
		return m.Obj(xTemp)
	}
	mls := uni.NewModel(lineFun, nil)

	for ; m.Iter < sol.IterMax; m.Iter++ {

		d.Copy(m.GradX)
		if m.Iter > 0 {
			yNew.Sub(m.GradX, gOld)
			sNew.Sub(m.X, xOld)

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

		gLin = mat.Dot(d, m.GradX)

		m.Time = time.Since(startT)
		m.DoCallbacks()

		if m.Time > sol.TimeMax {
			err = errors.New("Time limit reached")
		}
		if normG < sol.TolAbs || normG/normG0 < sol.TolRel {
			break
		}

		mls.SetX(stepSize)
		mls.SetLB(0, m.ObjX, gLin)
		mls.SetUB()
		err = sol.LineSearch.Solve(mls)
		if err != nil {
			fmt.Println(err)
			d.Copy(m.GradX)
			d.Scal(-1)
			mls.SetX(stepSize)
			mls.SetLB(0, m.ObjX, -normG)
			mls.SetUB()
			err = sol.LineSearch.Solve(mls)
			if err != nil {
				fmt.Println(err)
				break
			}
		}
		stepSize, m.ObjX = mls.X, mls.ObjX

		xOld.Copy(m.X)
		gOld.Copy(m.GradX)

		m.X.Axpy(stepSize, d)
		m.Grad(m.X, m.GradX)
		normG = m.GradX.Nrm2()
	}

	if m.Iter == sol.IterMax {
		err = errors.New("Maximum number of iterations reached")
	}
	return err
}
