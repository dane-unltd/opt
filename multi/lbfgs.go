package multi

import (
	"fmt"
	"github.com/dane-unltd/linalg/mat"
	"github.com/dane-unltd/opt/uni"
	"math"
)

type LBFGS struct {
	Mem        int
	LineSearch uni.Solver
}

func NewLBFGS() *LBFGS {
	s := &LBFGS{
		Mem:        5,
		LineSearch: uni.NewArmijo(),
	}
	return s
}

func (sol LBFGS) Solve(m *Model) Status {
	var status Status

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

	m.init()

	for {

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

		if status = m.update(); status != 0 {
			break
		}

		mls.SetX(stepSize)
		mls.SetLB(0, m.ObjX, gLin)
		mls.SetUB()
		lsStatus := sol.LineSearch.Solve(mls)
		if lsStatus < 0 {
			fmt.Println("Linesearch:", status)
			d.Copy(m.GradX)
			d.Scal(-1)
			mls.SetX(stepSize)
			mls.SetLB(0, m.ObjX, -m.GradX.Nrm2Sq())
			mls.SetUB()
			lsStatus = sol.LineSearch.Solve(mls)
			if lsStatus < 0 {
				fmt.Println("Linesearch:", status)
				status = Status(lsStatus)

				break
			}
		}
		stepSize, m.ObjX = mls.X, mls.ObjX

		xOld.Copy(m.X)
		gOld.Copy(m.GradX)

		m.X.Axpy(stepSize, d)
		m.Grad(m.X, m.GradX)
	}

	return status
}
