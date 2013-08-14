package multi

import (
	"fmt"
	"github.com/dane-unltd/linalg/mat"
	"github.com/dane-unltd/opt/uni"
)

type LBFGS struct {
	Mem        int
	LineSearch uni.DerivSolver
}

func NewLBFGS() *LBFGS {
	s := &LBFGS{
		Mem:        5,
		LineSearch: uni.DerivWrapper{uni.NewArmijo()},
	}
	return s
}

func (sol LBFGS) Solve(o Grad, in *Solution, p *Params, u ...Updater) *Result {
	r := NewResult(in)
	obj := ObjGradWrapper{r: r, o: o}
	r.initGrad(obj)
	h := newHelper(r.Solution, u)

	stepSize := 1.0
	gLin := 0.0
	n := len(r.X)

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

	lineFunc := NewLineFuncDeriv(obj, r.X, d)
	lsInit := uni.NewSolution()
	lsParams := uni.NewParams()

	for ; r.Status == NotTerminated; h.update(r, p) {
		d.Copy(r.GradX)
		if r.Iter > 0 {
			yNew.Sub(r.GradX, gOld)
			sNew.Sub(r.X, xOld)

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

		gLin = mat.Dot(d, r.GradX)

		lsInit.SetX(stepSize)
		lsInit.SetLB(0, r.ObjX, gLin)
		lsRes := sol.LineSearch.Solve(lineFunc, lsInit, lsParams)
		if lsRes.Status < 0 {
			fmt.Println("Linesearch:", lsRes.Status)
			d.Copy(r.GradX)
			d.Scal(-1)
			lsInit.SetLB(0, r.ObjX, -r.GradX.Nrm2Sq())
			lsRes = sol.LineSearch.Solve(lineFunc, lsInit, lsParams)
			if lsRes.Status < 0 {
				fmt.Println("Linesearch:", lsRes.Status)
				r.Status = Status(lsRes.Status)

				break
			}
		}
		stepSize, r.ObjX = lsRes.X, lsRes.ObjX

		xOld.Copy(r.X)
		gOld.Copy(r.GradX)

		r.X.Axpy(stepSize, d)
		obj.ValGrad(r.X, r.GradX)
	}
	return r
}
