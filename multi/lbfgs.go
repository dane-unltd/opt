package multi

import (
	"fmt"
	"github.com/dane-unltd/opt/uni"
	"github.com/dane-unltd/goblas"
	"time"
)

type LBFGS struct {
	Termination
	Mem        int
	LineSearch uni.FdFOptimizer
}

func NewLBFGS() *LBFGS {
	s := &LBFGS{
		Termination: Termination{
			IterMax: 1000,
			TimeMax: time.Minute,
		},
		Mem:        5,
		LineSearch: uni.NewCubic(),
	}
	return s
}

func (sol LBFGS) OptimizeFdF(o FdF, in *Solution, upd ...Updater) *Result {
	r := NewResult(in)
	obj := fdfWrapper{r: r, fdf: o}
	r.initFdF(obj)

	upd = append(upd, sol.Termination)

	initialTime := time.Now()

	gLin := 0.0
	n := len(r.X)

	S := make([]goblas.Vector, sol.Mem)
	Y := make([]goblas.Vector, sol.Mem)
	for i := 0; i < sol.Mem; i++ {
		S[i] = goblas.NewVector(make([]float64, n))
		Y[i] = goblas.NewVector(make([]float64, n))
	}

	x := goblas.NewVector(r.X)
	g := goblas.NewVector(r.Grad)
	d := goblas.NewVector(make([]float64, n))

	xOld := goblas.NewVector(make([]float64, n))
	gOld := goblas.NewVector(make([]float64, n))
	sNew := goblas.NewVector(make([]float64, n))
	yNew := goblas.NewVector(make([]float64, n))

	alphas := make([]float64, sol.Mem)
	betas := make([]float64, sol.Mem)
	rhos := make([]float64, sol.Mem)

	lineFunc := NewLineFdF(obj, r.X, d.Data)
	lsInit := uni.NewSolution()

	notFirst := false
	for doUpdates(r, initialTime, upd) == 0 {
		goblas.Dcopy(g, d)
		if notFirst {
			goblas.Dcopy(g, yNew)
			goblas.Daxpy(-1, gOld, yNew)
			goblas.Dcopy(x, sNew)
			goblas.Daxpy(-1, xOld, sNew)

			temp := S[len(S)-1]
			copy(S[1:], S)
			S[0] = temp
			goblas.Dcopy(sNew, S[0])

			temp = Y[len(S)-1]
			copy(Y[1:], Y)
			Y[0] = temp
			goblas.Dcopy(yNew, Y[0])

			copy(rhos[1:], rhos)
			rhos[0] = 1 / goblas.Ddot(sNew, yNew)
			for i := 0; i < sol.Mem; i++ {
				alphas[i] = rhos[i] * goblas.Ddot(S[i], d)
				goblas.Daxpy(-alphas[i], Y[i], d)
			}
			for i := sol.Mem - 1; i >= 0; i-- {
				betas[i] = rhos[i] * goblas.Ddot(Y[i], d)
				goblas.Daxpy(alphas[i]-betas[i], S[i], d)
			}
		}
		notFirst = true

		goblas.Dscal(-1, d)
		gLin = goblas.Ddot(d, g)

		wolfe := uni.Wolfe{
			Armijo:    0.2,
			Curvature: 0.9,
			X0:        0,
			F0:        r.Obj,
			Deriv0:    gLin,
		}

		lsInit.Set(1.0)
		lsInit.SetLower(0, r.Obj, gLin)
		lsRes := sol.LineSearch.OptimizeFdF(lineFunc, lsInit, wolfe)
		if lsRes.Status < 0 {
			fmt.Println("Linesearch:", lsRes.Status)
			goblas.Dcopy(g, d)
			goblas.Dscal(-1, d)
			lsInit.SetLower(0, r.Obj, goblas.Dnrm2(g))
			lsRes = sol.LineSearch.OptimizeFdF(lineFunc, lsInit, wolfe)
			if lsRes.Status < 0 {
				fmt.Println("Linesearch:", lsRes.Status)
				r.Status = Status(lsRes.Status)

				break
			}
		}
		stepSize := lsRes.X
		r.Obj = lsRes.Obj

		goblas.Dcopy(x, xOld)
		goblas.Dcopy(g, gOld)

		goblas.Daxpy(stepSize, d, x)
		obj.DF(r.X, r.Grad)
	}
	return r
}
