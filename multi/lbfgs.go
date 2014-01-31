package multi

import (
	"fmt"
	"github.com/dane-unltd/opt/uni"
	"github.com/gonum/blas/dblas"
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

	S := make([]dblas.Vector, sol.Mem)
	Y := make([]dblas.Vector, sol.Mem)
	for i := 0; i < sol.Mem; i++ {
		S[i] = dblas.NewVector(make([]float64, n))
		Y[i] = dblas.NewVector(make([]float64, n))
	}

	x := dblas.NewVector(r.X)
	g := dblas.NewVector(r.Grad)
	d := dblas.NewVector(make([]float64, n))

	xOld := dblas.NewVector(make([]float64, n))
	gOld := dblas.NewVector(make([]float64, n))
	sNew := dblas.NewVector(make([]float64, n))
	yNew := dblas.NewVector(make([]float64, n))

	alphas := make([]float64, sol.Mem)
	betas := make([]float64, sol.Mem)
	rhos := make([]float64, sol.Mem)

	lineFunc := NewLineFdF(obj, r.X, d.Data)
	lsInit := uni.NewSolution()

	notFirst := false
	for doUpdates(r, initialTime, upd) == 0 {
		dblas.Dcopy(g, d)
		if notFirst {
			dblas.Dcopy(g, yNew)
			dblas.Daxpy(-1, gOld, yNew)
			dblas.Dcopy(x, sNew)
			dblas.Daxpy(-1, xOld, sNew)

			temp := S[len(S)-1]
			copy(S[1:], S)
			S[0] = temp
			dblas.Dcopy(sNew, S[0])

			temp = Y[len(S)-1]
			copy(Y[1:], Y)
			Y[0] = temp
			dblas.Dcopy(yNew, Y[0])

			copy(rhos[1:], rhos)
			rhos[0] = 1 / dblas.Ddot(sNew, yNew)
			for i := 0; i < sol.Mem; i++ {
				alphas[i] = rhos[i] * dblas.Ddot(S[i], d)
				dblas.Daxpy(-alphas[i], Y[i], d)
			}
			for i := sol.Mem - 1; i >= 0; i-- {
				betas[i] = rhos[i] * dblas.Ddot(Y[i], d)
				dblas.Daxpy(alphas[i]-betas[i], S[i], d)
			}
		}
		notFirst = true

		dblas.Dscal(-1, d)
		gLin = dblas.Ddot(d, g)

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
			dblas.Dcopy(g, d)
			dblas.Dscal(-1, d)
			lsInit.SetLower(0, r.Obj, dblas.Dnrm2(g))
			lsRes = sol.LineSearch.OptimizeFdF(lineFunc, lsInit, wolfe)
			if lsRes.Status < 0 {
				fmt.Println("Linesearch:", lsRes.Status)
				r.Status = Status(lsRes.Status)

				break
			}
		}
		stepSize := lsRes.X
		r.Obj = lsRes.Obj

		dblas.Dcopy(x, xOld)
		dblas.Dcopy(g, gOld)

		dblas.Daxpy(stepSize, d, x)
		obj.DF(r.X, r.Grad)
	}
	return r
}
