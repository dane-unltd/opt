package multi

import (
	"fmt"
	"github.com/dane-unltd/opt/uni"
	"github.com/gonum/blas/dbw"
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

	S := make([]dbw.Vector, sol.Mem)
	Y := make([]dbw.Vector, sol.Mem)
	for i := 0; i < sol.Mem; i++ {
		S[i] = dbw.NewVector(make([]float64, n))
		Y[i] = dbw.NewVector(make([]float64, n))
	}

	x := dbw.NewVector(r.X)
	g := dbw.NewVector(r.Grad)
	d := dbw.NewVector(make([]float64, n))

	xOld := dbw.NewVector(make([]float64, n))
	gOld := dbw.NewVector(make([]float64, n))
	sNew := dbw.NewVector(make([]float64, n))
	yNew := dbw.NewVector(make([]float64, n))

	alphas := make([]float64, sol.Mem)
	betas := make([]float64, sol.Mem)
	rhos := make([]float64, sol.Mem)

	lineFunc := NewLineFdF(obj, r.X, d.Data)
	lsInit := uni.NewSolution()

	notFirst := false
	for doUpdates(r, initialTime, upd) == 0 {
		dbw.Copy(g, d)
		if notFirst {
			dbw.Copy(g, yNew)
			dbw.Axpy(-1, gOld, yNew)
			dbw.Copy(x, sNew)
			dbw.Axpy(-1, xOld, sNew)

			temp := S[len(S)-1]
			copy(S[1:], S)
			S[0] = temp
			dbw.Copy(sNew, S[0])

			temp = Y[len(S)-1]
			copy(Y[1:], Y)
			Y[0] = temp
			dbw.Copy(yNew, Y[0])

			copy(rhos[1:], rhos)
			rhos[0] = 1 / dbw.Dot(sNew, yNew)
			for i := 0; i < sol.Mem; i++ {
				alphas[i] = rhos[i] * dbw.Dot(S[i], d)
				dbw.Axpy(-alphas[i], Y[i], d)
			}
			for i := sol.Mem - 1; i >= 0; i-- {
				betas[i] = rhos[i] * dbw.Dot(Y[i], d)
				dbw.Axpy(alphas[i]-betas[i], S[i], d)
			}
		}
		notFirst = true

		dbw.Scal(-1, d)
		gLin = dbw.Dot(d, g)

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
			dbw.Copy(g, d)
			dbw.Scal(-1, d)
			lsInit.SetLower(0, r.Obj, dbw.Nrm2(g))
			lsRes = sol.LineSearch.OptimizeFdF(lineFunc, lsInit, wolfe)
			if lsRes.Status < 0 {
				fmt.Println("Linesearch:", lsRes.Status)
				r.Status = Status(lsRes.Status)

				break
			}
		}
		stepSize := lsRes.X
		r.Obj = lsRes.Obj

		dbw.Copy(x, xOld)
		dbw.Copy(g, gOld)

		dbw.Axpy(stepSize, d, x)
		obj.DF(r.X, r.Grad)
	}
	return r
}
